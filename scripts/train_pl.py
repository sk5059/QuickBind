import argparse
import yaml
import os
import glob
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader
from commons.utils import log, get_parameter_value
from dataset.dataimporter import DataImporter
from openfold.utils.seed import seed_globally
from quickbind import QuickBind_PL
from openfold.utils.rigid_utils import Rigid, Rotation

torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType(mode='r'), help='Path to YAML config file')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training from checkpoint')
    parser.add_argument('--id', type=str, default=None, help='W&B ID for resuming run')
    parser.add_argument('--finetune', type=bool, default=False, help='Finetune from pretrained model')
    return parser.parse_args()

def prepare_rigid_coords(coords, pseudo_n, pseudo_c, construct_frames):
    if construct_frames:
        return Rigid.from_3_points(pseudo_n, coords, pseudo_c)
    else:
        return Rigid(
            rots=Rotation.identity(shape=coords.shape[:-1], dtype=torch.float32, fmt="quat"),
            trans=coords
        )

def collate(data, construct_frames, use_topological_distance, one_hot_adj):
    assert len(data) == 1
    data = data[0]
    
    # Prepare protein data
    aatype = data.aatype.unsqueeze(0)
    rec_mask = torch.ones(data.aatype.shape[0]).unsqueeze(0)
    t_rec = data.c_alpha_coords.unsqueeze(0)
    N = data.n_coords.unsqueeze(0)
    C = data.c_coords.unsqueeze(0)
    
    # Prepare ligand data
    lig_atom_features = data.lig_atom_features.unsqueeze(0).to(dtype=torch.float32)
    lig_mask = torch.ones(data.lig_atom_features.shape[0]).unsqueeze(0)
    t_lig = data.lig_atom_coords.unsqueeze(0)
    t_true = data.true_lig_atom_coords.unsqueeze(0)
    
    # Handle adjacency representation
    if use_topological_distance:
        adj = torch.clamp(data.distance_matrix.unsqueeze(0), max=7)
        adj = torch.nn.functional.one_hot(adj.long(), num_classes=8).to(dtype=torch.float32)
    elif one_hot_adj:
        adj = data.adjacency_bo.unsqueeze(0).to(dtype=torch.int64)
    else:
        adj = data.adjacency.unsqueeze(0).unsqueeze(-1).to(dtype=torch.float32)
    
    # Prepare indices
    ri = data.residue_index.unsqueeze(0).to(dtype=torch.int64)
    chain_id = data.chain_ids_processed.unsqueeze(0).to(dtype=torch.int64)
    entity_id = data.entity_ids_processed.unsqueeze(0).to(dtype=torch.int64)
    sym_id = data.sym_ids_processed.unsqueeze(0).to(dtype=torch.int64)
    id_batch = (ri, chain_id, entity_id, sym_id)
    
    # Prepare frame reference points
    pseudo_N = data.pseudo_N.unsqueeze(0)
    pseudo_C = data.pseudo_C.unsqueeze(0)
    true_pseudo_N = data.true_pseudo_N.unsqueeze(0)
    true_pseudo_C = data.true_pseudo_C.unsqueeze(0)
    
    # Convert true coords to rigid frames if needed
    t_true = prepare_rigid_coords(t_true, true_pseudo_N, true_pseudo_C, construct_frames)

    model_inputs = (aatype, lig_atom_features, adj, rec_mask, lig_mask, 
                   N, t_rec, C, t_lig, id_batch, pseudo_N, pseudo_C)
    
    return model_inputs, t_true

def setup_data_loaders(config, construct_frames, use_topological_distance, one_hot_adj):
    log('Setting up data loaders')
    
    train_data = DataImporter(complex_names_path=config['train_names'], **config['dataset_params'])
    val_data = DataImporter(complex_names_path=config['val_names'], **config['dataset_params'])
    
    collate_fn = lambda x: collate(x, construct_frames, use_topological_distance, one_hot_adj)
    
    train_loader = DataLoader(
        train_data, 
        batch_size=config['batch_size'], 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['num_workers'], 
        prefetch_factor=12
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size=config['batch_size'],
        collate_fn=collate_fn,
        num_workers=config['num_workers']
    )
    
    return train_loader, val_loader, train_data.get_feature_dimensions()

def setup_model_and_trainer(config, feature_dims, wandb_logger):
    rec_feat_dim, lig_feat_dim = feature_dims
    
    model = QuickBind_PL(
        aa_feat=rec_feat_dim, 
        lig_atom_feat=lig_feat_dim, 
        **config['model_parameters'], 
        loss_config=config['loss_params'], 
        **config['optimizer_params'],
        chunk_size=None
    )
    
    # Setup callbacks
    checkpoint_dir = f'checkpoints/{config["name"]}'
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best_checkpoint',
        monitor='val_loss',
        save_top_k=3,
        mode="min",
    )
    checkpoint_callback.FILE_EXTENSION = ".pt"
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config.get('patience', 50)),
        checkpoint_callback,
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        logger=wandb_logger,
        accumulate_grad_batches=config['iters_to_accumulate'],
        precision=16,  # Mixed precision training
        gradient_clip_val=config.get('clip_grad', 0.0),
        callbacks=callbacks,
        default_root_dir=checkpoint_dir,
        deterministic=True, 
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        num_nodes=2
    )
    
    return model, trainer, checkpoint_dir

def train(config, args, wandb_logger):
    # Set seeds for reproducibility
    seed = config.get('seed', 0)
    seed_globally(seed)
    pl.seed_everything(seed, workers=True)
    
    # Adjust config for finetuning if needed
    if args.finetune:
        config['dataset_params']['crop_size'] = 512
    
    # Get model parameters
    one_hot_adj = get_parameter_value('one_hot_adj', config['model_parameters'])
    use_topological_distance = get_parameter_value('use_topological_distance', config['model_parameters'])
    construct_frames = get_parameter_value('construct_frames', config['model_parameters'])
    
    # Setup data loaders
    train_loader, val_loader, feature_dims = setup_data_loaders(
        config, construct_frames, use_topological_distance, one_hot_adj
    )
    
    # Setup model and trainer
    model, trainer, checkpoint_dir = setup_model_and_trainer(config, feature_dims, wandb_logger)
    
    # Create checkpoint dir if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Train model based on mode
    if args.resume:
        checkpoints = glob.glob(f'{checkpoint_dir}/best_checkpoint*.pt')
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        trainer.fit(model, train_loader, val_loader, ckpt_path=latest_checkpoint)
    elif args.finetune:
        model_state = torch.load(f'{checkpoint_dir}/best_checkpoint.pt')
        model.load_state_dict(model_state['state_dict'])
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader, val_loader)

def main():
    args = parse_args()
    
    # Optimize CUDA performance
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Load config
    config = yaml.safe_load(args.config)
    
    # Setup logger
    if args.resume:
        assert args.id is not None, log('Must provide W&B ID when resuming!')
        wandb_logger = WandbLogger(
            name=config['name'], 
            project=config['wandb']['project'], 
            id=args.id
        )
    else:
        wandb_logger = WandbLogger(
            name=config['name'], 
            project=config['wandb']['project']
        )
    
    train(config, args, wandb_logger)

if __name__ == '__main__':
    main()