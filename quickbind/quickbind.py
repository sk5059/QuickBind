import torch
from torch import nn
import pytorch_lightning as pl
from commons.modified_of_modules import (
    InputEmbedder, EvoformerStack, StructureModule,
    BackboneUpdate, GatedInvariantPointAttention,
    FullEvoformerStack
)
from openfold.model.structure_module import StructureModuleTransition, InvariantPointAttention
from openfold.model.primitives import Linear, LayerNorm
from openfold.utils.rigid_utils import Rigid, Rotation
from functools import partial
from openfold.model.heads import DistogramHead
from openfold.utils.loss import distogram_loss
from quickbind_loss import QuickBindLoss

class QuickBind(nn.Module):
    def __init__(self, config=None):
        """
        Initialize the QuickBind model.
        
        Args:
            config (dict or str, optional): Configuration dictionary or path to YAML config file.
                If None, the default config is used.
        """
        super(QuickBind, self).__init__()
        
        # Load configuration
        if config is None:
            # Use default config
            config = get_default_config()
        elif isinstance(config, str):
            # Load config from yaml file
            config = load_yaml_config(config)
        
        # Extract config sections
        input_cfg = config["input_embed"]
        evo_cfg = config["evoformer"]
        struct_cfg = config["structure"]
        recycling_cfg = config["recycling"]
        aux_cfg = config["aux_heads"]
        
        # Initialize input embedder
        self.inputembedder = InputEmbedder(
            input_cfg["aa_feat"], 
            input_cfg["lig_atom_feat"], 
            input_cfg["c_emb"], 
            input_cfg["c_s"], 
            input_cfg["c_z"], 
            input_cfg["use_op_edge_embed"], 
            input_cfg["use_pairwise_dist"], 
            input_cfg["use_radial_basis"],
            input_cfg["use_rel_pos"], 
            input_cfg["use_multimer_rel_pos"], 
            input_cfg["mask_off_diagonal"], 
            input_cfg["one_hot_adj"], 
            input_cfg["use_topological_distance"]
        )

        # EVOFORMER
        self.no_evo_blocks = evo_cfg["no_evo_blocks"]
        self.chunk_size = evo_cfg["chunk_size"]
        
        if self.no_evo_blocks > 0:
            if evo_cfg["use_full_evo_stack"]:
                self.evoformer = FullEvoformerStack(
                    input_cfg["c_s"], 
                    input_cfg["c_z"], 
                    evo_cfg["c_hidden_msa_att"], 
                    evo_cfg["c_hidden_opm"], 
                    evo_cfg["c_hidden_mul"], 
                    evo_cfg["c_hidden_pair_att"], 
                    evo_cfg["c_s_out"],
                    evo_cfg["no_heads_msa"], 
                    evo_cfg["no_heads_pair"], 
                    evo_cfg["no_evo_blocks"], 
                    evo_cfg["transition_n"], 
                    evo_cfg["msa_dropout"],
                    evo_cfg["pair_dropout"], 
                    opm_first=evo_cfg["opm_first"]
                )
            else:
                self.evoformer = EvoformerStack(
                    input_cfg["c_s"], 
                    input_cfg["c_z"], 
                    evo_cfg["c_hidden_msa_att"], 
                    evo_cfg["c_hidden_opm"], 
                    evo_cfg["c_hidden_mul"], 
                    evo_cfg["c_hidden_pair_att"], 
                    evo_cfg["c_s_out"],
                    evo_cfg["no_heads_msa"], 
                    evo_cfg["no_heads_pair"], 
                    evo_cfg["no_evo_blocks"], 
                    evo_cfg["transition_n"], 
                    evo_cfg["msa_dropout"],
                    evo_cfg["pair_dropout"], 
                    opm_first=evo_cfg["opm_first"]
                )
                
        # STRUCTURE MODULE
        self.layer_norm_s = LayerNorm(evo_cfg["c_s_out"])
        self.layer_norm_z = LayerNorm(input_cfg["c_z"])
        self.linear_in = Linear(evo_cfg["c_s_out"], evo_cfg["c_s_out"])
        self.num_struct_blocks = struct_cfg["num_struct_blocks"]
        self.share_ipa_weights = struct_cfg["share_ipa_weights"]
        
        if self.share_ipa_weights:
            self.structure_module_block = StructureModule(
                    evo_cfg["c_s_out"], 
                    input_cfg["c_z"], 
                    struct_cfg["c_hidden"], 
                    struct_cfg["no_heads"], 
                    struct_cfg["no_qk_points"], 
                    struct_cfg["no_v_points"], 
                    struct_cfg["dropout_rate"],
                    struct_cfg["no_transition_layers"], 
                    struct_cfg["sum_pool"], 
                    struct_cfg["mean_pool"], 
                    struct_cfg["att_update"], 
                    struct_cfg["use_gated_ipa"], 
                    struct_cfg["construct_frames"]
            )
        else:
            if struct_cfg["use_gated_ipa"]:
                self.ipa_blocks = nn.ModuleList([
                    GatedInvariantPointAttention(
                        input_cfg["c_s"], 
                        input_cfg["c_z"], 
                        struct_cfg["c_hidden"], 
                        struct_cfg["no_heads"], 
                        struct_cfg["no_qk_points"], 
                        struct_cfg["no_v_points"]
                    ) for _ in range(struct_cfg["num_struct_blocks"])
                ])
            else:
                self.ipa_blocks = nn.ModuleList([
                    InvariantPointAttention(
                        input_cfg["c_s"], 
                        input_cfg["c_z"], 
                        struct_cfg["c_hidden"], 
                        struct_cfg["no_heads"], 
                        struct_cfg["no_qk_points"], 
                        struct_cfg["no_v_points"]
                    ) for _ in range(struct_cfg["num_struct_blocks"])
                ])
            self.ipa_dropout = nn.Dropout(struct_cfg["dropout_rate"])
            self.layer_norm_ipa = LayerNorm(input_cfg["c_s"])
            self.transition = StructureModuleTransition(
                input_cfg["c_s"], 
                struct_cfg["no_transition_layers"], 
                struct_cfg["dropout_rate"]
            )
            self.bb_update = BackboneUpdate(
                input_cfg["c_s"], 
                struct_cfg["sum_pool"], 
                struct_cfg["mean_pool"], 
                struct_cfg["att_update"], 
                struct_cfg["construct_frames"]
            )

        # RECYCLING EMBEDDINGS
        self.recycle = recycling_cfg["recycle"]
        self.recycle_iters = recycling_cfg["recycle_iters"]
        if self.recycle:
            self.layer_norm_s_recycle = LayerNorm(input_cfg["c_s"])
            self.layer_norm_z_recycle = LayerNorm(input_cfg["c_z"])
            self.linear_z_recycle = Linear(1, input_cfg["c_z"])

        # AUXILIARY HEADS
        self.use_aux_head = aux_cfg["use_aux_head"]
        self.use_lig_aux_head = aux_cfg["use_lig_aux_head"]
        if self.use_aux_head:
            self.distogram = DistogramHead(input_cfg["c_z"], aux_cfg["no_dist_bins"])
        if self.use_lig_aux_head:
            self.lig_distogram = DistogramHead(input_cfg["c_z"], aux_cfg["no_dist_bins_lig"])

        # Other settings
        self.communicate = struct_cfg["communicate"]
        if self.communicate:
            self.linear_a_i = Linear(input_cfg["c_s"], input_cfg["c_z"])
            self.linear_b_i = Linear(input_cfg["c_s"], input_cfg["c_z"])
            self.linear_dist = Linear(1, input_cfg["c_z"])

        self.construct_frames = struct_cfg["construct_frames"]
        self.blackhole_init = struct_cfg["blackhole_init"]
        self.pooled_update = bool(struct_cfg["sum_pool"] or struct_cfg["mean_pool"] or struct_cfg["att_update"])
        self.output_s = aux_cfg["output_s"]
            
    def forward(self, aatype, lig_atom_features, adj, rec_mask, lig_mask, N, t_rec, C, t_lig, ri, pseudo_N, pseudo_C):
        is_grad_enabled = torch.is_grad_enabled()
        # RECYCLING
        s_prev, z_prev = None, None
        t_prev = torch.cat([t_rec, t_lig], dim=-2)
        mask = torch.cat([rec_mask, lig_mask], dim=-1).to(dtype=torch.float32)
        edge_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        for iteration in range(self.recycle_iters):
            self.is_final_iter = (iteration == (self.recycle_iters-1))
            with torch.set_grad_enabled(is_grad_enabled and self.is_final_iter):
                if self.is_final_iter and torch.is_autocast_enabled(): # Sidestep AMP bug (PyTorch issue #65766)
                    torch.clear_autocast_cache()
                if self.output_s:
                    outputs, s, s_prev, z_prev, t_prev, s_pre_struct = self.iteration(
                        aatype, lig_atom_features, adj, s_prev, z_prev, t_prev, ri, mask, edge_mask,
                        N, t_rec, C, rec_mask, lig_mask, pseudo_N, pseudo_C
                    )
                else:
                    outputs, s, s_prev, z_prev, t_prev = self.iteration(
                        aatype, lig_atom_features, adj, s_prev, z_prev, t_prev, ri, mask, edge_mask,
                        N, t_rec, C, rec_mask, lig_mask, pseudo_N, pseudo_C
                    )
                if not self.is_final_iter: del outputs, s

        if self.use_aux_head and self.use_lig_aux_head:
            distogram_logits_full = self.distogram(z_prev)
            distogram_logits_lig = self.lig_distogram(z_prev[:, rec_mask.shape[-1]:, rec_mask.shape[-1]:])
            distogram_logits = (distogram_logits_full, distogram_logits_lig)
            if self.output_s:
                return outputs, distogram_logits, s_pre_struct
            else:
                return outputs, distogram_logits
        elif self.use_aux_head:
            distogram_logits = self.distogram(z_prev)
            if self.output_s:
                return outputs, distogram_logits, s_pre_struct
            else:
                return outputs, distogram_logits
        elif self.use_lig_aux_head:
            distogram_logits = self.lig_distogram(z_prev[:, rec_mask.shape[-1]:, rec_mask.shape[-1]:])
            if self.output_s:
                return outputs, distogram_logits, s_pre_struct
            else:
                return outputs, distogram_logits
        else:
            if self.output_s:
                return outputs, s_pre_struct
            else:
                return outputs


class QuickBind_PL(pl.LightningModule):
    def __init__(self, config=None):
        """
        Initialize the QuickBind PyTorch Lightning model.
        
        Args:
            config (dict or str, optional): Configuration dictionary or path to YAML config file.
                If None, the default config is used.
        """
        super().__init__()
        
        # Load configuration
        if config is None:
            # Use default config
            config = get_default_config()
        elif isinstance(config, str):
            # Load config from yaml file
            config = load_yaml_config(config)
            
        self.config = config
        
        # Create QuickBind model and loss function with the same config
        self.model = QuickBind(config)
        self.loss = QuickBindLoss(config)
        
        # Extract relevant configurations
        aux_cfg = config["aux_heads"]
        training_cfg = config["training"]
        
        self.use_aux_head = aux_cfg["use_aux_head"]
        self.use_lig_aux_head = aux_cfg["use_lig_aux_head"]
        self.lr = training_cfg["lr"]
        self.weight_decay = training_cfg["weight_decay"]
        
        self.save_hyperparameters()

    def forward(self, batch):
        return self.model(*batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, idx):
        batch, t_true = batch
        _, _, _, rec_mask, lig_mask, _, _, _, _, _, _, _ = batch
        if self.use_aux_head or self.use_lig_aux_head:
            outputs, distogram_logits = self.model(*batch)
        else:
            outputs = self.model(*batch)
            distogram_logits = None
        loss, (
            lig_lig_loss, lig_rec_loss, aux_loss, steric_clash_loss, full_distogram_loss
        ), rmsd = self.loss(t_true, outputs, lig_mask, rec_mask, distogram_logits)
        self.log('train_loss', loss)
        self.log('train_lig_lig_loss', lig_lig_loss)
        self.log('train_lig_rec_loss', lig_rec_loss)
        self.log('train_aux_loss', aux_loss)
        self.log('train_steric_clash_loss', steric_clash_loss)
        self.log('train_full_distogram_loss', full_distogram_loss)
        self.log('train_rmsd', rmsd)
        return loss

    def validation_step(self, batch, idx):
        batch, t_true = batch
        _, _, _, rec_mask, lig_mask, _, _, _, _, _, _, _ = batch
        if self.use_aux_head or self.use_lig_aux_head:
            outputs, distogram_logits = self.model(*batch)
        else:
            outputs = self.model(*batch)
            distogram_logits = None
        loss, (
            lig_lig_loss, lig_rec_loss, aux_loss, steric_clash_loss, full_distogram_loss
        ), rmsd = self.loss(t_true, outputs, lig_mask, rec_mask, distogram_logits)
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_lig_lig_loss', lig_lig_loss, sync_dist=True)
        self.log('val_lig_rec_loss', lig_rec_loss, sync_dist=True)
        self.log('val_aux_loss', aux_loss, sync_dist=True)
        self.log('val_steric_clash_loss', steric_clash_loss, sync_dist=True)
        self.log('val_full_distogram_loss', full_distogram_loss, sync_dist=True)
        self.log('val_rmsd', rmsd, sync_dist=True)
        return loss
                
    def iteration(
            self, aatype, lig_atom_features, adj, s_prev, z_prev, t_prev, ri, mask, edge_mask,
            N, t_rec, C, rec_mask, lig_mask, pseudo_N, pseudo_C
    ):
        # INPUT EMBEDDINGS
        s, z = self.inputembedder(aatype, lig_atom_features, t_prev, edge_mask, adj, ri)
        t_lig = t_prev[:, rec_mask.shape[-1]:, :]
        if self.construct_frames and not self.blackhole_init:
            rigids = Rigid.cat(
                [
                    Rigid.from_3_points(N, t_rec, C),
                    Rigid.from_3_points(pseudo_N, t_lig, pseudo_C)
                ], dim=1
            )
        else:
            rigids = Rigid.cat(
                [
                    Rigid.from_3_points(N, t_rec, C),
                    Rigid(
                        rots = Rotation.identity(
                            shape=t_lig.shape[:-1], dtype = torch.float32, device=t_lig.device, fmt="quat"
                        ), trans = t_lig
                    )
                ], dim=1
            )

        # RECYCLING EMBEDDINGS
        if None not in [s_prev, z_prev]:
            s_prev = self.layer_norm_s_recycle(s_prev)
            pairwise_distance_prev = (torch.cdist(t_prev, t_prev, p=2) * edge_mask).unsqueeze(-1).to(dtype=torch.float32)
            z_prev = self.linear_z_recycle(pairwise_distance_prev) + self.layer_norm_z_recycle(z_prev)
            s = s + s_prev
            z = z + z_prev

        # EVOFORMER
        if self.no_evo_blocks > 0:
            s = s.unsqueeze(-3)
            msa_mask = mask.unsqueeze(-2)
            s, z = self.evoformer(
                    s, z,
                    msa_mask=msa_mask,
                    pair_mask=edge_mask,
                    chunk_size=self.chunk_size
            )
        if self.recycle:
            s_prev, z_prev = s, z

        if self.output_s:
            s_pre_struct = s

        # STRUCTURE MODULE
        s = self.layer_norm_s(s)
        z = self.layer_norm_z(z)
        s = self.linear_in(s)

        out = []
        if self.share_ipa_weights:
            blocks = [
                partial(
                    self.structure_module_block, mask=mask, rec_mask=rec_mask, lig_mask=lig_mask
                ) for _ in range(self.num_struct_blocks)
            ]
            for block in blocks:
                s, z, new_trans = block(s, z, rigids)
                if not self.pooled_update:
                    new_trans = new_trans[:, rec_mask.shape[-1]:, :]
                new_trans = new_trans * lig_mask.unsqueeze(-1)
                if self.construct_frames:
                    rigids_ligand = rigids[:, rec_mask.shape[-1]:]
                    rigids_protein = rigids[:, :rec_mask.shape[-1]]
                    rigids_ligand_updated = rigids_ligand.compose_q_update_vec(new_trans)
                    updated_rigids = Rigid.cat([rigids_protein, rigids_ligand_updated], dim=1)
                else:
                    update = torch.cat([torch.zeros_like(rigids.get_trans()[:, :rec_mask.shape[-1], :]), new_trans], dim=-2)
                    updated_rigids = Rigid(
                        rots = rigids.get_rots(),
                        trans = rigids.get_trans() + update
                    )
                rigids = updated_rigids
                out.append(updated_rigids)
                if self.construct_frames:
                    rigids = rigids.stop_rot_gradient()
        else:
            for ipa in self.ipa_blocks:
                s = s + ipa(s, z, rigids, mask)
                s = self.ipa_dropout(s)
                s = self.layer_norm_ipa(s)
                s = self.transition(s)
                new_trans = self.bb_update(s, rec_mask, lig_mask)
                if not self.pooled_update:
                    new_trans = new_trans[:, rec_mask.shape[-1]:, :]
                new_trans = new_trans * lig_mask.unsqueeze(-1)
                if self.construct_frames:
                    rigids_ligand = rigids[:, rec_mask.shape[-1]:]
                    rigids_protein = rigids[:, :rec_mask.shape[-1]]
                    rigids_ligand_updated = rigids_ligand.compose_q_update_vec(new_trans)
                    updated_rigids = Rigid.cat([rigids_protein, rigids_ligand_updated], dim=1)                      
                else:
                    update = torch.cat([torch.zeros_like(rigids.get_trans()[:, :rec_mask.shape[-1], :]), new_trans], dim=-2)
                    updated_rigids = Rigid(
                        rots = rigids.get_rots(),
                        trans = rigids.get_trans() + update
                    )
                rigids = updated_rigids
                out.append(updated_rigids)
                if self.communicate:
                    ti = rigids.get_trans()
                    a_i = self.linear_a_i(s)
                    b_i = self.linear_b_i(s)
                    pair_emb = a_i[..., None, :] + b_i[..., None, :, :]
                    dist = (torch.cdist(ti, ti, p=2) * edge_mask).unsqueeze(-1).to(dtype=torch.float32)
                    pairwise_distance = self.linear_dist(dist)
                    pair_emb = pair_emb + pairwise_distance
                    z = z + pair_emb
                if self.construct_frames:
                    rigids = rigids.stop_rot_gradient()

        if self.recycle: t_prev = rigids.get_trans()

        if (self.use_aux_head or self.use_lig_aux_head) and self.is_final_iter:
            if self.output_s:
                return out, s, s_prev, z, t_prev, s_pre_struct
            else:
                return out, s, s_prev, z, t_prev
        else:
            if self.output_s:
                return out, s, s_prev, z_prev, t_prev, s_pre_struct
            else:
                return out, s, s_prev, z_prev, t_prev