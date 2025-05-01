import argparse
import torch
from torch import nn
import numpy as np
import pickle
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from openfold.model.primitives import Linear, LayerNorm
from commons.utils import log
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--ckpt', type=str, default=None, help='Checkpoint for evaluation')
    return p.parse_args()

class BindingAffinityPredictor(nn.Module):
    def __init__(self, c_s=64):
        super().__init__()
        self.norm = LayerNorm(c_s).to(device='cuda')
        self.affinity_in = nn.Sequential(
            Linear(c_s, c_s),
            nn.SiLU(),
            Linear(c_s, c_s),
        ).to(device='cuda')
        self.binding_affinity_head = nn.Sequential(
            Linear(c_s, c_s),
            nn.ReLU(),
            Linear(c_s, c_s//2),
            nn.ReLU(),
            Linear(c_s//2, 1, init="final"),
        ).to(device='cuda')
    
    def forward(self, s):
        mask = torch.ones_like(s)
        mask[s == 0] = 0
        s = self.norm(s)
        s_aff = self.affinity_in(s)
        s_aff = torch.sum(s_aff, dim=-2) / torch.sum(mask, dim=-2)
        pred_affinity = self.binding_affinity_head(s_aff)
        return pred_affinity

class BindingAffinityData(Dataset):
    def __init__(self, data, names, target_dict):
        self.data = data
        self.target_dict = target_dict
        self.names = names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.names[idx]
        x = self.data[idx].to(device='cuda')
        y = torch.tensor(self.target_dict[name]).unsqueeze(-1).to(device='cuda')
        return x, y

def load_data(output_path, binding_dict):
    outputs = torch.load(output_path)
    affinities = {k: v for k, v in binding_dict.items() if k in outputs['names']}
    s = pad_sequence([s.squeeze() for s in outputs['s_pre_struct']], batch_first=True)
    return s, outputs['names'], affinities

def train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs, early_stopping_patience):
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.to(dtype=float), targets.to(dtype=float))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss = val_running_loss / len(valid_loader)
        log(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0            
            torch.save(model.state_dict(), 'curr_ckpt.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                log("Early stopping triggered.")
                break

def get_predictions(model, test_loader):
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions.append(outputs.item())
            true_values.append(targets.item())
    return predictions, true_values

def compute_metrics(true_values, predicted_values):
    rmsd = np.sqrt(mean_squared_error(true_values, predicted_values))
    pearson_corr, _ = pearsonr(true_values, predicted_values)    
    spearman_corr, _ = spearmanr(true_values, predicted_values)    
    mae = mean_absolute_error(true_values, predicted_values)
    return rmsd, pearson_corr, spearman_corr, mae

if __name__ == '__main__':
    args = parse_arguments()
    log(f'Using seed {args.seed}.')
    torch.manual_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    np.random.seed(args.seed)

    batch_size = 64
    learning_rate = 0.01
    num_epochs = 1000
    patience = 50
    base_dir = 'checkpoints/quickbind_default'
    save_dir = f'{base_dir}/binding_affinity_prediction'

    log('Loading binding affinity data.')
    with open('data/binding_affinity_dict.pkl', 'rb') as f:
        binding_affinity_dict = pickle.load(f)

    is_train_mode = args.ckpt is None
    if is_train_mode:
        log('Loading training data.')
        train_s, train_names, train_affinities = load_data(
            f'{base_dir}/train_predictions-w-single-rep.pt', 
            binding_affinity_dict
        )
        train_dataset = BindingAffinityData(train_s, train_names, train_affinities)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    
        log('Loading validation data.')
        valid_s, valid_names, valid_affinities = load_data(
            f'{base_dir}/val_predictions-w-single-rep.pt',
            binding_affinity_dict
        )
        valid_dataset = BindingAffinityData(valid_s, valid_names, valid_affinities)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
    log('Loading test data.')
    test_s, test_names, test_affinities = load_data(
        f'{base_dir}/predictions-w-single-rep.pt',
        binding_affinity_dict
    )
    test_dataset = BindingAffinityData(test_s, test_names, test_affinities)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = BindingAffinityPredictor(64)

    if is_train_mode:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        log('Starting model training.')
        train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs, patience)
        model.load_state_dict(torch.load('curr_ckpt.pt'))
    else:
        log(f'Loading model from checkpoint: {args.ckpt}')
        model.load_state_dict(torch.load(args.ckpt))

    log('Evaluating model on test data.')
    predictions, true_values = get_predictions(model, test_loader)
    rmsd, pearson_corr, spearman_corr, mae = compute_metrics(true_values, predictions)
    
    log(f'RMSD: {rmsd:.4f}')
    log(f'Pearson Correlation: {pearson_corr:.4f}')
    log(f'Spearman Correlation: {spearman_corr:.4f}')
    log(f'MAE: {mae:.4f}')

    if is_train_mode:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = f'{save_dir}/ckpt_seed{args.seed}.pt'
        torch.save(model.state_dict(), save_path)
        log(f'Model saved to {save_path}')