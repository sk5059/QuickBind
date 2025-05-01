"""
Loss functions for QuickBind model.
"""
import torch
from torch import nn
from openfold.utils.loss import distogram_loss

class QuickBindLoss(nn.Module):
    def __init__(self, config):
        """
        Initialize the QuickBind loss function.
        """
        super().__init__()
        loss_cfg = config["loss"]
        aux_cfg = config["aux_heads"]
        
        self.lig_lig_loss_weight = loss_cfg["lig_lig_loss_weight"]
        self.lig_rec_loss_weight = loss_cfg["lig_rec_loss_weight"]
        self.aux_loss_weight = loss_cfg["aux_loss_weight"]
        self.steric_clash_loss_weight = loss_cfg["steric_clash_loss_weight"]
        self.full_distogram_loss_weight = loss_cfg["full_distogram_loss_weight"]
        self.eps = loss_cfg["eps"]
        self.clamp_distance = loss_cfg["clamp_distance"]
        self.use_aux_head = aux_cfg["use_aux_head"]
        self.use_lig_aux_head = aux_cfg["use_lig_aux_head"]

    def compute_fape_lig_lig(
        self,
        pred_frames,
        target_frames,
        pred_positions,
        target_positions,
        mask
    ):
        """
        Compute FAPE (Frame Aligned Point Error) for ligand-ligand interactions.
        
        Args:
            pred_frames: Predicted rigid frames
            target_frames: Target rigid frames
            pred_positions: Predicted positions
            target_positions: Target positions
            mask: Mask for valid positions
            
        Returns:
            torch.Tensor: FAPE loss for ligand-ligand interactions
        """
        # [*, N_frames, N_frames, 3]
        local_pred_pos = pred_frames.invert()[..., None].apply(
            pred_positions[..., None, :, :],
        )
        local_target_pos = target_frames.invert()[..., None].apply(
            target_positions[..., None, :, :],
        )
        error = torch.sqrt(
            torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + self.eps
        )
        edge_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        error = error * edge_mask
        error = torch.sum(torch.sum(error, dim=-1), dim=-1) / torch.sum(mask, dim=-1)**2
        return torch.mean(error)

    def compute_fape_lig_rec(
        self,
        pred_positions,
        target_positions,
        protein_frames,
        lig_mask,
        rec_mask,
        clamp_distance = None, 
    ):
        """
        Compute FAPE for ligand-receptor interactions.
        
        Args:
            pred_positions: Predicted positions
            target_positions: Target positions
            protein_frames: Protein rigid frames
            lig_mask: Ligand mask
            rec_mask: Receptor mask
            clamp_distance: Maximum distance to consider (optional)
            
        Returns:
            torch.Tensor: FAPE loss for ligand-receptor interactions
        """
        # [*, N_protein_frames, N_lig_frames, 3]
        local_pred_pos = protein_frames.invert()[..., None].apply(
            pred_positions[..., None, :, :],
        )
        local_target_pos = protein_frames.invert()[..., None].apply(
            target_positions[..., None, :, :],
        )
        error = torch.sqrt(
            torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + self.eps
        )
        edge_mask = rec_mask.unsqueeze(-1) * lig_mask.unsqueeze(-2)
        error = error * edge_mask
        if clamp_distance is not None:
            error = torch.clamp(error, min=0, max=clamp_distance)
        error = torch.sum(torch.sum(error, dim=-1), dim=-1) / (torch.sum(rec_mask, dim=-1) * torch.sum(lig_mask, dim=-1))
        return torch.mean(error)

    def compute_rmsd(self, ti, t_true, mask):
        """
        Compute root mean square deviation (RMSD).
        
        Args:
            ti: Predicted positions
            t_true: Target positions
            mask: Mask for valid positions
            
        Returns:
            torch.Tensor: RMSD loss
        """
        error = (ti - t_true) * mask.unsqueeze(-1)
        error = torch.sum(torch.sum(error**2, dim=-1), dim=-1) / (torch.sum(mask, dim=-1))
        return torch.mean(torch.sqrt(error + self.eps))

    def compute_steric_clash_loss_lig(self, ti, lig_mask):
        """
        Compute steric clash loss for ligand.
        
        Args:
            ti: Predicted positions
            lig_mask: Ligand mask
            
        Returns:
            torch.Tensor: Steric clash loss
        """
        edge_mask = lig_mask.unsqueeze(-1) * lig_mask.unsqueeze(-2)
        pairwise_distances = torch.cdist(ti, ti, p=2) * edge_mask
        error = torch.nn.functional.relu(0.5 - pairwise_distances)
        error = torch.sum(torch.sum(torch.tril(error, diagonal=-1), dim=-1), dim=-1)
        return torch.mean(error)

    def compute_kabsch_rmsd(self, ti_batch, t_true_batch, mask):
        """
        Compute Kabsch-RMSD.
        
        Args:
            ti_batch: Batch of predicted positions
            t_true_batch: Batch of target positions
            mask: Mask for valid positions
            
        Returns:
            torch.Tensor: Kabsch-RMSD loss
        """
        transformed_coords = []
        for ti, t_true in zip(ti_batch, t_true_batch):
            try:
                lig_coords_pred_mean = ti.mean(dim=0, keepdim=True, dtype=torch.float32)  # (1,3)
                lig_coords_mean = t_true.mean(dim=0, keepdim=True, dtype=torch.float32)  # (1,3)
                A = ((ti - lig_coords_pred_mean).transpose(0, 1) @ (t_true - lig_coords_mean)).to(dtype=torch.float32)
                U, S, Vt = torch.linalg.svd(A)
                corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=ti.device))
                rotation = (U @ corr_mat) @ Vt
                translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)
                transformed_coords.append((rotation @ t_true.t()).t() + translation)
            except Exception:
                print('Computing Kabsch RMSD failed.')
                return torch.zeros(1, requires_grad=True, dtype=torch.float32, device=ti_batch.device)
        
        return self.compute_pos_loss(ti_batch, torch.stack(transformed_coords), mask)

    def compute_pos_loss(self, ti, t_true, mask):
        """
        Compute position loss.
        
        Args:
            ti: Predicted positions
            t_true: Target positions
            mask: Mask for valid positions
            
        Returns:
            torch.Tensor: Position loss
        """
        error = (ti - t_true) * mask.unsqueeze(-1)    
        error = torch.sum(torch.sum(error**2, dim=-1), dim=-1) / (3*torch.sum(mask, dim=-1))
        return torch.mean(error)       

    def forward(self, target_frames, outputs, lig_mask, rec_mask, distogram_logits):
        """
        Forward pass of the loss function.
        
        Args:
            target_frames: Target frames
            outputs: Model outputs
            lig_mask: Ligand mask
            rec_mask: Receptor mask
            distogram_logits: Distogram logits
            
        Returns:
            tuple: (Total loss, Individual losses, RMSD)
        """
        target_frames = target_frames.cuda()
        pred_frames = outputs[-1][:, rec_mask.shape[-1]:]
        rec_frames = outputs[-1][:, :rec_mask.shape[-1]]
        target_positions = target_frames.get_trans()
        pred_positions = pred_frames.get_trans()
        
        # Compute individual losses
        lig_lig_loss = self.compute_fape_lig_lig(pred_frames, target_frames, pred_positions, target_positions, lig_mask)
        lig_rec_loss = self.compute_fape_lig_rec(pred_positions, target_positions, rec_frames, lig_mask, rec_mask, self.clamp_distance)
        aux_loss = torch.mean(torch.stack([
            self.compute_fape_lig_rec(pred_frames[:, rec_mask.shape[-1]:].get_trans(), target_positions, rec_frames, lig_mask, rec_mask, self.clamp_distance) for pred_frames in outputs
        ]))
        steric_clash_loss = self.compute_kabsch_rmsd(pred_positions, target_positions, lig_mask) if self.steric_clash_loss_weight > 0 else 0.0
        
        # Compute RMSD for monitoring
        rmsd = self.compute_rmsd(pred_positions, target_positions, lig_mask)

        # Process distogram losses based on which auxiliary heads are used
        if self.use_aux_head and self.use_lig_aux_head:
            distogram_logits_full, distogram_logits_lig = distogram_logits
            pseudo_beta_mask = torch.cat([rec_mask, lig_mask], dim=-1)
            pseudo_beta = torch.cat([rec_frames.get_trans(), pred_positions], dim=-2)
            rec_lig_distogram_loss = distogram_loss(distogram_logits_full, pseudo_beta, pseudo_beta_mask, min_bin=2.3125, max_bin=21.6875, no_bins=64)
            lig_lig_distogram_loss = distogram_loss(distogram_logits_lig, pred_positions, lig_mask, min_bin=1., max_bin=5., no_bins=42)
            full_distogram_loss = rec_lig_distogram_loss + lig_lig_distogram_loss
        elif self.use_aux_head:
            pseudo_beta_mask = torch.cat([rec_mask, lig_mask], dim=-1)
            pseudo_beta = torch.cat([rec_frames.get_trans(), pred_positions], dim=-2)
            full_distogram_loss = distogram_loss(distogram_logits, pseudo_beta, pseudo_beta_mask, min_bin=2.3125, max_bin=21.6875, no_bins=64)
        elif self.use_lig_aux_head:
            full_distogram_loss = distogram_loss(distogram_logits, pred_positions, lig_mask, min_bin=1., max_bin=5., no_bins=42)
        else:
            full_distogram_loss = 0.0

        # Compute total loss
        loss = (
            self.lig_lig_loss_weight * lig_lig_loss + \
            self.lig_rec_loss_weight * lig_rec_loss + \
            self.aux_loss_weight * aux_loss +\
            self.steric_clash_loss_weight * steric_clash_loss +\
            self.full_distogram_loss_weight * full_distogram_loss
        )

        # Handle NaN values
        if torch.isnan(loss):
            print('Loss is nan, skipping...')
            loss = torch.zeros(1, requires_grad=True, dtype=torch.float32, device=lig_lig_loss.device)

        return loss, (lig_lig_loss, lig_rec_loss, aux_loss, steric_clash_loss, full_distogram_loss), rmsd