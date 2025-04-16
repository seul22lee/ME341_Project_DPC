import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
import pickle
from pickle import dump

window = 10

def DPC_loss(model_output: torch.Tensor, target: torch.Tensor, u_output: torch.Tensor, c_fut: torch.Tensor, return_all=False):
    model_output = model_output.median(dim=-1).values
    model_output_x1 = model_output[:, :, 0]
    target_x1 = target[:, :, 0]
    errors = (target_x1 - model_output_x1) ** 2

    u_diff = u_output[:, 1:, :] - u_output[:, :-1, :]
    squared_diff = u_diff ** 2

    model_output_x2 = model_output[:, :, 1]
    low_violation = F.relu(c_fut[:, :, 0] - model_output_x2) ** 2 
    up_violation = F.relu(model_output_x2 - c_fut[:, :, 1]) ** 2 
    constraint_loss = low_violation + up_violation  

    tracking_loss = torch.sqrt(errors.mean())
    smoothness_loss = 10 * torch.sqrt(squared_diff.mean())
    constraint_loss = 2 * torch.sqrt(constraint_loss.mean())

    total_loss = tracking_loss + smoothness_loss + constraint_loss

    if return_all:
        return total_loss, tracking_loss.item(), smoothness_loss.item(), constraint_loss.item()
    else:
        return total_loss



class DPC_PolicyTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, future_cov_dim, static_cov_dim,
                 input_chunk_length, output_chunk_length, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.output_dim = output_dim
        self.output_chunk_length = output_chunk_length
        self.seq_len = input_chunk_length + output_chunk_length

        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, x_in):
        x, x_future_covariates, c_fut, x_static_covariates = x_in

        def ensure_3d(x):
            if x is None:
                return None
            elif x.dim() == 2:
                return x.unsqueeze(-1)
            return x

        x = ensure_3d(x)
        x_future_covariates = ensure_3d(x_future_covariates)
        c_fut = ensure_3d(c_fut)

        time_sequence = [x]
        if x_future_covariates is not None:
            time_sequence.append(x_future_covariates)
        if c_fut is not None:
            time_sequence.append(c_fut)

        x_seq = torch.cat(time_sequence, dim=2)  # (batch, time, total_feature_dim)

        x_emb = self.embedding(x_seq)
        x_enc = self.transformer_encoder(x_emb)
        out = self.decoder(x_enc[:, -self.output_chunk_length:, :])
        return out.unsqueeze(-1)

