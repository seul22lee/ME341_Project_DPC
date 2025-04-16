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
    smoothness_loss = 0.1 * torch.sqrt(squared_diff.mean())
    constraint_loss = 3 * torch.sqrt(constraint_loss.mean())

    total_loss = tracking_loss + smoothness_loss + constraint_loss

    if return_all:
        return total_loss, tracking_loss.item(), smoothness_loss.item(), constraint_loss.item()
    else:
        return total_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class DPC_PolicyCNN(nn.Module):
    def __init__(self, input_channels=6, output_dim=1, output_chunk_length=10,
                 channel_list=[64, 64], kernel_size=3, dropout=0.0):
        super().__init__()
        self.output_dim = output_dim
        self.output_chunk_length = output_chunk_length
        self.kernel_size = kernel_size
        self.dropout_rate = dropout

        padding = kernel_size // 2
        conv_layers = []
        in_channels = input_channels

        for out_channels in channel_list:
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            conv_layers.append(nn.ReLU())
            if dropout > 0:
                conv_layers.append(nn.Dropout(p=dropout))
            in_channels = out_channels

        self.conv_net = nn.Sequential(*conv_layers)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channel_list[-1] * window, output_dim * output_chunk_length)

    def forward(self, x_in):
        x, x_future_covariates, c_fut, _ = x_in

        def ensure_3d(tensor, name=""):
            if tensor is None:
                return tensor
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(-1)
                print(f"[INFO] {name} was 2D, reshaped to 3D: {tensor.shape}")
            elif tensor.ndim == 1:
                tensor = tensor.unsqueeze(0).unsqueeze(-1)
                print(f"[INFO] {name} was 1D, reshaped to 3D: {tensor.shape}")
            return tensor

        x = ensure_3d(x, "x")
        x_future_covariates = ensure_3d(x_future_covariates, "x_future_covariates")
        c_fut = ensure_3d(c_fut, "c_fut")

        x_all = torch.cat([x, x_future_covariates, c_fut], dim=2)  # [B, 10, 6]
        x_all = x_all.permute(0, 2, 1)  # [B, C=6, T=10]
        x_feat = self.conv_net(x_all)
        x_feat = self.flatten(x_feat)
        x_out = self.fc(x_feat)

        return x_out.view(x_out.shape[0], self.output_chunk_length, self.output_dim, 1)