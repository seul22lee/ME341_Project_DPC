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

class DPC_PolicyCNN(nn.Module):
    def __init__(self, input_channels=6, output_dim=1, output_chunk_length=10,
                 n_layers=3, kernel_size=3):
        super().__init__()
        self.output_dim = output_dim
        self.output_chunk_length = output_chunk_length
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        # 적절한 padding: time dimension 유지
        padding = kernel_size // 2

        conv_layers = []

        # 첫 번째 layer
        conv_layers.append(nn.Conv1d(input_channels, 64, kernel_size=kernel_size, padding=padding))

        # 중간 hidden layer들
        for _ in range(n_layers - 2):
            conv_layers.append(nn.Conv1d(64, 64, kernel_size=kernel_size, padding=padding))

        # 마지막 convolution layer
        conv_layers.append(nn.Conv1d(64, 64, kernel_size=kernel_size, padding=padding))

        self.convs = nn.ModuleList(conv_layers)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * window, output_dim * output_chunk_length)

    def forward(self, x_in):
        x, x_future_covariates, c_fut, _ = x_in

        x_all = torch.cat([x, x_future_covariates, c_fut], dim=2)  # [B, 10, 6]
        x_all = x_all.permute(0, 2, 1)  # [B, C=6, T=10]

        for conv in self.convs:
            x_all = F.relu(conv(x_all))

        x = self.flatten(x_all)
        x = self.fc(x)

        return x.view(x.shape[0], self.output_chunk_length, self.output_dim, 1)
