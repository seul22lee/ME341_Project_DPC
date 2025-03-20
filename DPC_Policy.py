import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
import pickle
from pickle import dump

window = 10

def DPC_loss(model_output: torch.Tensor, target: torch.Tensor, u_output: torch.Tensor, c_fut: torch.Tensor):

    model_output = model_output.median(dim=-1).values
    model_output_x1 = model_output[:, :, 0]
    target_x1 = target[:, :, 0]
    errors = (target_x1 - model_output_x1) ** 2

    u_diff = u_output[:, 1:, :] - u_output[:, :-1, :]
    squared_diff = u_diff ** 2 #

    model_output_x2 = model_output[:, :, 1]
    low_violation = F.relu(c_fut[:, :, 0] - model_output_x2) ** 2 
    up_violation = F.relu(model_output_x2 - c_fut[:, :, 1]) ** 2 
    constraint_loss = low_violation + up_violation  

    tracking_loss_sqrt = torch.sqrt(errors.mean())
    smoothness_loss_sqrt = 0.1*torch.sqrt(squared_diff.mean())
    constraint_loss_sqrt = 3*torch.sqrt(constraint_loss.mean())
    return tracking_loss_sqrt + smoothness_loss_sqrt + constraint_loss_sqrt 

class DPC_PolicyNN(nn.Module):
    def __init__(
        self,
        input_dim: int,  
        output_dim: int,
        future_cov_dim: int,
        static_cov_dim: int,
        input_chunk_length: int,
        output_chunk_length: int,
        hidden_dim: int
    ):
        super(DPC_PolicyNN, self).__init__()

        self.input_dim = input_dim  
        self.output_dim = output_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

        self.fc1 = nn.Linear(60, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, output_dim * output_chunk_length)

        self.relu = nn.ReLU()

    def forward(
        self, x_in: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
    ) -> torch.Tensor:
        x, x_future_covariates, c_fut, x_static_covariates = x_in

        if len(x.shape) == 3:
            x = x.flatten(start_dim=1) 

        if c_fut is not None:
            c_fut = c_fut.flatten(start_dim=1)
            x = torch.cat([x, c_fut], dim=1)

        if x_future_covariates is not None:
            x_future_covariates = x_future_covariates.flatten(start_dim=1)
            x = torch.cat([x, x_future_covariates], dim=1)

        if x_static_covariates is not None:
            x_static_covariates = x_static_covariates.flatten(start_dim=1)
            x = torch.cat([x, x_static_covariates], dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.fc7(x)

        batch_size = x.shape[0]
        x = x.view(batch_size, self.output_chunk_length, self.output_dim, 1)

        return x
