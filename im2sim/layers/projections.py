import torch
import torch.nn.functional as F
from torch import nn


class TrilinearProjection(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, encoder_outputs, graph_coords):
        grid = torch.stack([(2*graph_coords[...,i]/(d-1)) - 1
                          for i,d in enumerate(encoder_outputs.shape[2:])], axis=-1)
        grid = grid.unsqueeze(0).unsqueeze(-2).unsqueeze(-2)
        grid = grid.type_as(encoder_outputs)
        projection = F.grid_sample(encoder_outputs,grid).squeeze()
        return projection.permute(1,0)
