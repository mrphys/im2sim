import torch
import torch.nn.functional as F
from torch import nn



class ImagetoGraph(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, encoder_outputs, graph_coords):
        grid = torch.stack([(2*graph_coords[...,i]/(d-1)) - 1
                          for i,d in enumerate(encoder_outputs.shape[2:])], axis=-1)
        
        projection = F.grid_sample(encoder_outputs,grid)
        return projection.permute(0,2,3,4,1)
