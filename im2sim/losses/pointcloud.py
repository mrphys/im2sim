# import torch
# import torch.nn.functional as F 
from kaolin.metrics.pointcloud import *



# def chamfer_loss(y_true, y_pred):
#     distances = torch.cdist(y_true.unsqueeze(0), y_pred.unsqueeze(0)).squeeze()
#     return distances.min(0).values.mean()+distances.min(1).values.mean()
    
    