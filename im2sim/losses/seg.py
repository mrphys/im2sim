import torch


def dice_calc(y1: torch.Tensor, y2: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """
    A function to compute the DICE coefficient for two masks 

    Args:
        y1 (torch.Tensor): True one-hot encoded mask of shape [B,C,H,D] or [B,C,H,D,W]
        y2 (torch.Tensor): Predicted one-hot encoded mask of shape [B,C,H,D] or [B,C,H,D,W]
        smooth (float): float value to avoid divide-by-zero

    Returns:
        dice_coeff (torch.Tensor): A tensor of shape [B,C] containing the DICE scores for each sample and channel 
    """
    intersection = torch.sum(y2 * y1, dim=2)  # (N, C)
    union = torch.sum(y2.pow(2), dim=2) + torch.sum(y1, dim=2)  # (N, C)
    ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
    dice_coef = (2 * intersection + smooth) / (union + smooth)  # (N, C)
    return dice_coef

def dice_loss(y1: torch.Tensor, y2: torch.Tensor, smooth: float = 1e-5, channel_weights: torch.Tensor = None) -> torch.Tensor:
    """
    A function to compute the DICE loss

    Args:
        y1 (torch.Tensor): True one-hot encoded mask of shape [B,C,H,D] or [B,C,H,D,W]
        y2 (torch.Tensor): Predicted one-hot encoded mask of shape [B,C,H,D] or [B,C,H,D,W]
        smooth (float): float value to avoid divide-by-zero
        channel_weights (torch.Tensor, optional): tensor of weights for each channels DICE score of shape [C]

    Returns:
        dice_coeff (torch.Tensor): A tensor of shape [B,C] containing the DICE scores for each sample and channel 
    """
    if channel_weights is None:
        return torch.mean(1 - dice_calc(y1,y2,smooth))
    else:
        return torch.mean(torch.mean(1-dice_calc(y1,y2,smooth), dim=0)*channel_weights)

