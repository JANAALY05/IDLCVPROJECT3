"""Losses for the ablation study (task 4). All take raw logits.

  bce   - plain binary cross entropy
  wbce  - BCE with pos_weight: foreground pixels count pos_weight times more
          (pos_weight = #background / #foreground in the training set)
  focal - focal loss (Lin et al. 2017): down-weights pixels the model already gets
          right, so training focuses on the hard ones. gamma = 2 as in the exercise.

The exercise (lib/losses.py) wrote BCE by hand as  y_pred - y*y_pred + log(1+exp(-y_pred)).
That is unstable for large negative y_pred (exp overflows). PyTorch's
binary_cross_entropy_with_logits computes the same thing in a stable way, so we use it.

Every loss is averaged only over pixels inside the FOV (all pixels for PH2).
"""
import torch
import torch.nn.functional as F


def _masked_mean(pixel_loss, fov):
    return (pixel_loss * fov).sum() / fov.sum().clamp(min=1)


class BCELoss:
    def __call__(self, logits, y, fov):
        return _masked_mean(F.binary_cross_entropy_with_logits(logits, y, reduction="none"), fov)


class WeightedBCELoss:
    def __init__(self, pos_weight):
        self.pos_weight = torch.tensor(pos_weight)

    def __call__(self, logits, y, fov):
        pw = self.pos_weight.to(logits.device)
        return _masked_mean(F.binary_cross_entropy_with_logits(
            logits, y, reduction="none", pos_weight=pw), fov)


class FocalLoss:
    def __init__(self, gamma=2.0):
        self.gamma = gamma

    def __call__(self, logits, y, fov):
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * y + (1 - p) * (1 - y)          # probability given to the correct class
        return _masked_mean((1 - p_t) ** self.gamma * bce, fov)


def get_loss(name, pos_weight=None):
    if name == "bce":
        return BCELoss()
    if name == "wbce":
        return WeightedBCELoss(pos_weight)
    if name == "focal":
        return FocalLoss(gamma=2.0)
    raise ValueError(f"unknown loss {name}")
