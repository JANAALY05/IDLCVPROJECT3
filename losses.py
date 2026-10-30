"""Losses for the ablation study (task 4). All take raw logits.

  bce   - plain binary cross entropy
  wbce  - BCE with pos_weight: foreground pixels count pos_weight times more
          (pos_weight = #background / #foreground in the training set)
  focal - focal loss (Lin et al. 2017): down-weights pixels the model already gets
          right, so training focuses on the hard ones. gamma = 2 as in the exercise.
  dice  - soft Dice loss, 1 - Dice computed on probabilities. It only looks at the overlap
          with the foreground, so "predict background everywhere" gives the worst loss (1)
          instead of a low one -> counters the class-imbalance collapse.
  bce_dice - BCE + Dice: BCE gives smooth per-pixel gradients, Dice fixes the imbalance.

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


class DiceLoss:
    """Soft Dice over the whole batch, only inside the FOV (exercise formula with +1 smoothing)."""

    def __init__(self, smooth=1.0):
        self.smooth = smooth

    def __call__(self, logits, y, fov):
        p = torch.sigmoid(logits) * fov
        y = y * fov
        inter = (p * y).sum()
        return 1 - (2 * inter + self.smooth) / (p.sum() + y.sum() + self.smooth)


class BCEDiceLoss:
    def __init__(self):
        self.bce, self.dice = BCELoss(), DiceLoss()

    def __call__(self, logits, y, fov):
        return self.bce(logits, y, fov) + self.dice(logits, y, fov)


def get_loss(name, pos_weight=None):
    if name == "bce":
        return BCELoss()
    if name == "wbce":
        return WeightedBCELoss(pos_weight)
    if name == "focal":
        return FocalLoss(gamma=2.0)
    if name == "dice":
        return DiceLoss()
    if name == "bce_dice":
        return BCEDiceLoss()
    raise ValueError(f"unknown loss {name}")
