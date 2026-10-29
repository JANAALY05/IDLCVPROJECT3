"""Segmentation metrics: Dice, IoU, Accuracy, Sensitivity, Specificity.

Idea: compare prediction and ground truth pixel by pixel and count
  TP = vessel/lesion predicted AND really there
  FP = predicted, but really background
  FN = missed (really there, not predicted)
  TN = background predicted AND really background
Only pixels inside the FOV are counted (for PH2 the FOV is everything).
All five metrics are simple formulas of these four counts.

Counts are summed over the whole split (all images together) before the
formulas are applied.

Test the formulas on tiny hand-made examples:  python3 metrics.py
"""
import torch

EPS = 1e-8  # avoids division by zero (e.g. an image with no foreground)


def confusion_counts(logits, target, fov, threshold=0.5):
    """logits: raw model output [B,1,H,W]; target, fov: 0/1 tensors [B,1,H,W]."""
    pred = (torch.sigmoid(logits) > threshold).float()
    target = target.float()
    fov = fov.float()
    tp = (pred * target * fov).sum()
    fp = (pred * (1 - target) * fov).sum()
    fn = ((1 - pred) * target * fov).sum()
    tn = ((1 - pred) * (1 - target) * fov).sum()
    return torch.stack([tp, fp, fn, tn])


def metrics_from_counts(counts):
    tp, fp, fn, tn = [c.item() for c in counts]
    return {
        "dice":        2 * tp / (2 * tp + fp + fn + EPS),
        "iou":         tp / (tp + fp + fn + EPS),
        "accuracy":    (tp + tn) / (tp + fp + fn + tn + EPS),
        "sensitivity": tp / (tp + fn + EPS),   # how much of the real foreground we found
        "specificity": tn / (tn + fp + EPS),   # how much of the real background we kept
    }


@torch.no_grad()
def evaluate(model, loader, device):
    """Run the model on a whole loader and return the five metrics."""
    model.eval()
    counts = torch.zeros(4)
    for X, Y, FOV in loader:
        logits = model(X.to(device)).cpu()
        counts += confusion_counts(logits, Y, FOV)
    return metrics_from_counts(counts)


# ---------------- Tests with known answers ----------------
if __name__ == "__main__":
    def as_logits(p):
        # turn a 0/1 prediction into logits: big positive -> sigmoid ~1, big negative -> ~0
        return (p * 2 - 1) * 10

    def check(name, pred, target, fov, expected):
        pred, target, fov = [torch.tensor(a, dtype=torch.float32).view(1, 1, 2, 4)
                             for a in (pred, target, fov)]
        m = metrics_from_counts(confusion_counts(as_logits(pred), target, fov))
        ok = all(abs(m[k] - v) < 1e-6 for k, v in expected.items())
        print(f"{'OK  ' if ok else 'FAIL'} {name}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))

    target = [1, 1, 0, 0,
              0, 0, 0, 0]
    all_in = [1] * 8

    # 1. perfect prediction -> everything 1
    check("perfect", target, target, all_in,
          {"dice": 1, "iou": 1, "accuracy": 1, "sensitivity": 1, "specificity": 1})

    # 2. predict nothing -> accuracy still 6/8 = 0.75, but dice/iou/sensitivity 0
    check("all background", [0] * 8, target, all_in,
          {"dice": 0, "iou": 0, "accuracy": 0.75, "sensitivity": 0, "specificity": 1})

    # 3. half right: TP=1, FN=1, FP=1, TN=5
    #    dice = 2/(2+1+1)=0.5, iou = 1/3, acc = 6/8, sens = 1/2, spec = 5/6
    check("partial", [1, 0, 1, 0, 0, 0, 0, 0], target, all_in,
          {"dice": 0.5, "iou": 1 / 3, "accuracy": 0.75, "sensitivity": 0.5, "specificity": 5 / 6})

    # 4. FOV: the wrong pixel (index 2) is outside the FOV -> ignored -> perfect
    check("error outside FOV", [1, 1, 1, 0, 0, 0, 0, 0], target, [1, 1, 0, 1, 1, 1, 1, 1],
          {"dice": 1, "iou": 1, "accuracy": 1, "sensitivity": 1, "specificity": 1})
