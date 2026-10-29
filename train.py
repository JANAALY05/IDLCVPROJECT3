"""Train one model on one dataset with one loss, then evaluate on train/val/test.

Based on train.py from the exercise, extended with: command-line options, a validation
check every epoch, keeping the best model (by validation Dice), and saving all results.

Examples:
  python3 train.py --dataset PH2   --model encdec --loss bce
  python3 train.py --dataset PH2   --model unet --loss bce --clicks random --pos 5 --neg 5   (part 2)
  python3 train.py --dataset DRIVE --model encdec --loss bce --epochs 200

Outputs (in results/):
  <run>.pt    best model weights
  <run>.json  settings, loss/val-Dice per epoch, final metrics on train/val/test
"""
import os
import json
import time
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import PH2, DRIVE, get_loaders
from metrics import evaluate
from models import get_model
from losses import get_loss

DEFAULTS = {  # batch size and epochs per dataset
    "PH2":   {"batch_size": 8, "epochs": 50},
    "DRIVE": {"batch_size": 2, "epochs": 200},  # few images -> few steps per epoch
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["PH2", "DRIVE"], required=True)
parser.add_argument("--model", default="encdec")
parser.add_argument("--loss", choices=["bce", "wbce", "focal"], default="bce")
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=0)
# Part 2 (weak supervision, PH2 only): train on simulated clicks instead of full masks
parser.add_argument("--clicks", choices=["random", "margin"], help="click strategy; omit for full masks")
parser.add_argument("--pos", type=int, default=5, help="positive clicks per image")
parser.add_argument("--neg", type=int, default=5, help="negative clicks per image")
parser.add_argument("--margin", type=int, default=10, help="min. distance to border (margin strategy)")
args = parser.parse_args()
args.epochs = args.epochs or DEFAULTS[args.dataset]["epochs"]
args.batch_size = args.batch_size or DEFAULTS[args.dataset]["batch_size"]

run = f"{args.dataset}_{args.model}_{args.loss}"
if args.clicks:
    assert args.dataset == "PH2", "clicks are only implemented for PH2"
    run += f"_clicks_{args.clicks}_p{args.pos}_n{args.neg}"
os.makedirs("results", exist_ok=True)
torch.manual_seed(args.seed)  # same weight initialisation for every run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Run {run} on {device}  ({args.epochs} epochs, batch size {args.batch_size})")

# ---------- Data ----------
train_loader, val_loader, test_loader = get_loaders(args.dataset, args.batch_size)
cls = PH2 if args.dataset == "PH2" else DRIVE
train_eval_loader = DataLoader(cls("train"), batch_size=args.batch_size)  # no flips, full masks, for metrics
if args.clicks:
    # weak supervision: the training loader returns (image, click labels, click weights);
    # the loss only sees clicked pixels. Validation/test still use the full masks.
    from clicks import PH2Clicks
    train_loader = DataLoader(PH2Clicks("train", args.pos, args.neg, args.clicks, args.margin,
                                        seed=args.seed, augment=True),
                              batch_size=args.batch_size, shuffle=True, num_workers=3)

# pos_weight = background / foreground pixels in the training set (inside FOV)
pos_weight = None
if args.loss == "wbce":
    pos = tot = 0
    for _, y, f in train_eval_loader:
        pos += (y * f).sum().item()
        tot += f.sum().item()
    pos_weight = (tot - pos) / pos
    print(f"pos_weight = {pos_weight:.2f}")

# ---------- Model, loss, optimizer ----------
model = get_model(args.model).to(device)
loss_fn = get_loss(args.loss, pos_weight)
opt = optim.Adam(model.parameters(), args.lr)

# ---------- Training loop ----------
history = []
best_dice = -1
for epoch in range(args.epochs):
    tic = time.time()
    model.train()
    avg_loss = 0
    for X, Y, FOV in train_loader:
        X, Y, FOV = X.to(device), Y.to(device), FOV.to(device)
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, Y, FOV)
        loss.backward()
        opt.step()
        avg_loss += loss.item() / len(train_loader)

    # validation after every epoch; keep the model with the best validation Dice
    val = evaluate(model, val_loader, device)
    history.append({"epoch": epoch + 1, "train_loss": avg_loss, "val_dice": val["dice"]})
    marker = ""
    if val["dice"] > best_dice:
        best_dice = val["dice"]
        torch.save(model.state_dict(), f"results/{run}.pt")
        marker = "  *best*"
    print(f"Epoch {epoch+1:3d}/{args.epochs}  loss {avg_loss:.4f}  "
          f"val dice {val['dice']:.4f}  ({time.time()-tic:.1f}s){marker}")

# ---------- Final evaluation with the best model ----------
model.load_state_dict(torch.load(f"results/{run}.pt", map_location=device))
final = {name: evaluate(model, loader, device)
         for name, loader in [("train", train_eval_loader), ("val", val_loader), ("test", test_loader)]}

print("\nBest model (by validation Dice):")
print(f"{'split':6s} " + " ".join(f"{k:>11s}" for k in final["test"]))
for split, m in final.items():
    print(f"{split:6s} " + " ".join(f"{v:11.4f}" for v in m.values()))

with open(f"results/{run}.json", "w") as f:
    json.dump({"args": vars(args), "pos_weight": pos_weight,
               "history": history, "final": final}, f, indent=2)
print(f"Saved results/{run}.json and results/{run}.pt")
