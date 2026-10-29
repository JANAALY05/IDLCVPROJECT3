"""Create fixed train/val/test splits for PH2 and DRIVE and save them to splits.json.

Run once:  python3 make_splits.py
Every other script loads splits.json, so all models and losses use the same images.
"""
import os
import json
import random

ROOT = "/dtu/datasets1/02516/"
SEED = 42

# ---------- 1. Collect the IDs ----------
# PH2: one folder per case, e.g. "IMD002"
ph2_ids = sorted(os.listdir(os.path.join(ROOT, "PH2_Dataset_images")))

# DRIVE: take the number from "21_training.tif" -> "21"
drive_files = os.listdir(os.path.join(ROOT, "DRIVE", "training", "images"))
drive_ids = sorted((f.split("_")[0] for f in drive_files), key=int)


# ---------- 2 + 3. Shuffle with a fixed seed and slice ----------
def split(ids, n_train, n_val, n_test, seed):
    assert n_train + n_val + n_test == len(ids), "split sizes must add up"
    ids = list(ids)
    random.Random(seed).shuffle(ids)  # own generator -> same result every run
    return {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train:n_train + n_val]),
        "test": sorted(ids[n_train + n_val:]),
    }


splits = {
    "seed": SEED,
    "PH2": split(ph2_ids, 160, 20, 20, SEED),
    "DRIVE": split(drive_ids, 12, 4, 4, SEED),
}

# ---------- Sanity checks ----------
for name in ["PH2", "DRIVE"]:
    s = splits[name]
    tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
    assert not (tr & va or tr & te or va & te), f"{name}: overlap between sets"
    print(f"{name}: train={len(tr)} val={len(va)} test={len(te)}")
print("DRIVE val:", splits["DRIVE"]["val"])
print("DRIVE test:", splits["DRIVE"]["test"])

# ---------- 4. Save ----------
with open("splits.json", "w") as f:
    json.dump(splits, f, indent=2)
print("Saved splits.json")
