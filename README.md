# Project 3 – Image Segmentation (02516 IDLCV, DTU)

Segmentation of skin lesions (**PH2**) and retinal vessels (**DRIVE**) with a simple
encoder–decoder and a U-Net, a loss-function ablation, and (part 2) **weak supervision
with simulated point clicks** on PH2.

## Main results (test set)

| Dataset | Model / labels | Loss | Dice | IoU | Acc | Sens | Spec |
|---|---|---|---|---|---|---|---|
| PH2 | Encoder–decoder | BCE | 0.888 | 0.799 | 0.933 | 0.872 | 0.960 |
| PH2 | U-Net | BCE | 0.883 | 0.790 | 0.933 | 0.829 | 0.979 |
| PH2 | U-Net | focal | 0.822 | 0.698 | 0.905 | 0.719 | 0.987 |
| PH2 | U-Net | weighted BCE | **0.898** | 0.815 | 0.940 | 0.869 | 0.971 |
| DRIVE | Encoder–decoder | BCE | 0.000 | 0.000 | 0.865 | 0.000 | 1.000 |
| DRIVE | U-Net | BCE | 0.810 | 0.680 | 0.951 | 0.779 | 0.977 |
| DRIVE | U-Net | focal | **0.811** | 0.682 | 0.950 | 0.793 | 0.975 |
| DRIVE | U-Net | weighted BCE | 0.792 | 0.655 | 0.940 | 0.845 | 0.955 |

Part 2 (U-Net + partial BCE on PH2, N positive + N negative clicks per image, test Dice):

| N | 1 | 3 | 5 | 10 | 25 | 50 | full masks |
|---|---|---|---|---|---|---|---|
| random clicks | 0.813 | 0.840 | 0.875 | 0.877 | 0.891 | 0.883 | 0.883 |
| margin clicks (≥10 px from border) | 0.775 | 0.792 | 0.876 | 0.867 | 0.878 | 0.879 | 0.883 |

Key findings: the encoder–decoder collapses to "all background" on DRIVE (no skip
connections, thin vessels lost in the 32×32 bottleneck), the U-Net fixes this; weighted BCE
raises sensitivity on both datasets; ~5+5 clicks per image get within 0.01 Dice of full
supervision, and random clicks beat "careful" margin clicks.

## Repository structure

```
make_splits.py      fixed train/val/test splits -> splits.json (PH2 160/20/20, DRIVE 12/4/4)
datasets.py         PH2 and DRIVE data loaders (based on the exercise loader); check: python3 datasets.py
metrics.py          Dice, IoU, accuracy, sensitivity, specificity (inside FOV); tests: python3 metrics.py
models.py           EncDec (from the exercise) and U-Net; layer shapes: python3 models.py
losses.py           BCE, weighted BCE, focal, Dice, BCE+Dice (all masked by the FOV / click weights)
train.py            train + validate (best epoch by val Dice) + evaluate on all splits -> results/
clicks.py           part 2: simulated clicks (random / margin) and the click training dataset
visualize.py        figure with test predictions of all models -> predictions.png
inspect_data.py     first look at the raw data
jobs/               LSF batch scripts for the DTU HPC (course queue c02516)
figure_scripts/     scripts that draw the U-Net sketch and the clicks ablation plot
figures/            result figures (U-Net sketch, predictions, clicks examples, clicks ablation)
```

## How to run (DTU HPC)

Data is read from `/dtu/datasets1/02516` (PH2_Dataset_images, DRIVE).

**1. Environment (once).** The home quota is 30 GB and PyTorch needs ~5 GB, so install
without the pip cache:

```bash
module load python3/3.11.9
python3 -m venv ~/idlcv-env
source ~/idlcv-env/bin/activate
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir matplotlib pillow numpy
```

Then change the `source ...` line in `jobs/*.sh` to your own environment.

**2. Splits and checks** (run from the repo folder):

```bash
python3 make_splits.py      # creates splits.json (deterministic, seed 42)
python3 datasets.py         # shapes, mask values, foreground fraction, check_*.png
python3 metrics.py          # 4 tests, all should print OK
python3 models.py           # U-Net feature-map sizes
python3 clicks.py           # clicks_examples.png
```

**3. Training on the GPU** (batch queue, one 10 GB GPU slice; logs in `logs/`):

```bash
mkdir -p logs
bsub -app c02516_1g.10gb < jobs/jobscript.sh            # EncDec, BCE, PH2 + DRIVE   (~5 min)
bsub -app c02516_1g.10gb < jobs/jobscript_unet.sh       # U-Net,  BCE, PH2 + DRIVE   (~8 min)
bsub -app c02516_1g.10gb < jobs/jobscript_ablation.sh   # U-Net, focal + wBCE        (~20 min)
bsub -app c02516_1g.10gb < jobs/jobscript_clicks.sh     # part 2, 12 click runs      (~30 min)
bsub -app c02516_1g.10gb < jobs/jobscript_encdec_drive_fix.sh  # EncDec on DRIVE with wBCE / Dice / BCE+Dice
bstat                                                   # job status
```

Each run writes `results/<dataset>_<model>_<loss>[_clicks_...].json` (settings, history,
final metrics) and `.pt` (best weights). For a quick test use the interactive GPU node
`02516sh` (only one session at a time).

**4. Figures:** `python3 visualize.py` (needs the trained `.pt` files);
`python3 figure_scripts/make_unet_sketch.py` and `python3 figure_scripts/plot_clicks_ablation.py`
(no GPU or data needed).

## Notes
- Two seeds: **42** for the train/val/test split (`make_splits.py`), **0** for training
  (weight initialisation, and click placement in part 2; `train.py --seed`).
- Every experiment uses the same `splits.json`, Adam (lr 1e-3), PH2: 192×256, batch 8,
  50 epochs; DRIVE: 512×512, batch 2, 200 epochs.
- DRIVE's `test/` folder has no vessel annotations, so all three DRIVE splits come from
  `training/` (12/4/4) — results on DRIVE have high variance.
- In part 2, validation and test still use the full masks (model selection with full
  masks is slightly optimistic for a real weak-label setting).

