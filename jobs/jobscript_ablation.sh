#!/bin/sh
### Loss ablation (task 4): U-Net with focal loss and weighted BCE on both datasets.
### (U-Net + plain BCE was already run with jobscript_unet.sh.)
### Submit with:   bsub -app c02516_1g.10gb < jobs/jobscript_ablation.sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J seg_ablation
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 03:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.11.9
source ~/yolo-env/bin/activate   # change to your own Python environment (see README)
cd "$LS_SUBCWD"   # the folder you ran bsub from (the repo root)

for loss in focal wbce; do
    python3 train.py --dataset PH2   --model unet --loss $loss
    python3 train.py --dataset DRIVE --model unet --loss $loss
done
