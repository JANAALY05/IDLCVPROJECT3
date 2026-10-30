#!/bin/sh
### Fixing the encoder-decoder collapse on DRIVE: same model, only the loss changes.
### (EncDec + plain BCE predicted background everywhere.)
### Submit with:   bsub -app c02516_1g.10gb < jobs/jobscript_encdec_drive_fix.sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J encdec_drive_fix
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 02:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.11.9
source ~/yolo-env/bin/activate   # change to your own Python environment (see README)
cd "$LS_SUBCWD"   # the folder you ran bsub from (the repo root)

for loss in wbce dice bce_dice; do
    python3 train.py --dataset DRIVE --model encdec --loss $loss
done
