#!/bin/sh
### GPU batch job for course 02516 (from the HPC lecture slides).
### Submit with:   bsub -app c02516_1g.10gb < jobs/jobscript.sh
### Check with:    bstat
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J seg_encdec
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 02:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.11.9
source ~/yolo-env/bin/activate   # change to your own Python environment (see README)
cd "$LS_SUBCWD"   # the folder you ran bsub from (the repo root)

python3 train.py --dataset PH2   --model encdec --loss bce
python3 train.py --dataset DRIVE --model encdec --loss bce
