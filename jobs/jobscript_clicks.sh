#!/bin/sh
### Part 2 ablation: U-Net + BCE on PH2 trained only on clicks.
### Number of clicks per image (same for positive and negative) x click strategy.
### Same splits, epochs, batch size, lr and seed as part 1 -> compare with full masks.
### Submit with:   bsub -app c02516_1g.10gb < jobs/jobscript_clicks.sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J seg_clicks
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 04:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.11.9
source ~/yolo-env/bin/activate   # change to your own Python environment (see README)
cd "$LS_SUBCWD"   # the folder you ran bsub from (the repo root)

for strategy in random margin; do
    for n in 1 3 5 10 25 50; do
        python3 train.py --dataset PH2 --model unet --loss bce --clicks $strategy --pos $n --neg $n
    done
done
