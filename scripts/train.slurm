#!/bin/bash -e
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=training
#SBATCH --output=/scratch/yl9539/mmdetection/scripts/slurm_train_24_%j.out
#SBATCH --gres=gpu:2
module purge

# Enter required modules

cd ../

bash ./tools/dist_train.sh ./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_datapath.py 2
