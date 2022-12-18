#!/bin/bash -e
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=training
#SBATCH --output=/scratch/yl9539/mmdetection/scripts/slurm_load_24_%j.out
#SBATCH --gres=gpu:4
module purge

# Enter required modules

cd ../

bash ./tools/dist_train.sh ./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_datapath_loadckpt.py 4
