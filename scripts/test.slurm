#!/bin/bash -e
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=new_train
#SBATCH --output=/scratch/yl9539/mmdetection/scripts/slurm_test_24_%j.out
#SBATCH --gres=gpu:1
module purge

# Enter required modules

cd ../

python tools/test.py \
	configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_datapath_loadckpt.py \
	work_dirs/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_datapath_loadckpt_new/epoch_1.pth \
	--show-dir mask_rcnn_swin-t-p4-w7_fpn_1x_coco_datapath_loadckpt_new_results \
	--eval bbox segm	
