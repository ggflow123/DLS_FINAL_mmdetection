# DLS_FINAL_mmdetection

Supervised Training in the New York University 2021 Intro To Deep Learning System Class final project, use mmdetection package

## Yuanzhe Liu yl9539@nyu.edu

## Chieh-Hsin Chen cc7204@nyu.edu

# ## Links

For Self Supervised Learning MoBY to train Swin Transformer, also the main part of the whole project, please consult:

https://github.com/ggflow123/DLS_Final_Project

For Self Supervised Learning DINO to train ResNet-50, please consult:

https://github.com/ggflow123/DLS_FINAL_DINO

# Usage

## Environment Setup

For the environment setup, please do directly to the official mmdetection github:
https://github.com/open-mmlab/mmdetection

I first consulted this page:
https://github.com/SwinTransformer/Swin-Transformer-Object-Detection

Then, the version of mmcv-full and pytorch are not compatible. Please follow the official installation of mmdetection. The link is available below:
https://mmdetection.readthedocs.io/en/stable/get_started.html

## Data Peparation

We use 2017 COCO Dataset. Mmdetection gives a nice download.
You can use coco2017, voc2007 or LVIS.
In the folder, do:

```
python tools/misc/download_dataset.py --dataset-name coco2017
python tools/misc/download_dataset.py --dataset-name voc2007
python tools/misc/download_dataset.py --dataset-name lvis
```

For more info, please consult https://github.com/open-mmlab/mmdetection/blob/master/docs/en/useful_tools.md#dataset-download

## Running

All the configuration files are in the **config** folder. For more info, please check https://github.com/open-mmlab/mmdetection/blob/master/docs/en/tutorials/config.md.

To submit a job in NYU Greene or Linux Cloud Computing environment, scripts are available in the **scripts** folder.

For example, to train a Mask R-CNN + FPN with Swin Transformer as the backbone, with 4 GPU and 24 hours, in the scripts folder, do:

```
sbatch train_load.slurm
```

To run directly, do:

```
bash ./tools/dist_train.sh ./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_datapath_loadckpt.py 4

```

where ``mask_rcnn_swin-t-p4-w7_fpn_1x_coco_datapath_loadckpt.py`` is the configuration file. You can customize the configuration file by yourself. For me, don't run with this configuration file directly. I changed the path of the data. Please be careful.

# For Professors and TAs:

If you are reading till here right now, please forgive me not showing the evaluation data. I keep updating the data. The data will be available shortly (within 24 hours).Thank you!
