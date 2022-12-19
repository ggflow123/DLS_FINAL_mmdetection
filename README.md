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

# Results

The results below are all based on the models with1 epoch training on COCO 2017 training dataset.

The results are the COCO 2017 test dataset.

### For Swin Backbone:

#### Bounding Box

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.264
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.486
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.263
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.150
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.287
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.349
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.418
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.418
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.418
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.249
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.446
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.543

#### Segmentation

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.268
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.458
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.276
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.118
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.289
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.401
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.412
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.412
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.412
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.232
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.444
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.554

![A living room](swin1.jpg "A living room")

![a bear](swin2.jpg "a bear with Swin")

![a bedroom](swin3.jpg "A Bedroom")

### For ResNet-50 Backbone:

#### Bounding Box

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.190


 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.362


 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.178


 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.102


 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.210


 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.245


 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.350


 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.350


 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.350


 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.185


 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.368


 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.455

#### Segmentation

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.191


 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.339


 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.192


 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.077


 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.207


 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.285


 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.342


 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.342


 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.342


 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.167


 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.367


 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.457

![A living room](resnet50-1.jpg "A living room")

![A Bear](resnet50-2.jpg "A Bear With ResNet-50")

![A Bedroom](resnet50-3.jpg "A Bedroom with ResNet 50")

# Model Files

| Mask R-CNN + FPN + Swin Transformer                                                                                                                                                      | Mask R-CNN + FPN + ResNet-50                                                                                                                                                           |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Epoch-1](https://drive.google.com/file/d/1ICCtjaznnKimzO2ec3w3kjs-OW-jUcm5/view?usp=sharing)  [Epoch-10](https://drive.google.com/file/d/1B2sc9oeTfCx0Jku3arNE3xFAD_wu6c8L/view?usp=sharing) | [Epoch-1](https://drive.google.com/file/d/1jH8BrW0dZfGt0h3aIp-9oo_zgCHw4jPG/view?usp=sharing) [Epoch-9](https://drive.google.com/file/d/1lotBccc4ZLFI_9v8ysnKaUlN143WHkkF/view?usp=sharing) |
