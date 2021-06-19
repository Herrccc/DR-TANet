# Dynamic Receptive Temporal Attention Network for Street Scene Change Detection

This is the official implementation of TANet and DR-TANet in "DR-TANet: Dynamic Receptive Temporal Attention Network for Street Scene Change Detection" (IEEE IV 2021). The preprint version is [here](https://arxiv.org/abs/2103.00879).

![img1](https://github.com/Herrccc/DR-TANet/blob/main/img/TANet:DR-TANet.png)

## Requirements

- python 3.7+
- opencv 3.4.2+
- pytorch 1.2.0+
- torchvision 0.4.0+
- tqdm 4.51.0
- tensorboardX 2.1

## Datasets

Our network is tested on two datasets for street-view scene change detection. 

- 'PCD' dataset from [Change detection from a street image pair using CNN features and superpixel segmentation](http://www.vision.is.tohoku.ac.jp/files/9814/3947/4830/71-Sakurada-BMVC15.pdf). 
  - You can find the information about how to get 'TSUNAMI', 'GSV' and preprocessed datasets for training and test [here](https://kensakurada.github.io/pcd_dataset.html).
- 'VL-CMU-CD' dataset from [Street-View Change Detection with Deconvolutional Networks](http://www.robesafe.com/personal/roberto.arroyo/docs/Alcantarilla16rss.pdf).
  -  'VL-CMU-CD': [[googledrive]](https://drive.google.com/file/d/0B-IG2NONFdciOWY5QkQ3OUgwejQ/view?resourcekey=0-rEzCjPFmDFjt4UMWamV4Eg)
  -  dataset for training and test in our work: [[googledrive]](https://drive.google.com/file/d/1GzQR9kQouH4_1PmFRTHl4dWTAzqz3ppH/view?usp=sharing)

## Training

Start training with TANet on 'PCD' dataset.
>The configurations for TANet
>- local-kernel-size:1, attn-stride:1, attn-padding:0, attn-groups:4.
>- local-kernel-size:3, attn-stride:1, attn-padding:1, attn-groups:4.
>- local-kernel-size:5, attn-stride:1, attn-padding:2, attn-groups:4.
>- local-kernel-size:7, attn-stride:1, attn-padding:3, attn-groups:4.

    python3 train.py --dataset pcd --datadir /path_to_dataset --checkpointdir /path_to_check_point_directory --max-epochs 100 --batch-size 16 --encoder-arch resnet18 --local-kernel-size 1

Start training with DR-TANet on 'VL-CMU-CD' dataset.

    python3 train.py --dataset vl_cmu_cd --datadir /path_to_dataset --checkpointdir /path_to_check_point_directory --max-epochs 150 --batch-size 16 --encoder-arch resnet18 --epoch-save 25 --drtam --refinement

## Evaluating

Start evaluating with DR-TANet on 'PCD' dataset.

    python3 eval.py --dataset pcd --datadir /path_to_dataset --checkpointdir /path_to_check_point_directory --resultdir /path_to_save_eval_result --encoder-arch resnet18 --drtam --refinement --store-imgs
  
