# FAME: Frequency and Motion Extrapolation for Robust Multimodal Facial Action Unit Detection

Pytorch pretraining & downstream-training code for **FAME**. We propose a novel self-supervised multimodal learning framework tailored for dynamic AU rep resentation learning. FAME features asynchronous masked pretraining through high pass filtered facial images and incomplete point clouds from the past. 

![image-20251227100337659](/figs/framework.jpg)

## Model Zoo

|      | task         | name | Dataset                | download                    |
| ---- | ------------ | ---- | ---------------------- | --------------------------- |
| 0    | Pretrain     | FAME | BP4D, DISFA, Aff-Wild2 | (released after acceptance) |
| 1    | AU Detection | -    | BP4D, DISFA, Aff-Wild2 | (released after acceptance) |

## Pretraining

### Install

First, clone this repository into your local machine.

```
git clone https://github.com/flotaas/FAME_code.git
```

Next, install required dependencies.

```
pip install -r requirements.txt
```

### Data Preparation

For each input image, we use [3DDFA](https://github.com/cleardusk/3DDFA_V2) to generate facial point clouds.

### Training

```
python pretrain.py
```

### Visualization

The framework supports qualitative visualization of reconstructed multimodal signals.

![image-20251227104319524](/figs/vis.jpg)

(Visualization scripts will be released together with the full paper.)

## Downstream

These components will be released after paper acceptance.

### Acknowledgement

This repository is based on [PiMAE](https://github.com/antonioo-c/PiMAE) and [MFM](https://www.mmlab-ntu.com/project/mfm/index.html) repositories, we thank them for their great work.