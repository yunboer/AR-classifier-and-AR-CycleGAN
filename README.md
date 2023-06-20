# AR-classifier-and-AR-CycleGAN

## Introduction

This repository contains PyTorch code for the paper: *Artifact Detection and Restoration in Histology Images with Stain-Style and Structural Preservation*

This repository contains two new models: **AR-classifier** and **AR-CycleGAN**. The **AR-classifier** is designed to classify artifacts, while the **AR-CycleGAN** is capable of artifact recovery. 

The **AR-classifier** is designed based on the ResNet18 architecture and is capable of classifying patches from whole slide images (WSIs) into three categories: "artifact," "normal," and "unrestorable."

<img src="./figs/stage1.png" width="100%" align=center>

The **AR-CycleGAN** is designed based on the CycleGAN and can remove artifacts from patches belonging to the "artifact" category.

<img src="./figs/stage2.png" width="100%" align=center>

## Datasets

We provide two download links here for the our artifacts classification and CycleGAN dataset.

[OneDrive](https://sjtueducn-my.sharepoint.com/:f:/g/personal/yx_sun_sjtu_edu_cn/EmL-Ek6v-ElNp9E96AiDVe0B_N12Beca57UrW_R-qZohMw?e=dLBrbq)

[Baidu Cloud](https://pan.baidu.com/s/1Pbws5T46uHdAEAgiOFcH1A?pwd=dx6t)


## Getting Started

Ensure your Python >= 3.8 (recommended 3.8)

1. Clone the repo
    ```
    git clone https://github.com/yunboer/AR-classifier-and-AR-CycleGAN.git
    ```

2. pip all the requirement packages.
    ```
    # if you want to try AR-classifier
    cd AR-classifier

    # if you want to try AR-CycleGAN
    cd AR-CycleGAN

    # pip install all the packages
    pip install -r requirements.txt
    ```

3. Prepare the yaml file for the RandStainNA



*This document is currently being updated...*

<!-- ---

The expriment images will be released soon.

<img src="./figs/cmp.png" width="60%" align=center> -->



