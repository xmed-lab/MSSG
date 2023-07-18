## MSSG

Qixiang ZHANG, Yi LI, Cheng XUE, Xiaomeng LI*, "Morphology-inspired Unsupervised Gland Segmentation via Selective Semantic Grouping", MICCAI 2023 (Accepted).



### 1. Introduction
This MSSG framework is designed for unsupervised gland segmentation on histology images, containing two modules. The first Selective Proposal Mining (SPM) module generates proposals for different gland sub-regions. And the Morphology-aware Semantic Grouping (MSG) module groups the semantics of the sub-region proposal to obtain comprehensive knowledge about glands.

![figure_2](C:\Users\ericZ\Desktop\figure_2.png)



### 2. Environment

This code has been tested with Python 3.9, PyTorch 1.12.0, CUDA 11.3 mmseg 0.8.0 and mmcv 1.4.0 on Ubuntu 20.04.



### 3. Preparation

Download GlaS dataset from [Official Website](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/), and place the dataset as following:

```shell
/your/directory/SPM/
      └── glas/
            ├── training_images/
            │     ├── xxxxxxxxx.bmp
            │     └── ...
            └── training_annotations/
                  ├── xxxxxxxxx.bmp
                  └── ...
```



**Or download [resources](https://pan.baidu.com/s/1htY5nZacceXj_m2FlY8uXw) (dataset, crop image, and weights) with extract code llb3, then link to codes.**

```shell
git clone https://github.com/XMed-Lab/MSSG.git
cd MSSG
ln -s MSSG_resources/glas_SPM SPM/glas
ln -s MSSG_resources/glas_MSG MSG/glas
ln -s MSSG_resources/weights MSG/weights
```

Install Python library dependencies
```shell
pip install -r requirements.txt
```

Install MMSegmentation codebase, see [documentation](https://mmsegmentation.readthedocs.io/en/latest/) from MMLab for details.



### 4. Training for SPM Module

**You can simply download the pre-generated proposal map from [resources](https://pan.baidu.com/s/1htY5nZacceXj_m2FlY8uXw) with extracted code kopl**

Or use the SPM module to generate candidate proposals for each histology image

```shell
python SPM/SPM.py
```

Utilize the empirical cue to select gland sub-region from the candidate proposals

```shell
python select_proposal.py
```



### 5. Training for MSG Module

Crop the training image and the proposal map

```shell
cd MSG
python tools/crop_img_and_gt.py MSG/glas/images SPM/proposal_map MSG/glas
```



Train the segmentation model with MSG modules.

```shell
cd MSG
bash tools/dist_train.sh configs/pspnet_mssg/pspnet_wres38-d8_10k_histo.py 4 runs/mssg
```
