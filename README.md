# Siamese Mask R-CNN

This is the official implementation of Siamese Mask R-CNN from [One-Shot Instance Segmentation](https://arxiv.org/abs/1811.11507). It is based on the [Mask R-CNN](https://arxiv.org/abs/1703.06870) implementation by [Matterport](https://github.com/matterport/Mask_RCNN).

The repository includes:
- [x] Source code of Siamese Mask R-CNN
- [x] Training code for MS COCO
- [x] Evaluation on MS COCO metrics (AP)
- [x] Training and evaluation of one-shot splits of MS COCO
- [x] Pre-trained weights for ImageNet
- [x] Training code to reproduce the results from the paper
- [ ] Pre-trained weights for MS COCO and the one-shot splits
- [ ] Code to evaluate models from the paper
- [ ] Code to generate paper figures

## One-Shot Instance Segmentation

![Teaser Image](figures/teaser_web.jpg)

One-shot instance segmentation can be summed up as: Given a query image and a reference image showing an object of a novel category, we seek to detect and segment all instances of the corresponding category (‘person’ on the left, ‘car’ on the right). Note that no ground truth annotations of reference categories are used during training.
This type of visual search task creates new challenges for computer vision algorithms, as methods from metric and few-shot learning have to be incorporated into the notoriously hard tasks ofobject identification and segmentation. 
Siamese Mask R-CNN extends Mask R-CNN - a state-of-the-art object detection and segmentation system - with a Siamese backbone and a matching procedure to perform this type of visual search.

## Installation

### Requirements

Linux, Python 3.4+, Tensorflow, Keras and other dependencies listed in install_requirements.ipynb

### Prepare COCO dataset.

The model requires [MS COCO](http://cocodataset.org/#home) and the [CocoAPI](https://github.com/waleedka/coco). It is recommended to symlink the dataset root and the CocoAPI to `/data`.

### Install dependencies

Run the install_requirements.ipynb notebook to install all relevant requirements.

## Training

## Evaluation

## Model description

## Citation
