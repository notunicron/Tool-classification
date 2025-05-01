# Cholec80 Surgical Tool Classification using YOLOv8

This project uses the Ultralytics YOLOv8 classification model, fine-tuned with PyTorch, to classify surgical tools (Clipper, Grasper, Hook, Scissor) present in images from a subset of the Cholec80 dataset.

## Overview

The goal is to train a convolutional neural network to automatically identify which of the four specified surgical tools is visible in a given image frame. This script handles:
1.  Loading a pre-trained YOLOv8 classification model (`yolov8n-cls.pt` by default).
2.  Fine-tuning the model on the provided training and validation sets.
3.  Evaluating the trained model's performance on the test set.
4.  Providing an example of how to predict the tool class for a single image.

## Features
*   Uses state-of-the-art YOLOv8 classification models.
*   Leverages transfer learning from ImageNet pre-training.
*   Simple setup using the `ultralytics` pip package.
*   Includes training, validation, and testing phases.
*   Calculates Top-1 accuracy on the full test set.
*   Demonstrates single-image prediction.

## Dataset

The project dataset was taken from

https://www.kaggle.com/datasets/aditya9790/chloectinytools

This project is designed for a specific subset of the Cholec80 dataset, structured as follows:
```
cholec-tinytools/
├── test/
│ ├── clipper/
│ │ └── *.png
│ ├── grasper/
│ │ └── *.png
│ ├── hook/
│ │ └── *.png
│ └── scissor/
│ └── *.png
├── train/
│ ├── clipper/
│ │ └── *.png
│ ├── grasper/
│ │ └── *.png
│ ├── hook/
│ │ └── *.png
│ └── scissor/
│ └── *.png
└── validation/
├── clipper/
│ └── *.png
├── grasper/
│ └── *.png
├── hook/
│ └── *.png
└── scissor/
└── *.png
```