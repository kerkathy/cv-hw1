# Use Pytorch to fine-tune object detection model using CNN based network

# import the necessary packages
from torchvision.models import detection
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as trns
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import numpy as np
import argparse
import json
import torch
import os
import sys
from pathlib import Path

import cv2

from dataset import MarineDataset

TRAIN = "train"
DEV = "valid"
SPLITS = [TRAIN, DEV]

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--stage", default="debug", 
                choices=["train", "valid", "debug"], help="train, valid, or debug mode")
ap.add_argument("-d", "--data_dir", required=True, type=Path,
                help="path to input dataset where the images are stored in folders `train`, `val` and `test`")
ap.add_argument("-m", "--model",  default="frcnn-resnet", choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"],
                help="name of the object detection model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-b", "--batch_size", type=int, default=8)
args = vars(ap.parse_args())

# load the class labels
# from the "categories" section in the json file called `.annotations.coco.json` in the directory assigned by `--dataset`


# TODO: collate_fn? transform?
# 使用 trns.ToTensor() 將影像大小為 (H x W x C) ，值的範圍介於 [0, 255] 的 PIL Image 或是 numpy.ndarray
# 轉換至影像大小為 (C x H x W) ，值的範圍介於 [0.0, 1.0] 的 torch.FloatTensor
data_paths = {split: args["data_dir"] / split for split in SPLITS}

datasets: dict[str, MarineDataset] = {
    split: MarineDataset(path, transforms=trns.ToTensor())
    for split, path in data_paths.items()
}
print(f"Loaded {len(datasets[TRAIN])} training images")
print(f"Loaded {len(datasets[DEV])} validation images")

# Group data with the same height and width together, so that we can batch them together

# 分組並按尺寸排序
image_sizes = {}
for img_size in datasets[TRAIN].image_sizes:
    id, size = img_size
    if size not in image_sizes:
        image_sizes[size] = []
    image_sizes[size].append(id)
sorted_indices = []
for size in image_sizes:
    sorted_indices.extend(image_sizes[size])
for size in image_sizes:
    print(f"Size: {size}, first img: {image_sizes[size][0]}, last img: {image_sizes[size][-1]}")
    print(f"Size: {size}, {len(image_sizes[size])} images")
# print(image_sizes)  
# print(sorted_indices)
sys.exit(0)


# 按排序後的索引創建數據加載器
batch_size = 16
# data_loader = DataLoader(image_folder, batch_size=batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(sorted_indices))


# create DataLoader for train / dev datasets
train_dataloader = DataLoader(datasets[TRAIN], shuffle=True, batch_size=args["batch_size"])
dev_dataloader = DataLoader(datasets[DEV], shuffle=False, batch_size=args["batch_size"])

# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE} device")

if args["stage"] == "debug":
    model = detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # For Training
    images, targets = next(iter(train_dataloader))
    print(f"Reading {len(images)} data")
    images = list(image for image in images)
    try:
        targets = [{k: v for k, v in t.items()} for t in targets]
        #TODO: fix bug here
    except AttributeError:
        print("len: ", len(targets))
        print("type: ", type(targets))
        for i, t in enumerate(targets):
            print(f"[{i}]")
            print(type(t))
            print(t)
        sys.exit(0)
    output = model(images, targets)   # Returns losses and detections
    # For inference
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)           # Returns predictions
    print("images:", images)
    print("targets:", targets)
    print("model:", model)
    sys.exit(0)

# initialize a dictionary containing model name and its corresponding
# torchvision function call
MODELS = {
    "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
    "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    "retinanet": detection.retinanet_resnet50_fpn
}
# load the model and set it to evaluation mode
num_classes = datasets[TRAIN].num_classes
model = MODELS[args["model"]](pretrained=True, progress=True,
                              num_classes=num_classes, pretrained_backbone=True)

# replace the classifier with a new one, that has num_classes which is user-defined

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.to(DEVICE)


# TODO: train and val mode

"""
Reference:
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
https://www.twblogs.net/a/5d720e21bd9eee5327ff7374
https://pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection

"""