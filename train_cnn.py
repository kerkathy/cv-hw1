# Use Pytorch to fine-tune object detection model using CNN based network

# import the necessary packages
from torchvision.models import detection
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import argparse
import torch
import sys
from pathlib import Path
from tqdm import tqdm

from dataset import MarineDataset

TRAIN = "train"
DEV = "valid"
SPLITS = [TRAIN, DEV]


# construct the argument parser and parse the arguments
def main(args):
    data_paths = {split: args["data_dir"] / split for split in SPLITS}

    datasets: dict[str, MarineDataset] = {
        split: MarineDataset(path)
        for split, path in data_paths.items()
    }
    print(f"Loaded {len(datasets[TRAIN])} training images")
    print(f"Loaded {len(datasets[DEV])} validation images")

    # create DataLoader for train / dev datasets
    train_dataloader = DataLoader(datasets[TRAIN], shuffle=True, batch_size=args["batch_size"], collate_fn=datasets[TRAIN].collate_fn)
    dev_dataloader = DataLoader(datasets[DEV], shuffle=False, batch_size=args["batch_size"], collate_fn=datasets[DEV].collate_fn)

    # set the device we will be using to run the model
    DEVICE = torch.device(f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu")
    print(f"Using {DEVICE} device")
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    if args["stage"] == "debug":
        model = detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        # For Training
        images, targets = next(iter(train_dataloader))
        print(f"Reading {len(images)} data")

        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        # print("images:", images)
        # print("targets:", targets)

        output = model(images, targets)   # Returns losses and detections
        # For inference
        model.eval()
        x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        predictions = model(x)           # Returns predictions
        # print("model:", model)
        print("output:", output)
        print("predictions:", predictions)
        sys.exit(0)

    # initialize a dictionary containing model name and its corresponding
    # torchvision function call
    MODELS = {
        "frcnn-resnet": detection.fasterrcnn_resnet50_fpn(weights="DEFAULT"),
        "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
        "retinanet": detection.retinanet_resnet50_fpn
    }
    # load the model and set it to evaluation mode
    num_classes = datasets[TRAIN].num_classes
    model = MODELS[args["model"]](pretrained=True)

    # replace the classifier with a new one, that has num_classes which is user-defined

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(DEVICE)


    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)

    num_epochs =  args["epoch"]

    for epoch in range(num_epochs):
        # train for one epoch
        train_one_epoch(model, optimizer, train_dataloader, DEVICE)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, dev_dataloader, device=DEVICE)


def train_one_epoch(model, optimizer, train_dataloader, device):
    """
    Train for one epoch
    """
    # set model to training mode
    model.train()
    # initialize the loss
    loss = 0
    # iterate over the data
    for images, targets in tqdm(train_dataloader):
        # move the images and targets to the device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        output = model(images, targets)
        # backward pass
        loss = sum(loss for loss in output.values())
        loss.backward()
        # update the weights
        optimizer.step()
        # update the loss
        loss += loss.item()
    # print the loss
    print(f"Training loss: {loss / len(train_dataloader)}")


def evaluate(model, dev_dataloader, device):
    """
    Evaluate a model on a dataset
    """
    # set model to evaluation mode
    model.eval()
    # initialize the loss
    loss = 0
    # iterate over the data
    for images, targets in tqdm(dev_dataloader):
        # move the images and targets to the device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # forward pass
        with torch.no_grad():
            output = model(images, targets)
        # update the loss
        loss += sum(loss for loss in output.values()).item()
    # print the loss
    print(f"Validation loss: {loss / len(dev_dataloader)}")


def parse():
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
    ap.add_argument("--gpu", type=str, default= "0", help="GPU ID to use")
    ap.add_argument("-e", "--epoch", type=int, default=10, help="number of epochs to train")
    args = vars(ap.parse_args())
    return args

if __name__ == "__main__":
    args = parse()
    main(args)

"""
Reference:
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
https://www.twblogs.net/a/5d720e21bd9eee5327ff7374
https://pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection

"""


"""
Archive:

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

# 按排序後的索引創建數據加載器
batch_size = 16
# data_loader = DataLoader(image_folder, batch_size=batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(sorted_indices))

"""