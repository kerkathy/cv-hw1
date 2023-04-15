from transformers import DetrImageProcessor, DetrForObjectDetection
from dataset import CocoDetection
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import torch
import os
import sys
from datasets.coco_eval import CocoEvaluator 

model_checkpoint = "../detr-object-detection-finetuned"
data_path = Path('../hw1_dataset/images')

processor = DetrImageProcessor.from_pretrained(model_checkpoint)

val_dataset = CocoDetection(img_folder=data_path / 'valid', processor=processor, train=False)
val_dataloader = DataLoader(val_dataset, collate_fn=val_dataset.collate_fn, batch_size=2)

id2label = {x["id"]: x["name"] for x in val_dataset.categories}
label2id = {v: k for k, v in id2label.items()}
print(f"id2label: {id2label}")

model = DetrForObjectDetection.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

""" Evaluation using Map"""
def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

import numpy as np

# initialize evaluator with ground truth (gt)
evaluator = CocoEvaluator(coco_gt=val_dataset.coco, iou_types=["bbox"])

print("Running evaluation...")
for idx, batch in enumerate(tqdm(val_dataloader)):
    # get the inputs
    # pixel_values = batch["pixel_values"].to(device)
    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

    # forward pass
    with torch.no_grad():
      outputs = model(pixel_values=batch["pixel_values"].to(device), pixel_mask=batch["pixel_mask"].to(device))
    # print(f"dim of pixel_values = {batch['pixel_values'].shape}")

    # print(f"Shape of output logits: {outputs.logits.shape}, output pred_boxes: {outputs.pred_boxes.shape}")
    # turn into a list of dictionaries (one item for each example in the batch)
    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=orig_target_sizes)

    # provide to metric
    # metric expects a list of dictionaries, each item 
    # containing image_id, category_id, bbox and score keys 
    predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
    evaluator.update(predictions)

print("[debugging]")
evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()
