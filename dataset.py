import torchvision
import os
import json
from pathlib import Path

# TODO: modify json file name
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(img_folder, "_annotations.coco.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

        self.categories = json.loads(Path(ann_file).read_text())['categories']

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return {"pixel_values": pixel_values, "labels": target}
     
    def collate_fn(self, batch):
        pixel_values = [item["pixel_values"] for item in batch]
        encoding = self.processor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
        labels = [item["labels"] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        # batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch
    
    
