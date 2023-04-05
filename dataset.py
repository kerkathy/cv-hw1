import torch
from PIL import Image
import json
from utils import resize_bbox, transform, xywh2xyxy
import cv2

class MarineDataset(torch.utils.data.Dataset):
    def __init__(self, path):

        ann_path = path / "_annotations.coco.json"
        data = json.loads(ann_path.read_text())

        self.root = path
        self.categories = data['categories']
        self.imgs = data['images']

        # label 0 is reserved for background, so we modify the labels by adding 1
        for i in range(len(self.categories)):
            self.categories[i]['id'] += 1

        # Group the annotations by their image_id
        # i.e. convert to a dictionary, where (key, value) = (image_id, list of annotations)
        # Also modify the category_id by adding 1
        self.annotations_dict = {}
        for i in range(len(data['annotations'])):
            data['annotations'][i]['category_id'] += 1
            image_id = data['annotations'][i]['image_id']
            if image_id in self.annotations_dict:
                self.annotations_dict[image_id].append(data['annotations'][i])
            else:
                self.annotations_dict[image_id] = [data['annotations'][i]]
        

        # Draw image from tensor of size (3, 1024, 768) and save it to disk
        # DEBUGGNG
        # img = self.imgs[0]
        # img_name = img['file_name']
        # img_id = int(img['id'])
        # img_path = self.root / img_name
        # img = cv2.imread(str(img_path))
        # print("value of img[0]: ", img)
        # print("img_id: ", img_id)
        # # print("size of img[0]: ", img.size())
        # # img = img.permute(1, 2, 0).numpy()
        # # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # for anno in self.annotations_dict[img_id]:
        #     box = anno['bbox']
        #     cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        # cv2.imwrite(f"debug-{img_name}.jpg", img)
        # # DEBUGGNG
        


        # TODO: transform data for data aug if needed

    @property
    # read-only, can't modify
    def num_classes(self) -> int:
        return len(self.categories) + 1 # +1 for background
    
    def __getitem__(self, idx):
        # load images and masks
        img_path = self.root / self.imgs[idx]['file_name']
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        img_id = self.imgs[idx]['id']

        # get bounding box coordinates for each mask
        num_objs = len(self.annotations_dict[img_id])
        boxes = []
        labels = []
        iscrowd = []
        area = 0
        for i in range(num_objs):
            boxes.append(xywh2xyxy(self.annotations_dict[img_id][i]['bbox']))   # convert the coordinates to xyxy
            labels.append(self.annotations_dict[img_id][i]['category_id'])
            iscrowd.append(self.annotations_dict[img_id][i]['iscrowd'])
            area = self.annotations_dict[img_id][i]['area']

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # boxes = resize_bbox(boxes, self.imgs[idx]['width'], self.imgs[idx]['height'])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([img_id])
        area = torch.as_tensor(area, dtype=torch.float32)

        target = dict()
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # debug
        # print("img.shape: ", img.shape)
        # print(f"target: {len(target['boxes'])} boxes")

        return img, target

    def __len__(self):
        return len(self.imgs)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        return tuple(zip(*batch))
    
        # imgs = list()
        # targets = list()

        # for b in batch:
        #     imgs.append(b[0])
        #     targets.append(b[1])

        # imgs = torch.stack(imgs, dim=0)

        # return imgs, targets # tensor (N, 3, 300, 300), 3 lists of N tensors each