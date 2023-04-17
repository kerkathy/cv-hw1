# Object Detection Using CNN-based Network

## YOLO
### Before you start
We should already have `yolov5` folder in our repo. But in case you don't have one, clone it from official repo.
```
# run this only if you didn't see `yolov5` folder in current directory
git clone https://github.com/ultralytics/yolov5
```

### Environment
First, change working directory to `yolov5`. Then do the below.
```
conda env create -f environment.yml
conda actiavte yolo
pip install -r requirements.txt
```
### Preprocess
Create required file structure for yolo. Then, create yolo txts from coco json files.
```
mkdir ../hw1_datasets/labels
python ../create_yolo_labels.py
```

### Train
train the model from pretrained yolov5 large weights.
```
python train.py --img-size 640 --batch 16 --epochs 200 --data marine.yaml --weights yolov5l.pt

```

### Inference
Provide 
- {1}: the path to test img folder
- {2}: path to output file
To get the Map numbers, run the below:
```
python mydetect.py ${1} ${2}
# for example
# python mydetect.py images/valid output.json

```
To draw bbox on each image, run the below:
```
# some other options are provided, go to official repo to see.
python detect.py --weights "<path_to_weight.pt>" --source "images/test/*" --save-txt
# for example,
# python detect.py --weights "runs/train/exp/weights/best.pt" --source "images/test/*" --save-txt

```


### Data Structure for YOLO
When training YOLO, make sure the dataset folder contains
- images
--- train
--- valid
--- test
- labels
--- train
--- valid
Also, state this directory path as required in your mydataset.yaml (in this repo, it's called marine.yaml)

### Original Data 
The provided directory structure looks like the below:
```
.
├── train
├── valid
│   ├── XXXX.jpg
│   ├── XXXX.jpg
│   ├── XXXX.jpg
│   └── .annotations.coco.json
├── test
├── sample_submission.json
```

where the structure of `.annotations.coco.json` looks like
```
{
    "categories": [
        {
            "id": 0,
            "name": "creatures",
            "supercategory": "none"
        }, ...
    ], 
    "images":[
        {
            "id": 0,
            "license": 1,
            "file_name": "IMG_3120_jpeg_jpg.rf.05e302318ebf9502b3467828b1f1e45a.jpg",
            "height": 1024,
            "width": 768,
            "date_captured": "2020-11-18T19:53:47+00:00"
        }, ...
    ], 
    "annotations": [
        {
            "id": 0,
            "image_id": 0,
            "category_id": 6,
            "bbox": [
                138,
                316,
                391,
                364
            ],
            "area": 142324,
            "segmentation": [],
            "iscrowd": 0
        }, ...
    ]
}
``` 
