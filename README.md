# Object Detection Using CNN-based Network

## Data
The directory structure should look like the below:
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