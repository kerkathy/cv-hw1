import torch
import os
import json

# Model
model_path = "runs/train/exp7/weights/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', model_path)  # custom model

# Images
folder = "../hw1_dataset/images/valid"  # or file, Path, URL, PIL, OpenCV, numpy, list
img_names = [name for name in os.listdir(folder) if name.endswith(".jpg")]
im = [os.path.join(folder, img) for img in img_names]
print(f"{len(im)} test images in total.")
if len(im) == 0:
    print(os.listdir(folder))
    raise NotImplementedError

# Inference
results = model(im)
# Results

# results.xyxy[0]  # im predictions (tensor)
with open("eval_output.json", 'w') as f:
    all_results = {}	
    for img, result in zip(img_names, results.pandas().xyxy):
        result["boxes"] = result[["xmin", "ymin", "xmax", "ymax"]].apply(lambda x: ', '.join(x.astype(str)), axis=1)
        out_dict = {}
        out_dict["boxes"] = result["boxes"].tolist()
        out_dict["scores"] = result["confidence"].tolist()
        out_dict["labels"] = result["class"].tolist()
        all_results[img] = out_dict
    json.dump(all_results, f)  
    
# im predictions (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
