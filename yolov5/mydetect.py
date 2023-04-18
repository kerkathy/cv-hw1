import torch
import os
import json
import sys

if(len(sys.argv) < 2):
    raise ValueError("Specify the data location and output path. Should be `mydetect.py <path_to_data> <path_to_output>` ")
folder = sys.argv[1]  # or file, Path, URL, PIL, OpenCV, numpy, list
output_path = sys.argv[2]

# Model
model_path = "runs/train/exp7/weights/best.pt"
# sys.path.insert(0, './runs')
# model = torch.hub.load('ultralytics/yolov5', 'custom', model_path)  # custom model
model = torch.hub.load('.', 'custom', model_path, source='local')  # custom model
print(f"Inference using {model_path}")
# Images
img_names = [name for name in os.listdir(folder) if name.endswith(".jpg")]
im = [os.path.join(folder, img) for img in img_names]
print(f"{len(im)} test images in {os.path.abspath(folder)}.\n")
if len(im) == 0:
    print(os.listdir(folder))
    raise ValueError(f"No image to detect in {os.path.abspath(folder)}")

# Inference
results = model(im)
# Results

# results.xyxy[0]  # im predictions (tensor)
with open(output_path, 'w') as f:
    all_results = {}	
    for img, result in zip(img_names, results.pandas().xyxy):
    	result["boxes"] = result[["xmin", "ymin", "xmax", "ymax"]].values.tolist()
    	out_dict = {}
    	out_dict["boxes"] = result["boxes"].tolist()
    	out_dict["scores"] = result["confidence"].tolist()
    	out_dict["labels"] = result["class"].tolist()
    	all_results[img] = out_dict
    json.dump(all_results, f)  
print(f"Results saved to {os.path.abspath(output_path)}")

# im predictions (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
