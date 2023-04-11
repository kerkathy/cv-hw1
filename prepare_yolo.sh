# create env
conda env create -f environment.yml

# clone repo and install
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install

# create required directory
# now at yolov5, want to create dir at ../hw1_datasets/
mkdir ../hw1_datasets/labels

# create yolo txts from coco json
python create_yolo_labels.py
