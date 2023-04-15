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

# training
python train.py --img-size 640 --batch 16 --epochs 60 --data marine.yaml --weights yolov5s.pt

# inference and draw bbox on each image
# python detect.py --weights "runs/train/exp7/weights/best.pt" --source "../hw1_dataset/images/test/*" --save-txt

# inference and generate output json file
# TODO: modify this to command line args
python mydetect.py

