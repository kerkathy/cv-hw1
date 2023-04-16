# create env
conda env create -f environment.yml

# clone repo and install
# git clone https://github.com/ultralytics/yolov5  # clone

cd yolov5
pip install -r requirements.txt  # install
# TODO: specify torch / torchvision version

# create required directory
# now at yolov5, want to create dir at ../hw1_datasets/
mkdir ../hw1_datasets/labels

# create yolo txts from coco json
python ../create_yolo_labels.py

# train from pretrained wolov5 large weight
# python train.py --img-size 640 --batch 16 --epochs 200 --data marine.yaml --weights yolov5l.pt

# TODO model weight link
wget <path_to_dropbox>
unzip cv_hw1.zip

cd ..
