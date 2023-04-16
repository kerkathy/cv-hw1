# create env
conda env create -f environment.yml
conda activate yolo
pip install -r requirements.txt  # install

# clone repo and install
# git clone https://github.com/ultralytics/yolov5  # clone

cd yolov5

# create required directory
# now at yolov5, want to create dir at ../hw1_datasets/
# mkdir ../hw1_datasets/labels

# create yolo txts from coco json
# python ../create_yolo_labels.py

# train from pretrained wolov5 large weight
# python train.py --img-size 640 --batch 16 --epochs 200 --data marine.yaml --weights yolov5l.pt

# TODO model weight link
wget https://www.dropbox.com/s/upkacxo9ejia5kz/model.zip
unzip cv_hw1.zip

cd ..
