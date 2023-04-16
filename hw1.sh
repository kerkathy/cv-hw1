# run by using `bash hw1.sh <test_img_dir> <path_to_output_file>
# inference and generate output json file
cd yolov5
python mydetect.py ${1} ${2}  

