#!/bin/bash

pwd

#installing pytest packages 
pip install pytest

#installing packages for generating excel and html reports
pip install pytest-html
pip install pytest-excel

OVTF_PATH=$PWD
cd examples/
pwd

pip3 install -r requirements.txt

#download model for TF2 object detection
chmod +x convert_yolov4.sh
./convert_yolov4.sh

#download model for TF1 classification
curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" | tar -C ${OVTF_PATH}/examples/data -xz

#download model for TF1 object detection
cd TF_1_x
chmod +x convert_yolov4.sh
./convert_yolov4.sh

cd ../..
pwd