from ensurepip import version
from importlib.resources import path
import pytest
from importlib import reload 
import os
import sys
import csv
import subprocess
import shlex
import pathlib
import tensorflow
import openvino_tensorflow
#import numpy as np

ovtf_path = pathlib.Path(__file__).resolve().parents[1]

#Path for  input image, input labels
image_path = pathlib.Path(__file__).parents[1].joinpath('examples/data/grace_hopper.jpg').resolve()
print(image_path)

labels_path = pathlib.Path(__file__).parents[1].joinpath('examples/data/coco.names').resolve()
labels_path_cpp = pathlib.Path(__file__).resolve().parents[1].joinpath('examples/data/imagenet_slim_labels.txt')
print(labels_path)

image_net_path = pathlib.Path(__file__).parents[1].joinpath('examples/data/imagenet_slim_labels.txt').resolve()

input_model_file_path = pathlib.Path(__file__).parents[1].joinpath('examples/data/inception_v3_2016_08_28_frozen.pb')

@pytest.mark.classification
def test_ovtf_default_clf(sample_fixture):
    assert "OVTF Summary" in sample_fixture[0]

@pytest.mark.object_detection
def test_ovtf_default_obj(sample_fixture_obj):
    assert "OVTF Summary" in sample_fixture_obj[0]

def test_default_cpp():
    binary_exe = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'+str(ovtf_path)+\
                '/build_cmake/artifacts/lib:'+str(ovtf_path)+'/build_cmake/artifacts/tensorflow'
    p0 = subprocess.call(binary_exe, shell=True)

    cmd_str = './build_cmake/examples/classification_sample/infer_image'
    print(cmd_str)
    command = shlex.split(cmd_str)
    spout = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = spout.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        print("The process ran more than 10 minutes : ")
        spout.kill()
        out, err = spout.communicate()
    
    encoding = 'utf-8'
    out = out.decode(encoding)
    out = out.split('\n')
    print(*out, sep = '\n')
    assert "OVTF Summary" in out[9] 

@pytest.mark.classification
def test_imagenet_labels():
    count = 0
    f = open(str(image_net_path), 'r')
    for line in f:
        count+=1
    f.close()
    assert count == 1001

@pytest.mark.classification   
@pytest.mark.parametrize('flag', ['ovtf', 'tf', 'ovtf_tf1', 'tf_tf1', 'ovtf_cpp'])
def test_all_flags(flag):
    if 'tf1' in flag: 
        cmd_str = "python3 examples/TF_1_x/classification_sample.py"   
        cmd_str += ' --graph='+str(input_model_file_path)
        cmd_str += ' --input_layer=input'
        cmd_str += ' --output_layer=InceptionV3/Predictions/Reshape_1'
        cmd_str += ' --input='+str(image_path)
        cmd_str += ' --labels='+str(image_net_path)   #for tf1 classification imagenet_slim_labels!!
        cmd_str += ' --input_height='+str(299)
        cmd_str += ' --input_width='+str(299)
        cmd_str += ' --input_mean='+str(0)
        cmd_str += ' --input_std='+str(255)
        cmd_str += ' --no_show'

    elif 'cpp' in flag:
        binary_exe = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'+str(ovtf_path)+\
                    '/build_cmake/artifacts/lib:'+str(ovtf_path)+'/build_cmake/artifacts/tensorflow'
        p0 = subprocess.call(binary_exe, shell=True)   
        cmd_str = './build_cmake/examples/classification_sample/infer_image'
        cmd_str += ' --image='+str(image_path)
        cmd_str += ' --graph='+str(input_model_file_path)
        cmd_str += ' --labels='+str(labels_path_cpp)
        cmd_str += ' --input_width=299'
        cmd_str += ' --input_height=299'
        cmd_str += ' --input_mean=0.00'
        cmd_str += ' --input_std=255.00'
        cmd_str += ' --input_layer=input'
        cmd_str += ' --output_layer=InceptionV3/Predictions/Reshape_1'
     
    else: 
        cmd_str = "python3 examples/classification_sample.py"
        cmd_str += ' --input='+str(image_path)
        cmd_str += ' --labels='+str(labels_path)
        cmd_str += ' --input_height='+str(299)
        cmd_str += ' --input_width='+str(299)
        cmd_str += ' --input_mean='+str(0)
        cmd_str += ' --input_std='+str(255)
        cmd_str += ' --no_show'
        
    if flag == 'tf' or flag == 'tf_tf1':
        cmd_str += ' --disable_ovtf'
    cmd_str += ' --backend='+str('CPU')

    print(cmd_str)
    command = shlex.split(cmd_str)
    spout = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = spout.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        print("The process ran more than 10 minutes : ")
        spout.kill()
        out, err = spout.communicate()
    
    encoding = 'utf-8'
    out = out.decode(encoding)
    out = out.split('\n')
    print(*out, sep='\n')
    
    if flag == 'ovtf' or flag == 'ovtf_tf1':
        assert "OVTF Summary" in out[0]
    elif flag == 'tf' or flag == 'tf_tf1':
        assert "Inference time in ms" in out[0]

model_path = pathlib.Path(__file__).parents[1].resolve().joinpath('examples/data/yolo_v4').resolve()
labels_file_path = pathlib.Path(__file__).parents[1].resolve().joinpath('examples/data/coco.names').resolve()

@pytest.mark.object_detection
@pytest.mark.parametrize('flag', ['ovtf', 'tf', 'ovtf_tf1', 'tf_tf1'])
def test_all_flags_obj(flag):
    if 'tf1' in flag:
        cmd_str = "python3 examples/TF_1_x/object_detection_sample.py" 
        cmd_str += ' --input_mean='+str(0)
        cmd_str += ' --input_std='+str(255)  
        cmd_str += ' --graph='+str(model_path)+str('.pb') 
        cmd_str += ' --input_layer=image_input'
        cmd_str += ' --output_layer="conv2d_109/BiasAdd,conv2d_101/BiasAdd,conv2d_93/BiasAdd"'
    else:
        cmd_str = "python3 examples/object_detection_sample.py"
        cmd_str += ' --model='+str(model_path)
    cmd_str += ' --input='+str(image_path)
    cmd_str += ' --labels='+str(labels_path)
    cmd_str += ' --input_height='+str(416)
    cmd_str += ' --input_width='+str(416)
    cmd_str += ' --backend='+str('CPU')
    cmd_str += ' --no_show'
    #cmd_str += ' --rename'  #--rename is creating problems!!
    cmd_str += ' --conf_threshold='+str(0.6)
    cmd_str += ' --iou_threshold='+str(0.5)
    if flag == 'tf' or flag == 'tf_tf1':
        cmd_str += ' --disable_ovtf'


    print(cmd_str)
    command = shlex.split(cmd_str)
    spout = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = spout.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        print("The process ran more than 10 minutes : ")
        spout.kill()
        out, err = spout.communicate()
    
    encoding = 'utf-8'
    out = out.decode(encoding)
    out = out.split('\n')
    print(*out, sep='\n')
    
    if flag == 'ovtf' or flag == 'ovtf_tf1':
        assert "OVTF Summary" in out[0]
    elif flag == 'tf' or flag == 'tf_tf1':
        assert "Inference time in ms" in out[0]   