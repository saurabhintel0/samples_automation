import cmd
from faulthandler import disable
import pytest
from importlib import reload 
import os
import sys
import csv
import subprocess
import shlex
import pathlib
#import numpy as np

ovtf_path = pathlib.Path(__file__).resolve().parents[1]

labels_file_path = pathlib.Path(__file__).parents[1].resolve().joinpath('examples/data/coco.names')

#tf1 clf model used =examples/data/inception_v3_2016_08_28_frozen.pb
tf1_clf_model_path = pathlib.Path(__file__).parents[1].resolve().joinpath('examples/data/inception_v3_2016_08_28_frozen.pb')

#tf1_clf_labels_path = 
tf1_clf_labels_path = pathlib.Path(__file__).parents[1].resolve().joinpath('examples/data/imagenet_slim_labels.txt')

#input_image
tf1_input_image = pathlib.Path(__file__).parents[1].resolve().joinpath('examples/data/grace_hopper.jpg')

@pytest.mark.classification
@pytest.mark.parametrize('flag', ['no_show', 'help', 'labels', 'height', 'width', 'mean', 'std', 'disable_ovtf', 'backend',
                        'graph_tf1', 'input_layer_tf1', 'output_layer_tf1', 'input_tf1', 'no_show_tf1', 'help_tf1', 'labels_tf1', 
                        'height_tf1', 'width_tf1', 'mean_tf1', 'std_tf1', 'disable_ovtf_tf1', 'backend_tf1', 'backend_cpp',
                        'image_cpp', 'graph_cpp', 'labels_cpp', 'input_width_cpp', 'input_height_cpp', 'input_mean_cpp', 'input_std_cpp',
                        'input_layer_cpp', 'output_layer_cpp', 'help_cpp'])  
def test_help_flags(flag):
    if 'tf1' in flag:
        cmd_str = "python3 examples/TF_1_x/classification_sample.py"
        if flag == 'graph_tf1':
            cmd_str += ' --graph='+str(tf1_clf_model_path)
            cmd_str += ' --input_layer=input'
            cmd_str += ' --output_layer=InceptionV3/Predictions/Reshape_1'
            cmd_str += ' --labels='+str(tf1_clf_labels_path)
    elif 'cpp' in flag:
        binary_exe = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'+str(ovtf_path)+\
                '/build_cmake/artifacts/lib:'+str(ovtf_path)+'/build_cmake/artifacts/tensorflow'
        p0 = subprocess.call(binary_exe, shell=True)
        cmd_str = './build_cmake/examples/classification_sample/infer_image'
    else:
        cmd_str = "python3 examples/classification_sample.py"

    if flag == 'no_show' or flag =='no_show_tf1':
        cmd_str += " --no_show"
    elif flag == 'help' or flag == 'help_tf1' or flag == 'help_cpp':
        cmd_str += " --help"
    elif flag == 'labels':
        cmd_str += ' --labels='+str(labels_file_path)
    elif flag == 'labels_tf1' or flag == 'labels_cpp':
        cmd_str += ' --labels='+str(tf1_clf_labels_path)
    elif flag == 'height' or flag == 'height_tf1' or flag == 'input_height_cpp':
        cmd_str += ' --input_height='+str(299)
    elif flag == 'width' or flag == 'width_tf1' or flag == 'input_width_cpp':
        cmd_str += ' --input_width='+str(299)
    elif flag == 'mean' or flag == 'mean_tf1' or flag == 'input_mean_cpp':
        cmd_str += ' --input_mean='+str(0)
    elif flag == 'std' or flag == 'std_tf1' or flag == 'input_std_cpp':
        cmd_str += ' --input_std='+str(255)
    elif flag == 'disable_ovtf' or flag == 'disable_ovtf_tf1':
        cmd_str += ' --disable_ovtf'
    elif flag == 'backend' or flag == 'backend_tf1' or flag == 'backend_cpp':
        cmd_str += ' --backend=CPU'
    elif flag == 'input_layer_tf1' or flag == 'input_layer_cpp':
        cmd_str += ' --input_layer=input'
    elif flag == 'output_layer_tf1' or flag == 'output_layer_cpp':
        cmd_str += ' --output_layer=InceptionV3/Predictions/Reshape_1'
    elif flag == 'input_tf1':
        cmd_str += ' --input='+str(tf1_input_image)
    elif flag == 'image_cpp':
        cmd_str += ' --image='+str(tf1_input_image)
    elif flag == 'graph_cpp':
        cmd_str += ' --graph='+str(tf1_clf_model_path)
    
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
   
    err = err.decode(encoding)
    err = err.split('\n')
    
    if flag =='help' or flag == 'help_tf1' or flag == 'help_cpp':
        assert 'usage:' in out[0]
    elif flag == 'disable_ovtf' or flag == 'disable_ovtf_tf1':
        assert "Inference time in ms" in out[0]
    elif 'cpp' in flag:
        assert "OVTF Summary" in out[9] 
    else:
        assert "OVTF Summary" in out[0]


model_path = pathlib.Path(__file__).parents[1].resolve().joinpath('examples/data/yolo_v4').resolve()
labels_file_path = pathlib.Path(__file__).parents[1].resolve().joinpath('examples/data/coco.names').resolve()
model_path_tf1_obj =  pathlib.Path(__file__).parents[1].resolve().joinpath('examples/data/yolo_v4.pb').resolve()
data_folder_path = pathlib.Path(__file__).parents[1].resolve().joinpath('examples/data')

@pytest.mark.object_detection
@pytest.mark.parametrize('flag', ['model', 'no_show', 'help', 'labels', 'height', 'width', 'disable_ovtf', 'conf_threshold', 'iou_threshold', 'input', 'backend',
                                'graph_tf1', 'input_layer_tf1', 'output_layer_tf1', 'input_tf1', 'no_show_tf1', 'help_tf1', 'labels_tf1', 'height_tf1', 
                                'width_tf1', 'disable_ovtf_tf1', 'conf_threshold_tf1', 'iou_threshold_tf1', 'backend_tf1', 'mean_tf1', 'std_tf1'])
def test_help_flags_obj(flag):
    if 'tf1' in flag:
        cmd_str = "python3 examples/TF_1_x/object_detection_sample.py"
        if flag == 'graph_tf1':
            cmd_str += ' --graph='+str(model_path_tf1_obj)
            cmd_str += ' --input_layer=image_input'
            cmd_str += ' --output_layer="conv2d_109/BiasAdd,conv2d_101/BiasAdd,conv2d_93/BiasAdd"'
            cmd_str += ' --labels='+str(labels_file_path)
    else:
        cmd_str = "python3 examples/object_detection_sample.py"
    if flag == 'model':
        cmd_str += ' --model='+str(model_path)+' --labels='+str(labels_file_path)  
    elif flag == 'no_show' or flag=='no_show_tf1':
        cmd_str += " --no_show"
    elif flag == 'help' or flag=='help_tf1':
        cmd_str += " --help"
    elif flag == 'labels' or flag == 'labels_tf1':
        cmd_str += ' --labels='+str(labels_file_path)
    elif flag == 'height' or flag == 'height_tf1':
        cmd_str += ' --input_height='+str(416)
    elif flag == 'width' or flag == 'width_tf1':
        cmd_str += ' --input_width='+str(416)
    elif flag == 'disable_ovtf' or flag == 'disable_ovtf_tf1':
        cmd_str += ' --disable_ovtf'
    elif flag == 'conf_threshold' or flag == 'conf_threshold_tf1':
        cmd_str += ' --conf_threshold='+str(0.6)
    elif flag == 'iou_threshold' or flag == 'iou_threshold_tf1':
        cmd_str += ' --iou_threshold='+str(0.5)
    elif flag == 'input' or flag == 'input_tf1':
        cmd_str += ' --input='+str(tf1_input_image)
    elif flag == 'mean_tf1':
        cmd_str += ' --input_mean='+str(0)
    elif flag == 'std_tf1':
        cmd_str += ' --input_std='+str(255)
    elif flag == 'input_layer_tf1':
        cmd_str += ' --input_layer=input_image'
    elif flag == 'output_layer_tf1':
        cmd_str += ' --output_layer="conv2d_109/BiasAdd,conv2d_101/BiasAdd,conv2d_93/BiasAdd"'
    elif flag == 'backend' or flag == 'backend_tf1':
        cmd_str += ' --backend=CPU'  
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
    #print(out[0])
    
    err = err.decode(encoding)
    err = err.split('\n')
    
    if flag =='help' or flag == 'help_tf1': 
        assert 'usage:' in out[0]
    elif flag == 'disable_ovtf' or flag == 'disable_ovtf_tf1':
        assert "Inference time in ms" in out[0]
    else:
        assert "OVTF Summary" in out[0]
"""
    elif flag == 'rename' or flag == 'rename_tf1':
        path1 = data_folder_path.joinpath('grace_hopper-person-tie-IntelOpenVINO.jpg')
        if path1.is_file():
            cmd = 'mv -v '+str(data_folder_path)+str("/grace_hopper-person-tie-IntelOpenVINO.jpg ")+str(data_folder_path)+str("/grace_hopper.jpg")
            p1 = subprocess.call(cmd, shell=True)        
        assert "OVTF Summary" in out[0]
    else:
        assert "OVTF Summary" in out[0]
"""

#grace_hopper-person-tie-IntelOpenVINO.jpg
@pytest.mark.object_detection
def test_rename_image():
    cmd_str = "yes | python3 examples/object_detection_sample.py --rename"
    p1 = subprocess.call(cmd_str, shell=True)  

    path1 = data_folder_path.joinpath('grace_hopper-person-tie-IntelOpenVINO.jpg')
    if path1.is_file():
        cmd = 'mv -v '+str(data_folder_path)+str("/grace_hopper-person-tie-IntelOpenVINO.jpg ")+str(data_folder_path)+str("/grace_hopper.jpg")
        p1 = subprocess.call(cmd, shell=True)        
        
    assert p1 == 0