from importlib.resources import path
import pytest
from importlib import reload 
import os
import sys
import csv
import subprocess
import shlex
import pathlib
#import numpy as np

ovtf_dir_path = pathlib.Path(__file__).parents[1].resolve() #openvino tensorflow dir path

#creating empty .pb model file for providing wrong model path
empty_model_file_path = pathlib.Path(__file__).parent.joinpath('empty_model.pb').resolve().touch()

#creating empty .jpg file for providing wrong image path
empty_image_file_path = pathlib.Path(__file__).parent.joinpath('empty_image.jpg').resolve().touch()

@pytest.mark.classification
@pytest.mark.parametrize('wrong_path', ['empty_model_file_path', 'empty_image_file_path',
                                        'empty_model_file_path_tf1', 'empty_image_file_path_tf1',
                                        'empty_model_file_path_cpp', 'empty_image_file_path_cpp'])
def test_negative_scenarios(wrong_path):
    if 'tf1' in wrong_path:
        cmd_str = "python3 examples/TF_1_x/classification_sample.py"       
    elif 'cpp' in wrong_path:
        binary_exe = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'+str(ovtf_dir_path)+\
                '/build_cmake/artifacts/lib:'+str(ovtf_dir_path)+'/build_cmake/artifacts/tensorflow'
        p0 = subprocess.call(binary_exe, shell=True)
        cmd_str = './build_cmake/examples/classification_sample/infer_image'
    else: 
        cmd_str = "python3 examples/classification_sample.py"
        
    if wrong_path == 'empty_model_file_path':
        cmd_str += ' --model='+str(pathlib.Path(__file__).parent.joinpath('empty_model.pb').resolve())
    elif wrong_path == 'empty_model_file_path_cpp':
        cmd_str += ' --graph='+str(pathlib.Path(__file__).parent.joinpath('empty_model.pb').resolve())
    elif wrong_path == 'empty_model_file_path_tf1':
        cmd_str += ' --graph='+str(pathlib.Path(__file__).parent.joinpath('empty_model.pb').resolve())
        cmd_str += ' --input_layer=input'
        cmd_str += ' --output_layer=InceptionV3/Predictions/Reshape_1'
        cmd_str += ' --labels='+str(pathlib.Path(__file__).parents[1].resolve().joinpath('examples/data/imagenet_slim_labels.txt'))
    elif wrong_path == 'empty_image_file_path' or wrong_path == 'empty_image_file_path_tf1':
        cmd_str += ' --input='+str(pathlib.Path(__file__).parent.joinpath('empty_image.jpg').resolve())
    elif wrong_path == 'empty_image_file_path_cpp':
        cmd_str += ' --image='+str(pathlib.Path(__file__).parent.joinpath('empty_image.jpg').resolve())
    
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
    print(*err, sep='\n')

    if wrong_path == 'empty_model_file_path' or wrong_path == 'empty_model_file_path_tf1' or wrong_path == 'empty_model_file_path_cpp':
        with pytest.raises(AssertionError):
            assert "OVTF Summary" in out[0] 
    elif wrong_path == 'empty_image_file_path' or wrong_path == 'empty_image_file_path_tf1' or wrong_path == 'empty_image_file_path_cpp':
        with pytest.raises(Exception):
            assert "OVTF Summary" in out[0]

@pytest.mark.classification
@pytest.mark.parametrize('format', ['image', 'video', 'image_tf1', 'video_tf1', 'image_cpp', 'video_cpp'])
def test_diff_input_img_format(format):  
    if 'tf1' in format:
        cmd_str = "python3 examples/TF_1_x/classification_sample.py"       
    elif 'cpp' in format:
        binary_exe = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'+str(ovtf_dir_path)+\
                '/build_cmake/artifacts/lib:'+str(ovtf_dir_path)+'/build_cmake/artifacts/tensorflow'
        p0 = subprocess.call(binary_exe, shell=True)
        cmd_str = './build_cmake/examples/classification_sample/infer_image'
    else: 
        cmd_str = "python3 examples/classification_sample.py"

    if format == 'image' or format == 'image_tf1':
        cmd_str += ' --input='+str(ovtf_dir_path.joinpath('examples/data/coco.names'))
    elif format == 'image_cpp':
        cmd_str += ' --image='+str(ovtf_dir_path.joinpath('examples/data/coco.names'))
    elif format == 'video' or format == 'video_tf1':
        cmd_str += ' --input='+str(ovtf_dir_path.joinpath('examples/data/people-detection1.abc'))  
    elif format == 'video_cpp':
        cmd_str += ' --image='+str(ovtf_dir_path.joinpath('examples/data/people-detection1.abc'))
    #cmd_str += ' --no_show'

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
    with pytest.raises(Exception):
        print(Exception)
        assert "OVTF Summary" in out[0]
    

@pytest.mark.object_detection
@pytest.mark.parametrize('wrong_path', ['empty_model_file_path', 'empty_image_file_path',
                                        'empty_model_file_path_tf1', 'empty_image_file_path_tf1'])
def test_negative_scenarios_obj(wrong_path):
    if 'tf1' in wrong_path:
        cmd_str = "python3 examples/TF_1_x/object_detection_sample.py"       
    else: 
        cmd_str = "python3 examples/object_detection_sample.py"
    if wrong_path == 'empty_model_file_path' :
        cmd_str += ' --model='+str(pathlib.Path(__file__).parent.joinpath('empty_model.pb').resolve())
    elif wrong_path == 'empty_model_file_path_tf1':
        cmd_str += ' --graph='+str(pathlib.Path(__file__).parent.joinpath('empty_model.pb').resolve())
        cmd_str += ' --input_layer=image_input'
        cmd_str += ' --output_layer="conv2d_109/BiasAdd,conv2d_101/BiasAdd,conv2d_93/BiasAdd"'
        cmd_str += ' --labels='+str(pathlib.Path(__file__).parents[1].resolve().joinpath('examples/data/coco.names'))
    elif wrong_path == 'empty_image_file_path' or wrong_path == 'empty_image_file_path_tf1':
        cmd_str += ' --input='+str(pathlib.Path(__file__).parent.joinpath('empty_image.jpg').resolve())
    
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
    print(*err, sep='\n')


    if wrong_path == 'empty_model_file_path' or wrong_path == 'empty_model_file_path_tf1':
        with pytest.raises(AssertionError):
            assert "OVTF Summary" in out[0] 
    elif wrong_path == 'empty_image_file_path' or wrong_path == 'empty_image_file_path_tf1':
        with pytest.raises(Exception):
            assert "OVTF Summary" in out[0]


@pytest.mark.object_detection
@pytest.mark.parametrize('format', ['image', 'video', 'image_tf1', 'video_tf1'])
def test_diff_input_img_format_obj(format):  
    if 'tf1' in format:
        cmd_str = 'python3 examples/TF_1_x/object_detection_sample.py'
    else:    
        cmd_str = 'python3 examples/object_detection_sample.py'
    if format == 'image' or format == 'image_tf1':
        cmd_str += ' --input='+str(ovtf_dir_path.joinpath('examples/data/coco.names'))
    elif format == 'video' or format == 'video_tf1':
        cmd_str += ' --input='+str(ovtf_dir_path.joinpath('examples/data/people-detection1.abc'))  
    cmd_str += ' --no_show'

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
    with pytest.raises(Exception):
        print(Exception)
        assert "OVTF Summary" in out[0]
