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
non_default_test_image = ovtf_path.joinpath('automation')

#cat.jpg, car.jpg, fish.jpg, people_detection.mp4

@pytest.mark.classification
@pytest.mark.parametrize('input_image_or_video', ['cat', 'car', 'fish', 'people-detection',
                                                'cat_tf1', 'car_tf1', 'fish_tf1', 'people-detection_tf1',
                                                'cat_cpp', 'car_cpp', 'fish_cpp'])
def test_non_default_images_video(input_image_or_video):
    if 'tf1' in str(input_image_or_video):
        cmd_str = "python3 examples/TF_1_x/classification_sample.py"  
    elif 'cpp' in input_image_or_video:
        binary_exe = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'+str(ovtf_path)+\
                '/build_cmake/artifacts/lib:'+str(ovtf_path)+'/build_cmake/artifacts/tensorflow'
        p0 = subprocess.call(binary_exe, shell=True)
        cmd_str = './build_cmake/examples/classification_sample/infer_image'
    else:  
        cmd_str = "python3 examples/classification_sample.py"
    #cmd_str += " --no_show"
    if input_image_or_video == 'cat' or input_image_or_video == 'cat_tf1':
        cmd_str += " --input="+str(non_default_test_image.joinpath('cat.jpg'))
    elif input_image_or_video == 'car' or input_image_or_video == 'car_tf1':
        cmd_str += " --input="+str(non_default_test_image.joinpath('car.jpg'))
    elif input_image_or_video == 'fish' or input_image_or_video == 'fish':
        cmd_str += " --input="+str(non_default_test_image.joinpath('fish.jpg'))
    elif 'people-detection' in str(input_image_or_video):
        cmd_str += " --input="+str(non_default_test_image.joinpath('people-detection.mp4'))
    elif input_image_or_video == 'cat_cpp':
        cmd_str += " --image="+str(non_default_test_image.joinpath('cat.jpg'))
    elif input_image_or_video == 'car_cpp':
        cmd_str += " --image="+str(non_default_test_image.joinpath('car.jpg'))
    elif input_image_or_video == 'fish_cpp':
        cmd_str += " --image="+str(non_default_test_image.joinpath('fish.jpg'))

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
    if 'cpp' in input_image_or_video:
        assert "OVTF Summary" in out[9]
    else:    
        assert "OVTF Summary" in out[0] 

@pytest.mark.object_detection
@pytest.mark.parametrize('input_image_or_video', ['cat', 'car', 'fish', 'people-detection',
                                                'cat_tf1', 'car_tf1', 'fish_tf1', 'people-detection_tf1'  ])
def test_non_default_images_video_obj(input_image_or_video):
    if 'tf1' in input_image_or_video:
        cmd_str = "python3 examples/object_detection_sample.py"
    else:    
        cmd_str = "python3 examples/object_detection_sample.py"
    cmd_str += " --no_show"
    if 'cat' in str(input_image_or_video):
        cmd_str += " --input="+str(non_default_test_image.joinpath('cat.jpg'))
    elif 'car' in str(input_image_or_video):
        cmd_str += " --input="+str(non_default_test_image.joinpath('car.jpg'))
    elif 'fish' in str(input_image_or_video):
        cmd_str += " --input="+str(non_default_test_image.joinpath('fish.jpg'))
    elif 'people-detection' in str(input_image_or_video):
        cmd_str += " --input="+str(non_default_test_image.joinpath('people-detection.mp4'))

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
    assert "OVTF Summary" in out[0] 