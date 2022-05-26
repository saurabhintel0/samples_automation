from turtle import back
import pytest
from importlib import reload 
import os
import sys
import csv
import subprocess
import shlex
#import numpy as np
import pathlib

tenk_images_path = pathlib.Path(__file__).resolve().parents[0].joinpath('tenk_images')
ten_videos_path = pathlib.Path(__file__).resolve().parents[0].joinpath('people-detection.mp4')
no_of_iterations = 10

@pytest.mark.classification
@pytest.mark.parametrize('backend',  ["CPU", "GPU" ,"MYRIAD" ,"VAD-M", "CPU_tf1", "GPU_tf1" ,"MYRIAD_tf1" ,"VAD-M_tf1",
                                        "CPU_tf", "CPU_tf1_tf"])
def test_10k_images(backend):
    #copy create_images.py from automation folder to automation/tenk_images folder
    cmd = 'cp automation/create_images.py automation/tenk_images'
    p0 = subprocess.call(cmd, shell=True) 

    #creating 10k images in 'automation/tenk_images' folder
    create_10k_images = 'python3 automation/tenk_images/create_images.py'
    p1 = subprocess.call(create_10k_images, shell=True)

    #delete create_images.py file
    delete_create_images_file = 'rm automation/tenk_images/create_images.py'
    p2 = subprocess.call(delete_create_images_file, shell=True)

    if 'tf1' in backend:
        cmd_str = "python3 examples/TF_1_x/classification_sample.py"
    else:    
        cmd_str = "python3 examples/classification_sample.py"
    cmd_str += " --no_show"
    cmd_str += " --input="+str(tenk_images_path)
    if backend[-4:] == '_tf1':
        cmd_str += ' --backend='+str(backend[:-4])
    elif backend == 'CPU_tf' or backend == 'CPU_tf1_tf':
        cmd_str += ' --backend=CPU'
        cmd_str += ' --disable_ovtf'
    else:
        cmd_str += " --backend="+str(backend)

    print(cmd_str)

    command = shlex.split(cmd_str)
    spout = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = spout.communicate(timeout=6000)
    except subprocess.TimeoutExpired:
        print("The process ran more than 10 minutes : ")
        spout.kill()
        out, err = spout.communicate()
    
    encoding = 'utf-8'
    out = out.decode(encoding)
    out = out.split('\n')
    print(*out, sep = '\n')
    if backend[-3:] == '_tf':
        assert "Inference time in ms" in out[0]
    else:
        assert "OVTF Summary" in out[0] 

@pytest.mark.classification
@pytest.mark.parametrize('backend',  ["CPU", "GPU" ,"MYRIAD" ,"VAD-M",  "CPU_tf1", "GPU_tf1" ,"MYRIAD_tf1" ,"VAD-M_tf1"])
def test_10k_images_10_iterations(backend):
    if 'tf1' in backend:
        cmd_str = "python3 examples/TF_1_x/classification_sample.py"
    else:    
        cmd_str = "python3 examples/classification_sample.py"
    cmd_str += " --no_show"
    cmd_str += " --input="+str(tenk_images_path)
    if 'tf1' in backend:
        cmd_str += ' --backend='+str(backend[:-4])
    else:
        cmd_str += " --backend="+str(backend)

    print(cmd_str)

    for i in range(no_of_iterations):
        command = shlex.split(cmd_str)
        spout = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = spout.communicate(timeout=6000)
        except subprocess.TimeoutExpired:
            print("The process ran more than 10 minutes : ")
            spout.kill()
            out, err = spout.communicate()
    
    encoding = 'utf-8'
    out = out.decode(encoding)
    out = out.split('\n')
    print(*out, sep = '\n')
    assert "OVTF Summary" in out[0] 


#stress tests for videos
@pytest.mark.classification
@pytest.mark.parametrize('vid', ['tf1', 'tf2'])
def test_stress_video(vid):
    if 'tf1' in vid:
        cmd_str = "python3 examples/TF_1_x/classification_sample.py"
    else:    
        cmd_str = "python3 examples/classification_sample.py"
    cmd_str += " --no_show"
    cmd_str += " --input="+str(ten_videos_path)

    print(cmd_str)

    command = shlex.split(cmd_str)
    for i in range(no_of_iterations):
        spout = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = spout.communicate(timeout=600000)
        except subprocess.TimeoutExpired:
            print("The process ran more than 10 minutes : ")
            spout.kill()
            out, err = spout.communicate()
    
    encoding = 'utf-8'
    out = out.decode(encoding)
    out = out.split('\n')
    print(*out, sep = '\n')
    assert "OVTF Summary" in out[0] 



@pytest.mark.object_detection
@pytest.mark.parametrize('backend',  ["CPU", "GPU" ,"MYRIAD" ,"VAD-M",  "CPU_tf1", "GPU_tf1" ,"MYRIAD_tf1" ,"VAD-M_tf1",
                                        "CPU_tf","CPU_tf1_tf"])
def test_10k_images_obj(backend):
    if 'tf1' in backend:
        cmd_str = "python3 examples/TF_1_x/object_detection_sample.py"
    else:    
        cmd_str = "python3 examples/object_detection_sample.py"
    cmd_str += " --no_show"
    cmd_str += " --input="+str(tenk_images_path)

    if backend == 'CPU_tf1' or backend == 'GPU_tf1' or backend == 'MYRIAD_tf1' or backend == 'VAD-M_tf1':
        cmd_str += ' --backend='+str(backend[:-4])
    elif backend == 'CPU_tf' or backend == 'CPU_tf1_tf':
        cmd_str += ' --backend=CPU'
        cmd_str += ' --disable_ovtf'
    else:
        cmd_str += " --backend="+str(backend)

    print(cmd_str)

    command = shlex.split(cmd_str)
    spout = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = spout.communicate(timeout=6000)
    except subprocess.TimeoutExpired:
        print("The process ran more than 10 minutes : ")
        spout.kill()
        out, err = spout.communicate()
    
    encoding = 'utf-8'
    out = out.decode(encoding)
    out = out.split('\n')
    print(*out, sep = '\n')
    if backend[-3:] == '_tf':
        assert "Inference time in ms" in out[0]
    else:
        assert "OVTF Summary" in out[0] 

@pytest.mark.object_detection
@pytest.mark.parametrize('backend',  ["CPU", "GPU" ,"MYRIAD" ,"VAD-M",  "CPU_tf1", "GPU_tf1" ,"MYRIAD_tf1" ,"VAD-M_tf1"])
def test_10k_images_10_iterations_obj(backend):
    if 'tf1' in backend:
        cmd_str = "python3 examples/TF_1_x/object_detection_sample.py"
    else:    
        cmd_str = "python3 examples/object_detection_sample.py"
    cmd_str += " --no_show"
    cmd_str += " --input="+str(tenk_images_path)
    if 'tf1' in backend:
        cmd_str += ' --backend='+str(backend[:-4])
    else:
        cmd_str += " --backend="+str(backend)

    print(cmd_str)

    for i in range(no_of_iterations):
        command = shlex.split(cmd_str)
        spout = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = spout.communicate(timeout=6000)
        except subprocess.TimeoutExpired:
            print("The process ran more than 10 minutes : ")
            spout.kill()
            out, err = spout.communicate()
    
    encoding = 'utf-8'
    out = out.decode(encoding)
    out = out.split('\n')
    print(*out, sep = '\n')
    assert "OVTF Summary" in out[0] 

#stress tests for videos
@pytest.mark.object_detection
@pytest.mark.parametrize('vid',  ['tf2', 'tf1'])
def test_stress_video_obj(vid):
    if 'tf1' in vid:
        cmd_str = "python3 examples/TF_1_x/object_detection_sample.py"
    else:    
        cmd_str = "python3 examples/object_detection_sample.py"
    cmd_str += " --no_show"
    cmd_str += " --input="+str(ten_videos_path)

    print(cmd_str)

    command = shlex.split(cmd_str)
    #kept no_of_iterations=5 coz takes a lot of time
    for i in range(5):
        spout = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = spout.communicate(timeout=600000)
        except subprocess.TimeoutExpired:
            print("The process ran more than 10 minutes : ")
            spout.kill()
            out, err = spout.communicate()
    
    encoding = 'utf-8'
    out = out.decode(encoding)
    out = out.split('\n')
    print(*out, sep = '\n')
    assert "OVTF Summary" in out[0] 

@pytest.mark.parametrize('flag', ['tf1', 'tf'])
def test_images_diff_format(flag):
    #copy create_images.py and create_images_ext.py from automation folder to automation/tenk_images folder
    cmd = 'cp automation/create_images.py automation/tenk_images'
    p0 = subprocess.call(cmd, shell=True) 
    cmd1 = 'cp automation/create_images_ext.py automation/tenk_images'
    p1 = subprocess.call(cmd1, shell=True) 


    #creating 10k images in 'automation/tenk_images' folder
    create_10k_images = 'python3 automation/tenk_images/create_images.py'
    p1 = subprocess.call(create_10k_images, shell=True)
    create_10k_images1 = 'python3 automation/tenk_images/create_images_ext.py'
    p2 = subprocess.call(create_10k_images1, shell=True)

    #delete create_images.py file
    delete_create_images_file = 'rm automation/tenk_images/create_images.py'
    p2 = subprocess.call(delete_create_images_file, shell=True)   
    delete_create_images_file1 = 'rm automation/tenk_images/create_images_ext.py'
    p2 = subprocess.call(delete_create_images_file1, shell=True)  
    if flag == 'tf1':
        cmd_str = "python3 examples/TF_1_x/classification_sample.py"
    else:    
        cmd_str = "python3 examples/classification_sample.py"
    cmd_str += " --no_show"
    cmd_str += " --input="+str(tenk_images_path)
    print(cmd_str)
    command = shlex.split(cmd_str)
    spout = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = spout.communicate(timeout=6000)
    except subprocess.TimeoutExpired:
        print("The process ran more than 10 minutes : ")
        spout.kill()
        out, err = spout.communicate()
    
    encoding = 'utf-8'
    out = out.decode(encoding)
    out = out.split('\n')
    print(*out, sep = '\n')
    assert "OVTF Summary" in out[0] 