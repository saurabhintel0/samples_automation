import pytest
from importlib import reload 
import os
import sys
import csv
import subprocess
import shlex
import pathlib
#import numpy as np

#Another option is to test only those backends which are present in the device by 
#openvino_tensorflow.set_backend('<backend_name>')

ovtf_path = pathlib.Path(__file__).resolve().parents[1]
@pytest.mark.classification
@pytest.mark.parametrize('backend',  ["CPU", "GPU" ,"MYRIAD" ,"VAD-M" ,"STOCK_TF_CPU", "CPUU",
                                    "CPU_tf1", "GPU_tf1" ,"MYRIAD_tf1" ,"VAD-M_tf1" ,"STOCK_TF_CPU_tf1", "CPUU_tf1",
                                    "CPU_cpp", "GPU_cpp" ,"MYRIAD_cpp" ,"VAD-M_cpp", "CPUU_cpp"])   #CPUU for wrong backend name test case
def test_backend(backend):
    if 'tf1' in backend:
        cmd_str = 'python3 examples/TF_1_x/classification_sample.py'
    elif 'cpp' in backend:
        binary_exe = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'+str(ovtf_path)+\
                '/build_cmake/artifacts/lib:'+str(ovtf_path)+'/build_cmake/artifacts/tensorflow'
        p0 = subprocess.call(binary_exe, shell=True)
        cmd_str = './build_cmake/examples/classification_sample/infer_image'
    else:
        cmd_str = "python3 examples/classification_sample.py"

    if backend == "STOCK_TF_CPU" or backend == "STOCK_TF_CPU_tf1":
        cmd_str += " --disable_ovtf"
    else:
        if 'tf1' in backend:
            cmd_str += " --backend="+str(backend[:-4])
        elif 'cpp' in backend:
            cmd_str += " --backend="+str(backend[:-4])
        else:
            cmd_str += " --backend="+str(backend)

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

    
    print("So that we can debug this problem!!!!!!!!!!", type(out))
    print("out =========== ", out)
    res = [i for i in out if "OVTF Summary" in i]
    print("res if this is truw we r gud to go!!!", res)
    
    #print(out[0])
    
    err = err.decode(encoding)
    err = err.split('\n')
    print(*err, sep='\n')
    if str(backend) == "STOCK_TF_CPU" or backend == "STOCK_TF_CPU_tf1":
        res = [i for i in out if "Inference time in ms" in i]
        print("length of res!!!!!!!!!!!!!!!!!!", len(res))
        assert len(res)>0
        #assert "Inference time in ms" in out[0]
    elif str(backend) == "CPUU" or str(backend) == "CPUU_tf1" or str(backend) == 'CPUU_cpp':
        with pytest.raises(Exception):
            assert "OVTF Summary" in out[0] 
    elif 'cpp' in backend:
        assert "OVTF Summary" in out[9] 
    else:
        res = [i for i in out if "OVTF Summary" in i]
        print("length of res!!!!!!!!!!!!!!!!!!", len(res))
        assert len(res)>0
        #assert "OVTF Summary" in out[0] 

@pytest.mark.object_detection
@pytest.mark.parametrize('backend', ["CPU", "GPU" ,"MYRIAD" ,"VAD-M" ,"STOCK_TF_CPU", "CPUU",
                                    "CPU_tf1", "GPU_tf1" ,"MYRIAD_tf1" ,"VAD-M_tf1" ,"STOCK_TF_CPU_tf1", "CPUU_tf1"])  
def test_backend_obj(backend):
    if 'tf1' in backend:
        cmd_str = 'python3 examples/TF_1_x/object_detection_sample.py'
    else:
        cmd_str = "python3 examples/object_detection_sample.py"
    cmd_str += " --no_show"
    if backend == "STOCK_TF_CPU" or backend == "STOCK_TF_CPU_tf1":
        cmd_str += " --disable_ovtf"
    else:
        if 'tf1' in backend:
            cmd_str += " --backend="+str(backend[:-4])
        else:
            cmd_str += " --backend="+str(backend)

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
    print(*err, sep='\n')
    if str(backend) == "STOCK_TF_CPU" or backend == "STOCK_TF_CPU_tf1":
        assert "Inference time in ms" in out[0]
    elif str(backend) == "CPUU" or str(backend) == "CPUU_tf1":
        with pytest.raises(Exception):
            assert "OVTF Summary" in out[0] 
    else:
        assert "OVTF Summary" in out[0] 