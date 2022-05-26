import pytest
from importlib import reload 
import os
import sys
import csv
import subprocess
import shlex
import pytest
#import numpy as np

@pytest.fixture()
def sample_fixture():
    cmd_str = "python3 examples/classification_sample.py"
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
    yield out

@pytest.fixture()
def sample_fixture_obj():
    cmd_str = "python3 examples/object_detection_sample.py"

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
    yield out
    