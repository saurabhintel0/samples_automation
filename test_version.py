import tensorflow as tf
import openvino_tensorflow as ovtf

def test_tf_version():
    print(tf.__version__)
    assert tf.__version__ == '2.7.0'
    print(ovtf.__version__)

def test_ovtf_version():
    assert '1.1.0' in ovtf.__version__

def test_ov_version():
    assert 'master' in ovtf.__version__

def test_abi_flag_version():
    assert '0' in ovtf.__version__ 
