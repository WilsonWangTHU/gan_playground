# --------------------------------------------------------
#   In this file we init the path
#   Written by Tingwu Wang, 2016/Sep/22
# --------------------------------------------------------


import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

_this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
_base_dir = osp.join(_this_dir, '..')
add_path(_base_dir)

def bypass_frost_warning():
    return 0

def get_base_dir()
	return _base_dir