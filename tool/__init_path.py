# --------------------------------------------------------
#   In this file we init the path
#   Written by Tingwu Wang, 2016/Sep/22
# --------------------------------------------------------


import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
base_dir = osp.join(this_dir, '..')
add_path(base_dir)
