import argparse
import tensorflow as tf
from tqdm import tqdm
import os
from os.path import join, exists

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('-stype_dir', type=str, default='style_images')
parser.add_argument('-save_dir', type=str, default='saved_models')
parser.add_argument('-lr', default=0.001, type=float)
parser.add_argument('-exp_name', type=str, required=True)
parser.add_argument('-batch_size', type=int , default=8)
parser.add_argument('-num_epochs', type=int, default=100)
parser.add_argument('-style_weight', type=float, default=10.0)
parser.add_argument('-content_weight', type=float, default=1.0)
parser.add_argument('-save_model_weights', type=str2bool, default=False)

