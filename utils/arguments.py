"""
Author: Talip Ucar
Email: ucabtuc@gmail.com

Loads arguments and configuration for GNN-based encoder used in NESS.
"""

import os
import pprint
from argparse import ArgumentParser
from os.path import abspath, dirname

import torch

from utils.utils import get_runtime_and_model_config, print_config


def get_arguments():
    # Initialize parser
    parser = ArgumentParser()
    # Dataset can be provided via command line
    parser.add_argument("-d", "--dataset", type=str, default="cora")
    # Encoder type
    parser.add_argument("-gnn", "--gnn", type=str, default="GNAE")
    # Random seed
    parser.add_argument("-seed", "--seed", type=int, default=57)
    # Whether to use contrastive loss
    parser.add_argument("-cl", "--cl", type=bool, default=False)
    # Whether to add noise to input
    parser.add_argument("-an", "--an", type=bool, default=True)
    # Whether to use GPU.
    parser.add_argument("-g", "--gpu", dest='gpu', action='store_true', 
                        help='Used to assign GPU as the device, assuming that GPU is available')
    
    parser.add_argument("-ng", "--no_gpu", dest='gpu', action='store_false', 
                        help='Used to assign CPU as the device')
    parser.set_defaults(gpu=True)
    
    # GPU device number as in "cuda:0". Defaul is 0.
    parser.add_argument("-dn", "--device_number", type=str, default='0', 
                        help='Defines which GPU to use. It is 0 by default')
    
    # Experiment number
    parser.add_argument("-ex", "--experiment", type=int, default=1)
    # Load model saved at specific epoch
    parser.add_argument("-m", "--model_at_epoch", type=int, default=None)
    # Return parser arguments
    return parser.parse_args()

def get_config(args):
    # Load runtime config from config folder: ./config/
    config = get_runtime_and_model_config(args)
    # Define which device to use: GPU or CPU
    config["device"] = torch.device('cuda:'+args.device_number if torch.cuda.is_available() and args.gpu else 'cpu')
    # Model at specific epoch
    config["model_at_epoch"] = args.model_at_epoch
    # Indicate which device is being used
    print(f"Device being used is {config['device']}")
    # Return
    return config

def print_config_summary(config, args=None):
    """Prints out summary of options and arguments used"""
    # Summarize config on the screen as a sanity check
    print(100 * "=")
    print(f"Here is the configuration being used:\n")
    print_config(config)
    print(100 * "=")
    if args is not None:
        print(f"Arguments being used:\n")
        print_config(args)
        print(100 * "=")
