"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Utility functions.
"""
import argparse
import cProfile
import os
import pstats
import random as python_random
import sys

import numpy as np
import torch
import yaml
from numpy.random import seed
from sklearn import manifold
from texttable import Texttable
from typing import Dict, Tuple, Union, List, Any, Callable


def set_seed(options: Dict[str, Union[int, str]]) -> None:
    """
    Sets seed to ensure reproducibility.

    Parameters
    ----------
    options : dict
        Dictionary with the seed option. The key is "seed".
    """
    seed(options["seed"])
    np.random.seed(options["seed"])
    python_random.seed(options["seed"])
    torch.manual_seed(options["seed"])


def create_dir(dir_path: str) -> None:
    """
    Creates a directory if it does not exist.

    Parameters
    ----------
    dir_path : str
        The path to the directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def set_dirs(config: Dict[str, Any]) -> None:
    """
    It sets up directory that will be used to load processed_data and src as well as saving results.

    Directory structure example:
        results > dataset > training  -------> > model
                          > evaluation         > plots
                                               > loss

    Parameters
    ----------
    config : dict
        Dictionary that defines options to use.
    """
    # Set main results directory using database name. Exp:  processed_data/dpp19
    paths = config["paths"]
    # results
    results_dir = make_dir(paths["results"], "")
    # results > framework
    results_dir = make_dir(results_dir, config["experiment"])
    # results > framework > training
    training_dir = make_dir(results_dir, "training")
    # results > framework > evaluation
    evaluation_dir = make_dir(results_dir, "evaluation")
    # results > framework > evaluation > clusters
    clusters_dir = make_dir(evaluation_dir, "clusters")
    # results > framework > evaluation > reconstruction
    recons_dir = make_dir(evaluation_dir, "reconstructions")
    # results > framework > training >  model
    training_model_dir = make_dir(training_dir, "model")
    # results > framework > training >  plots
    training_plot_dir = make_dir(training_dir, "plots")
    # results > framework > training > loss
    training_loss_dir = make_dir(training_dir, "loss")
    # Print a message.
    print("Directories are set.")


def make_dir(directory_path: str, new_folder_name: str) -> str:
    """
    Creates an expected directory if it does not exist.

    Parameters
    ----------
    directory_path : str
        Path to the directory.
    new_folder_name : str
        Name of the new folder.

    Returns
    -------
    str
        Path to the created directory.
    """
    directory_path = os.path.join(directory_path, new_folder_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


def get_runtime_and_model_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Returns runtime and model/dataset specific config file.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.

    Returns
    -------
    dict
        Dictionary with the configuration.
    """
    try:
        with open(f"./config/{args.dataset}.yaml", "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        sys.exit(f"Error reading {args.dataset} config file")
    
    # Copy dataset names to config to use later
    config["dataset"] = args.dataset
    
    # Return 
    return config


def update_config_with_model_dims(dataset: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates options by adding the dimension of input features as the dimension of first hidden layer of the model.

    Parameters
    ----------
    dataset : Any
        Dataset object.
    config : dict
        Dictionary with the configuration.

    Returns
    -------
    dict
        Updated dictionary with the configuration.
    """
    # Get data
    train_data = dataset.train_data[-1]
    # Get the number of features
    config["in_channels"] = train_data.x.size(1)
    # Update the output dimension of decoder
    config["out_dim"] = train_data.x.size(0)
    return config


def run_with_profiler(main_fn: Callable, config: Dict[str, Any]) -> None:
    """
    Runs function with profile to see how much time each step takes.

    Parameters
    ----------
    main_fn : Callable
        Main function to profile.
    config : dict
        Dictionary with the configuration.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    # Run the main
    main_fn(config)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats.print_stats()


def print_config(args: Union[Dict[str, Any], argparse.Namespace]) -> None:
    """
    Prints out options and arguments.

    Parameters
    ----------
    args : dict or argparse.Namespace
        Configuration options either as a dictionary or argparse Namespace.
    """
    # Yaml config is a dictionary while parser arguments is an object. Use vars() only on parser arguments.
    if type(args) is not dict:
        args = vars(args)
    # Sort keys
    keys = sorted(args.keys())
    # Initialize table
    table = Texttable()
    # Add rows to the table under two columns ("Parameter", "Value").
    table.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    # Print the table.
    print(table.draw())
