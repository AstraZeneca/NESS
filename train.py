"""
Author: Talip Ucar
Email: ucabtuc@gmail.com

Main function for training for GNN-based encoder using NESS.
"""

import copy
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.model import NESS
from utils.arguments import get_arguments, get_config, print_config_summary
from utils.load_data import GraphLoader
from utils.utils import set_dirs, update_config_with_model_dims


def train(config, data_loader, save_weights=True):
    """
    :param dict config: Dictionary containing options.
    :param IterableDataset data_loader: Pytorch data loader.
    :param bool save_weights: Saves model if True.
    :return:

    Utility function for saving on one training fold.
    """
    # Instantiate model
    model = NESS(config)
    # Start the clock to measure the training time
    start = time.process_time()
    # Fit the model to the data
    model.fit(data_loader)
    # Total time spent on training
    training_time = time.process_time() - start
    # Report the training time
    print("Done with training...")
    print(f"Training time:  {training_time//60} minutes, {training_time%60} seconds")
    # Return the best Test set AUC
    return model.test_auc, model.test_ap


def main(config):
    # Ser directories (or create if they don't exist)
    set_dirs(config)
    # Get data loader for first dataset.
    ds_loader = GraphLoader(config, dataset_name=config["dataset"])
    # Add the number of features in a dataset as the first dimension of the model
    config = update_config_with_model_dims(ds_loader, config)
    # Start training and save model weights at the end
    test_auc, test_ap = train(config, ds_loader, save_weights=True)
    # Return best test auc
    return test_auc, test_ap


if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    
    # Get configuration file
    config = get_config(args)

    # By default, we are using the name of the dataset. This can be customized.
    config["experiment"] = config["dataset"]

    # File name to use when saving results as csv. This can be customized
    config["file_name"] = config["experiment"] + "_sub" + str(config["n_subgraphs"]) + '_seed' + str(config["seed"])

    # Summarize config and arguments on the screen as a sanity check
    print_config_summary(config, args)

    # Run the main
    test_auc,test_ap = main(config)
    
    print(f"Test AUC: {test_auc}, AP: {test_ap}")
