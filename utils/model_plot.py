"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Plot utilities. Used to plot losses recorded during training.
"""

import matplotlib.pyplot as plt
from typing import Dict, List


def save_loss_plot(losses: Dict[str, List[float]], plots_path: str) -> None:
    """
    Saves loss plot. 

    If validation loss is present, the plot includes both training and validation loss; otherwise, it includes only
    the training loss.

    Parameters
    ----------
    losses : dict
        A dictionary contains lists of losses. The keys are "tloss_e" for training loss and "vloss_e" for validation 
        loss. The values are lists of recorded loss values.
    plots_path : str
        Path to save the loss plot.
    """
    x_axis = list(range(len(losses["tloss_e"])))
    plt.plot(x_axis, losses["tloss_e"], c='r', label="Training")
    title = "Training"
    if len(losses["vloss_e"]) >= 1:
        # If validation loss is recorded less often, we need to adjust x-axis values by the factor of difference
        beta = len(losses["tloss_e"]) / len(losses["vloss_e"])
        x_axis = list(range(len(losses["vloss_e"])))
        # Adjust the values of x-axis by beta factor
        x_axis = [beta * i for i in x_axis]
        plt.plot(x_axis, losses["vloss_e"], c='b', label="Validation")
        title += " and Validation "
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title(title + " Loss", fontsize=12)
    plt.tight_layout()
    plt.savefig(plots_path + "/loss.png")
    plt.clf()
    

def save_auc_plot(summary: Dict[str, List[float]], plots_path: str) -> None:
    """
    Saves AUC (Area Under the ROC Curve) plot during training.

    Parameters
    ----------
    summary : dict
        A dictionary contains list of loss and AUC values stored during training. The key for AUC values is 
        "val_auc".
    plots_path : str
        Path to save the AUC plot.
    """
    if len(summary["val_auc"]) > 1:
        
        x_axis = list(range(len(summary["val_auc"])))
        plt.plot(x_axis, summary["val_auc"], c='r', label="Validation")
            
        title = "AUCs during training"
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.legend(loc="upper right")
        plt.title(title, fontsize=12)
        plt.tight_layout()
        plt.savefig(plots_path + "/val_auc.png")
        plt.clf()