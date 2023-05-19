"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Plot utilities. Used to plot losses recorded during training.
"""

import matplotlib.pyplot as plt


def save_loss_plot(losses, plots_path):
    """Saves loss plot

    Args:
        losses (dict): A dictionary contains list of losses
        plots_path (str): Path to use when saving loss plot

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
    

def save_auc_plot(summary, plots_path):
    """Saves AUC plot

    Args:
        summary (dict): A dictionary contains list of loss and auc values stored during training
        plots_path (str): Path to use when saving loss plot

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