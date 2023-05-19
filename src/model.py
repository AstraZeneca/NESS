"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: NESS class, the framework used to learn node embeddings from static subgraphs.
"""

import csv
import gc
import itertools
import os

import numpy as np
import pandas as pd
import torch as th
import torch_geometric.transforms as T
from torch_geometric.seed import seed_everything as th_seed
from torch_geometric.utils import dropout_adj
from tqdm import tqdm

from utils.loss_functions import JointLoss
from utils.model_plot import save_auc_plot, save_loss_plot
from utils.model_utils import GAEWrapper
from utils.utils import set_dirs, set_seed

th.autograd.set_detect_anomaly(True)

class NESS:
    """
    Model: Trains an Autoencoder with a Projection network, using NESS framework.
    """

    def __init__(self, config):
        """Class to train an autoencoder model with projection in NESS framework.

        Args:
            config (dict): Configuration dictionary.

        """
        # Get config
        self.config = config
        # Define which device to use: GPU, or CPU
        self.device = config["device"]
        # Create empty lists and dictionary
        self.model_dict, self.summary = {}, {}
        # Set random seed
        set_seed(self.config)
        # Set paths for results and initialize some arrays to collect data during training
        self._set_paths()
        # Set directories i.e. create ones that are missing.
        set_dirs(self.config)
        # ------Network---------
        # Instantiate networks
        print("Building the models for training and evaluation in NESS framework...")
        # Set Autoencoders i.e. setting loss, optimizer, and device assignment (GPU, or CPU)
        self.set_autoencoder()
        # Set scheduler (its use is optional)
        self._set_scheduler()
        # Print out model architecture
        self.print_model_summary()

    def set_autoencoder(self):
        """Sets up the autoencoder model, optimizer, and loss"""    
        # Instantiate the model for the text Autoencoder
        self.autoencoder = GAEWrapper(self.config)
        # Add the model and its name to a list to save, and load in the future
        self.model_dict.update({"autoencoder": self.autoencoder})
        # Assign autoencoder to a device
        for _, model in self.model_dict.items(): model.to(self.device)
        # Get model parameters
        parameters = [model.parameters() for _, model in self.model_dict.items()]
        # Joint loss including contrastive, reconstruction and distance losses
        self.joint_loss = None if self.config["dataset"][:4] == "ogbl" else JointLoss(self.config)
        # Set optimizer for autoencoder
        self.optimizer_ae = self._adam(parameters, lr=self.config["learning_rate"])
        # Add items to summary to be used for reporting later
        self.summary.update({"recon_loss": []})

    def set_parallelism(self, model):
        """Sets up parallelism in training."""
        # If we are using GPU, and if there are multiple GPUs, parallelize training
        if th.cuda.is_available() and th.cuda.device_count() > 1:
            print(th.cuda.device_count(), " GPUs will be used!")
            model = th.nn.DataParallel(model)
        return model

    def fit(self, data_loader):
        """Fits model to the data"""

        # Get data loaders
        train_data = data_loader.train_data
        validation_data = data_loader.validation_data
        test_data = data_loader.test_data

        # Placeholders to record losses per batch
        self.metrics = {"tloss_e": [], "vloss_e": [], "rloss_e": [], "zloss_e": [], "val_auc": [], "tr_auc": []}
        self.val_auc = "NA"
        self.tr_auc = "NA"

        # Turn on training mode for the model.
        self.set_mode(mode="training")

        # Reset best test auc
        self.best_val_auc = 0
        self.best_epoch = 0
        self.patient = 0

        
        # Start joint training of Autoencoder with Projection network
        for epoch in range(self.config["epochs"]):
            
            if self.patient == self.config["patience"]:
                break
            
            # Keep a record of epoch
            self.epoch = epoch
            
            # Change random seed back to original
            th_seed(self.config["seed"])
            
            # 0 - Update Autoencoder
            self.update_autoencoder(train_data)
            
            # 1 - Update log message using epoch and batch numbers
            self.update_log(epoch)
            
            # 2 - Clean-up for efficient memory usage.
            gc.collect()                
                
            # 3 - Run Validation
            self.run_validation(train_data, validation_data)

            # 4 - Change learning rate if scheduler==True
            _ = self.scheduler.step() if self.config["scheduler"] else None
            
        # Get the test performance
        self.test_auc, self.test_ap = self.autoencoder.single_test(train_data, test_data)

        # Save plots of training loss and validation auc
        save_loss_plot(self.metrics, self._plots_path)
        save_auc_plot(self.metrics, self._plots_path)
        
        # Convert loss dictionary to a dataframe
        loss_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in self.metrics.items()]))
        
        # Save loss dataframe as csv file for later use
        loss_df.to_csv(self._loss_path + "/losses.csv")
        
            
    def run_validation(self, train_data, validation_data):
        
        # Set the evaluation mode
        self.set_mode(mode="evaluation")
        
        # Validate every nth epoch. n=1 by default, but it can be changed in the config file
        if self.config["validate"]:

            # Compute validation AUCs
            self.val_auc, _ = self.autoencoder.single_test(train_data, validation_data)
                
            # Append auc's to the list to use for plots
            self.metrics["val_auc"].append(self.val_auc)
                
        # Save intermediate model on regular intervals
        if self.epoch >=self.config["nth_epoch"] and self.epoch % self.config["nth_epoch"] == 0:
            
            # Check the test auc at this epoch
            self.config["model_at_epoch"] = self.epoch
            val_auc, _ = self.autoencoder.single_test(train_data, validation_data)

            # Update the metrics.
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_epoch = self.epoch 
                self.save_weights()
                self.patient = 0
            else:
                self.patient += 1
                    
        # Set training mode back
        self.set_mode(mode="training")
    

    def update_autoencoder(self, subgraphs):
        """Updates autoencoder model using combinations of subgraphs of features

        Args:
            feature_label_batch_list (list): A list that contains combinations of pairs of subgraphs
            Xorig (torch.tensor): Ground truth data used to generate subgraphs

        """
        total_loss, contrastive_loss, recon_loss, zrecon_loss = [], [], [], []
        
        # Last element of the list is the original whole training graph
        graph = subgraphs[-1]
        
        # If len(subgraphs) > 1, it means that we sampled subgraphs from the graph. Else, we have a standard GAE
        subgraphs = subgraphs[:-1] if len(subgraphs) > 1 else subgraphs
        
        # A list to hold list of latents --- will be used to compute contrastive loss
        z_list = []
        
        # Initialize total loss
        tloss = None
        
        # pass subgraphs through model to reconstruct the original graph from subgraphs
        for sg in subgraphs:
            
            # Reference graph
            ref_graph = graph if self.config["full_graph"] else sg
                        
            # Drop edges if True
            if self.config["add_noise"]:
                sg.edge_index, sg.edge_attr = dropout_adj(sg.edge_index, edge_attr= sg.edge_attr, p=self.config["p_noise"])
            
            # Forwards pass
            z, latent = self.autoencoder(sg.x, sg.edge_index)           

            # Reconstruction loss by using GAE's native function
            rloss = self.autoencoder.gae.recon_loss(latent, ref_graph.pos_edge_label_index)
            
            # If the model is a variational model
            if self.autoencoder.variational:
                rloss = rloss + (1 / ref_graph.num_nodes) * self.autoencoder.gae.kl_loss()
            
            # Store z to the list
            z_list.append(z)
            
            # total loss
            tloss = tloss + rloss if tloss  is not None else rloss
            
            # Accumulate losses
            total_loss.append(tloss)
            recon_loss.append(rloss)
            
        # Clean up
        del rloss, tloss
        gc.collect()

        # Compute the losses
        n = len(total_loss)
        total_loss = sum(total_loss) / n
        recon_loss = sum(recon_loss) / n
        
        # If the graph is large such as pubmed, push the losses to cpu.
        if self.config["dataset"] == "pubmed":
            total_loss = total_loss.cpu()
            recon_loss = recon_loss.cpu()
            
        # Initiliaze contrastive loss
        closs = None
        zloss = None
        
        if self.config["contrastive_loss"] and len(subgraphs)>1:
                    
            # Generate combinations of z's to compute contrastive loss
            z_combinations = self.get_combinations_of_subgraphs(z_list)
            
            # Compute the contrastive loss for each pair of latent vectors
            for z in z_combinations:
                # Contrastive loss
                zloss = self.joint_loss(z)
                
                # Total contrastive loss
                closs = closs + zloss if closs is not None else zloss

            # Mean constrative loss
            closs = closs/len(z_combinations)
        
        # Update total loss
        total_loss = total_loss + closs if closs is not None else total_loss
        
        # Record losses
        self.metrics["tloss_e"].append(total_loss.item())
        self.metrics["rloss_e"].append(recon_loss.item())
        self.metrics["zloss_e"].append(closs.item() if closs is not None else 0)
        
        # Update Autoencoder params
        self._update_model(total_loss, self.optimizer_ae, retain_graph=True)
        
        # Delete loss and associated graph for efficient memory usage
        del total_loss, recon_loss, closs, zloss
        gc.collect()



    def get_combinations_of_subgraphs(self, z_list):
        """Generate a list of combinations of subgraphs from the list of subgraphs

        Args:
            z_list (list): List of subgraphs e.g. [z1, z2, z3, ...]
        
        Returns:
            (list): A list of combinations of subgraphs e.g. [(z1, z2), (z1, z3), ...]

        """        
                            
        # Compute combinations of subgraphs [(z1, z2), (z1, z3)...]
        subgraph_combinations = list(itertools.combinations(z_list, 2))
        # List to store the concatenated subgraphs
        concatenated_subgraphs_list = []
        
        # Go through combinations
        for (zi, zj) in subgraph_combinations:
            # Concatenate xi, and xj, and turn it into a tensor
            z = th.cat((zi, zj), dim=0)
            
            # Add it to the list
            concatenated_subgraphs_list.append(z)
        
        # Return the list of combination of subgraphs
        return concatenated_subgraphs_list
    
    def clean_up_memory(self, losses):
        """Deletes losses with attached graph, and cleans up memory"""
        for loss in losses: del loss
        gc.collect()

    def process_batch(self, xi, xj):
        """Concatenates two transformed inputs into one, and moves the data to the device as tensor"""
        # Combine xi and xj into a single batch
        Xbatch = np.concatenate((xi, xj), axis=0)
        # Convert the batch to tensor and move it to where the model is
        Xbatch = self._tensor(Xbatch)
        # Return batches
        return Xbatch

    def update_log(self, epoch):
        """Updates the messages displayed during training and evaluation"""
        # For the first epoch, add losses for batches since we still don't have loss for the epoch
        if epoch < 1:
            description = f"Epoch:[{epoch - 1}], Total loss:{self.metrics['tloss_e'][-1]:.4f}"
            description += f", X recon loss:{self.metrics['rloss_e'][-1]:.4f}"
            if self.config["contrastive_loss"]:
                description += f", contrastive loss:{self.metrics['zloss_e'][-1]:.6f}"
            description += f", val auc:{self.val_auc}"

        # For sub-sequent epochs, display only epoch losses.
        else:
            description = f"Epoch:[{epoch - 1}] training loss:{self.metrics['tloss_e'][-1]:.4f}"
            description += f", X recon loss:{self.metrics['rloss_e'][-1]:.4f}"
            if self.config["contrastive_loss"]:
                description += f", contrastive loss:{self.metrics['zloss_e'][-1]:.6f}"
            # Add validation auc
            description += f", val auc:{self.val_auc}"

        # Update the displayed message
        print(description)


    def set_mode(self, mode="training"):
        """Sets the mode of the models, either as .train(), or .eval()"""
        for _, model in self.model_dict.items():
            model.train() if mode == "training" else model.eval()

    def save_weights(self, with_epoch = False):
        """Used to save weights."""
        for model_name in self.model_dict:
            
            # Check if we want to save the model at a specific epoch
            file_name = model_name + "_" + str(self.epoch) if with_epoch else model_name
            
            # Save the model
            th.save(self.model_dict[model_name], self._model_path + "/" + file_name + ".pt")
        
        print("Done with saving models.")

    def load_models(self, epoch = None):
        """Used to load weights saved at the end of the training."""
        for model_name in self.model_dict:
            
            # Check if we want to load the model saved at a specific epoch
            file_name = model_name + "_" + str(epoch) if epoch is not None else model_name

            # Load the model
            model = th.load(self._model_path + "/" + file_name + ".pt", map_location=self.device)
            
            # Register model to the class
            setattr(self, model_name, model.eval())
            print(f"--{model_name} is loaded")
        
        print("Done with loading models.")

    def print_model_summary(self):
        """Displays model architectures as a sanity check to see if the models are constructed correctly."""
        # Summary of the model
        description = f"{40 * '-'}Summary of the models:{40 * '-'}\n"
        description += f"{34 * '='} NESS Architecture {34 * '='}\n"
        description += f"{self.autoencoder}\n"
        # Print model architecture
        print(description)

    def _update_model(self, loss, optimizer, retain_graph=True):
        """Does backprop, and updates the model parameters

        Args:
            loss (): Loss containing computational graph
            optimizer (torch.optim): Optimizer used during training
            retain_graph (bool): If True, retains graph. Otherwise, it does not.

        """
        # Reset optimizer
        optimizer.zero_grad()
        # Backward propagation to compute gradients
        loss.backward(retain_graph=retain_graph)
        # Update weights
        optimizer.step()

    def _set_scheduler(self):
        """Sets a scheduler for learning rate of autoencoder"""
        # Set scheduler (Its use will be optional)
        self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer_ae, step_size=1, gamma=0.99)

    def _set_paths(self):
        """ Sets paths to bse used for saving results at the end of the training"""
        # Top results directory
        self._results_path = os.path.join(self.config["paths"]["results"], self.config["experiment"])
        # Directory to save model
        self._model_path = os.path.join(self._results_path, "training", "model")
        # Directory to save plots as png files
        self._plots_path = os.path.join(self._results_path, "training", "plots")
        # Directory to save losses as csv file
        self._loss_path = os.path.join(self._results_path, "training", "loss")

    def _adam(self, params, lr=1e-4):
        """Sets up AdamW optimizer using model params"""
        return th.optim.AdamW(itertools.chain(*params), lr=lr, betas=(0.9, 0.999), eps=1e-07)

    def _tensor(self, data):
        """Turns numpy arrays to torch tensors"""
        if type(data).__module__ == np.__name__:
            data = th.from_numpy(data)
        return data.to(self.device).float()