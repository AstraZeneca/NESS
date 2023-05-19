"""
Author: Talip Ucar
Email: ucabtuc@gmail.com

A library of models used in NESS framework.
"""

import copy
import os

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F
import torch_geometric
import torchvision
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from torch_geometric.data import Data
from torch_geometric.nn import (APPNP, ARGA, ARGVA, GAE, VGAE, GATConv,
                                GCNConv, GINConv, global_add_pool)
from torch_geometric.nn.models.mlp import MLP
from torch_geometric.seed import seed_everything as th_seed
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import (add_self_loops, negative_sampling,
                                   remove_self_loops)
from typing import Dict, Tuple


class GAEWrapper(nn.Module):
    """
    A wrapper class for Graph Autoencoder (GAE) models. Depending on the configuration
    provided during initialization, different types of encoders and decoders can be used.
    
    Attributes
    ----------
    options : Dict[str, any]
        Configuration options for the model.
    variational : bool
        Indicates whether the encoder is variational or not.
    gae : VGAE | GAE | ARGA | ARGVA
        Graph Autoencoder model.
    linear_layer1 : nn.Linear
        The first linear layer used in the contrastive loss calculation.
    linear_layer2 : nn.Linear
        The second linear layer used in the contrastive loss calculation.
    """
    
    def __init__(self, options: Dict[str, any]):
        """
        Initializes the GAEWrapper.

        Parameters
        ----------
        options : Dict[str, any]
            Configuration dictionary.
        """
        super(GAEWrapper, self).__init__()
        self.options = options
        
        if options["encoder_type"] in  ["GNAE"]:
            encoder = GNAEEncoder(options)
        elif options["encoder_type"] in  ["VGNAE"]:
            encoder = VGNAEEncoder(options)
        elif options["encoder_type"] == "GCN":
            encoder = GCNEncoder(options)
        elif options["encoder_type"] == "VGCN":
            encoder = VariationalGCNEncoder(options)
        elif  options["encoder_type"] == "GAT":
            encoder = GATEncoder(options)
        elif options["encoder_type"] == "Linear":
            encoder = LinearEncoder(options)
        elif options["encoder_type"] == "VariationalLinear":
            encoder = VariationalLinearEncoder(options)
        elif options["encoder_type"] in ["ARGA", "ARGVA"]:
            discriminator = Discriminator(options)
            encoder = GCNEncoder(options) if options["encoder_type"] == "ARGA" else VariationalGCNEncoder(options)
        else:
            print("Encoder type could not be found. Please check for suppported models")
            exit()

        if options["encoder_type"] in ["VariationalLinear", "VGNAE", "VGCN", "ARGVA"]:
            self.variational = True 
        else:
            self.variational = False
        
        decoder = None
        
        # TODO: Use decoder = MLPModel(options) for custom decoder
        self.gae = VGAE(encoder = encoder, decoder=decoder) if self.variational else GAE(encoder = encoder, decoder=decoder)
        
        # Overwrite it if the model is ARGA or ARGVA
        if options["encoder_type"] == "ARGA":
            self.gae = ARGA(encoder = encoder, discriminator=discriminator) 
        if options["encoder_type"] == "ARGVA":
            self.gae = ARGVA(encoder = encoder, discriminator=discriminator) 
            
        if self.options["contrastive_loss"]:
            # Two-Layer Projection Network
            # First linear layer, which will be followed with non-linear activation function in the forward()
            self.linear_layer1 = nn.Linear(options["z_dim"], options["z_dim"])
            # Last linear layer for final projection
            self.linear_layer2 = nn.Linear(options["z_dim"], options["z_dim"])

    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the autoencoder.

        Parameters
        ----------
        x : Tensor
            Input features.
        edge_index : Tensor
            Edge index tensor defining the graph structure.
            
        Returns
        -------
        Tuple[Tensor, Tensor]
            Output tensor z and latent tensor.
        """
        # Forward pass on Encoder
        latent = self.gae.encode(x, edge_index)
        
        if self.options["contrastive_loss"]:
            # Forward pass on Projection
            # Apply linear layer followed by non-linear activation to decouple final output, z, from representation layer h.
            z = F.leaky_relu(self.linear_layer1(latent))
            # Apply final linear layer
            z = self.linear_layer2(z)
            # Do L2 normalization
            z = F.normalize(z, p=self.options["p_norm"], dim=1) if self.options["normalize"] else z
            # Return 
        else:
            z = latent
            
        return z, latent

    
    def single_test(self, train_data: Data, eval_data: Data) -> Tuple[float, float]:
        """
        Computes AUC and AP metrics for evaluation.

        Parameters
        ----------
        train_data : Data
            Training data.
        eval_data : Data
            Evaluation data.

        Returns
        -------
        Tuple[float, float]
            ROC AUC score and Average Precision (AP) score.
        """
        test_pos_edge_index = eval_data.pos_edge_label_index
        test_neg_edge_index = eval_data.neg_edge_label_index
        
        self.gae.eval()
        self.gae.encoder.eval()

        with torch.no_grad():
            
            # List to hold latents of subgraphs
            z_list = []
            subgraphs = train_data[:-1] if len(train_data) > 1 else train_data
            
            # Generate subgraphs - TODO method to generate sub-graphs
            for sg in subgraphs:
                z = self.gae.encode(sg.x, sg.pos_edge_label_index)
                
                # Append the latent
                z_list.append(z)
                
            # Mean aggregation of z's
            z = sum(z_list)/len(z_list)            

        roc_auc_score, average_precision_score = self.gae.test(z, test_pos_edge_index, test_neg_edge_index)
        self.gae.train()
        self.gae.encoder.train()

        return roc_auc_score, average_precision_score
    
    
class GCNEncoder(nn.Module):
    """
    Simple GCN Encoder class.
    """
    def __init__(self, options: Dict) -> None:
        """
        Initializes the GCNEncoder class.

        Parameters
        ----------
        options : dict
            Dictionary of options or configurations.
        """
        super().__init__()
        
        self.options = options
        self.conv1 = GCNConv(options["in_channels"], 2*options["z_dim"])
        self.conv2 = GCNConv(2*options["z_dim"], options["z_dim"])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward function for the GCNEncoder.

        Parameters
        ----------
        x : torch.Tensor
            Input features tensor.
        edge_index : torch.Tensor
            Tensor describing edge connections.
        
        Returns
        -------
        torch.Tensor
            Output of the GCN encoder.
        """
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
    

class VariationalGCNEncoder(nn.Module):
    """
    Variational GCN Encoder class.
    """
    def __init__(self, options: Dict) -> None:
        """
        Initializes the VariationalGCNEncoder class.

        Parameters
        ----------
        options : dict
            Dictionary of options or configurations.
        """
        super().__init__()
        
        self.options = options
        in_channels = options["in_channels"]
        out_channels = options["z_dim"]
        
        self.conv1 = GCNConv(in_channels, 2*out_channels)
        self.conv_mu = GCNConv(2*out_channels, out_channels)
        self.conv_logstd = GCNConv(2*out_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward function for the VariationalGCNEncoder.

        Parameters
        ----------
        x : torch.Tensor
            Input features tensor.
        edge_index : torch.Tensor
            Tensor describing edge connections.
        
        Returns
        -------
        tuple of torch.Tensor
            Output mean and log variance of the Variational GCN encoder.
        """
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

    
class LinearEncoder(nn.Module):
    """
    Linear Encoder class.
    """
    def __init__(self, options: Dict[str, any]) -> None:
        """
        Initialize the LinearEncoder class.

        Parameters
        ----------
        options : Dict[str, any]
            Dictionary of options or configurations.
        """
        super().__init__()
        
        self.options = options
        in_channels = options["in_channels"]
        out_channels = options["z_dim"]
        
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward function for the LinearEncoder.

        Parameters
        ----------
        x : torch.Tensor
            Input features tensor.
        edge_index : torch.Tensor
            Tensor describing edge connections.

        Returns
        -------
        torch.Tensor
            Output tensor of the encoder.
        """
        return self.conv(x, edge_index)


class VariationalGCNEncoder(nn.Module):
    """
    Variational GCN Encoder class.
    """
    def __init__(self, options: Dict[str, any]) -> None:
        """
        Initialize the VariationalGCNEncoder class.

        Parameters
        ----------
        options : Dict[str, any]
            Dictionary of options or configurations.
        """
        super().__init__()
        
        self.options = options
        in_channels = options["in_channels"]
        out_channels = options["z_dim"]
        
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward function for the VariationalGCNEncoder.

        Parameters
        ----------
        x : torch.Tensor
            Input features tensor.
        edge_index : torch.Tensor
            Tensor describing edge connections.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Output tensor of the encoder, returns both mean and log standard deviation.
        """
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    
    
class GATEncoder(nn.Module):
    """
    Graph Attention Networks (GAT) Encoder module.
    """
    def __init__(self, options: Dict[str, any]) -> None:
        """
        Initialize the GATEncoder module.

        Parameters
        ----------
        options : Dict[str, any]
            Dictionary of options or configurations.
        """
        super().__init__()
        
        in_channels = options["in_channels"]
        out_channels = options["z_dim"]

        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.)
        
        # On the Pubmed dataset, use heads=8 in conv2.
        heads = 8 if options["dataset"]=="pubmed" else 1
        self.conv2 = GATConv(8 * 8, out_channels, heads=heads, concat=False, dropout=0.)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GATEncoder module.

        Parameters
        ----------
        x : torch.Tensor
            Input node features tensor.
        edge_index : torch.Tensor
            Edge index tensor.

        Returns
        -------
        torch.Tensor
            The output tensor of node embeddings after passing through the GAT layers.
        """
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x #F.log_softmax(x, dim=-1)
    
    
class GNAEEncoder(nn.Module):
    """
    GNAE Encoder module. (Source: https://github.com/SeongJinAhn/VGNAE/blob/main/main.py)
    """
    def __init__(self, options: Dict[str, any]) -> None:
        """
        Initialize the GNAEEncoder module.

        Parameters
        ----------
        options : Dict[str, any]
            Dictionary of options or configurations.
        """
        super(GNAEEncoder, self).__init__()
        
        in_channels = options["in_channels"]
        out_channels = options["z_dim"]
        
        self.options = options
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=1, alpha=0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, not_prop: int = 0) -> torch.Tensor:
        """
        Forward pass through the GNAEEncoder module.

        Parameters
        ----------
        x : torch.Tensor
            Input node features tensor.
        edge_index : torch.Tensor
            Edge index tensor.
        not_prop : int, optional
            Flag for propagation, by default 0

        Returns
        -------
        torch.Tensor
            The output tensor of node embeddings after passing through the GNAEEncoder layers.
        """
        x = self.linear1(x)
        x = F.normalize(x,p=2,dim=1)  * 1.8 #args.scaling_factor
        x = self.propagate(x, edge_index)
        return x
    
    
class VGNAEEncoder(nn.Module):
    """
    VGNAE Encoder module. (Source: https://github.com/SeongJinAhn/VGNAE/blob/main/main.py)
    """
    def __init__(self, options: Dict[str, any]) -> None:
        """
        Initialize the VGNAEEncoder module.

        Parameters
        ----------
        options : Dict[str, any]
            Dictionary of options or configurations.
        """
        super(VGNAEEncoder, self).__init__()
        
        in_channels = options["in_channels"]
        out_channels = options["z_dim"]
        
        self.options = options
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=1, alpha=0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, not_prop: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VGNAEEncoder module.

        Parameters
        ----------
        x : torch.Tensor
            Input node features tensor.
        edge_index : torch.Tensor
            Edge index tensor.
        not_prop : int, optional
            Flag for propagation, by default 0

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of output tensors of node embeddings after passing through the VGNAEEncoder layers.
        """
        x_ = self.linear1(x)
        x_ = self.propagate(x_, edge_index)

        x = self.linear2(x)
        x = F.normalize(x,p=2,dim=1)*1.8
        x = self.propagate(x, edge_index)
        return x, x_

    
class Discriminator(nn.Module):
    """
    Discriminator module.
    """
    def __init__(self, options: Dict[str, any]) -> None:
        """
        Initialize the Discriminator module.

        Parameters
        ----------
        options : Dict[str, any]
            Dictionary of options or configurations.
        """
        super().__init__()
        
        self.options = options
        in_channels = options["z_dim"] 
        hidden_channels = 2*in_channels
        out_channels = 1
        
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Discriminator module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after passing through the Discriminator layers.
        """
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        return self.lin3(x)