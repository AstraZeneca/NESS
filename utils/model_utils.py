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
import torch as th
import torch.nn.functional as F
import torch_geometric
import torchvision
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch_geometric.nn import (APPNP, ARGA, ARGVA, GAE, VGAE, GATConv,
                                GCNConv, GINConv, global_add_pool)
from torch_geometric.nn.models.mlp import MLP
from torch_geometric.seed import seed_everything as th_seed
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import (add_self_loops, negative_sampling,
                                   remove_self_loops)


class GAEWrapper(nn.Module):
    """
    Autoencoder wrapper class
    """

    def __init__(self, options):
        """

        Args:
            options (dict): Configuration dictionary.
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

    
    def forward(self, x, edge_index):
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

    
    def single_test(self, train_data, eval_data):
        """Computes AUC and AP metrics"""
        
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
    
    
class GCNEncoder(torch.nn.Module):
    def __init__(self, options):
        super().__init__()
        
        self.options = options
        self.conv1 = GCNConv(options["in_channels"], 2*options["z_dim"])
        self.conv2 = GCNConv(2*options["z_dim"], options["z_dim"])

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
    

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, options):
        super().__init__()
        
        self.options = options
        in_channels = options["in_channels"]
        out_channels = options["z_dim"]
        
        self.conv1 = GCNConv(in_channels, 2*out_channels)
        self.conv_mu = GCNConv(2*out_channels, out_channels)
        self.conv_logstd = GCNConv(2*out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

    
class LinearEncoder(torch.nn.Module):
    def __init__(self, options):
        super().__init__()
        
        self.options = options
        in_channels = options["in_channels"]
        out_channels = options["z_dim"]
        
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, options):
        super().__init__()
        
        self.options = options
        in_channels = options["in_channels"]
        out_channels = options["z_dim"]
        
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    
    
class GATEncoder(torch.nn.Module):
    def __init__(self, options):
        super().__init__()
        
        in_channels = options["in_channels"]
        out_channels = options["z_dim"]

        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.)
        
        # On the Pubmed dataset, use heads=8 in conv2.
        heads = 8 if options["dataset"]=="pubmed" else 1
        self.conv2 = GATConv(8 * 8, out_channels, heads=heads, concat=False, dropout=0.)

    def forward(self, x, edge_index):
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x #F.log_softmax(x, dim=-1)
    
    
class GNAEEncoder(torch.nn.Module):
    """Source: https://github.com/SeongJinAhn/VGNAE/blob/main/main.py"""
    
    def __init__(self, options):
        super(GNAEEncoder, self).__init__()
        
        in_channels = options["in_channels"]
        out_channels = options["z_dim"]
        
        self.options = options
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=1, alpha=0)

    def forward(self, x, edge_index,not_prop=0):
        x = self.linear1(x)
        x = F.normalize(x,p=2,dim=1)  * 1.8 #args.scaling_factor
        x = self.propagate(x, edge_index)
        return x
    
class VGNAEEncoder(torch.nn.Module):
    """Source: https://github.com/SeongJinAhn/VGNAE/blob/main/main.py"""
    
    def __init__(self, options):
        super(VGNAEEncoder, self).__init__()
        
        in_channels = options["in_channels"]
        out_channels = options["z_dim"]
        
        self.options = options
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=1, alpha=0)

    def forward(self, x, edge_index,not_prop=0):

        x_ = self.linear1(x)
        x_ = self.propagate(x_, edge_index)

        x = self.linear2(x)
        x = F.normalize(x,p=2,dim=1)*1.8
        x = self.propagate(x, edge_index)
        return x, x_

    
class Discriminator(torch.nn.Module):
    def __init__(self, options):
        super().__init__()
        
        self.options = options
        in_channels = options["z_dim"] 
        hidden_channels = 2*in_channels
        out_channels = 1
        
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        return self.lin3(x)