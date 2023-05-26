"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: A library for data loaders.
"""

import os

import datatable as dt
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
from torch_geometric.nn import GAE, VGAE, GCNConv
from torch_geometric.seed import seed_everything as th_seed
from typing import Any, Dict, List, Tuple


class GraphLoader:
    """
    Data loader class for graph data.
    """

    def __init__(self, config: Dict[str, Any], dataset_name: str, kwargs: Dict[str, Any] = {}) -> None:
        """
        Initializes the GraphLoader.

        Parameters
        ----------
        config : Dict[str, Any]
            Dictionary containing options and arguments.
        dataset_name : str
            Name of the dataset to load.
        kwargs : Dict[str, Any], optional
            Dictionary for additional parameters if needed, by default {}.
        """
        # Get config
        self.config = config
        # Set the seed
        th_seed(config["seed"])
        # Set the paths
        paths = config["paths"]
        # data > dataset_name
        file_path = os.path.join(paths["data"], dataset_name)
        # Get the datasets
        self.train_data, self.validation_data, self.test_data = self.get_dataset(dataset_name, file_path)        
        

    def get_dataset(self, dataset_name: str, file_path: str) -> Tuple[Data, Data, Data]:
        """
        Returns the training, validation, and test datasets.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load.
        file_path : str
            Path to the dataset.

        Returns
        -------
        Tuple[Data, Data, Data]
            Training, validation, and test datasets.
        """

        # Initialize Graph dataset class
        graph_dataset = GraphDataset(self.config, datadir=file_path, dataset_name=dataset_name)
        
        # Load Training, Validation, Test datasets
        train_data, val_data, test_data = graph_dataset._load_data()
        
        # Generate static subgraphs from training set
        train_data = self.generate_subgraphs(train_data)
  
        # Return
        return train_data, val_data, test_data
    
    
    def generate_subgraphs(self, train_data: Data) -> List[Data]:
        """
        Generates subgraphs from the training data.

        Parameters
        ----------
        train_data : Data
            Training data containing the graph.

        Returns
        -------
        List[Data]
            List of subgraphs generated from the training data.
        """
        # Initialize list to hold subgraphs
        subgraphs = [train_data]

        # Check if we are generating subgraphs from the graph. If False, we are in standard GAE mode
        if self.config["n_subgraphs"] > 1:
                
            # Generate subgraphs
            for i in range(self.config["n_subgraphs"]):
                
                # Change random seed
                th_seed(i)
                
                partition = 1.0/(self.config["n_subgraphs"]-i)
                
                # For the last subgraph, get 95% of the remaining graph. if num_val=1.0, RandomLinkSplit will raise error
                if partition == 1.0:
                    partition = 0.95
                    
                random_link_split = T.RandomLinkSplit(num_val=partition, 
                                                      num_test=0, 
                                                      is_undirected=True, 
                                                      split_labels=True, 
                                                      add_negative_train_samples=False)

                # get a subgraph from training data
                train_data, train_subgraph, _ = random_link_split(train_data)
                
                # Make sure that we are using only the nodes within the subgraph by overwriting the edge index 
                # with positive edge index + positive edge index reversed in direction (to make it undirected)
                pos_swapped = train_subgraph.pos_edge_label_index[[1,0],:] 
                train_subgraph.edge_index = torch.cat((train_subgraph.pos_edge_label_index, pos_swapped), dim=1)
                
                # Remove negative edge attributes. We want to sample negative samples during training
                # Masks are also not needed
                if hasattr(train_subgraph, "neg_edge_label_index"):
                    delattr(train_subgraph, "neg_edge_label_index")
                    delattr(train_subgraph, "neg_edge_label")
                    
                if hasattr(train_subgraph, "train_mask"):
                    delattr(train_subgraph, "train_mask")
                    delattr(train_subgraph, "val_mask")
                    delattr(train_subgraph, "test_mask")


                # store the sampled subgraph
                subgraphs = [train_subgraph] + subgraphs
                       
        # Change random seed back to original
        th_seed(self.config["seed"])
        
        # Return all subgraphs and original larger graph
        return subgraphs

    
def get_transform(options):
    """Splits data to train, validation and test, and moves them to the device"""
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(options["device"]),
        T.RandomLinkSplit(num_val=0.05, 
                          num_test=0.15, 
                          is_undirected=True,
                          split_labels=True, 
                          add_negative_train_samples=False),
        ])
        
    return transform


class GraphDataset:
    """
    Dataset class for graph data format.
    """

    def __init__(self, config: Dict[str, Any], datadir: str, dataset_name: str) -> None:
        """
        Initializes the GraphDataset.

        Parameters
        ----------
        config : Dict[str, Any]
            Dictionary containing options and arguments.
        datadir : str
            The path to the data directory.
        dataset_name : str
            Name of the dataset to load.
        """
        self.config = config
        self.paths = config["paths"]
        self.dataset_name = dataset_name
        self.data_path = os.path.join(self.paths["data"], 'Planetoid')
        self.transform = get_transform(config)

        
    def _load_data(self) -> Tuple[Data, Data, Data]:
        """
        Loads one of many available datasets and returns features and labels.

        Returns
        -------
        Tuple[Data, Data, Data]
            Training, validation, and test datasets.
        """
        if self.dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
            # Get the dataset
            dataset = Planetoid(self.data_path, self.dataset_name, split="random", transform = self.transform)
        elif  self.dataset_name.lower() in ['chameleon']:
            # Get the dataset
            dataset = WikipediaNetwork(root=self.data_path, name=self.dataset_name, transform = self.transform)
        elif  self.dataset_name.lower() in ["cornell", "texas", "wisconsin"]:
            # Get the dataset
            dataset = WebKB(root=self.data_path, name=self.dataset_name, transform = self.transform) 
        else:
            print(f"Given dataset name is not found. Check for typos, or missing condition ")
            exit()
            
        # Data splits
        train_data, val_data, test_data = dataset[0]
        
        # Return
        return train_data, val_data, test_data
