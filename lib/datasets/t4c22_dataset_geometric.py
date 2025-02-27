#  Copyright 2022 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch_geometric
from torch_geometric.utils import subgraph

import os
import json

import numpy as np
import pandas as pd
import scipy.sparse as sps

from lib.datasets.road_graph_mapping import TorchRoadGraphMapping
from lib.datasets.t4c22_dataset import T4c22Competitions
from t4c22.t4c22_config import cc_dates
from t4c22.t4c22_config import day_t_filter_to_df_filter
from t4c22.t4c22_config import day_t_filter_weekdays_daytime_only
#from t4c22.t4c22_config import load_inputs
from ..utils import sample_mask

config_path = 'D:/TU Dortmund/Semesters/Summer Semester 2024/Thesis/Experiment run/grin_final_after_thesis/t4c22/t4c22_config.json'

with open(config_path, 'r') as file:
    config = json.load(file)

# Extract BASEDIR from the JSON configuration
base_dir = config["BASEDIR"]

# Construct the paths dynamically
road_graph_nodes_path = os.path.join(base_dir, "road_graph/london/road_graph_nodes.parquet")
road_graph_edges_path = os.path.join(base_dir, "road_graph/london/road_graph_edges.parquet")


save_path = 'distance_matrix_london'


# https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-larger-datasets
class T4c22GeometricDataset(torch_geometric.data.Dataset):
    def __init__(
        self,
        root: Path,
        city: str, 
        device=None,     
        edge_attributes=None,
        cachedir: Optional[Path] = None,
        limit: int = None,
        day_t_filter=day_t_filter_weekdays_daytime_only,
        competition: T4c22Competitions = T4c22Competitions.CORE,
        val_len=0.15,  
        test_len=0.15,  
        window=0,  # Optional window parameter - it ensures that there is no overlap or information leakage between the training, validation, and test sets.
        mask=None,
    ):
        """Dataset for t4c22 core competition (congestion classes) for one
        city.

        Get 92 items a day (last item of the day then has x loop counter
        data at 91, 92, 93, 94 and y congestion classes at 95) I.e.
        overlapping sampling, but discarding samples going over midnight.

        Missing values in input or labels are represented as nans, use `torch.nan_to_num`.

        CC labels are shift left by one in tensor as model outputs only green,yellow,red but not unclassified and allows for direct use in `torch.nn.CrossEntropy`
            # 0 = green
            # 1 = yellow
            # 2 = red

        Parameters
        ----------
        root: basedir for data
        city: "london" / "madrid" / "melbourne"
        edge_attributes: any numerical edge attribute from `road_graph_edges.parquet`
                - parsed_maxspeed
                - speed_kph
                - importance
                - oneway
                - lanes
                - tunnel
                - length_meters
        cachedir: location for single item .pt files (created on first access if cachedir is given)
        limit: limit the dataset to at most limit items (for debugging)
        day_t_filter: filter taking day and t as input for filtering the data. 
        """
        super().__init__(root)
        #self._mask = None
        self.root: Path = root

        self.cachedir = cachedir
        self.city = city
        self.device = device or torch.device("cpu")  # Default to CPU if no device is specified
        self.limit = limit
        self.day_t_filter = day_t_filter 
        self.competition = competition

        # Load the distance matrix
        self.dist = self.load_distance_matrix(road_graph_nodes_path, road_graph_edges_path, save_path, limit=limit)

        self.torch_road_graph_mapping = TorchRoadGraphMapping(
            city=city,
            edge_attributes=edge_attributes,
            root=root,
            df_filter=partial(day_t_filter_to_df_filter, filter=day_t_filter) if self.day_t_filter is not None else None,
            skip_supersegments=self.competition == T4c22Competitions.CORE,
        )

        # `day_t: List[Tuple[Y-m-d-str,int_0_96]]`
        # TODO most days have even 96 (rolling window over midnight), but probably not necessary because of filtering we do.
#         if split == "test":
#             num_tests = load_inputs(basedir=self.root, split="test", city=city, day="test", df_filter=None)["test_idx"].max() + 1
#             self.day_t = [("test", t) for t in range(num_tests)]
#         else:
#             self.day_t = [(day, t) for day in cc_dates(self.root, city=city, split=self.split) for t in range(4, 96) if self.day_t_filter(day, t)]'
        
        self.day_t = [(day, t) for day in cc_dates(self.root, city=city) for t in range(4, 96) if self.day_t_filter(day, t)]  
        print(f"Total number of time steps: {len(self.day_t)}")   
        
        # Apply the limit immediately after creating day_t 
        if self.limit is not None:
            self.day_t = self.day_t[:self.limit]
           
        self.full_day_t = self.day_t.copy()  # Store the full dataset
        # Apply the splitter method to get the train, validation, and test indices
        self.train_idxs, self.val_idxs, self.test_idxs = self.splitter(self.full_day_t, val_len, test_len, window)
        print(f"Training samples: {len(self.train_idxs)}, Validation samples: {len(self.val_idxs)}, Test samples: {len(self.test_idxs)}")
              
        if mask is not None:
            mask = np.asarray(mask).astype('uint8')
        self._mask = mask
    
        # Generate training and evaluation masks
        self._training_mask, self._eval_mask = self.generate_masks()
        print(f"training mask shape: {self._training_mask.shape}, eval mask shape: {self._eval_mask.shape}")

    @property
    def raw_file_names(self):
        return []  # No raw files

    @property
    def processed_file_names(self):
        return []  # Return an empty list

    @property
    def training_mask(self):
        return self._training_mask

    @property
    def eval_mask(self):
        return self._eval_mask
    
    @property
    def has_mask(self):
        return self._mask is not None
    
    @property
    def mask(self):
        if self._mask is None:
            return np.ones_like(self.day_t.values).astype('uint8')
        return self._mask
    
    #@property
    #def split_indices(self):
        #"""Expose train, val, and test indices."""
        #return self.train_idxs, self.val_idxs, self.test_idxs
    
    def get_full_data(self, idx: int) -> torch_geometric.data.Data:
        """Fetch data from the full dataset using original indices."""
        if self.limit is not None and idx >= self.limit:
            raise IndexError(f"Index {idx} is out of range for full dataset of size {len(self.full_day_t)}.")
        day, t = self.full_day_t[idx]
        return self._load_data(day, t, idx)  # Load the data using a helper method
            
    def get(self, idx: int, limit: Optional[int] = None) -> torch_geometric.data.Data:
        """Fetch data from the current split."""
        if idx >= len(self.day_t):
            raise IndexError(f"Index {idx} is out of range for current dataset of size {len(self.day_t)}.")
        day, t = self.day_t[idx]
        data = self._load_data(*self.day_t[idx], idx, limit=limit)
        # Move data to the specified device
        return data.to(self.device)


    def splitter(self, dataset, val_len=0, test_len=0, window=0):
        """Method to split the dataset indices into training, validation, and test sets."""
        
        print(f"==================================")
        print(f"Splitting the dataset....") 
        print(f"==================================")
        
        idx = np.arange(len(dataset))
        print(f"Total dataset size: {len(idx)}")
        
        if test_len < 1:
            test_len = int(test_len * len(idx))
        if val_len < 1:
            val_len = int(val_len * (len(idx) - test_len))
            
        test_start = len(idx) - test_len
        val_start = test_start - val_len
        
        return [idx[:val_start - window], idx[val_start:test_start - window], idx[test_start:]]

    
    def generate_masks(self):
        """
        Generate training and evaluation masks for the dataset using the masks created
        during raw data loading.

        Parameters:
        - split (str): Indicates the dataset split ('train', 'val', or 'test').

        Returns:
        - training_mask (torch.Tensor): Mask for training samples.
        - eval_mask (torch.Tensor): Mask for evaluation samples.
        """

        # Mask for non-zero values 
        valid_mask = []
        eval_mask = []

        for idx in range(len(self.full_day_t)):

                # Fetch the data for the current sample
                data = self.get(idx, limit=self.limit)  

                # Binary mask for valid values (non-NaN and non-zero values)
                mask = (~torch.isnan(data.x)) & (data.x != 0)
    
                # Generate random fault and noise masks (using sample_mask logic)
                fault_mask = sample_mask(
                    shape=data.x.shape,
                    p=0.0015,  # Probability of fault
                    p_noise=0.05,  # Probability of noise
                    min_seq=12,  # Minimum sequence length for fault
                    max_seq=12 * 4,  # Maximum sequence length for fault
                    rng=np.random.default_rng(56789)  # Reproducibility
                )

                # Convert fault mask to tensor and apply to the valid mask
                fault_mask_tensor = torch.tensor(fault_mask, dtype=torch.bool)
                eval_mask_tensor = mask & fault_mask_tensor

                # Append masks
                valid_mask.append(mask)
                eval_mask.append(eval_mask_tensor)

        # Calculate training masks without concatenation
        training_mask = torch.stack([
            valid_mask & ~eval_mask
            for valid_mask, eval_mask in zip(valid_mask, eval_mask)
             if valid_mask is not None and eval_mask is not None
        ], dim=0) 

        eval_mask = torch.stack(
            [eval for eval in eval_mask if eval is not None],
            dim=0
        )     
 
        return training_mask, eval_mask

    
    def load_distance_matrix(self, road_graph_nodes_path: str, road_graph_edges_path: str, save_path: str = None, limit: int = None):
        """
        Load or create a sparse distance matrix based on road graph edges. Euclidean distance between nodes.
        
        This method calculates the distance only for connected nodes based on the road graph edges,
        and stores the result in a sparse matrix format to save memory.

        Parameters:
        - road_graph_nodes_path (str): Path to the road graph nodes file (.parquet).
        - road_graph_edges_path (str): Path to the road graph edges file (.parquet).
        - save_path (str, optional): Path to save the calculated sparse distance matrix. If not provided, the default is not to save.

        Returns:
        - dist_matrix (scipy.sparse.coo_matrix): Sparse distance matrix between connected nodes.
        """
        
        # Check if the road graph nodes and edges files exist
        if not os.path.exists(road_graph_nodes_path):
           raise FileNotFoundError(f"Road graph nodes file not found: {road_graph_nodes_path}")
    
        if not os.path.exists(road_graph_edges_path):
           raise FileNotFoundError(f"Road graph edges file not found: {road_graph_edges_path}")
    
        if save_path and os.path.exists(save_path):
            # Load the precomputed sparse distance matrix
            print(f"Loading precomputed distance matrix from {save_path}")
            dist_matrix = sps.load_npz(save_path)
        else:
            # Load the road graph nodes and edges data
            df_nodes = pd.read_parquet(road_graph_nodes_path)
            df_edges = pd.read_parquet(road_graph_edges_path)


       # Subset the nodes if `nodes_limit` is provided
        if limit is not None:
           df_nodes = df_nodes.iloc[:limit]
           node_ids = set(df_nodes['node_id'])
           df_edges = df_edges[df_edges['u'].isin(node_ids) & df_edges['v'].isin(node_ids)]
                      
        # Create a mapping from node_id to its row index in df_nodes
        node_id_to_index = {node_id: idx for idx, node_id in enumerate(df_nodes['node_id'])}

        # Create edge_index
        edge_index = torch.tensor(
            [[node_id_to_index[row['u']], node_id_to_index[row['v']]] for _, row in df_edges.iterrows()],
            dtype=torch.long
        ).T

        # Calculate edge_attr (Euclidean distances between nodes)
        node_positions = df_nodes[['x', 'y']].values
        edge_attr = torch.tensor(
            [
                 np.linalg.norm(node_positions[node_id_to_index[row['u']]] - node_positions[node_id_to_index[row['v']]])
                 for _, row in df_edges.iterrows()
            ],
            dtype=torch.float32
        )

        # Determine the number of nodes in the filtered edge_index
        num_nodes = edge_index.max().item() + 1  # Largest node index + 1

        # Create the subset tensor based on the adjusted limit
        subset = torch.arange(num_nodes, dtype=torch.long)  # Use the valid range from edge_index

        # Apply the subgraph function
        edge_index, edge_attr, _ = subgraph(
            subset=subset, edge_index=edge_index, edge_attr=edge_attr, relabel_nodes=True, return_edge_mask=True
        )
        
        # Initialize lists for distances
        row_indices, col_indices, distances = [], [], []

        # Determine the working node set
        if subset is not None:
            # If a subset is provided, map it to the filtered nodes
            subset = subset.numpy() if isinstance(subset, torch.Tensor) else subset
            node_id_to_index = {node_id: idx for idx, node_id in enumerate(subset)}
            node_positions = df_nodes[df_nodes['node_id'].isin(subset)][['x', 'y']].values
            df_edges = df_edges[df_edges['u'].isin(subset) & df_edges['v'].isin(subset)]
        else:
            # Use the full graph data
            node_positions = df_nodes[['x', 'y']].values
            node_id_to_index = {node_id: idx for idx, node_id in enumerate(df_nodes['node_id'])}

        # Calculate distances for connected nodes
        for _, edge in df_edges.iterrows():
            node1 = node_id_to_index[edge['u']]
            node2 = node_id_to_index[edge['v']]
            dist = np.linalg.norm(node_positions[node1] - node_positions[node2])
            row_indices.append(node1)
            col_indices.append(node2)
            distances.append(dist)

        # Determine the number of total nodes (from limit or df_nodes)
        num_nodes = limit if limit is not None else len(df_nodes)

        # Ensure edge_index includes all nodes up to the limit
        missing_nodes = set(range(num_nodes)) - set(edge_index.unique().tolist())
        for node in missing_nodes:
            # Add self-loops for disconnected nodes
            row_indices.append(node)
            col_indices.append(node)
            distances.append(0.0)

        # Create sparse distance matrix with all nodes
        dist_matrix = sps.coo_matrix((distances, (row_indices, col_indices)), shape=(num_nodes, num_nodes))

        # Optionally save the distance matrix
        if save_path:
            sps.save_npz(save_path, dist_matrix)
            print(f"Sparse distance matrix saved to {save_path}")

        print(f"Sparse distance matrix calculated. Shape: {dist_matrix.shape}")
        self.dist = dist_matrix
        return dist_matrix
 
   
    def get_similarity(self, method='rbf', thr=0.1, sigma=None, force_symmetric=False, sparse=True):
        """
        Calculate similarity using various methods like Pearson Correlation, Gaussian RBF, etc.

        Parameters:
        - method (str): Similarity calculation method ('pearson', 'rbf', 'attention', 'graph_learning').
        - thr (float): Threshold for filtering small similarity scores.
        - sigma (float): Standard deviation parameter for Gaussian RBF kernel.
        - force_symmetric (bool): Whether to enforce symmetry on the adjacency matrix.

        Returns:
        - adj (scipy.sparse.coo_matrix): Sparse adjacency matrix based on the chosen similarity calculation method.
        """
        
            
        if method == 'pearson':
            adj = self.pearson_correlation_similarity(thr, force_symmetric)
        elif method == 'rbf':
            adj = self.rbf_similarity(thr, sigma, force_symmetric, sparse)
        elif method == 'attention':
            adj = self.spatial_attention_similarity(force_symmetric)
        elif method == 'graph_learning':
            adj = self.graph_learning_similarity(force_symmetric)
        else:
            raise ValueError(f"Unknown similarity calculation method: {method}")
        return adj
    

    def pearson_correlation_similarity(self, thr, force_symmetric):
        pass

    def rbf_similarity(self, thr, sigma, force_symmetric, sparse):
        """
        Calculate similarity using a Gaussian RBF kernel. dist from the load_distance_matrix method.
        
        """
        
        if not isinstance(self.dist, sps.spmatrix):
            raise ValueError("Distance matrix `self.dist` must be a sparse matrix!")
            
        if self.dist.shape[0] != self.dist.shape[1]:
            raise ValueError("Distance matrix must be square!")    
        
        # Use the sparse data directly for calculations
        finite_dist = self.dist.data[~np.isinf(self.dist.data)]

        if len(finite_dist) == 0:
            print("[Warning] Distance matrix contains no finite values. Falling back to a default adjacency matrix.")
           # Create a default adjacency matrix
            if sparse:
                num_nodes = self.dist.shape[0]
                adj = sps.coo_matrix(([], ([], [])), shape=(num_nodes, num_nodes))  # Empty adjacency matrix
            else:
                num_nodes = self.dist.shape[0]
                adj = np.zeros((num_nodes, num_nodes))  # Dense zero matrix
            return adj

        if sigma is None:
             sigma = finite_dist.std() #calculating the std

        # Apply the RBF formula
        exp_dist = np.exp(-np.square(self.dist.data / sigma))
        exp_dist[exp_dist < thr] = 0.0 #making the distance zero which are smaller than the threshold

        # Initialize adj based on sparse flag
        if sparse:
            adj = sps.coo_matrix((exp_dist, (self.dist.row, self.dist.col)), shape=self.dist.shape)
        else:
            adj = np.zeros(self.dist.shape)
            adj[self.dist.row, self.dist.col] = exp_dist

        # Enforce symmetry if required
        if force_symmetric:
            if sparse:
                adj = adj.maximum(adj.T)
            else:
                adj = np.maximum(adj, adj.T)

        return adj 

    def spatial_attention_similarity(self, force_symmetric):
        pass

    def graph_learning_similarity(self, force_symmetric):
        pass
        

    @property
    def mask(self):
        return self._mask

    @property
    def df(self):
        """
        Generate a DataFrame representation of the dataset.

        Returns:
        - pd.DataFrame: DataFrame containing dataset details (e.g., `day_t` and indices).
        """
        if not hasattr(self, "full_day_t"):
            raise AttributeError("Dataset does not have the 'full_day_t' attribute.")
        
        data = {
            "index": range(len(self.full_day_t)),
            "day_t": self.full_day_t,
        }
        
        # Include additional columns if needed
        return pd.DataFrame(data)

    def data_timestamps(self, indices, flatten=True):
        """
        Retrieve timestamps corresponding to specific indices.

        Parameters:
        - indices (list or np.array): Indices to fetch timestamps for.
        - flatten (bool): If True, return a flattened list of timestamps. Otherwise, return as-is.

        Returns:
        - dict: A dictionary with relevant timestamp information.
        """
        if not hasattr(self, "full_day_t"):
            raise AttributeError("Dataset does not have the 'full_day_t' attribute.")

        print(f"[data_timestamps Debug] Original indices length: {len(indices)}")
        if self.limit is not None:
           print(f"[data_timestamps Debug] Applying limit: {self.limit}")
           indices = indices[:self.limit]
        print(f"[data_timestamps Debug] Indices length after applying limit: {len(indices)}")

        
        # Extract timestamps for the provided indices
        timestamps = [self.full_day_t[idx] for idx in indices]
        print(f"[data_timestamps Debug] Retrieved timestamps for {len(timestamps)} indices")

        if flatten:
            flattened_horizons = [t[1] for t in timestamps]
            print(f"[data_timestamps Debug] Flattened timestamps length: {len(flattened_horizons)}")
            return {"horizon": flattened_horizons}
        else:
            print(f"[data_timestamps Debug] Returning unflattened timestamps")
            return {"horizon": timestamps}
        
    def len(self) -> int:
        if self.limit is not None:
            return min(self.limit, len(self.day_t))
        print(f"[t4c22 geometric] total number of time steps {len(self.day_t)}")
        return len(self.day_t)
    

    def __getitem__(self, idx: int) -> torch_geometric.data.Data:
        if idx >= len(self.day_t):
            IndexError(f"Index {idx} is out of range for dataset of size {len(self.day_t)}.")
        return self.get(idx)


    #def get(self, idx: int, subset_nodes: Optional[int] = None) -> torch_geometric.data.Data:
    def _load_data(self, day, t, idx, limit: Optional[int] = None):
        """
        Load data dynamically, optionally subsetting nodes for debugging or efficiency.

        Parameters:
        - day: The day string.
        - t: The time index.
        - idx: The sample index.
        - limit: Number of nodes to include in the subset (None for all nodes).
        """
        # Debugging: Print the day and timestamp for the sample being loaded
        #print(f"[t4c22 geometric] Loading data for day: {day}, time index: {t}, sample index: {idx}")

        #day, t = self.day_t[idx]

        city = self.city
        basedir = self.root
        
        # Handle cached data
        if self.cachedir is not None:
            cache_file = self.cachedir / (
                f"cc_labels_{self.city}_{day}_{t}.pt" if self.competition == T4c22Competitions.CORE else f"eta_labels_{self.city}_{day}_{t}.pt"
            )

            if cache_file.exists():
                data = torch.load(cache_file)
                print(f"[Cache] Loaded data from {cache_file}")
                return data

        # x: 4 time steps of loop counters on nodes
        x, mask = self.torch_road_graph_mapping.load_inputs_day_t(basedir=basedir, city=city, day=day, t=t, idx=idx)

        # Apply dynamic subsetting if limit is provided
        if limit is not None:
           x = x[:limit]
           mask = mask[:limit]

        # Transfer tensors to the dataset's 

        x = x.clone().detach().to(torch.float32).to(self.device)
        mask = mask.clone().detach().to(torch.bool).to(self.device)
        
        # Load node_id mapping (only for debugging purposes)
        if not hasattr(self, "node_id_mapping"):
            node_df = pd.read_parquet(road_graph_nodes_path)
            self.node_id_mapping = {idx: row.node_id for idx, row in node_df.iterrows()}

        # y: congestion classes on edges at +60'
        y = None
        #if self.split != "test":
        if self.competition == T4c22Competitions.CORE:
            y = self.torch_road_graph_mapping.load_cc_labels_day_t(basedir=basedir, city=city, day=day, t=t, idx=idx)
        else:
            y = self.torch_road_graph_mapping.load_eta_labels_day_t(basedir=basedir, city=city, day=day, t=t, idx=idx)

            # # Subset y dynamically based on the number of nodes
            # if limit is not None and y is not None:
            #     y = y[:limit]

       # Convert y to tensor
        if y is not None:
            y = y.clone().detach().to(torch.float32).to(self.device)

        # Dynamically adjust edge_index for the subset
        edge_index = self.torch_road_graph_mapping.edge_index
        if limit is not None:
            edge_index_mask = (edge_index[0, :] < limit) & (edge_index[1, :] < limit)
            edge_index = edge_index[:, edge_index_mask]

        # Create data object
        data = torch_geometric.data.Data(
            x=x,
            edge_index=edge_index,
            edge_attr=self.torch_road_graph_mapping.edge_attr,
            y=y,
            mask=mask
        )    
                    
        if self.cachedir is not None:
            self.cachedir.mkdir(exist_ok=True, parents=True)
            torch.save(data, cache_file)
            print(f"[Cache] Saved data to {cache_file}")

        return data
    