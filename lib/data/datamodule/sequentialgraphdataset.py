import torch
from torch_geometric.data import Data

class SequentialGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, window, horizon, stride=1, device=None, limit=None):
        """
        Parameters:
        ----------
        data_list: list of torch_geometric.data.Data
            List of graph objects (assumes time-order).
        window: int
            Number of past graphs to use as input.
        horizon: int
            Number of future graphs to predict.
        stride: int
            Step size between sequences.
        """
        print(f"============================================")
        print(f"SequentialGraphDataset................")
        print(f"============================================")

        self.data_list = data_list
        self.window = window
        self.horizon = horizon
        self.stride = stride
        self.limit = limit
        self.device = device or torch.device("cpu")

        # Ensure sufficient length for sequences
        self.sequence_length = window + horizon

        if len(data_list) < self.sequence_length:
            raise ValueError(f"Insufficient data for the specified window ({window}) and horizon ({horizon}) lengths. "
                     f"Dataset size: {len(data_list)}, required: {window + horizon}.")
        
        # Generate valid starting indices
        self.indices = list(range(0, len(data_list) - self.sequence_length + 1, stride))
        if len(self.indices) < 1:
            raise ValueError(
                f"Not enough sequences can be generated. "
                f"Dataset size: {len(data_list)}, required sequence length: {self.sequence_length}."
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        window_data = self.data_list[start_idx : start_idx + self.window]
        horizon_data = self.data_list[start_idx + self.window : start_idx + self.sequence_length]

        def apply_limit(graphs):
            limited_graphs = []
            for g in graphs:
                if self.limit is not None:
                    # Filter valid edges for the node subset
                    valid_edges = (g.edge_index[0] < self.limit) & (g.edge_index[1] < self.limit)
                    edge_index = g.edge_index[:, valid_edges]
                    edge_attr = g.edge_attr[valid_edges] if g.edge_attr is not None else None

                    # Adjust `y` to match the subset
                    if g.y is not None:
                        if g.y.shape[0] == g.x.shape[0]:  # Node-level `y`
                            y = g.y[:self.limit]
                        elif g.y.shape[0] == g.edge_index.shape[1]:  # Edge-level `y`
                            y = g.y[valid_edges]
                        else:  # Graph-level `y`
                            y = g.y
                    else:
                        y = None

                    # Subset the graph
                    g = Data(
                        x=g.x[:self.limit],
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=y,
                        mask=g.mask[:self.limit] if g.mask is not None else None,
                    )
                limited_graphs.append(g.to(self.device))
            return limited_graphs

        window_data = apply_limit(window_data)
        horizon_data = apply_limit(horizon_data)
        
    # Combine window and horizon into a single torch_geometric.data.Data object
        combined_data = Data()
        combined_data.window = window_data
        combined_data.horizon = horizon_data

    # Combine masks from the window graphs
        try:
            combined_data.mask = torch.cat([getattr(g, 'mask', torch.empty(0).to(self.device)) for g in combined_data.window], dim=0)
        except AttributeError:
            combined_data.mask = None

        # Combine targets (y) from the horizon graphs
        try:
            combined_data.y = torch.cat([getattr(g, 'y', torch.empty(0).to(self.device)) for g in combined_data.horizon], dim=0)
        except AttributeError:
            combined_data.y = None         

        return combined_data
