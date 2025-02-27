import torch
from torch_geometric.data import Batch

def custom_collate_fn(batch):
    """
    Custom collate function for batching PyTorch Geometric Data objects.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Separate `window` and `horizon` batches
    window_batch = [data.window for data in batch if hasattr(data, 'window')]
    horizon_batch = [data.horizon for data in batch if hasattr(data, 'horizon')]

    if not window_batch or not horizon_batch:
        print("[custom_collate_fn] Empty or invalid batch.")
        return None

    window_batch = [item for sublist in window_batch for item in sublist]
    horizon_batch = [item for sublist in horizon_batch for item in sublist]

    batched_window = Batch.from_data_list(window_batch).to(device)
    batched_horizon = Batch.from_data_list(horizon_batch).to(device)

    # Collect adjacency matrices from individual graphs
    adj_list = [data.adj for data in batch if hasattr(data, 'adj')]

    if adj_list:
        batched_adj = batch_sparse_adj(adj_list)  # Use block diagonal sparse adjacency batching
    else:
        batched_adj = None

    return {
        'window': batched_window,
        'horizon': batched_horizon,
        'adj': batched_adj.to(device) if batched_adj is not None else None  
    }


def batch_sparse_adj(adj_list):
    """
    Given a list of sparse adjacency matrices, returns a batched sparse tensor
    by stacking them as a block diagonal sparse matrix.
    """
    indices, values, sizes = [], [], []
    row_offset = 0

    for adj in adj_list:
        coo = adj.coalesce()
        indices.append(coo.indices() + row_offset)  # Offset indices for block diagonal stacking
        values.append(coo.values())
        sizes.append(coo.size()[0])  # Number of nodes per graph

        row_offset += adj.shape[0]

    stacked_indices = torch.cat(indices, dim=1)
    stacked_values = torch.cat(values)
    total_size = (row_offset, row_offset)

    return torch.sparse_coo_tensor(stacked_indices, stacked_values, total_size)

def expand_adjacency(adj, batch_size):
    """
    Expands an adjacency matrix dynamically to match the batch size.

    Args:
        adj (torch.sparse_coo_tensor): Original adjacency matrix of shape [num_nodes, num_nodes].
        batch_size (int): Number of graphs in the batch.

    Returns:
        torch.sparse_coo_tensor: Expanded adjacency matrix of shape [batch_size * num_nodes, batch_size * num_nodes].
    """
    if not adj.is_sparse:
        raise ValueError("Input adjacency matrix must be a sparse tensor.")

    num_nodes = adj.shape[0]  # Original number of nodes
    total_nodes = batch_size * num_nodes  # New total nodes in batched adjacency

    indices, values = adj.coalesce().indices(), adj.coalesce().values()

    batch_indices = []
    batch_values = []

    for i in range(batch_size):
        offset = i * num_nodes  # Offset indices for each batch graph
        new_indices = indices + offset  # Shift node indices
        batch_indices.append(new_indices)
        batch_values.append(values)

    # Concatenate all indices and values
    stacked_indices = torch.cat(batch_indices, dim=1)
    stacked_values = torch.cat(batch_values)

    # Create block-diagonal sparse adjacency matrix
    batched_adj = torch.sparse_coo_tensor(stacked_indices, stacked_values, (total_nodes, total_nodes))

    return batched_adj.coalesce()  # Ensure the final tensor is coalesced
