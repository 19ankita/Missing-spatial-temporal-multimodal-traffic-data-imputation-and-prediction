import torch
from torch import nn
import torch.nn.functional as F


from ... import epsilon

class SpatialConvOrderK(nn.Module):
    """
    Efficient implementation inspired from graph-wavenet codebase
    """

    def __init__(self, c_in, c_out, support_len=3, order=2, include_self=True):
        super(SpatialConvOrderK, self).__init__()
        self.include_self = include_self 
            
        c_in = (order * support_len + (1 if include_self else 0)) * c_in 

        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)

        self.order = order 

    @staticmethod 
    def compute_support(adj, device=None):
        if device is not None:
            adj = adj.to(device)
        
        adj = adj.coalesce()  # Ensure the adjacency matrix is in COO format
        
        # Transpose the adjacency matrix
        adj_bwd = torch.sparse_coo_tensor(
            torch.stack([adj._indices()[1], adj._indices()[0]]),  # Transpose the indices
            adj._values(),
            adj.size()
        ).coalesce()
        
        # Normalize the forward and backward adjacency matrices
        adj_fwd = SpatialConvOrderK.normalize_sparse(adj, device)
        adj_bwd = SpatialConvOrderK.normalize_sparse(adj_bwd, device)

        # Return the normalized forward and backward adjacency matrices
        support = [adj_fwd, adj_bwd]

        return support
        
        
    @staticmethod #thesis
    def normalize_sparse(adj, device):
        # Sum the values in each row of the sparse matrix
        row_sum = torch.sparse.sum(adj, dim=1).to_dense() + epsilon + 1e-5
        
        # Create a diagonal matrix with 1 / row_sum
        inv_row_sum = torch.pow(row_sum, -1).view(-1, 1)
        
        # Broadcast the inverse of row sum to multiply with adjacency values
        new_values = adj._values() * inv_row_sum[adj._indices()[0]].squeeze()

        # Recreate the sparse matrix with normalized values
        normalized_adj = torch.sparse_coo_tensor(adj._indices(), new_values, adj.size()).coalesce()
        return normalized_adj    

    @staticmethod
    def compute_support_orderK(adj, k, include_self=False, device=None):
        if isinstance(adj, (list, tuple)):
            support = adj
        else:
            support = SpatialConvOrderK.compute_support(adj, device)
        supp_k = []
        for a in support:
            ak = a
            for i in range(k - 1):
                if ak.shape != a.shape:
                    print(f"[compute_support_orderK] Shape mismatch at order {i+1}: {ak.shape} vs {a.shape}")
                ak = torch.sparse.mm(ak, a)  # Use sparse matrix multiplication
                if not include_self:
                    ak = ak.to_dense()  # Convert to dense to zero out diagonals #thesis
                    ak.fill_diagonal_(0.)
                    ak = ak.to_sparse()  # Convert back to sparse #thesis
                supp_k.append(ak)

                print(f"[compute_support_orderK] Order {i+2}, shape of ak: {ak.shape}")
        return support + supp_k

    def forward(self, x, support):

        # [batch, features, nodes, steps]
        if x.dim() < 4:
            squeeze = True
            x = torch.unsqueeze(x, -1)
        else:
            squeeze = False
            
        out = [x] if self.include_self else []

        # Ensure support is a list
        if not isinstance(support, list):
            support = [support]

        for a in support:  
             if a.is_sparse:
                x1 = torch.stack([ 
                    torch.sparse.mm(a, x[n, c]) for n in range(x.shape[0]) for c in range(x.shape[1])
                ]).view(x.shape[0], x.shape[1], a.shape[0], x.shape[-1])  # Reshape back to (N, C, W, L)
             else:
                x1 = torch.einsum('ncvl,wv->ncwl', (x, a)).contiguous()

             out.append(x1)

        for k in range(2, self.order + 1):
            if a.is_sparse:
                x2 = torch.stack([ 
                    torch.sparse.mm(a, x[n, c]) for n in range(x.shape[0]) for c in range(x.shape[1])
                ]).view(x.shape[0], x.shape[1], a.shape[0], x.shape[-1])  # Reshape back to (N, C, W, L)
            else:
                x2 = torch.einsum('ncvl,wv->ncwl', (x, a)).contiguous()

            out.append(x2)
            x1=x2
    
        out = torch.cat(out, dim=1)

        expected_channels = self.mlp.weight.shape[1]  # Get expected input channels (128)
        current_channels = out.shape[1]  # Current input channels (136)
        channels_to_pad = expected_channels - current_channels  # Compute required padding

        # Apply padding only if needed
        if channels_to_pad > 0:
            out = F.pad(out, (0, 0, 0, 0, 0, channels_to_pad))  # Pad dynamically
        elif channels_to_pad < 0:
            out = out[:, :expected_channels, :, :]  # Trim extra channels

        out = self.mlp(out)

        if squeeze:
            out = out.squeeze(-1)
        return out