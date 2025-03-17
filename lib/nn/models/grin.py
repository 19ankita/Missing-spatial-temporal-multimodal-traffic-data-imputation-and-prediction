import torch
from einops import rearrange
from torch import nn
from scipy.sparse import coo_matrix

from ..layers import BiGRIL
from ...utils.parser_utils import str_to_bool
from ...utils.data_utils import expand_adjacency


class GRINet(nn.Module):
    def __init__(self,
                 adj,
                 d_in,
                 d_hidden,
                 d_ff,
                 ff_dropout,
                 n_layers=1,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 d_u=0,
                 d_emb=0,
                 layer_norm=False,
                 merge='mlp',
                 impute_only_holes=True,
                 window=None,
                 batch_size=None,
                 mask=None,
                 **kwargs):
        super(GRINet, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_u = int(d_u) if d_u is not None else 0 
        self.d_emb = int(d_emb) if d_emb is not None else 0
        self.impute_only_holes = impute_only_holes
        self.batch_size = batch_size
        self.mask = mask
        self.window = window 
        
        #Convert adjacency matrix to PyTorch tensor correctly
        if isinstance(adj, coo_matrix):
            i = torch.LongTensor([adj.row, adj.col])
            v = torch.FloatTensor(adj.data)
            sparse_adj = torch.sparse_coo_tensor(i, v, adj.shape).coalesce()
        elif isinstance(adj, torch.Tensor) and adj.is_sparse:
            sparse_adj = adj.coalesce()
        else:
            sparse_adj = torch.tensor(adj).float()

        self.register_buffer('adj', sparse_adj)  

        # Expand adjacency matrix dynamically
        self.adj = expand_adjacency(adj, batch_size)

        print(f"[GRINet] Expanded adj shape: {self.adj.shape}")      

        print(f"======================================")
        print(f"Starting GRINet....")
        print(f"======================================")

        self.bigrill = BiGRIL(input_size=self.d_in,
                              ff_size=d_ff,
                              ff_dropout=ff_dropout,
                              hidden_size=self.d_hidden,
                              embedding_size=self.d_emb,
                              n_nodes=self.adj.shape[0],
                              n_layers=n_layers,
                              kernel_size=kernel_size,
                              decoder_order=decoder_order,
                              global_att=global_att,
                              layer_norm=layer_norm,
                              merge=merge)


    def forward(self, x, edge_index, mask=None, u=None, batch_size=None, steps=None, **kwargs):
        """
        Handles reshaping x dynamically without exception-based validation.
        """
        # Default values
        batch_size = batch_size or self.batch_size
        steps = steps or self.window  # Default to self.window if steps is None

        # Extract input dimensions
        x_shape = x.shape
        x_dim = x.dim()

        print(f"[GRIN forward] Initial x shape: {x_shape}")

        # Ensure x always has the correct shape dynamically
        if x_dim == 2:  # Case: [nodes, channels]
            x = x.unsqueeze(0).unsqueeze(0)  # Shape -> [1, 1, nodes, channels]
            x = x.expand(batch_size, steps, -1, -1)  # Expand batch and steps
        
        elif x_dim == 3:  # Case: [steps, nodes, channels]
            x = x.unsqueeze(0)  # Add batch dimension -> [1, steps, nodes, channels]
            x = x.expand(batch_size, -1, -1, -1)  # Expand batch size

        elif x_dim == 4:  # Case: Already in [batch_size, steps, nodes, channels]
            pass  # No changes needed

        else:  
            # If x has an unexpected dimension, reshape it to (batch, steps, nodes, channels)
            x = x.view(batch_size, steps, -1, x.shape[-1])

        print(f"[GRIN forward] Reshaped x shape: {x.shape}")

        # Convert to expected format: [batch, channels, nodes, steps]
        x = rearrange(x, 'b s n c -> b c n s')

        print(f"[GRIN forward] Rearranged x shape: {x.shape}")

        # Handle `mask` in the same way as `x`
        if mask is not None:
            mask_shape = mask.shape
            mask_dim = mask.dim()

            if mask_dim == 2:  
                mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, steps, -1, mask.shape[-1])
            elif mask_dim == 3:  
                mask = mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
            elif mask_dim == 4:  
                pass  # Already correct
            else:
                mask = mask.view(batch_size, steps, -1, mask.shape[-1])

            mask = rearrange(mask, "b s n c -> b c n s")

        print(f"[GRIN forward] Final mask shape: {mask.shape if mask is not None else 'None'}")

        # Handle `u` (optional input)
        if u is not None:
            u = rearrange(u, 'b s n c -> b c n s')  

        # Process adjacency dynamically based on node count
        adj_batch = self.adj if x.shape[2] == self.adj.shape[0] else self.adj[:x.shape[2], :x.shape[2]]

        print(f"[GRIN forward] Using adjacency shape: {adj_batch.shape}")

        # Forward pass through BiGRIL
        imputation, prediction = self.bigrill(x, adj_batch, mask=mask, u=u, cached_support=self.training)

        # Convert back to original shape: [batches, steps, nodes, channels]
        imputation = rearrange(imputation, "b c n s -> b s n c")
        prediction = rearrange(prediction, "b c n s -> b s n c")

        print(f"[GRIN forward] Final imputation shape: {imputation.shape}")
        print(f"[GRIN forward] Final prediction shape: {prediction.shape}")

        return imputation if not self.training else (imputation, prediction)

        
    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-hidden', type=int, default=64)
        parser.add_argument('--d-ff', type=int, default=64)
        parser.add_argument('--ff-dropout', type=int, default=0.)
        parser.add_argument('--n-layers', type=int, default=1)
        parser.add_argument('--kernel-size', type=int, default=2)
        parser.add_argument('--decoder-order', type=int, default=1)
        parser.add_argument('--d-u', type=int, default=0)
        parser.add_argument('--d-emb', type=int, default=8)
        parser.add_argument('--layer-norm', type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--global-att', type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--merge', type=str, default='mlp')
        parser.add_argument('--impute-only-holes', type=str_to_bool, nargs='?', const=True, default=True)
        return parser
