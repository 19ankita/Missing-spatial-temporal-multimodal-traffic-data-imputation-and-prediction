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

        training_mode = kwargs.get('training_mode', True)
        
        print(f"======================================")
        print(f"GRIN forward....")
        print(f"======================================")
        
        # x: [batches, steps, nodes, channels] -> [batches, channels, nodes, steps]
            
        # Use self.batch_size if batch_size is not provided
        batch_size = batch_size or self.batch_size
        
        steps = steps or self.window  
            
        num_nodes = x.shape[-2]  
        channels = x.shape[-1]  # Assume last dimension is always channels
     
        print(f"Batch size: {batch_size}, Steps: {steps}, Num nodes: {num_nodes}, Channels: {channels}")
        print(f"Mask shape: {mask.shape}")

        # **Fix: Use a mask to extract relevant nodes**
        if num_nodes == self.adj.shape[0]:  # If adj already matches batch nodes, use it
            adj_batch = self.adj
        else:
            indices = self.adj.indices()
            values = self.adj.values()

            # Mask indices for the valid nodes
            mask_rows = indices[0] < num_nodes
            mask_cols = indices[1] < num_nodes
            valid_mask = mask_rows & mask_cols  # Keep only valid edges

            # Apply mask
            masked_indices = indices[:, valid_mask]
            masked_values = values[valid_mask]

            # Construct new sparse adjacency matrix
            adj_batch = torch.sparse_coo_tensor(
                masked_indices, masked_values, (num_nodes, num_nodes), device=self.adj.device
            ).coalesce()

        print(f"[GRIN forward] Using adjacency shape: {adj_batch.shape}")
        
        # Validate dimensions and reshape `x` dynamically
        if x.dim() == 2:  # Case [nodes, channels]
           num_nodes, channels = x.shape
           steps = steps or 1  # If steps is not provided, assume a single time step
           total_elements = x.numel()
           expected_elements = 1 * steps * num_nodes * channels
           if total_elements != expected_elements:
              raise ValueError(f"[GRIN] Invalid reshape dimensions for x. Expected {expected_elements}, got {total_elements}.")

           x = x.view(1, steps, num_nodes, channels).repeat(batch_size, 1, 1, 1)
        elif x.dim() == 3:  # Case [steps, nodes, channels]
            steps, num_nodes, channels = x.shape
            total_elements = x.numel()
            expected_elements = 1 * steps * num_nodes * channels
            if total_elements != expected_elements:
               raise ValueError(f"[GRIN] Invalid reshape dimensions for x. Expected {expected_elements}, got {total_elements}.")

            x = x.view(1, steps, num_nodes, channels).repeat(batch_size, 1, 1, 1)
        elif x.dim() == 4:  # Case [batch_size, steps, nodes, channels]
            batch_size, steps, num_nodes, channels = x.shape
        else:
            raise ValueError(f"[GRIN] Unexpected input shape for x: {x.shape}")
        
        # Rearrange for convolution
        x = rearrange(x, 'b s n c -> b c n s')

        print(f"[GRIN forward] Rearranged x shape: {x.shape}")

        # Reshape mask
        if mask is not None:
            if mask.dim() == 2:  # Case [nodes, channels]
                mask = mask.view(1, steps, num_nodes, mask.shape[-1]).repeat(batch_size, 1, 1, 1)
            elif mask.dim() == 3:  # Case [steps, nodes, channels]
                mask = mask.view(1, steps, num_nodes, mask.shape[-1]).repeat(batch_size, 1, 1, 1)
            elif mask.dim() == 4:  # Case [batch_size, steps, nodes, channels]
                pass  # Already in desired format
            else:
                raise ValueError(f"[GRIN] Unexpected input shape for mask: {mask.shape}")

            mask = rearrange(mask, "b s n c -> b c n s")  # [batch, steps, nodes, channels] -> [batch, channels, nodes, steps]   
       
        print(f"[GRIN forward] After rearrange - mask shape: {mask.shape}")

        if u is not None:
           u = rearrange(u, 'b s n c -> b c n s')       
        
        print(f"BiGRIL from GRIN....") 
        # imputation: [batches, channels, nodes, steps] prediction: [4, batches, channels, nodes, steps]
        imputation, prediction = self.bigrill(x, adj_batch, mask=mask, u=u, cached_support=self.training)

        print(f"[GRIN forward] Before imputation - mask shape: {mask.shape}, x shape: {x.shape}, imputation shape: {imputation.shape}")
       
        if self.impute_only_holes and not self.training:
           imputation = torch.where(mask, x, imputation)

        # Check imputation and prediction before transpose
        print(f"[GRIN forward] Before transpose:")
        print(f"[GRIN forward] imputation shape: {imputation.shape}")
        print(f"[GRIN forward] prediction shape: {prediction.shape}")

        # out: [batches, channels, nodes, steps] -> [batches, steps, nodes, channels]
        imputation = torch.transpose(imputation, -3, -1)
        prediction = torch.transpose(prediction, -3, -1)

        # Check imputation and prediction after transpose
        print(f"[GRIN forward] After transpose:")
        print(f"[GRIN forward] imputation shape: {imputation.shape}")
        print(f"[GRIN forward] prediction shape: {prediction.shape}")

        if self.training:
            print(f"Training mode - returning both imputation and prediction")
            return imputation, prediction
        return imputation

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
