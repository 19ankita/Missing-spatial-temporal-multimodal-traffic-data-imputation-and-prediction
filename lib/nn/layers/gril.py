import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .spatial_conv import SpatialConvOrderK
from .gcrnn import GCGRUCell
from ..utils.ops import reverse_tensor


class SpatialDecoder(nn.Module):
    
    def __init__(self, d_in, d_model, d_out, support_len, order=1, attention_block=False, nheads=2, dropout=0.):


        super(SpatialDecoder, self).__init__()
        self.order = order 

        self.lin_in = nn.Conv1d(d_in, d_model, kernel_size=1) 

        # Initial SpatialConvOrderK
        self.graph_conv = SpatialConvOrderK(c_in=d_model, c_out=d_model,
                                            support_len=support_len * order, order=1, include_self=False)

        # Attention - optional
        if attention_block:
            self.spatial_att = SpatialAttention(d_in=d_model, d_model=d_model, nheads=nheads, dropout=dropout)
            self.lin_out = nn.Conv1d(3 * d_model, d_model, kernel_size=1)
        else:
            self.register_parameter('spatial_att', None)
            self.lin_out = nn.Conv1d(2 * d_model, d_model, kernel_size=1)

        self.read_out = nn.Conv1d(2 * d_model, d_out, kernel_size=1)
        self.activation = nn.PReLU()
        self.adj = None
        

    def forward(self, x, m, h, u, adj, cached_support=False):

        num_nodes = x.size(2)  
        m = m.expand(-1, -1, num_nodes)

        # Ensure `h` matches `x` in num_nodes
        if h.shape[2] != num_nodes:
            h = h[:, :, :num_nodes]  # Adjust hidden state if needed

        x_in = [x, m, h] if u is None else [x, m, u, h]
        x_in = torch.cat(x_in, 1)

        if self.order > 1:
            if cached_support and (self.adj is not None):
                adj = self.adj
            else:
                adj = SpatialConvOrderK.compute_support_orderK(adj, self.order, include_self=False, device=x_in.device) 
                self.adj = adj if cached_support else None

        x_in = self.lin_in(x_in)
 
        out = self.graph_conv(x_in, adj)
        
        if self.spatial_att is not None:
            x_in = rearrange(x_in, 'b f n -> b 1 n f')
            out_att = self.spatial_att(x_in, torch.eye(x_in.size(2), dtype=torch.bool, device=x_in.device))
            out_att = rearrange(out_att, 'b s n f -> b f (n s)')
            out = torch.cat([out, out_att], 1)

        if out.shape[0] != h.shape[0]:
            
            if out.shape[0] < h.shape[0]:  # Expand `out`
                out = out.repeat(h.shape[0] // out.shape[0], 1, 1)  # Repeat along batch dimension
            else:  # Slice `out` to match `h`
                out = out[:h.shape[0]]

        out = torch.cat([out, h], 1) 

        out = self.activation(self.lin_out(out))

        out = torch.cat([out, h], 1)

        return self.read_out(out), out

class GRIL(nn.Module):
    
    def __init__(self,
                 input_size,
                 hidden_size,
                 u_size=None,
                 n_layers=1,
                 dropout=0.,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 support_len=2,
                 n_nodes=None,
                 layer_norm=False):

        super(GRIL, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.u_size = int(u_size) if u_size is not None else 0
        self.n_layers = int(n_layers)
        rnn_input_size = 2 * self.input_size + self.u_size  # input + mask + (eventually) exogenous
    
        # Spatio-temporal encoder (rnn_input_size -> hidden_size)
        self.cells = nn.ModuleList() #GCGRU cell to capture both the spatial and temporal data
        self.norms = nn.ModuleList()
        for i in range(self.n_layers):
            d_in = rnn_input_size if i == 0 else self.hidden_size
            print(f"d_in at layer {i}: {d_in}")  # Print the d_in for each layer
            self.cells.append(GCGRUCell(d_in=rnn_input_size if i == 0 else self.hidden_size,
                                        num_units=self.hidden_size, support_len=support_len, order=kernel_size))
            if layer_norm:
                self.norms.append(nn.GroupNorm(num_groups=1, num_channels=self.hidden_size))
            else:
                self.norms.append(nn.Identity())

        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # Fist stage readout - initial imputation based on the current hidden state
        self.first_stage = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.input_size, kernel_size=1)

        self.spatial_decoder = SpatialDecoder(d_in=rnn_input_size + self.hidden_size,
                                              d_model=self.hidden_size,
                                              d_out=self.input_size,
                                              support_len=2,
                                              order=decoder_order,
                                              attention_block=global_att)
        

        # Hidden state initialization embedding
        if n_nodes is not None:
            self.h0 = self.init_hidden_states(n_nodes)
        else:
            self.register_parameter('h0', None)

    def init_hidden_states(self, n_nodes):
        h0 = []
        for l in range(self.n_layers):
            std = 1. / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
            vals = torch.distributions.Normal(0, std).sample((self.hidden_size, n_nodes))
            h0.append(nn.Parameter(vals))
        return nn.ParameterList(h0)

    def get_h0(self, x):
        if self.h0 is not None:
            return [h.expand(x.shape[0], -1, -1) for h in self.h0]
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2])).to(x.device)] * self.n_layers

    def update_state(self, x, h, adj):
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            print(f"Inside the loop...")
            rnn_in = h[layer] = norm(cell(rnn_in, h[layer], adj))
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h
    
    
    def forward(self, x, adj, mask=None, u=None, h=None, cached_support=False, training_mode=True):

        # x: [batch, features, nodes, steps]
        # Extract batch size, channels, num_nodes, and steps dynamically
        batch_size, channels, num_nodes, steps = x.shape

        # Infer all valid if mask is None
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.uint8)

        # If no hidden state is provided, initialize one
        if h is None:
            h = self.get_h0(x)
        elif not isinstance(h, list):
            h = [*h]

        # Temporal conv
        predictions, imputations, states, representations = [], [], [], []

        for step in range(steps): 

            x_s = x[..., step]  
            m_s = mask[..., step]  

            h_s = h[-1]  
            u_s = u[..., step] if u is not None else None  

            # First-stage imputation: Predict missing values using the current hidden state
            xs_hat_1 = self.first_stage(h_s) 


            # Fill missing values in `x_s` with the predicted values `xs_hat_1`
            x_s = torch.where(m_s, x_s, xs_hat_1)

            # Use spatial decoder to get more refined imputation and representation
            xs_hat_2, repr_s = self.spatial_decoder(x=x_s, m=m_s, h=h_s, u=u_s, adj=adj, cached_support=cached_support) 

            # readout of imputation state + mask to retrieve imputations
            # prepare inputs
            x_s = torch.where(m_s, x_s, xs_hat_2)

            # Prepare the input for the next RNN update -Concatenate the imputed features, mask, and optional exogenous inputs.
            inputs = [x_s, m_s]

            if u_s is not None:
                inputs.append(u_s)
    
            inputs = torch.cat(inputs, dim=1)  # x_hat_2 + mask + exogenous

          # update state with original sequence filled using imputations
            h = self.update_state(inputs, h, adj) #Use the GCGRU cell to compute the new hidden state.
            
            # store imputations and states
            imputations.append(xs_hat_2) 
            predictions.append(xs_hat_1) 
            states.append(torch.stack(h, dim=0)) 
            representations.append(repr_s) 

        # Aggregate outputs -> [batch, features, nodes, steps]
        imputations = torch.stack(imputations, dim=-1)
        predictions = torch.stack(predictions, dim=-1)
        states = torch.stack(states, dim=-1)
        representations = torch.stack(representations, dim=-1)

        return imputations, predictions, representations, states


class BiGRIL(nn.Module):

    def __init__(self,
                 input_size, #feature dimension
                 hidden_size,
                 ff_size,
                 ff_dropout,
                 n_layers=1,
                 dropout=0.,
                 n_nodes=None,
                 support_len=2,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 u_size=0,
                 embedding_size=0,
                 layer_norm=False,
                 merge='mlp'):

        super(BiGRIL, self).__init__()

        self.fwd_rnn = GRIL(input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            dropout=dropout,
                            n_nodes=n_nodes,
                            support_len=support_len,
                            kernel_size=kernel_size,
                            decoder_order=decoder_order,
                            global_att=global_att,
                            u_size=u_size,
                            layer_norm=layer_norm)

        self.bwd_rnn = GRIL(input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            dropout=dropout,
                            n_nodes=n_nodes,
                            support_len=support_len,
                            kernel_size=kernel_size,
                            decoder_order=decoder_order,
                            global_att=global_att,
                            u_size=u_size,
                            layer_norm=layer_norm)

        if n_nodes is None:
            embedding_size = 0
        if embedding_size > 0:
            self.emb = nn.Parameter(torch.empty(embedding_size, n_nodes))
            nn.init.kaiming_normal_(self.emb, nonlinearity='relu')
        else:
            self.register_parameter('emb', None)

        #merging both the forward and backward model output   
        if merge == 'mlp':
            self._impute_from_states = True
            self.out = nn.Sequential(
                nn.Conv2d(in_channels=4 * hidden_size + input_size + embedding_size,
                          out_channels=ff_size, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Conv2d(in_channels=ff_size, out_channels=input_size, kernel_size=1)
            )
        elif merge in ['mean', 'sum', 'min', 'max']:
            # if the merge is happening through MLP, then all the hidden states and embeddings are combined too
            self._impute_from_states = False 
            self.out = getattr(torch, merge)
        else:
            raise ValueError("Merge option %s not allowed." % merge)
        self.supp = None

    def forward(self, x, adj, mask=None, u=None, cached_support=False):

        if cached_support and (self.supp is not None):
            supp = self.supp
        else:
            supp = SpatialConvOrderK.compute_support(adj, x.device)
            self.supp = supp if cached_support else None
             
        # Forward - this overrides the GRIL class forward method
        fwd_out, fwd_pred, fwd_repr, _ = self.fwd_rnn(x, supp, mask=mask, u=u, cached_support=cached_support)
        
        # Backward
        rev_x, rev_mask, rev_u = [reverse_tensor(tens) for tens in (x, mask, u)]
        
        *bwd_res, _ = self.bwd_rnn(rev_x, supp, mask=rev_mask, u=rev_u, cached_support=cached_support)
        bwd_out, bwd_pred, bwd_repr = [reverse_tensor(res) for res in bwd_res]
    
        if self._impute_from_states:
            inputs = [fwd_repr, bwd_repr, mask]

            if self.emb is not None:
                b, *_, s = fwd_repr.shape  # fwd_h: [batches, channels, nodes, steps]

                # Ensure emb matches the number of nodes
                num_nodes = fwd_repr.shape[2]  # Extract the number of nodes from fwd_repr

                if self.emb.shape[1] < num_nodes:
                    padding = (0, num_nodes - self.emb.shape[1])  
                    emb_padded = F.pad(self.emb, padding, mode="constant", value=0)  # Pad missing nodes
                elif self.emb.shape[1] > num_nodes:
                    emb_padded = self.emb[:, :num_nodes]  # Trim excess nodes
                else:
                    emb_padded = self.emb  # No modification needed

            # Ensure the modified embedding remains a Parameter
            if not isinstance(self.emb, nn.Parameter):
                self.emb = nn.Parameter(emb_padded, requires_grad=True)

            # Expand embedding for batch and steps
            emb_expanded = self.emb.view(1, *self.emb.shape, 1).expand(b, -1, -1, s)

            inputs += [emb_expanded]

            # Ensure all tensors in `inputs` have the same shape before concatenation
            min_nodes = min(t.shape[2] for t in inputs)  # Find smallest num_nodes across inputs
            inputs = [t[:, :, :min_nodes, :] for t in inputs]  # Trim to min_nodes

            imputation = torch.cat(inputs, dim=1)

            imputation = self.out(imputation)
        else:
            # Stack forward and backward outputs

            imputation = torch.stack([fwd_out, bwd_out], dim=1)

            imputation = self.out(imputation, dim=1)


        predictions = torch.stack([fwd_out, bwd_out, fwd_pred, bwd_pred], dim=0)

        return imputation, predictions
