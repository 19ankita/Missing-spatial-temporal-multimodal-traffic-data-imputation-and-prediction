from functools import partial

import torch
#from pytorch_lightning.metrics import Metric
from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape

#based on PyTorch's Metric class
#metric computation module designed for handling scenarios where certain parts of 
#the data need to be excluded from metric computation (e.g., due to masking, missing values, 
#or irrelevant regions).
class MaskedMetric(Metric):
    def __init__(self,
                 metric_fn,
                 mask_nans=False,
                 mask_inf=False,
                 #compute_on_step=True,
                 dist_sync_on_step=False,
                 process_group=None,
                 dist_sync_fn=None,
                 metric_kwargs=None,
                 at=None):
        super(MaskedMetric, self).__init__(#compute_on_step=compute_on_step,
                                           dist_sync_on_step=dist_sync_on_step,
                                           process_group=process_group,
                                           dist_sync_fn=dist_sync_fn)

        if metric_kwargs is None:
            metric_kwargs = dict()
        self.metric_fn = partial(metric_fn, **metric_kwargs) #A function (like mean squared 
        #error or mean absolute error) that computes the metric between predictions (y_hat) and targets (y).
        
        self.mask_nans = mask_nans #If True, ignore NaN values in the metric computation.
        self.mask_inf = mask_inf #If True, ignore infinite values in the metric computation.
        if at is None:
            self.at = slice(None) # can compute the metric only for a specific slice of the time steps.
        else:
            self.at = slice(at, at + 1)
            
        #Accumulates the sum of the metric values across all updates    
        self.add_state('value', dist_reduce_fx='sum', default=torch.tensor(0.).float())
        
        #Tracks the number of valid elements considered for the metric computation.
        self.add_state('numel', dist_reduce_fx='sum', default=torch.tensor(0))

    def _check_mask(self, mask, val):
        #Ensures that the mask matches the shape of the predicted values (y_hat).
        if mask is None:
            #mask = torch.ones_like(val).byte()
            mask = torch.ones_like(val, dtype=torch.bool)
        else:
            # Align mask shape with val - thesis
            if mask.shape != val.shape:
                mask = mask.view_as(val)
            _check_same_shape(mask, val)
            
        #Applies additional filtering to the mask if mask_nans or mask_inf is set, 
        #ignoring NaN or inf values in y_hat.    
        
        if self.mask_nans: 
            mask = mask * ~torch.isnan(val)
        if self.mask_inf:
            mask = mask * ~torch.isinf(val)
        return mask

    def _compute_masked(self, y_hat, y, mask):
        #Computes the metric value only for the elements allowed by the mask.
        # Align y_hat to match y shape - thesis
        if y_hat.numel() != y.numel():
            y_hat = y_hat.flatten()[:y.numel()].view_as(y)
            
        _check_same_shape(y_hat, y)
        val = self.metric_fn(y_hat, y)
        mask = self._check_mask(mask, val)
        val = torch.where(mask, val, torch.tensor(0., device=val.device).float())
        return val.sum(), mask.sum()

    def _compute_std(self, y_hat, y):
        #Computes the metric value without any masking, assuming all elements are valid.
        # Align y_hat to match y shape - thesis
        if y_hat.numel() != y.numel():
            y_hat = y_hat.flatten()[:y.numel()].view_as(y)
            
        _check_same_shape(y_hat, y)
        val = self.metric_fn(y_hat, y)
        return val.sum(), val.numel()

    def is_masked(self, mask):
        #Checks if a mask is required, based on the presence of a mask or the mask_nans/mask_inf flags.
        return self.mask_inf or self.mask_nans or (mask is not None)

    def update(self, y_hat, y, mask=None):
        #This method is called during training or evaluation to update the accumulated metric states (value and numel).
        #Handles alignment issues between y_hat, y, and mask, ensuring that they all have compatible shapes.
        #Applies the metric function to compute values for valid elements and accumulates the results.
        
        if isinstance(y_hat, tuple):
            y_hat = y_hat[0]

        # Align y_hat with y for metric computation
        if y_hat.numel() != y.numel():
            y_hat = y_hat.flatten()[:y.numel()].view_as(y)
            
        if len(y.shape) == 1:
        # Skip slicing if y is 1D
            self.at = slice(None)    

        # Slice or reshape mask to match y_hat
        if mask is not None and mask.numel() != y.numel():
            mask = mask.flatten()[:y.numel()].view_as(y)
        #y_hat = y_hat[:, self.at]
        #y = y[:, self.at]
        #if mask is not None:
            #mask = mask[:, self.at]
            
        if self.is_masked(mask):
            val, numel = self._compute_masked(y_hat, y, mask)
        else:
            val, numel = self._compute_std(y_hat, y)
        self.value += val
        self.numel += numel

    def compute(self):
        if self.numel > 0:
            return self.value / self.numel
        return self.value
