import torch

from . import Filler
from ..nn.models import GRINet


class GraphFiller(Filler):
    """
    Inherits from Filler but introduces graph-specific logic.    
    """

    def __init__(self,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 scaled_target=False,
                 whiten_prob=0.05,
                 pred_loss_weight=1.,
                 warm_up=0,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None,
                 batch_size=None,  
                 mask=None,
                 adj=None,
                 **kwargs):
        super(GraphFiller, self).__init__(model_class=model_class,
                                          model_kwargs=model_kwargs,
                                          optim_class=optim_class,
                                          optim_kwargs=optim_kwargs,
                                          loss_fn=loss_fn,
                                          scaled_target=scaled_target,
                                          whiten_prob=whiten_prob, 
                                          metrics=metrics,
                                          scheduler_class=scheduler_class,
                                          scheduler_kwargs=scheduler_kwargs,
                                          batch_size=batch_size,
                                          mask=mask,
                                          adj=adj,
                                          **kwargs)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if adj is not None:
            self.adj = adj.to(self._device)  
        else:
            self.adj = None  

        self.tradeoff = pred_loss_weight
        self.save_hyperparameters("adj")  # Ensures `adj` persists across training steps
    
        if model_class is GRINet: #it trims predictions from both the start and end of the sequence by warm_up steps.
            self.trimming = (warm_up, warm_up)

    def trim_seq(self, *seq):
        trimmed_seq = []
        for s in seq:
            if s is None:  # Skip None tensors
                trimmed_seq.append(None)
            elif len(s.shape) == 1:  # Skip trimming for 1D tensors like `y` - target should be left untouched
                trimmed_seq.append(s)
            else:
                trimmed_seq.append(s[:, self.trimming[0]:s.size(1) - self.trimming[1]])
        if len(trimmed_seq) == 1:
            return trimmed_seq[0]
        return trimmed_seq

    def check_nan_inf(self, tensor, name="Tensor"):
        """Check for NaN or Inf values in tensors."""
        if tensor is not None:
            if torch.isnan(tensor).any():
                print(f"[Warning] {name} contains NaN values!")
            if torch.isinf(tensor).any():
                print(f"[Warning] {name} contains Inf values!")

    def training_step(self, batch, batch_idx): 

        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Compute masks
        mask = batch_data['mask'].clone().detach()
        batch_data['mask'] = torch.bernoulli(mask.clone().detach().float() * self.keep_prob).byte()
        
        eval_mask = batch_data.pop('eval_mask', None)
        if eval_mask is None:
            eval_mask = torch.zeros_like(mask, dtype=torch.bool)  # Default: all False (unseen data)
            
        eval_mask = (mask | eval_mask) & ~batch_data['mask']  # Bitwise OR and exclude current mask
        
        y = batch_data.pop('y')

        # **Check for NaN values**
        self.check_nan_inf(mask, "Mask")
        self.check_nan_inf(eval_mask, "Eval Mask")
        self.check_nan_inf(y, "Target (y)")

        if hasattr(self, "adj") and self.adj is not None:
           print(f"[GraphFiller] Training step - Adjacency matrix shape: {self.adj.shape}")
        else:
           print("[GraphFiller] Training step - Warning: `self.adj` is None or not set!")

        if 'adj' in batch_data:
            print(f"[GraphFiller] Using batched adjacency shape: {batch_data['adj'].shape}")
        else:
            print("[GraphFiller] Warning: No adjacency matrix provided!")

        batch_data['adj'] = self.adj

        # Compute predictions and compute loss
        res = self.predict_batch(batch, preprocess=False, postprocess=False)
        imputation, predictions = (res[0], res[1:]) if isinstance(res, (list, tuple)) else (res, [])

        # Check for NaNs before loss computation
        self.check_nan_inf(imputation, "Imputation")
        
        if mask is not None and len(mask.shape) == 1:
            mask = mask.unsqueeze(1)
        if eval_mask is not None and len(eval_mask.shape) == 1:
            eval_mask = eval_mask.unsqueeze(1)

        # trim to imputation horizon len
        imputation, mask, eval_mask, y = self.trim_seq(imputation, mask, eval_mask, y) 
                
        predictions = self.trim_seq(*predictions) 

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)
            for i, _ in enumerate(predictions):
                predictions[i] = self._postprocess(predictions[i], batch_preprocessing)

        loss = self.loss_fn(imputation, target, mask)  #Main Loss (Imputation Loss)
        self.check_nan_inf(loss, "Loss")
        
        for pred in predictions:
            loss += self.tradeoff * self.loss_fn(pred, target, mask)

        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)

        self.train_metrics.update(imputation.detach(), y, eval_mask)  # all unseen data
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):

        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        mask = batch_data.get('mask')
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')

        # **Check for NaN values**
        self.check_nan_inf(mask, "Mask")
        self.check_nan_inf(eval_mask, "Eval Mask")
        self.check_nan_inf(y, "Target (y)")

        # Compute predictions and compute loss
        imputation = self.predict_batch(batch, preprocess=False, postprocess=False)

        self.check_nan_inf(imputation, "Imputation")
        
        if mask is not None and len(mask.shape) == 1:
            mask = mask.unsqueeze(1)
        if eval_mask is not None and len(eval_mask.shape) == 1:
            eval_mask = eval_mask.unsqueeze(1)
            
        # trim to imputation horizon len
        imputation, mask, eval_mask, y = self.trim_seq(imputation, mask, eval_mask, y)

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)
            
        # Compute loss only on relevant parts
        val_loss = self.loss_fn(imputation, target, eval_mask)
        self.check_nan_inf(val_loss, "Validation Loss")

        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)

        self.val_metrics.update(imputation.detach(), y, eval_mask)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_loss', val_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return val_loss

  