import inspect
from copy import deepcopy

import pytorch_lightning as pl
import torch

from pytorch_lightning.utilities import rank_zero_warn
from torchmetrics import MetricCollection
from pytorch_lightning.utilities import move_data_to_device

from .. import epsilon
from ..nn.utils.metric_base import MaskedMetric
from ..utils.utils import ensure_list


class Filler(pl.LightningModule):
    def __init__(self,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 scaled_target=False,
                 whiten_prob=0.05,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None,
                 batch_size=None,  
                 mask=None,
                 **kwargs):
        """
        A generalized class for imputation tasks.
        
        :param model_class: Class of pytorch nn.Module implementing the imputer.
        :param model_kwargs: Model's keyword arguments.
        :param optim_class: Optimizer class.
        :param batch_size: batch size.
        :param mask: mask.
        :param optim_kwargs: Optimizer's keyword arguments.
        :param loss_fn: Loss function used for training.
        :param scaled_target: Whether to scale target before computing loss using batch processing information.
        :param whiten_prob: Probability of removing a value and using it as ground truth for imputation.
        :param metrics: Dictionary of type {'metric1_name':metric1_fn, 'metric2_name':metric2_fn ...}.
        :param scheduler_class: Scheduler class.
        :param scheduler_kwargs: Scheduler's keyword arguments.
        """
        super(Filler, self).__init__()
 
        self.save_hyperparameters(model_kwargs)

        self.model_cls = model_class #model setup
        self.model_kwargs = model_kwargs
        self.batch_size = batch_size if batch_size is not None else 32
        self.mask = mask
        
        self.model_kwargs['mask'] = mask

        self.optim_class = optim_class #optimizer setup
        self.optim_kwargs = optim_kwargs #optimizer setup
        self.scheduler_class = scheduler_class
        if scheduler_kwargs is None:
            self.scheduler_kwargs = dict()
        else:
            self.scheduler_kwargs = scheduler_kwargs

        if loss_fn is not None:
            self.loss_fn = self._check_metric(loss_fn, on_step=True)
        else:
            self.loss_fn = None

        self.scaled_target = scaled_target

        # during training whiten ground-truth values with this probability
        assert 0. <= whiten_prob <= 1.
        self.keep_prob = 1. - whiten_prob

        if metrics is None:
            metrics = dict()
        self._set_metrics(metrics) #setup metrics to track the performance during training, validation and testing

        # instantiate model
        self.model = self.model_cls(**self.model_kwargs).to(self._device)
        
        if 'mask' not in self.model_kwargs:
            raise ValueError("[Filler] Mask not found in model_kwargs during initialization.")

    def reset_model(self): #reset the model to it's initial state
        self.model = self.model_cls(**self.model_kwargs)

    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def forward(self, *args, **kwargs):

        if "mask" not in kwargs:
            rank_zero_warn(
                "[Filler.forward] Missing 'mask' in arguments. Available keys: {}".format(kwargs.keys()),
                stacklevel=2
            )
            raise ValueError("Missing 'mask' in arguments.")

	# Ensure inputs are moved to the appropriate device (if necessary)
        device = self._device
        args = [arg.to(device) if hasattr(arg, 'to') else arg for arg in args]
        kwargs = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in kwargs.items()}

        return self.model(*args, **kwargs)

    @staticmethod
    def _check_metric(metric, on_step=False):
        if not isinstance(metric, MaskedMetric):
            if 'reduction' in inspect.getfullargspec(metric).args:
                metric_kwargs = {'reduction': 'none'}
            else:
                metric_kwargs = dict()
            return MaskedMetric(metric, compute_on_step=on_step, metric_kwargs=metric_kwargs)
        return deepcopy(metric)

    def _set_metrics(self, metrics): #setup metrics to track the performance during training, validation and testing
        self.train_metrics = MetricCollection(
            {f'train_{k}': self._check_metric(m, on_step=True) for k, m in metrics.items()})
        self.val_metrics = MetricCollection({f'val_{k}': self._check_metric(m) for k, m in metrics.items()})
        self.test_metrics = MetricCollection({f'test_{k}': self._check_metric(m) for k, m in metrics.items()})

    def _preprocess(self, data, batch_preprocessing):
        """
        Perform preprocessing of a given input.

        :param data: pytorch tensor of shape [batch, steps, nodes, features] to preprocess
        :param batch_preprocessing: dictionary containing preprocessing data
        :return: preprocessed data
        """
        if isinstance(data, (list, tuple)):
            return [self._preprocess(d, batch_preprocessing) for d in data]

        trend = batch_preprocessing.get('trend', 0.)
        bias = batch_preprocessing.get('bias', 0.)
        scale = batch_preprocessing.get('scale', 1.0)

        if not isinstance(trend, torch.Tensor):
           trend = torch.tensor(trend, device=self._device)
        if not isinstance(bias, torch.Tensor):
           bias = torch.tensor(bias, device=self._device)
        if not isinstance(scale, torch.Tensor):
           scale = torch.tensor(scale, device=self._device)

        return (data - trend - bias) / (scale + epsilon)

    def _postprocess(self, data, batch_preprocessing):
        """
        Perform postprocessing (inverse transform) of a given input.

        :param data: pytorch tensor of shape [batch, steps, nodes, features] to trasform
        :param batch_preprocessing: dictionary containing preprocessing data
        :return: inverse transformed data
        """

        if isinstance(data, (list, tuple)):
            return [self._postprocess(d, batch_preprocessing) for d in data]
 
        trend = batch_preprocessing.get('trend', 0.)
        bias = batch_preprocessing.get('bias', 0.)
        scale = batch_preprocessing.get('scale', 1.)

        if not isinstance(trend, torch.Tensor):
           trend = torch.tensor(trend, device=self._device)
        if not isinstance(bias, torch.Tensor):
           bias = torch.tensor(bias, device=self._device)
        if not isinstance(scale, torch.Tensor):
           scale = torch.tensor(scale, device=self._device)

        return data * (scale + epsilon) + bias + trend

    def predict_batch(self, batch, preprocess=False, postprocess=True, return_target=False, **kwargs):
        """
        This method takes as an input a batch as a two dictionaries containing tensors and outputs the predictions.
        Prediction should have a shape [batch, nodes, horizon]

        :param batch: list dictionary following the structure [data:
                                                                {'x':[...], 'y':[...], 'u':[...], ...},
                                                              preprocessing:
                                                                {'bias': ..., 'scale': ..., 'x_trend':[...], 'y_trend':[...]}]
        :param preprocess: whether the data need to be preprocessed (note that inputs are by default preprocessed before creating the batch)
        :param postprocess: whether to postprocess the predictions (if True we assume that the model has learned to predict the trasformed signal)
        :param return_target: whether to return the prediction target y_true and the prediction mask
        :return: (y_true), y_hat, (mask)
        """
        # Move batch to the correct device
        batch = move_data_to_device(batch, self._device)
           
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        if batch_preprocessing is None:
           batch_preprocessing = {}  

        # Extract x, edge_index, and mask from the batch
        x = batch_data.get('x', None)
        edge_index = batch_data.get('edge_index', None)
        mask = batch_data.get('mask', None) 
       
        if preprocess:
            x = self._preprocess(x, batch_preprocessing)
            
        mask = batch.get('mask', None)
        if mask is None:
        # Generate a default mask (all valid) if mask is missing
           mask = torch.ones_like(x, dtype=torch.bool)
        
        # Forward pass
        y_hat = self.forward(x=x, edge_index=edge_index, mask=mask)
            
        # Postprocess if needed
        if postprocess:
            y_hat = self._postprocess(y_hat, batch_preprocessing)

        if return_target:
            y = batch_data.get('y', None)
            return y, y_hat, mask

        return y_hat

    def predict_loader(self, loader, preprocess=False, postprocess=True, return_mask=True):
        """
        Makes predictions for an input dataloader. Returns both the predictions and the predictions targets.

        :param loader: torch dataloader
        :param preprocess: whether to preprocess the data
        :param postprocess: whether to postprocess the data
        :param return_mask: whether to return the valid mask (if it exists)
        :return: y_true, y_hat
        """
        targets, imputations, masks = [], [], []
        batch_count = 0  # Counter for batches
        
        for batch in loader:
            print(f"[predict_loader] Processing batch {batch_count}...")

            batch = move_data_to_device(batch, self._device)
            print(f"[predict_loader] Batch moved to device: {self.device}")
            print(f"[predict_loader] Batch keys: {batch.keys() if isinstance(batch, dict) else 'Not a dictionary'}")

            batch_data, batch_preprocessing = self._unpack_batch(batch)

            # Extract mask and target
            eval_mask = batch_data.pop('eval_mask', None) 
            y = batch_data.pop('y')

            y_hat = self.predict_batch(batch, preprocess=preprocess, postprocess=postprocess, return_target=True)

            if isinstance(y_hat, (list, tuple)):
                y_hat = y_hat[0]

            targets.append(y)
            imputations.append(y_hat)
            masks.append(eval_mask)

            batch_count += 1

        y = torch.cat(targets, 0) if targets else None
        y_hat = torch.cat(imputations, 0) if imputations else None

        mask = torch.cat(masks, dim=0) if return_mask and masks and masks[0] is not None else None

        if return_mask:
            return y, y_hat, mask 
        return y, y_hat

    def _unpack_batch(self, batch):
        """
        Unpack a batch into data and preprocessing dictionaries.

        :param batch: the batch
        :return: batch_data, batch_preprocessing
        """

        if isinstance(batch, (tuple, list)) and (len(batch) == 2): 
            batch_data, batch_preprocessing = batch
        else:
            batch_data = batch #this is our case
            batch_preprocessing = dict()
    
        if "window" in batch_data:
            print(f"Batch `window`: {batch_data['window']}")
            window = batch_data["window"]

            # Flatten nested lists if necessary
            if isinstance(window, list) and isinstance(window[0], list):  #Training Dataset=[Data(x=[59110, 4], edge_index=[2, 132414], y=[132414], mask=[59110, 4])]               
               window = [item for sublist in window for item in sublist] #Validation Dataset = [[Data(x=[59110, 4], edge_index=[2, 132414], y=[132414], mask=[59110, 4])]]


            # Ensure `window` contains valid `Data` objects with `x`
            if len(window) > 0 and hasattr(window[0], "x"):
                x = torch.cat([data.x for data in window], dim=0)
                edge_index = torch.cat([data.edge_index for data in window], dim=1)
                # Add x and edge_index to batch_data without overwriting other keys
                batch_data["x"] = x
                batch_data["edge_index"] = edge_index
            else:
                raise ValueError("[filler] Batch does not have a valid 'window' with 'x' attribute.")
        else:
            raise ValueError("[filler] Batch does not contain the 'window' attribute.")
        
        
        # Handle `horizon`
        if "horizon" in batch_data:
            horizon = batch_data["horizon"]

            # Flatten nested lists if necessary
            if isinstance(horizon, list) and isinstance(horizon[0], list):
                horizon = [item for sublist in horizon for item in sublist]

            # Ensure `horizon` contains valid elements with `y`
            if len(horizon) > 0 and all(hasattr(data, "y") for data in horizon):
                y = torch.cat([data.y for data in horizon], dim=0)
                batch_data["y"] = y  # Add processed `y` to batch_data
            else:
                raise ValueError("[filler] Batch does not have a valid 'horizon' with 'y' attribute.")
        else:
            raise ValueError("[filler] Batch does not contain the 'horizon' attribute.")

        return batch_data, batch_preprocessing

    def training_step(self, batch, batch_idx):

        # Move batch to the correct device
        batch = move_data_to_device(batch, self._device)

        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        
        # Extract mask and target
        mask = batch_data['mask'].clone().detach() #detach(): Ensures no gradients are tracked for this operation. This is useful when working with tensors that are part of a computation graph (e.g., in training).
        
        #The mask is first converted to a floating-point tensor with .float() to perform probabilistic operations.  Random Bernoulli Sampling
        batch_data['mask'] = torch.bernoulli(mask.clone().detach().float() * self.keep_prob).byte()

        eval_mask = batch_data.pop('eval_mask', torch.zeros_like(mask))  # Default: no eval_mask
        eval_mask = (mask | eval_mask) & ~batch_data['mask']

        y = batch_data.pop('y')
        
        # Compute predictions and compute loss
        imputation = self.predict_batch(batch, preprocess=False, postprocess=False, batch_size=self.batch_size)

        # Handle tuple output
        if isinstance(imputation, tuple):
            imputation = imputation[0]
    
        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)

        loss = self.loss_fn(imputation, target, mask)

        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        self.train_metrics.update(imputation.detach(), y, eval_mask)  # all unseen data
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):

        # Move batch to the correct device
        batch = move_data_to_device(batch, self._device)

        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')

        # Compute predictions and compute loss
        imputation = self.predict_batch(batch, preprocess=False, postprocess=False, batch_size=self.batch_size)

        if isinstance(imputation, tuple):
            imputation = imputation[0]
    
        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)

        val_loss = self.loss_fn(imputation, target, eval_mask)

        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        self.val_metrics.update(imputation.detach(), y, eval_mask)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_loss', val_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return val_loss

    def on_train_epoch_start(self) -> None:

        optimizers = ensure_list(self.optimizers())
        for i, optimizer in enumerate(optimizers):
            lr = optimizer.optimizer.param_groups[0]['lr']
            self.log(f'lr_{i}', lr, on_step=False, on_epoch=True, logger=True, prog_bar=False)

    def configure_optimizers(self):
        cfg = dict()
        optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        cfg['optimizer'] = optimizer
        if self.scheduler_class is not None:
            metric = self.scheduler_kwargs.pop('monitor', None)
            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
            cfg['lr_scheduler'] = scheduler
            if metric is not None:
                cfg['monitor'] = metric
        return cfg
