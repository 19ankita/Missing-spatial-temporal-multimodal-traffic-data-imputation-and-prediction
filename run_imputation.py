import copy
import datetime
import os
from pathlib import Path
from argparse import ArgumentParser
import json

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml 
import scipy.sparse as sps
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from lib import fillers, datasets, config
from lib.data.datamodule import SequentialGraphDataset
from lib.nn import models
from lib.nn.utils.metric_base import MaskedMetric
from lib.nn.utils.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from lib.utils import parser_utils, numpy_metrics, ensure_list, prediction_dataframe
from lib.utils.parser_utils import str_to_bool

from lib.utils.data_utils import custom_collate_fn


def load_base_dir_from_json(json_path):
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config['BASEDIR']

def has_graph_support(model_cls):
    return model_cls == models.GRINet


def get_model_classes(model_str):
    if model_str == 'grin':
        model, filler = models.GRINet, fillers.GraphFiller
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, filler


def get_dataset(dataset_name, base_dir, limit=None):
    root = Path(base_dir)
    if dataset_name in ['traffic_block', 'traffic_point']: 
        city = "london"
        dataset = datasets.T4c22GeometricDataset(root=root, city=city, limit=limit)
        
        # # Apply the limit to the valid time points
        # if limit is not None:
        #     dataset.full_day_t = dataset.full_day_t[:limit]
        
        # Create a dictionary to hold datasets for each split
        datasets_dict = {
            "train": copy.deepcopy(dataset),
            "val": copy.deepcopy(dataset),
            "test": copy.deepcopy(dataset)
        }
        
        # Set the split attribute and day_t list based on the split type
        datasets_dict["train"].day_t = [dataset.full_day_t[i] for i in dataset.train_idxs]
        datasets_dict["val"].day_t = [dataset.full_day_t[i] for i in dataset.val_idxs]
        datasets_dict["test"].day_t = [dataset.full_day_t[i] for i in dataset.test_idxs]
        
        return datasets_dict
    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")


def parse_args(): 
    # Argument parser
    parser = ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=1, help='Stride for the sliding window')
    parser.add_argument('--window', type=int, default=24, help='Size of the sliding window in the past')
    parser.add_argument('--horizon', type=int, default=24, help='Size of the prediction horizon')
    parser.add_argument('--delay', type=int, default=0, help='Offset between end of window and start of horizon')

    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument("--model-name", type=str, default='grin')
    parser.add_argument("--dataset-name", type=str, default='traffic_block')
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument('--nodes-per-sample', type=int, default=500, help='Number of nodes per sample.')


    # Splitting/aggregation params
    parser.add_argument('--in-sample', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)
    parser.add_argument('--aggregate-by', type=str, default='mean')
    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")  
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--scaled-target', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--grad-clip-algorithm', type=str, default='norm')
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--consistency-loss', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--whiten-prob', type=float, default=0.05)
    parser.add_argument('--pred-loss-weight', type=float, default=1.0)
    parser.add_argument('--warm-up', type=int, default=0)
    # graph params
    parser.add_argument("--adj-threshold", type=float, default=0.1)
    # gain hparams
    parser.add_argument('--alpha', type=float, default=10.)
    parser.add_argument('--hint-rate', type=float, default=0.7)
    parser.add_argument('--g-train-freq', type=int, default=1)
    parser.add_argument('--d-train-freq', type=int, default=5)

    known_args, _ = parser.parse_known_args()
    model_cls, _ = get_model_classes(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)


    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    return args

def run_experiment(args):

    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    model_cls, filler_cls = get_model_classes(args.model_name)
    
    print(f"==================================")
    print("[Main script] Loading dataset...")
    print(f"==================================")

    # Dynamically assign limit
    limit = args.nodes_per_sample if hasattr(args, 'nodes_per_sample') else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets_dict = get_dataset(args.dataset_name, base_dir, limit=limit) 
    train_dataset = datasets_dict["train"]
    val_dataset = datasets_dict["val"]
    test_dataset = datasets_dict["test"]
    
    print(f"train_dataset: {len(train_dataset)}, val_dataset: {len(val_dataset)}, test_dataset: {len(test_dataset)}")
    print(f"==================================")
    print("[Main script] Dataset loaded.")
    print(f"==================================")
    
    train_mask = train_dataset.training_mask.to(device) 
    eval_mask = val_dataset.eval_mask.to(device) 
    print(f"train_mask: {train_mask.shape}, eval_mask: {eval_mask.shape}")
    
    #no overlap between the two masks
    overlap = train_mask & eval_mask
    print(f"Overlap between train_mask and eval_mask: {overlap.sum()}")
    
    # Access train, val, and test indices directly
    #train_idxs, val_idxs, test_idxs = datasets_dict.split_indices
    # Access train, val, and test indices directly
    train_idxs = train_dataset.train_idxs
    val_idxs = val_dataset.val_idxs
    test_idxs = test_dataset.test_idxs
    print(f"train_idxs: {len(train_idxs)}, val_idxs: {len(val_idxs)}, test_idxs: {len(test_idxs)}")

        
    ########################################
    # create logdir and save configuration #
    ########################################

    counter = 1
    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(config['logs'], args.dataset_name, args.model_name, exp_name)

    # Add a suffix if the directory already exists
    while os.path.exists(logdir):
        logdir = f"{original_logdir}_{counter}"
        counter += 1

    # save config for logging
    Path(logdir).mkdir(parents=True)

    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4, sort_keys=True)
           
    # Prepare SequentialGraphDataset
    train_data_list = [train_dataset.get_full_data(idx) for idx in train_idxs]
    val_data_list = [val_dataset.get_full_data(idx) for idx in val_idxs]
    test_data_list = [test_dataset.get_full_data(idx) for idx in test_idxs]
    print(f"train_data_list: {len(train_data_list)}, val_data_list: {len(val_data_list)}, test_data_list: {len(test_data_list)}")
    
    # Compute Batched Sparse Adjacency Matrix using the original dataset class
    adj_dense = train_dataset.get_similarity(thr=args.adj_threshold).toarray()
    print(f"adj_dense shape: {adj_dense.shape}")
    print(f"adj_dense sample:\n{adj_dense[:5, :5]}")
    
    adj_sparse = sps.csr_matrix(adj_dense)
    adj_sparse.setdiag(0)  # Remove self-loops
    print(f"adj_sparse shape: {adj_sparse.shape}")
    print(f"adj_sparse sample:\n{adj_sparse[:5, :5].toarray()}")
    
    # Convert sparse matrix to PyTorch COO format
    coo = adj_sparse.tocoo()
    values = torch.tensor(coo.data, dtype=torch.float32)
    indices = torch.tensor([coo.row, coo.col], dtype=torch.int64)
    adj_single = torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape)).to(device)
    print(f"adj_single shape: {adj_single.shape}")
    print(f"adj_single indices sample:\n{adj_single._indices()[:, :5]}")
    print(f"adj_single values sample:\n{adj_single._values()[:5]}")
    
    if not adj_single.is_sparse:
        raise TypeError("The adjacency matrix is not a PyTorch sparse tensor after conversion")

    # Stack adjacency matrices for batching
    batch_size = args.batch_size
    adj_list = [adj_single.clone() for _ in range(batch_size)]  # Clone adj for each sample in batch

    # Create batched sparse adjacency matrix
    batch_indices = []
    batch_values = []

    for batch_idx, adj in enumerate(adj_list):
        adj_indices = adj._indices()
        adj_values = adj._values()

        # Offset node indices for batching
        batch_indices.append(adj_indices + batch_idx * adj.shape[0])
        batch_values.append(adj_values)
        
    # Concatenate adjacency matrices
    batched_adj_indices = torch.cat(batch_indices, dim=1)
    batched_adj_values = torch.cat(batch_values)

    batched_adj = torch.sparse_coo_tensor(
        batched_adj_indices, batched_adj_values, 
        torch.Size((batch_size * adj.shape[0], batch_size * adj.shape[1]))
    ).to(device)
    
    print(f"batched_adj shape: {batched_adj.shape}")
    print(f"batched_adj indices sample:\n{batched_adj._indices()[:, :5]}")
    print(f"batched_adj values sample:\n{batched_adj._values()[:5]}") 


    # Prepare the graph dataset as a sequential dataset
    train_dataset = SequentialGraphDataset(
        data_list=train_data_list,
        window=args.window,
        horizon=args.horizon,
        stride=args.stride,
        device=device,
        limit=limit
    )

    val_dataset = SequentialGraphDataset(
        data_list=val_data_list,
        window=args.window,
        horizon=args.horizon,
        stride=args.stride,
        device=device,
        limit=limit
    )
    
    test_dataset = SequentialGraphDataset(
    data_list=test_data_list,
    window=args.window,
    horizon=args.horizon,
    stride=args.stride,
    device=device,
    limit=limit
    )
    
    print(f"train_dataset: {len(train_dataset)}, val_dataset: {len(val_dataset)}, test_dataset: {len(test_dataset)}")
    
    
    # Use torch_geometric's DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, drop_last=False)
    print(f"train_loader: {len(train_loader)}, val_loader: {len(val_loader)}, test_loader: {len(test_loader)}")
 
    ########################################
    # predictor                            #
    ########################################
    
    # model's inputs
    additional_model_hparams = dict(adj=adj, d_in=train_dataset[0].window[0].x.shape[-1], n_nodes=adj.shape[0], batch_size=args.batch_size, window=args.window, device=device) 
    
    model_kwargs = parser_utils.filter_args(args={**vars(args), **additional_model_hparams}, 
                                            target_cls=model_cls,
                                            return_dict=True)
    
    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(F, args.loss_fn),
                           metric_kwargs={'reduction': 'none'})


    metrics = {'mae': MaskedMAE(),
               'mape': MaskedMAPE(),
               'mse': MaskedMSE(),
               'mre': MaskedMRE()}

    
    # filler's inputs
    scheduler_class = CosineAnnealingLR if args.use_lr_schedule else None
    additional_filler_hparams = dict(model_class=model_cls,
                                     adj=adj,
                                     model_kwargs=model_kwargs,
                                     batch_size=args.batch_size, 
                                     optim_class=torch.optim.Adam,
                                     optim_kwargs={'lr': args.lr,
                                                   'weight_decay': args.l2_reg},
                                     loss_fn=loss_fn,
                                     metrics=metrics,
                                     device=device,
                                     scheduler_class=scheduler_class,
                                     scheduler_kwargs={
                                         'eta_min': 0.0001,
                                         'T_max': args.epochs
                                     },
                                     alpha=args.alpha,
                                     hint_rate=args.hint_rate,
                                     g_train_freq=args.g_train_freq,
                                     d_train_freq=args.d_train_freq)

    filler_kwargs = parser_utils.filter_args(args={**vars(args), **additional_filler_hparams},
                                             target_cls=filler_cls,
                                             return_dict=True)

    filler_kwargs['adj'] =  batched_adj  

    if filler_kwargs.get('adj') is not None:
       print(f"[Main script] Adjacency matrix (before GraphFiller): {filler_kwargs['adj'].shape}")
    else:
       print("[Main script] ERROR: 'adj' is None before passing to GraphFiller!")
 
    # Explicitly add batch_size if missing
    if 'batch_size' not in filler_kwargs:
        filler_kwargs['batch_size'] = args.batch_size
        print("[Main script] Added batch_size explicitly to filler_kwargs")
        
    # Add the masks explicitly
    filler_kwargs['mask'] = {
        'train_mask': train_mask,  
        'eval_mask': eval_mask     
    }
    
    print("[Main script] Mask keys in filler_kwargs:", filler_kwargs.get('mask', {}).keys())

    filler = filler_cls(**filler_kwargs)
    
    ########################################
    # training                             #
    ########################################

    # callbacks
    early_stop_callback = EarlyStopping(monitor='val_mae', patience=args.patience, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1, monitor='val_mae', mode='min')

    logger = TensorBoardLogger(logdir, name="model")

    trainer = pl.Trainer(max_epochs=args.epochs,
                         logger=logger,
                         default_root_dir=logdir,
                         devices=args.gpus if torch.cuda.is_available() else 1,
                         accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         gradient_clip_val=args.grad_clip_val,
                         gradient_clip_algorithm=args.grad_clip_algorithm,
                         callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(filler, train_dataloaders=train_loader, val_dataloaders=val_loader) 

    # Logging best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    print(f"[Training Completed] Best model saved at: {best_model_path}")

    # Extract the best metrics
    best_metrics = trainer.logged_metrics

    print(f"========== Final Training & Validation Metrics ==========")
    print(f"Best Validation MAE: {best_metrics.get('val_mae', 'N/A'):.4f}")
    print(f"Best Validation MSE: {best_metrics.get('val_mse', 'N/A'):.4f}")
    print(f"Best Validation MAPE: {best_metrics.get('val_mape', 'N/A'):.4f}")
    print(f"=============================================")

    # Logging training history to a file
    log_file_path = os.path.join(logdir, "training_log.txt")
    with open(log_file_path, "a") as f:
        f.write(f"\n======== Training Summary ========\n")
        f.write(f"Best Model Path: {best_model_path}\n")
        f.write(f"Epoch {trainer.current_epoch}\n")
        f.write(f"Train Loss: {trainer.callback_metrics.get('train_loss', 'N/A'):.4f}\n")
        f.write(f"Validation MAE: {trainer.callback_metrics.get('val_mae', 'N/A'):.4f}\n")
        f.write(f"Validation MSE: {trainer.callback_metrics.get('val_mse', 'N/A'):.4f}\n")
        f.write(f"Validation MAPE: {trainer.callback_metrics.get('val_mape', 'N/A'):.4f}\n")
        f.write(f"=============================================\n")

    print(f"[Logging] Training results saved to {log_file_path}")

if __name__ == '__main__':
    base_dir = load_base_dir_from_json('t4c22/t4c22_config.json') 
    args = parse_args()
    run_experiment(args)

