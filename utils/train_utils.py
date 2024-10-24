from ast import Lambda
import numpy as np
import os

#----> pytorch imports
import torch
import torch.optim as optim


from .optim_utils import Lamb
from models.model_ABMIL import ABMIL
from models.model_CLAM import CLAM_SB
from models.transformer import Transformer
from models.model_TransMIL import TransMIL


from utils.general_utils import _get_split_loader, _print_network
from utils.core_utils import EarlyStopping, train_loop, validate, summary
from utils.loss_func import WeightedFocalMSELoss
from utils.utils import l1_reg_all


def _get_splits(datasets, cur):
    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val splits...', end=' ')
    train_split, val_split, test_split = datasets
    # _save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    return train_split, val_split, test_split


def _init_loss_function(loss_func=None, device="cpu"):
    r"""
    Init the loss function
    Returns:
        - loss_fn : NLLSurvLoss or NLLRankSurvLoss
    """
    print('\nInit loss function...', end=' ')
    if loss_func == 'CE':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif loss_func == 'MSE':
        loss_fn = torch.nn.MSELoss()
    elif loss_func == 'L1':
        loss_fn = torch.nn.L1Loss()    
    elif loss_func == 'wFocalMSE':
        loss_fn = WeightedFocalMSELoss()
    else:
        raise NotImplementedError  
    loss_fn = loss_fn.to(device)  
    return loss_fn


def _init_optim(model, optim_func=None, lr=1e-4, reg=1e-5, scheduler_func=None, lr_adj_iteration=100):
    print('\nInit optimizer ...', end=' ')

    if optim_func == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    elif optim_func == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=reg)
    elif optim_func == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=reg)
    elif optim_func == "radam":
        optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=reg) 
    elif optim_func == "lamb":
        optimizer = Lamb(model.parameters(), lr=lr, weight_decay=reg)
    else:
        raise NotImplementedError

    if scheduler_func == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         T_max=lr_adj_iteration, eta_min=lr*0.1, verbose=True) #设置余弦退火算法调整学习率，每个epoch调整
    elif scheduler_func == "CyclicLR":
        scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer,
                                                base_lr=lr*0.25, max_lr=lr, step_size_up=lr_adj_iteration//6, 
                                                cycle_momentum=False, verbose=True) #
    elif scheduler_func == "LinearLR":
        scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, 
                                                start_factor=1, end_factor=0.1, total_iters=lr_adj_iteration//2, verbose=True)
    elif scheduler_func == "OneCycleLR":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, 
                                                  max_lr=lr, total_steps=lr_adj_iteration, pct_start=0.2, div_factor=10, final_div_factor=10, verbose=True)
    elif scheduler_func == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                              step_size=int(lr_adj_iteration*0.75), gamma=0.3, verbose=True)
    elif scheduler_func is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                              step_size=lr_adj_iteration, gamma=0.3, verbose=True) # no change for lr
    # 不同的scheduler策略可参考 https://zhuanlan.zhihu.com/p/538447997

    return optimizer, scheduler


def _init_model(model_type=None, model_size="ccl2048", input_size=2048, drop_out=0., n_classes=2, 
                device="cpu", **kwargs):
    print('\nInit Model...', end=' ')
    if model_type == "ABMIL":
        model_dict = {"size_arg":model_size, "dropout" : drop_out, "n_classes" : n_classes, "device" : device}
        model = ABMIL(**model_dict)
    
    elif model_type == "CLAM":
        model = CLAM_SB(gate=True, size_arg=model_size, dropout=False, k_sample=8, instance_eval=True, n_classes=n_classes)
                
    elif model_type == "TransMIL":
        model = TransMIL(input_dim=input_size, hidden_dim=512, n_classes=n_classes, num_heads=8, dropout=drop_out)
         
    elif model_type == "Transformer":
        model = Transformer(num_classes=n_classes, input_dim=input_size, depth=1, 
                            heads=4, dim_head=64, hidden_dim=512, 
                            pool='cls', dropout=drop_out, emb_dropout=0., 
                            pos_enc=None, 
                            )

    elif model_type == "CPMP":
        model = Transformer(num_classes=n_classes, input_dim=input_size, depth=1, 
                            heads=4, dim_head=64, hidden_dim=512, 
                            pool='cls', dropout=drop_out, emb_dropout=0., 
                            pos_enc=None, 
                            agent_n=kwargs['agent_num'], 
                            pos_ppeg_flag=False, # ppeg spatial positional encoding or not
                            )        
    
    else:
        raise ValueError('Unsupported model_type:', model_type)
    model = model.to(device)
    
    return model


def _init_loaders(args, train_split, val_split, test_split):

    print('\nInit Loaders...', end='\n')
    if train_split is not None:
        train_loader = _get_split_loader(args, train_split, training=True, testing=False, weighted=args.weighted_sample, batch_size=args.batch_size)
    else:
        train_loader = None

    if val_split is not None:
        val_loader = _get_split_loader(args, val_split,  testing=False, batch_size=1)
    else:
        val_loader = None

    if test_split is not None:
        test_loader = _get_split_loader(args, test_split,  testing=False, batch_size=1)
    else:
        test_loader = None        

    return train_loader, val_loader, test_loader


def _init_writer(save_dir, cur, log_data=True):
    writer_dir = os.path.join(save_dir, str(cur))
    
    os.makedirs(writer_dir, exist_ok=True)

    if log_data:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None
    return writer


def train_val(timeidx, cur, args, **kwargs):
    dataset_factory = kwargs["dataset_factory"]
    if "dataset_independent" in kwargs.keys():
        dataset_independent = kwargs["dataset_independent"]
        indep_loader = _get_split_loader(args, dataset_independent,  testing=False, batch_size=1)
    else:
        indep_loader = None

    print(f"Created train and val datasets for time {timeidx} fold {cur}")
    if args.split_dir.split("/")[-1][-12:] == "TrainValTest":
        csv_path = f'{args.split_dir}/splits_time{timeidx}.csv'
    else:
        csv_path = f'{args.split_dir}/splits_time{timeidx}_fold{cur}.csv'
        
    datasets = dataset_factory.return_splits(from_id=False, csv_path=csv_path)
    datasets[0].set_split_id(split_id=cur)
    datasets[1].set_split_id(split_id=cur)

    train_split, val_split, test_split = _get_splits(datasets, cur)
    train_loader, val_loader, test_loader = _init_loaders(args, train_split, val_split, test_split)

    
    loss_fn_main = _init_loss_function(args.loss_func, args.device)
    loss_fn = {'main': loss_fn_main}

    if args.loss_func_aux is not None:
        loss_fn['aux'] = _init_loss_function(args.loss_func_aux, args.device)


    model = _init_model(args.model_type, args.model_size, args.encoding_dim, 
                        args.drop_out, args.n_classes,
                        device=args.device,
                        agent_num=args.agent_token_num)
    
    optimizer, scheduler = _init_optim(model, args.optim, args.lr, args.reg, args.scheduler, args.max_epochs)
    _print_network(args.results_dir, model)

    writer = _init_writer(args.results_dir, cur, args.log_data)
    
    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 50, stop_epoch=300, verbose = True, loss_or_score=args.early_stopping_metric)
    else:
        early_stopping = None
    
    print('Done!')
    
    for epoch in range(args.max_epochs):
        train_loop(epoch, model, train_loader, optimizer, scheduler, args.n_classes, writer, loss_fn, l1_reg_all, args.lambda_reg, args.gc,
                   device=args.device)
        stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir, device=args.device)
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)), map_location=args.device))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    val_results_dict, val_error, val_auc, acc_logger = summary(model, val_loader, args.n_classes, device=args.device)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    if test_loader is not None: # test_loader or dataset_independent
        test_results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes, device=args.device)
    elif indep_loader is not None:
        test_results_dict, test_error, test_auc, acc_logger = summary(model, indep_loader, args.n_classes, device=args.device)
    
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(2):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
    writer.close()
    return val_results_dict, test_results_dict, val_auc, test_auc, 1-val_error, 1-test_error