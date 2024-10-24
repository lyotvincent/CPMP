import numpy as np
import torch
from utils.utils import *

import os
from wsi_datasets.dataset_generic import split_slideinfo
from models.model_MIL import MIL_fc, MIL_fc_mc
from models.model_CLAM import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc



class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False, loss_or_score='loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        assert loss_or_score in ['loss', 'score'], "only support 'loss' or 'score' for loss_or_score"
        self.loss_or_score = loss_or_score
        if loss_or_score == 'loss':
            self.val_loss_min = np.Inf
        elif loss_or_score == 'score':
            self.val_loss_min = -np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss if self.loss_or_score == 'loss' else val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease or score increase.'''
        if self.verbose:
            if self.loss_or_score == 'loss':
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            elif self.loss_or_score == 'score':
                print(f'Validation score increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train_loop(epoch, model, loader, optimizer, scheduler, n_classes, writer = None, loss_fn = None, reg_fn=None, lambda_reg=0.0, gc=16, **kwargs):   
    device=kwargs['device'] 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=2)
    train_loss = 0.
    train_error = 0.

    print('\n')
    optimizer.zero_grad() # zero grad before training

    for batch_idx, (data, label, coords, slide_id) in enumerate(loader):
        data = data.to(device).type(torch.float32)
        coords = coords.to(device).type(torch.float32)

        reg_label = label[:, 0].to(device)
        category_label = (label[:, 0] > 0.5).to(device)

        if n_classes == 1:
            Y_prob, _, results_dict = model(data, coords=coords, label=category_label.long(), **kwargs)
            logits = Y_prob
            Y_hat = Y_prob > 0.5
    
            if 'aux' in loss_fn and loss_fn['aux'].__class__.__name__ == "MSELoss": 
                loss_aux = loss_fn['aux'](Y_prob, category_label.float().unsqueeze(0)) * 5
                
            elif 'aux' in loss_fn and loss_fn['aux'].__class__.__name__ == "CrossEntropyLoss":
                loss_aux = loss_fn['aux'](torch.cat((1-Y_prob, Y_prob), dim=1), category_label.long())
            
            else:
                loss_aux = 0.0
            results_dict["loss_aux"] = loss_aux

        else:
            logits, Y_prob, Y_hat, _, results_dict = model(data, coords=coords, label=category_label.long(), **kwargs)
            results_dict["loss_aux"] = 0.0
        
        acc_logger.log(Y_hat, category_label)

        if loss_fn['main'].__class__.__name__ == "MSELoss":
            loss = loss_fn['main'](logits, reg_label.unsqueeze(0))*100
        elif loss_fn['main'].__class__.__name__ == "CrossEntropyLoss":
            loss = loss_fn['main'](logits, category_label.long())
        else:
            raise NotImplementedError

        loss = loss + results_dict["loss_aux"]
        loss += results_dict["instance_loss"] if "instance_loss" in results_dict else 0. # only for CLAM model

        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 10 == 0:
            print('batch {}, MSE loss: {:.4f}, reg_label: {}, bag_size: {}'.format(batch_idx, loss_value, reg_label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, category_label)
        train_error += error
        
        # backward pass
        loss = loss / gc + loss_reg 
        loss.backward() # Only the gradient is calculated,  parameter update is performed when the gc condition is met

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    scheduler.step()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_total_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(2):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


@torch.no_grad()
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None, **kwargs):
    device=kwargs['device'] 

    model.eval()
    acc_logger = Accuracy_Logger(n_classes=2)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    for batch_idx, (data, label, coords, _) in enumerate(loader):
        data = data.to(device, non_blocking=True).type(torch.float32)
        coords = coords.to(device, non_blocking=True).type(torch.float32)


        reg_label = label[:, 0].to(device)
        category_label = (label[:, 0] > 0.5).to(device)
        
        if n_classes == 1:
            Y_prob, _, results_dict = model(data, coords=coords, label=category_label.long(), **kwargs)
            logits = Y_prob
            Y_hat = Y_prob > 0.5

            if 'aux' in loss_fn and loss_fn['aux'].__class__.__name__ == "MSELoss":
                loss_aux = loss_fn['aux'](Y_prob, category_label.float().unsqueeze(0)) * 5
            elif 'aux' in loss_fn and loss_fn['aux'].__class__.__name__ == "CrossEntropyLoss":
                loss_aux = loss_fn['aux'](torch.cat((1-Y_prob, Y_prob), dim=1), category_label.long())
            else:
                loss_aux = 0.0
            results_dict["loss_aux"] = loss_aux

        else:
            logits, Y_prob, Y_hat, _, results_dict = model(data, coords=coords, label=category_label.long(), **kwargs)
            results_dict["loss_aux"] = 0.0
        
        acc_logger.log(Y_hat, category_label)

        if loss_fn['main'].__class__.__name__ == "MSELoss":
            loss = loss_fn['main'](logits, reg_label.unsqueeze(0))*100
        elif loss_fn['main'].__class__.__name__ == "CrossEntropyLoss":
            loss = loss_fn['main'](logits, category_label.long())
        else:
            raise NotImplementedError
        
        loss = loss + results_dict["loss_aux"]
        loss += results_dict["instance_loss"] if "instance_loss" in results_dict else 0. # only for CLAM model

        prob[batch_idx] = Y_prob.cpu().numpy()
        labels[batch_idx] = category_label.item()
        
        val_loss += loss.item()
        error = calculate_error(Y_hat, category_label)
        val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 1:
        auc = roc_auc_score(labels, prob)

    elif n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(2):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        val_metric = val_loss if early_stopping.loss_or_score == "loss" else auc
        early_stopping(epoch, val_metric, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


@torch.no_grad()
def summary(model, loader, n_classes, **kwargs):
    device=kwargs['device'] 
    acc_logger = Accuracy_Logger(n_classes=2)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = []

    for batch_idx, (data, label, coords, _) in enumerate(loader):
        data = data.to(device).type(torch.float32)
        coords = coords.to(device).type(torch.float32)

        reg_label = label[:, 0].to(device)
        category_label = (label[:, 0] > 0.5).to(device)

        slide_id = slide_ids.iloc[batch_idx]

        if n_classes == 1:
            Y_prob, _, results_dict = model(data, coords=coords, **kwargs)
            logits = Y_prob
            Y_hat = Y_prob > 0.5

        else:
            logits, Y_prob, Y_hat, _, _ = model(data, coords=coords, **kwargs)
        
        acc_logger.log(Y_hat, category_label)
        
        probs = Y_prob.cpu().numpy()
        probs = probs.squeeze(0)
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = category_label.item()
        
        # patient_results.append({'slide_id': np.array(slide_id), 'prob_neg': probs[0], 'prob_pos': probs[1], 'label': label.item()})
        if n_classes == 1:
            patient_results.append({'slide_id': np.array(slide_id), 'prob': probs[0], 'reg_label': reg_label.item()})
        elif n_classes == 2:
            patient_results.append({'slide_id': np.array(slide_id), 'prob': probs[1], 'reg_label': reg_label.item()}) # only for binary classification
        
        error = calculate_error(Y_hat, category_label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 1:
        auc = roc_auc_score(all_labels, all_probs)
        aucs = []
    elif n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger