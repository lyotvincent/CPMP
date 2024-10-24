from typing import Any
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F



def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - torch.sigmoid(hazards), dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y)+eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss

# def nll_loss(hazards, Y, c, S=None, alpha=0.4, eps=1e-8):
#   batch_size = len(Y)
#   Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
#   c = c.view(batch_size, 1).float() #censorship status, 0 or 1
#   if S is None:
#       S = 1 - torch.cumsum(hazards, dim=1) # surival is cumulative product of 1 - hazards
#   uncensored_loss = -(1 - c) * (torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
#   censored_loss = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps))
#   loss = censored_loss + uncensored_loss
#   loss = loss.mean()
#   return loss

class CrossEntropySurvLoss(nn.Module):
    def __init__(self, alpha=0.15):
        super().__init__()
        self.alpha = alpha

    def __call__(self, h, y, t, c, alpha=None): 
        if alpha is None:
            return ce_loss(hazards=h, S=None, Y=y, c=c, alpha=self.alpha)
        else:
            return ce_loss(hazards=h, S=None, Y=y, c=c, alpha=alpha)


# https://github.com/jiawei-ren/BalancedMSE
def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss



class WeightedFocalMSELoss(nn.Module):
    """
    beta: scale factor for torch.abs(inputs - targets), the interval of  torch.abs(inputs - targets) is [0, 1], so beta should be set to a value > 1
    gamma: focusing parameter on exp, gamma >= 1
    """
    def __init__(self, activate='tanh', beta=2, gamma=2): #weights=None,
        super().__init__()
        self.activate = activate
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets, weight=None): #added weights as targets[1]
        return weighted_focal_mse_loss(inputs=inputs, targets=targets, weights=weight,
                                       activate=self.activate, beta=self.beta, gamma=self.gamma)


def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_huber_loss(inputs, targets, weights=None, beta=1.):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss