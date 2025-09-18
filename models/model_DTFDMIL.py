import os
import numpy as np
from random import random
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x



class Attention2(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention2, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x, isNorm=True):
        ## x: N x L
        A = self.attention(x)  ## N x K
        A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
        return A  ### K x N


class Attention(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)
    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        pred = self.classifier(afeat) ## K x num_cls
        return pred




class DTFDMIL(nn.Module):
    def __init__(self, in_chn = 1024, **params):
        super(DTFDMIL, self).__init__()
        params = SimpleNamespace(**params)

        self.classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
        self.attention = Attention(params.mDim).to(params.device)
        self.dimReduction = DimReduction(in_chn, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)
        self.attCls = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2).to(params.device)

        # initialize_weights(self)

    def forward(self, x, numGroup=4, train=False, **kwargs):
        if train:
            return self.train_forward(x, numGroup=numGroup, **kwargs)
        else:
            return self.eval_forward(x, numGroup=numGroup, **kwargs)

    def eval_forward(self, tfeat_tensor, numGroup=4, **kwargs):
        device = tfeat_tensor.device

        slide_sub_preds = []
        slide_d_feat = []

        midFeat = self.dimReduction(tfeat_tensor)
        AA = self.attention(midFeat, isNorm=False).squeeze(0)  ## N

        num_instances = tfeat_tensor.size(0)
        indices = torch.randperm(num_instances)
        index_chunk_list = torch.chunk(indices, numGroup)
        
        ## for each group
        for tindex in index_chunk_list:
            tmidFeat = torch.index_select(midFeat, dim=0, index=tindex.to(device))
            tAA = torch.index_select(AA, dim=0, index=tindex.to(device))
            tAA = torch.softmax(tAA, dim=0)
            
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

            tPredict = self.classifier(tattFeat_tensor)  ### 1 x 2
            
            slide_sub_preds.append(tPredict)
            slide_d_feat.append(tattFeat_tensor)

        ## for the second tier
        slide_d_feat = torch.cat(slide_d_feat, dim=0)
        gSlidePred = self.attCls(slide_d_feat)
        Y_prob = torch.softmax(gSlidePred, dim=1)
        Y_hat = torch.argmax(Y_prob, dim=1).unsqueeze(1)  # 1 x num_cls

        results_dict = {"embedding": slide_d_feat.flatten()}
        return gSlidePred, Y_prob, Y_hat, slide_d_feat, results_dict

    def train_forward(self, tfeat_tensor, numGroup=4, **kwargs):
        device = tfeat_tensor.device

        slide_sub_preds = [] # subbag pred
        slide_pseudo_feat = [] # subbag agg. feats

        num_instances = tfeat_tensor.size(0)
        indices = torch.randperm(num_instances)
        index_chunk_list = torch.chunk(indices, numGroup)
        
        ## for each group
        for tindex in index_chunk_list:
            subFeat_tensor = torch.index_select(tfeat_tensor, dim=0, index=tindex.to(device))
            tmidFeat = self.dimReduction(subFeat_tensor)
            tAA = self.attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            tPredict = self.classifier(tattFeat_tensor)  ### 1 x 2
            
            slide_sub_preds.append(tPredict)
            slide_pseudo_feat.append(tattFeat_tensor)

        ## for the first tier
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x fs

        ## for the second tier
        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs
        global_SlidePred = self.attCls(slide_pseudo_feat)

        Y_prob = F.softmax(global_SlidePred, dim = 1)
        Y_hat = torch.argmax(Y_prob, dim=1).unsqueeze(1)  # 1 x num_cls

        return global_SlidePred, Y_prob, Y_hat, slide_sub_preds, {}


if __name__ == "__main__":
    model = DTFDMIL(in_chn=1024, mDim=512, num_cls=2, 
                    droprate=0., droprate_2=0., numLayer_Res=0,
                    device='cuda')