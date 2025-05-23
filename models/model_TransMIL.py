"""
Shao, Z. et al. TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification.

codes borrowed from https://github.com/szc19990412/TransMIL/tree/main/models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .transformer_utils import NystromAttention

class TransLayer(nn.Module):
    def __init__(self, head=8, dim=512, dropout=0.1, pinv_iterations=6, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//head,
            heads = head,
            num_landmarks = dim//2,             # number of landmarks
            pinv_iterations = pinv_iterations,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,                    # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=dropout
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, num_heads=8, dropout=0.):
        super(TransMIL, self).__init__()
        
        self._fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.pos_layer = PPEG(dim=hidden_dim)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))        
        
        self.layer1 = TransLayer(head=num_heads, dim=hidden_dim, dropout=dropout)
        self.layer2 = TransLayer(head=num_heads, dim=hidden_dim, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        self.n_classes = n_classes
        self._fc2 = nn.Linear(hidden_dim, self.n_classes)


    def forward(self, x, **kwargs):
        x = x.unsqueeze(0)

        h = x.float() #[B, n, 1024]
        
        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        results_dict = {'embedding': h}

        #---->predict
        logits = self._fc2(h) #[B, n_classes]

        if self.n_classes == 1:
            Y_prob = torch.sigmoid(logits)
            return Y_prob, 0.0, results_dict

        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)

        return logits, Y_prob, Y_hat, 0.0, results_dict


if __name__ == "__main__":
    data = torch.randn((1, 5000, 1024)).cuda()
    model = TransMIL(input_dim=1024, hidden_dim=512, n_classes=2, num_heads=8, dropout=0.1).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)