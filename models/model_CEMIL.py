"""
CEMIL (Rahman et al., WACV 2025) — PyTorch reproduction

Author: Moriya Teiji
"""

from __future__ import annotations
import math
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def top_k_indices(scores: torch.Tensor, k_frac: float) -> torch.Tensor:
    """Return indices of top-k_frac scores in descending order.
    scores: shape [N]
    k_frac: 0 < k_frac <= 1
    """
    N = scores.shape[0]
    k = max(1, int(math.ceil(k_frac * N)))
    vals, idx = torch.topk(scores, k, largest=True, sorted=True)
    return idx


# ------------------------------------------------------------
# Patch Representation Module (PRM)
# ------------------------------------------------------------

class PatchRepresentationModule(nn.Module):
    """Simple MLP over patch features.
    Input:  [N, d_in]
    Output: [N, d_hid]
    """
    def __init__(self, d_in: int, d_hid: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_hid),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------
# Contextual Attention Module (CAM)
# ------------------------------------------------------------

class ContextualAttentionMIL(nn.Module):
    """Contextual Attention for MIL with optional gating.

    Implements two variants of attention scoring over instance embeddings h_i
    following the paper:
      - Standard contextual attention (Eq.(2) style): tanh(U h') -> softmax.
      - Gated attention (Eq.(4)): (tanh(U h') ⊙ σ(V h')) -> softmax.

    The context injection is multiplicative:
        h'_i = h_i ⊙ (W_c c)
    where c is a learned context vector, W_c is linear projection to match dim.

    Args:
        d_hid: hidden dim of patch embeddings h_i
        gated: if True, use gated attention (Eq.(4)); else standard
    Returns:
        z   : bag representation [d_hid]
        a   : attention weights [N]
        h   : (possibly) transformed patch reps used to compute attention [N, d_hid]
    """
    def __init__(self, d_hid: int, gated: bool = False):
        super().__init__()
        self.d_hid = d_hid
        self.gated = gated

        # Context vector and its projection
        self.context = nn.Parameter(torch.randn(d_hid))  # c
        self.Wc = nn.Linear(d_hid, d_hid, bias=False)    # W_c

        # Attention parameters
        self.U = nn.Linear(d_hid, d_hid, bias=False)
        self.wa = nn.Linear(d_hid, 1, bias=False)

        if gated:
            self.V = nn.Linear(d_hid, d_hid, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute contextual attention over instances.
        h: [N, d_hid]
        Returns (z, a, h), where z: [d_hid], a: [N]
        """
        N = h.shape[0]
        # Compute context-modulated hidden representation: h' = h ⊙ (W_c c)
        context_vec = self.Wc(self.context)              # [d_hid]
        h_prime = h * context_vec.unsqueeze(0)           # broadcast multiply

        # Attention logits
        if self.gated:
            # Eq.(4): w_a^T ( tanh(U h') ⊙ σ(V h') )
            t = torch.tanh(self.U(h_prime))              # [N, d_hid]
            g = torch.sigmoid(self.V(h_prime))           # [N, d_hid]
            logits = self.wa(t * g).squeeze(-1)          # [N]
        else:
            # Standard: w_a^T tanh(U h')
            logits = self.wa(torch.tanh(self.U(h_prime))).squeeze(-1)

        a = torch.softmax(logits, dim=0)                 # [N]
        # Bag representation z = Σ a_i * h_i   (use original h per paper Eq.(3))
        z = torch.sum(a.unsqueeze(-1) * h, dim=0)        # [d_hid]
        return z, a, h


# ------------------------------------------------------------
# Classification Head
# ------------------------------------------------------------

class BagClassifier(nn.Module):
    def __init__(self, d_hid: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_hid, n_classes)
        self.n_classes = n_classes

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.fc(z)
        if self.n_classes == 1:
            probs = torch.sigmoid(logits).squeeze(-1)
        else:
            probs = torch.softmax(logits, dim=-1)
        return logits, probs


# ------------------------------------------------------------
# Instructor & Learner models
# ------------------------------------------------------------

class InstructorMIL(nn.Module):
    def __init__(self, d_in: int, d_hid: int, n_classes: int, gated: bool = False):
        super().__init__()
        self.prm = PatchRepresentationModule(d_in, d_hid)
        self.cam = ContextualAttentionMIL(d_hid, gated=gated)
        self.cls = BagClassifier(d_hid, n_classes)
        self.n_classes = n_classes
        self.name = "instructor"

    @torch.no_grad()
    def compute_attention_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention weights a over all instances (for selection)."""
        h = self.prm(x)
        z, a, _ = self.cam(h)
        return a

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.prm(x)                    # [N, d_hid]
        z, a, _ = self.cam(h)              # z:[d_hid], a:[N]
        results_dict = {"embedding": z, "patch_repr": h, "attn": a}
        logits, probs = self.cls(z)         # [C]

        if self.n_classes == 1:
            return probs, 0.0, results_dict
        
        Y_hat = torch.argmax(logits, dim=1)

        return logits, probs, Y_hat, 0.0, results_dict


class LearnerMIL(nn.Module):
    def __init__(self, d_in: int, d_hid: int, n_classes: int, 
                 gated: bool = False, shared_prm: Optional[PatchRepresentationModule] = None):
        super().__init__()
        # Optionally share PRM parameters with instructor (paper mentions sharing PRM params)
        self.prm = shared_prm if shared_prm is not None else PatchRepresentationModule(d_in, d_hid)
        self.cam = ContextualAttentionMIL(d_hid, gated=gated)
        self.cls = BagClassifier(d_hid, n_classes)
        self.n_classes = n_classes
        self.name = "learner"

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.prm(x)
        z, a, _ = self.cam(h)
        results_dict = {"embedding": z, "patch_repr": h, "attn": a}
        logits, probs = self.cls(z)         # [C]

        if self.n_classes == 1:
            return probs, 0.0, results_dict
        
        Y_hat = torch.argmax(logits, dim=1)

        return logits, probs, Y_hat, 0.0, results_dict


# ------------------------------------------------------------
# Patch Selection Module (PSM)
# ------------------------------------------------------------

class PatchSelectionModule:
    """Select top-k patches per bag using the (frozen) instructor PRM+CAM.

    Given a bag x (N, d_in):
      1) Compute attention scores a_i via instructor.
      2) Select indices of top k_frac fraction.
      3) Return x_selected and idx.
    """
    def __init__(self, instructor: InstructorMIL, k_frac: float):
        assert 0 < k_frac <= 1.0
        self.instructor = instructor
        self.k_frac = k_frac
        # Freeze instructor components for selection
        for p in self.instructor.parameters():
            p.requires_grad_(False)
        self.instructor.eval()

    @torch.no_grad()
    def select(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # attention weights from instructor
        a = self.instructor.compute_attention_scores(x)  # [N]
        idx = top_k_indices(a, self.k_frac)              # [k]
        x_sel = x.index_select(0, idx)
        a_sel = a.index_select(0, idx)
        return x_sel, idx, a_sel


# ------------------------------------------------------------
# Losses
# ------------------------------------------------------------

def ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # target: scalar class index or one-hot; logits: [C]
    if target.ndim == 0:
        target = target.view(1)
    loss = F.cross_entropy(logits.unsqueeze(0), target.long())
    return loss


def pr_loss(h_I: torch.Tensor, h_L: torch.Tensor) -> torch.Tensor:
    """Patch representation MSE between instructor and learner on selected patches.
    h_I, h_L: [k, d_hid]
    """
    return F.mse_loss(h_L, h_I)


def pp_loss(p_I: torch.Tensor, p_L: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Prediction probability KL divergence: D_KL(P_I || P_L).
    Use log for p_I (detach) vs p_L: F.kl_div requires log-probs for input.
    """
    log_pL = (p_L + eps).log()
    return F.kl_div(log_pL, p_I.detach(), reduction='batchmean')