import torch
import torch.nn as nn
import torch.nn.functional as F

import math


"""
# borrowed from https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
"""
def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

"""
absolute position embeddings: fixed sin cos embeddings 
"""
def positionalencoding2d_mil(token_dim, coords):
    """
    :param token_dim: dimension of the model
    :param coords: array of 2d coordinates, N * 2 
    :return: token_dim*instance_num position matrix N * dim
    """
    if token_dim % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(token_dim))
    
    inst_num = coords.shape[0]
    device = coords.device

    pe = torch.zeros(token_dim, inst_num, device=device)
    # Each dimension use half of token_dim
    token_dim = int(token_dim / 2)
    div_term = torch.exp(torch.arange(0., token_dim, 2) *
                         -(math.log(1000.0) / token_dim)).to(device) #*0.1*math.pi
    pos_w = coords[:, 0].unsqueeze(1)
    pos_h = coords[:, 1].unsqueeze(1)
    pe[0:token_dim:2, :] = torch.sin(div_term * pos_w).transpose(0, 1)
    pe[1:token_dim:2, :] = torch.cos(div_term * pos_w).transpose(0, 1)
    pe[token_dim::2, :] = torch.sin(div_term * pos_h).transpose(0, 1)
    pe[token_dim + 1::2, :] = torch.cos(div_term * pos_h).transpose(0, 1)

    return pe.transpose(1, 0)


"""
borrowed from https://github.com/microsoft/Cream/blob/main/iRPE/DeiT-with-iRPE/irpe.py#L291
ICCV2021, Rethinking and Improving Relative Position Encoding for Vision Transformer
"""
class RelativePositionEmbedding1(nn.Module):
    """ Relative Position Embedding
        https://arxiv.org/abs/1910.10683
        https://github.com/bojone/bert4keras/blob/db236eac110a67a587df7660f6a1337d5b2ef07e/bert4keras/layers.py#L663
        https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py#L344
    """
    def __init__(self, heads_num, bidirectional = True, num_buckets = 32, max_distance = 128):
        super(RelativePositionEmbedding1, self).__init__()
        self.num_buckets = num_buckets
        self.bidirectional = bidirectional
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(self.num_buckets, heads_num)
 
    def forward(self, encoder_hidden, decoder_hidden):
        """
        Compute binned relative position bias
        Args:
            encoder_hidden: [batch_size x seq_length x emb_size]
            decoder_hidden: [batch_size x seq_length x emb_size]
        Returns:
            position_bias: [1 x heads_num x seq_length x seq_length]
        """
        query_length = encoder_hidden.size()[1]
        key_length = decoder_hidden.size()[1]
 
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self.relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values
 
    def relative_position_bucket(self, relative_position, bidirectional, num_buckets, max_distance):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)
 
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
 
        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )
 
        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets