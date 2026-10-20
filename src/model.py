import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(Q, K , V, mask=None, dropout=None):
    dk = K.size(-1)
    scores = torch.matmul(Q,K.transpose(-2,-1)) / torch.sqrt(dk)
    if mask:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn = torch.softmax(scores)

    if dropout:
        attn = dropout(attn)

    attn2 = torch.matmul(attn,V)

    return attn2, attn 


class MultiHeadAttention(nn.Module):
    def __init__(self, h, dmodel, dropout_rate=0.1):
        super().__init__()

        self.dmodel = dmodel
        self.dk = dmodel // h
        self.dv = self.dk
        self.h = h

        self.dropout = nn.Dropout(dropout_rate)
        self.linears = [nn.Linear(dmodel,dmodel) for i in range(4)] # Wq, Wk, Wv, W0

        self.attn = None

    def forward(self, Q, K, V, mask=None):

        nbatch = Q.size(0)

        # 1) Apply linear projections W

        # Q, K, V = original dims [batch, ntokens, dmodel]
        # Q, K, V must go to [batch, nheads, ntokens, dk]
        Q = self.linears[0](Q).view(nbatch, -1, self.h, self.dk).transpose(1,2)
        K = self.linears[1](K).view(nbatch, -1, self.h, self.dk).transpose(1,2)
        V = self.linears[2](V).view(nbatch, -1, self.h, self.dk).transpose(1,2)

        if mask:
            mask = mask.unsqueeze(1)

        # 2) apply scaled dot attention
        applied_attn, attn = attention(Q,K,V, mask)

        # 3) "concat" -> reshape tensors so that all head tensors are concetenated together
        applied_attn = applied_attn.transpose(1, 2).contiguous().view(nbatch, -1, self.h * self.dk) 

        return applied_attn, attn


class FeedForwardNetwork(nn.Module):
    def __init__(self, input=512, output=512, hidden=2048):
        super().__init__()

        self.flatten = nn.Flatten()
        self.ffn = nn.Sequential(
            nn.Linear(input,hidden),
            nn.ReLU(),
            nn.Linear(hidden,output)
        )

    def forward(self, x):
        x = self.flatten(x)
        y = x.ffn(x)
        return y


class AddNorm(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_res):
        norm = self.dropout(nn.LayerNorm(x + x_res))
        return norm


class Embedding(nn.Module):
    def __init__(self, vocab_size, dmodel, drop_out=0.1):
        super().__init__()

        self.dmodel = dmodel
        self.embed = nn.Embedding(vocab_size, dmodel)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):

        scores = self.embed(x) * math.sqrt(self.dmodel)
        return self.dropout(scores)


class PositionalEncoder(nn.Module):
    def __init__(self, dmodel, dropout=0.1, max_size=5000):
        self.dmodel = dmodel
        self.dropout = nn.Dropout(dropout)

        self.pe = torch.zeros((max_size, dmodel))
        position = torch.arange(0,max_size).unsqueeze(1)

        divisor = -1/(10000.0 ** (torch.arange(0,dmodel,2)/dmodel))

        self.pe[:,0::2] = torch.sin(position*divisor)
        self.pe[:,1::2] = torch.cos(position*divisor)

        self.pe = self.pe.unsqueeze(0)
        self.register_buffer("pe", self.pe)

    def forward(self, x):
        x = x + self.pe[:, :self.pe.size(1)].requires_grad_(False)
        return self.dropout(x)







class EncoderBlock(nn.Module):
    def __init__(self, dmodel):
        super().__init__()

        self.dmodel = dmodel

        self.mha = MultiHeadAttention(dmodel)
        self.addnorm1 = AddNorm()
        self.ffn = FeedForwardNetwork(dmodel,dmodel)
        self.addnorm2 = AddNorm()

    def forward(self, x):
        pass



    










