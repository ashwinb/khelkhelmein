import argparse
import copy
import logging
import math
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    # dim_value = dims.x (word embedding dimensionality)
    def __init__(self, dim_key, dim_value, num_layers, num_heads=1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_key = dim_key

        self.query = nn.Linear(dim_value, dim_key, bias=False)
        self.key = nn.Linear(dim_value, dim_key, bias=False)
        self.value = nn.Linear(dim_value, dim_value, bias=False)

        self.ff = nn.Sequential(
            nn.Linear(dim_value, dim_value), nn.ReLU(), nn.Linear(dim_value, dim_value)
        )

    def forward(self, x):
        e_in = x
        for _i in range(self.num_layers):
            q, k, v = self.query(e_in), self.key(e_in), self.value(e_in)

            dot = q.dot(k)
            softmax = nn.Softmax(dot * 1.0 / math.sqrt(self.dim_key))
            attention = softmax.matmul(v)

            # TODO: concat attentions for multi-head
            residual = e_in + attention
            # TODO: layer norm
            e_out = residual + self.ff(residual)
            # TODO: layer norm

            e_in = e_out
        return e_out


def main(args):
    enc =
    pass


def test(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Annotated transformer playground")
    main(parser.parse_args())
