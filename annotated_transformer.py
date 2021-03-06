import argparse
from collections import namedtuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperParams(
    namedtuple(
        "_HP", ["d_key", "d_value", "num_heads", "num_enc_layers", "num_dec_layers"]
    )
):
    def d_model(self):
        return self.d_value * self.num_heads


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_dim):
        self.ff = nn.Sequential(
            nn.Linear(d_model, inner_dim), nn.ReLU(), nn.Linear(inner_dim, d_model)
        )

    def forward(self, x):
        y = x + self.ff(x)
        # TODO: layer norm
        return y


class MultiHeadAttention(nn.Module):
    # d_model = d_value * num_heads = dimension of each embedding
    def __init__(self, d_key, d_value, num_heads=1):
        super(MultiHeadAttention, self).__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.num_heads = num_heads

        d_model = self.d_model()
        self.query = nn.Linear(d_model, d_key * num_heads, bias=False)
        self.key = nn.Linear(d_model, d_key * num_heads, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

        self.projection = nn.Linear(d_model, d_model, bias=False)

    def d_model(self):
        return self.d_value * self.num_heads

    def forward(self, x):
        is_batch = len(x.shape) == 3
        dim_offset = 1 if is_batch else 0

        K = self.d_key
        V = self.d_value
        H = self.num_heads
        N = x.shape[dim_offset]  # num_words
        SCALE = 1.0 / math.sqrt(K)

        xpos_dims = (1, 2) if is_batch else (0, 1)

        q, k, v = self.query(x), self.key(x), self.value(x)
        print(x.shape)
        print(q.shape)
        print(k.shape)
        print(v.shape)

        # Ignoring the batch dimension, q and k are tensors of shape (d_model, d_k * H)
        # we can think of them as concatenated results
        #     q = Q1 || Q2 || Q3 ... where Qi is the ith query head
        #     k = K1 || K2 || K3 ...
        #
        # we need to compute Q1 * K1', Q2 * K2', etc.
        #
        # To implement this operation, we will vertically stack the Qi matrices,
        # and horizontally stack the transposed Ki matrices. When we multiply them,
        # we will get a large matrix whose diagonal blocks contain useful results
        # and the rest need to be ignored during a softmax. We do that by setting
        # them to -inf
        q_vertical = torch.cat(
            [q[..., i : i + K] for i in range(0, K * H, K)], dim=dim_offset
        )

        k_horiz = torch.cat(
            [
                k.transpose(*xpos_dims)[..., i : i + K, :]
                for i in range(0, K * H, K)
            ],
            dim=1 + dim_offset,
        )

        print("q_vertical", q_vertical.shape, q_vertical)
        print("k_horiz", k_horiz.shape, k_horiz)

        dot_big = q_vertical.matmul(k_horiz)

        # mask all the unnecessary entries to -inf so they softmax to zero
        masked_dot = torch.empty_like(dot_big).fill_(-math.inf)
        for i in range(0, N * H, N):
            masked_dot[..., i : i + N, i : i + N] = dot_big[
                ..., i : i + N, i : i + N
            ]

        print(masked_dot)

        softmax = F.softmax(masked_dot * SCALE, dim=1 + dim_offset)
        print("softmax", softmax)

        # Now we do a similar vertical stacking for the value tensors.
        # Multiplying the result by softmax gives us attention but the result
        # is stacked vertically which we need to stack horizontally.
        v_vertical = torch.cat(
            [v[..., range(i, i + V)] for i in range(0, V * H, V)], dim=dim_offset
        )
        attention_vertical = softmax.matmul(v_vertical)

        attention = torch.cat(
            [attention_vertical[..., i : i + N, :] for i in range(0, N * H, N)],
            dim=1 + dim_offset,
        )
        assert attention.shape == x.shape

        # finally, we do a projection as specified in the paper
        projected = self.projection(attention)

        y = x + projected
        # TODO: layer norm

        return y


class Encoder(nn.Module):
    def __init__(self, d_key, d_value, num_layers):
        super(Encoder, self).__init__()

        self.net = nn.Sequential()
        for _ in range(num_layers):
            self.net.add(nn.Sequential(
                MultiHeadAttention(),
                FeedForward(d_model, inner_dim=d_value),  # paper mentions d_ff = 2048
            ))

    def forward(self, x):
        return self.net(x)


def main(args):
    d_value = 5
    atn = MultiHeadAttention(d_key=7, d_value=d_value, num_heads=2)
    nn.init.uniform_(atn.query.weight, 0.5, 0.55)
    nn.init.uniform_(atn.key.weight, 0.1, 0.12)
    nn.init.uniform_(atn.value.weight, 0.1, 1.0)

    w = torch.stack([torch.empty(atn.d_model()).uniform_(0, 1) for _ in range(3)])
    print(atn(w.reshape(1, -1, atn.d_model())))


def test(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Annotated transformer playground")
    main(parser.parse_args())
