"""Causal transformer over sequences of frozen template embeddings.

Predicts the NEXT template's embedding at each position. Anomaly = surprise, the
distance between the predicted next embedding and the actual one. Operating on a
shared embedding space (not token ids) is what lets a model trained on some
systems score a system it never saw.
"""
import torch
import torch.nn as nn


class LogTransformer(nn.Module):
    def __init__(
        self,
        vectors,                 # numpy [V, d_in] frozen, L2-normalized template space
        d_model=128,
        nhead=4,
        nlayers=3,
        dim_ff=256,
        max_len=512,
        dropout=0.1,
    ):
        super().__init__()
        V, d_in = vectors.shape
        self.emb = nn.Embedding.from_pretrained(
            torch.as_tensor(vectors, dtype=torch.float32), freeze=True
        )
        self.inp = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, activation="gelu", batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.out = nn.Linear(d_model, d_in)
        self.max_len = max_len

    def forward(self, ids, pad_mask):
        target = self.emb(ids)                      # [B, L, d_in] frozen
        L = ids.size(1)
        h = self.inp(target) + self.pos[:, :L]
        causal = torch.triu(torch.full((L, L), float("-inf"), device=ids.device), 1)
        h = self.enc(h, mask=causal, src_key_padding_mask=pad_mask)
        return self.out(h), target
