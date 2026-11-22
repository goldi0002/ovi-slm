"""Tiny GPT (~6M params). Stored as int8 -> well under 50 MB."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass, asdict


@dataclass
class Config:
    vocab: int = 4096
    ctx: int = 256
    d: int = 256
    layers: int = 6
    heads: int = 4


class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.n1, s.n2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        s.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        s.proj = nn.Linear(c.d, c.d, bias=False)
        s.ff = nn.Sequential(nn.Linear(c.d, 4 * c.d, bias=False), nn.GELU(),
                             nn.Linear(4 * c.d, c.d, bias=False))
        s.h = c.heads

    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, T, 3, s.h, C // s.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + s.ff(s.n2(x))


class GPT(nn.Module):
    def __init__(s, c: Config):
        super().__init__()
        s.c = c
        s.tok = nn.Embedding(c.vocab, c.d)
        s.pos = nn.Embedding(c.ctx, c.d)
        s.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        s.nf = nn.LayerNorm(c.d)
        s.apply(s._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(s, idx, targets=None):
        T = idx.shape[1]
        x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device))
        for b in s.blocks:
            x = b(x)
        logits = s.nf(x) @ s.tok.weight.T  # tied output head
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       targets.view(-1), ignore_index=-100)

    @torch.no_grad()
    def generate(s, idx, n, temp=0.8, top_k=40, stop=None, rep_pen=1.15):
        for _ in range(n):
            lg = s(idx[:, -s.c.ctx:])[:, -1] / temp
            for t in set(idx[0, -64:].tolist()):  # mild repetition penalty
                lg[0, t] = lg[0, t] / rep_pen if lg[0, t] > 0 else lg[0, t] * rep_pen
            v, _ = lg.topk(top_k)
            lg[lg < v[:, -1:]] = -float("inf")
            nxt = torch.multinomial(F.softmax(lg, -1), 1)
            idx = torch.cat([idx, nxt], 1)
            if stop is not None and nxt.item() == stop:
                break
        return idx


# ---- compact storage: int8 per-row quantisation of big matrices ----
def save_compact(model, path, extra=None):
    out = {"cfg": asdict(model.c), "extra": extra or {}, "w": {}}
    for k, p in model.state_dict().items():
        if p.ndim == 2:
            sc = p.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
            out["w"][k] = ((p / sc).round().to(torch.int8), sc.half())
        else:
            out["w"][k] = p.half()
    torch.save(out, path)


def load_compact(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    m = GPT(Config(**d["cfg"]))
    sd = {k: (v[0].float() * v[1].float() if isinstance(v, tuple) else v.float())
          for k, v in d["w"].items()}
    m.load_state_dict(sd)
    return m, d["extra"]
