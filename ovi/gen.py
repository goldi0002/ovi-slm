"""Ovi's small generator: a ~6M-parameter causal transformer (CPU, 4 threads) that writes the reply
from the last few turns plus (optionally) facts Ovi remembers about the user.

Prompt:   [memory lines]  <|user|> u1 <|ovi|> o1 <|user|> u2 <|ovi|>   ->   reply <|end|>
The memory lines come from Ovi's own stores (slots / conversation memory), so the model never has
to *know* anything: it only has to phrase what it is handed."""
import re, math, torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

TOK_PATH = "data/tokenizer.json"
USER, OVI, END = 0, 1, 2
CTX = 192
_tok = None


def tok():
    global _tok
    if _tok is None:
        _tok = Tokenizer.from_file(TOK_PATH)
    return _tok


def enc(s):
    return tok().encode(s).ids


def dec(ids):
    return tok().decode(list(ids))


def build_prompt(history, memory=()):
    """history: [(speaker, text)], speaker in 'u' / 'o', last one is the user's message. memory: list of strings."""
    ids = []
    if memory:
        ids += enc("Memory: " + " | ".join(memory)) + [END]
    for who, text in history[-6:]:
        ids += [USER if who == "u" else OVI] + enc(text)
    return ids[-(CTX - 40):] + [OVI]


class Block(nn.Module):
    def __init__(s, d, h, ff, p):
        super().__init__()
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        s.h, s.p = h, p

    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=s.p if s.training else 0.0)
        x = x + F.dropout(s.o(a.transpose(1, 2).reshape(B, T, D)), s.p, s.training)
        return x + F.dropout(s.ff(s.n2(x)), s.p, s.training)


class Gen(nn.Module):
    def __init__(s, V=4096, d=256, L=6, h=4, ff=1024, p=0.1):
        super().__init__()
        s.emb, s.pos = nn.Embedding(V, d), nn.Embedding(CTX + 64, d)
        s.blocks = nn.ModuleList(Block(d, h, ff, p) for _ in range(L))
        s.nf, s.p = nn.LayerNorm(d), p
        nn.init.normal_(s.emb.weight, std=0.02); nn.init.normal_(s.pos.weight, std=0.02)

    def forward(s, x):
        h = F.dropout(s.emb(x) + s.pos(torch.arange(x.size(1))[None]), s.p, s.training)
        for b in s.blocks:
            h = b(h)
        return s.nf(h) @ s.emb.weight.T

    @torch.no_grad()
    def reply(s, history, memory=(), max_new=40, temp=0.7, top_p=0.9, rep=1.15, seed=None):
        s.eval()
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        ids = build_prompt(history, memory); n0 = len(ids)
        for _ in range(max_new):
            lg = s(torch.tensor([ids]))[0, -1].clone()
            for t in set(ids[n0:]):
                lg[t] = lg[t] / rep if lg[t] > 0 else lg[t] * rep
            lg[USER] = lg[OVI] = -1e9
            if temp <= 0:
                nxt = int(lg.argmax())
            else:
                pr = F.softmax(lg / temp, -1); sp, si = pr.sort(descending=True)
                keep = sp.cumsum(0) - sp < top_p; sp = sp * keep; sp = sp / sp.sum()
                nxt = int(si[torch.multinomial(sp, 1, generator=g)])
            if nxt == END:
                break
            ids.append(nxt)
        return dec(ids[n0:]).strip()

    def nparams(s):
        return sum(p.numel() for p in s.parameters())
