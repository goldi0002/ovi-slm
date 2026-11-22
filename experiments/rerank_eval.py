"""Retrieve 30 candidates by question match, re-rank with CrossPMI. Tune weight on half the test turns, report on the other half.
Run: PYTHONPATH=. python experiments/rerank_eval.py"""
import random, numpy as np
from ovi import dialogue as D
from ovi.memory import STOP as ST, toks
from ovi.retrieval import Index
from ovi.replyrank import CrossPMI
random.seed(0)
train, test = D.splits(); train = D.dedupe(train); test = random.sample(test, 3000)
m = CrossPMI(dim=128).fit(train)
replies = [a for _, a, _ in train]; RM = m.reply_matrix(replies)
ix = Index()
for q, a, _ in train: ix.add(q, a)
cw = lambda s: {w for w in toks(s) if w not in ST and len(w) > 1}
def f1(a, b):
    a, b = cw(a), cw(b); i = len(a & b)
    return 0.0 if not a or not b or i == 0 else 2 * i / (len(a) + len(b))
cache = []
for q, g, _ in test:
    top = ix.topk(q, 30)
    if not top: cache.append(None); continue
    idxs = [i for i, _ in top]; iv = np.array([v for _, v in top]); cv = RM[idxs] @ m.msg_vec(q)
    cache.append((idxs, iv, cv, g))
def score(sub, w):
    tot = 0.0
    for c in sub:
        if c is None: continue
        idxs, iv, cv, g = c
        s = (1 - w) * (iv / (iv.max() + 1e-9)) + w * cv
        tot += f1(replies[idxs[int(s.argmax())]], g)
    return tot / len(sub)
A, B = cache[:1500], cache[1500:]
print("weight on CrossPMI | F1 tune-half | F1 held-out half")
best = max((score(A, w), w) for w in (0, .2, .4, .5, .6, .8, 1.0))
for w in (0, .2, .4, .5, .6, .8, 1.0): print(f"   {w:4}            | {score(A,w):.3f}      | {score(B,w):.3f}" + ("   <- chosen on tune half" if w == best[1] else ""))
