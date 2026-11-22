"""Does using the previous turns help pick the right reply?  Clean protocol:
 CrossPMI + index fit on TRAIN conversations; ranker trained on VALIDATION conversations; scored on TEST conversations.
 Run: PYTHONPATH=. python experiments/context_eval.py"""
import random, math, numpy as np
from ovi import dialogue as D
from ovi.memory import STOP as ST, toks
from ovi.retrieval import Index
from ovi.replyrank import CrossPMI, Ranker, words
random.seed(0); np.random.seed(0)
train = D.dedupe(D.dailydialog("train") + D.empathetic("train"))
cross = CrossPMI(dim=128).fit(train)
replies = [a for _, a, _ in train]; RM = cross.reply_matrix(replies)
rk = Ranker(cross); info = rk.prep_pool(replies)
ix = Index()
for q, a, _ in train: ix.add(q, a)
cw = lambda s: {w for w in toks(s) if w not in ST and len(w) > 1}
NC = 6   # extra context features
def ctx_feats(hist, rvecs, rinfo):
    prev = hist[:-1]; cv = cross._embed(" ".join(hist[-3:]), cross.Wm)
    cc = rvecs @ cv
    p1 = cw(prev[-1]) if len(prev) >= 1 else set(); p2 = cw(prev[-2]) if len(prev) >= 2 else set()
    out = np.zeros((len(rinfo), NC), np.float32)
    for k, (_, _, _, _, _, rs) in enumerate(rinfo):
        out[k] = [cc[k], len(p1 & rs), len(p2 & rs), len(p1 & rs) / (1 + len(rs)), (len(p1 & rs) > 0), 0]
    return out
def groups(trips, n, with_ctx):
    Xs = []
    for hist, gold in random.sample(trips, min(n, len(trips))):
        msg = hist[-1]
        top = [i for i, _ in ix.topk(msg, 32) if replies[i] != gold][:29]
        if len(top) < 20: continue
        rv = np.vstack([cross._embed(gold, cross.Wr)[None], RM[top]]); ri = rk.prep_pool([gold]) + [info[i] for i in top]
        F = rk.feats(msg, rv, ri)
        if with_ctx: F = np.hstack([F[:, :-1], ctx_feats(hist, rv, ri), F[:, -1:]])
        Xs.append(F)
    return Xs
trips = lambda d, e: D.triples(d, "daily") + D.triples(e, "emp")
va, te = trips("validation", "valid"), trips("test", "test")
# the pool was filtered to generic replies (all words seen >=5 times); the REAL replies must pass the same filter,
# otherwise "has a rare word" would give the answer away (that leak gave a fake 98-100% in an earlier run)
from ovi.dialogue import _COMMON, _WORD
ok = lambda r: all(rk.rdf.get(w, 0) >= 5 for w in _WORD.findall(r.lower()) if w not in _COMMON)
va, te = [t for t in va if ok(t[1])], [t for t in te if ok(t[1])]
print(f"validation triples {len(va)}, test triples {len(te)}")
res = {}
for name, wc in (("message only", False), ("message + history", True)):
    rk2 = Ranker(cross); rk2.fit_listwise(groups(va, 5000, wc)); rk = rk2
    Xt = groups(te, 2500, wc)
    assert np.isfinite(rk.w).all(), f"non-finite weights: {rk.w}"
    def win(X):
        sc = rk.score(X)
        return bool(np.isfinite(sc).all() and (sc[1:] >= sc[0]).sum() == 0)    # ties and NaNs count as LOSSES
    res[name] = np.mean([win(X) for X in Xt]); print("   weights:", np.round(rk.w, 2))
    print(f"learned ranker, {name:18}: R@1 {res[name]:.1%} over {len(Xt)} test turns (1 real + ~29 plausible; chance 3.3%)")
