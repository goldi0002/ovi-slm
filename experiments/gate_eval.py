"""Step 2 measurement: when is the ranker confident enough to answer a statement instead of a template?
Clean protocol (CrossPMI+index on TRAIN, ranker on VALIDATION, scored on TEST statements).
For each test turn: pool A = gold + 29 mixed distractors (gold present), pool B = 30 distractors only (gold absent).
Gate = ranker softmax probability of its top choice >= t.
 precision = P(top == gold | gate passes, gold present)      coverage = share of turns passing the gate
 false-confidence = share of gold-ABSENT turns that still pass the gate (it answers although nothing is right)
Run: PYTHONPATH=. python -u experiments/gate_eval.py"""
import random, time, numpy as np
from ovi import dialogue as D
from ovi.retrieval import Index
from ovi.replyrank import CrossPMI, Ranker, mixed_pool
from ovi.dialogue import _COMMON, _WORD
random.seed(0); np.random.seed(0); t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:4.0f}s]", *a, flush=True)
train = D.dedupe(D.dailydialog("train") + D.empathetic("train"))
cross = CrossPMI().fit(train); log("crosspmi fit")
reps = [a for _, a, _ in train]; RM = cross.reply_matrix(reps)
rk = Ranker(cross); info = rk.prep_pool(reps); ix = Index()
for q, a, _ in train: ix.add(q, a)
log("index built")
def grp(hist, gold, with_gold):
    top = mixed_pool(ix, cross, RM, reps, hist[-1], gold, n=29 if with_gold else 30)
    if len(top) < 20: return None
    if with_gold:
        rv = np.vstack([cross._embed(gold, cross.Wr)[None], RM[top]]); ri = rk.prep_pool([gold]) + [info[i] for i in top]
    else:
        rv = RM[top]; ri = [info[i] for i in top]
    return rk.feats(hist[-1], rv, ri, hist)
trips = lambda d, e: D.triples(d, "daily") + D.triples(e, "emp")
va, te = trips("validation", "valid"), trips("test", "test")
ok = lambda r: all(rk.rdf.get(w, 0) >= 5 for w in _WORD.findall(r.lower()) if w not in _COMMON)
va, te = [t for t in va if ok(t[1])], [t for t in te if ok(t[1])]
G = [g for g in (grp(h, y, True) for h, y in random.sample(va, min(3000, len(va)))) if g is not None]
r = Ranker(cross); r.rdf = rk.rdf; r.fit_listwise(G, iters=250); log(f"ranker trained on {len(G)} groups")
stm = [t for t in te if not t[0][-1].strip().endswith("?")]; random.shuffle(stm); stm = stm[:1500]
def probs(X): s = r.score(X); s = s - s.max(); p = np.exp(s); return p / p.sum()
A = []; B = []
for h, y in stm:
    xa, xb = grp(h, y, True), grp(h, y, False)
    if xa is None or xb is None: continue
    pa, pb = probs(xa), probs(xb); A.append((pa.max(), int(pa.argmax()) == 0)); B.append(pb.max())
A = np.array(A, float); B = np.array(B); log(f"{len(A)} statement turns; overall R@1 {A[:,1].mean():.1%} (chance 3.3%)")
print(f"{'gate p>=':>9} {'coverage':>9} {'precision':>10} {'false-conf (gold absent)':>26}")
for t in (0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
    m = A[:, 0] >= t
    print(f"{t:9.2f} {m.mean():9.1%} {(A[m,1].mean() if m.any() else float('nan')):10.1%} {(B>=t).mean():26.1%}")
