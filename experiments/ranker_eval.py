"""Hard-negative test: can the ranker pick the REAL reply out of 29 plausible replies (retrieved for the same message)?
chance = 3.3%.   Run: PYTHONPATH=. python experiments/ranker_eval.py"""
import random, numpy as np
from ovi import dialogue as D
from ovi.memory import STOP as ST, toks
from ovi.retrieval import Index
from ovi.replyrank import CrossPMI, Ranker, NFEAT
random.seed(0); np.random.seed(0)
train, test = D.splits(); train = D.dedupe(train)
random.shuffle(train); held = train[:6000]; train = train[6000:]     # ranker-training turns are NEVER seen by CrossPMI or the index
cross = CrossPMI(dim=128).fit(train)
replies = [a for _, a, _ in train]; RM = cross.reply_matrix(replies)
rk = Ranker(cross); info = rk.prep_pool(replies)
ix = Index()
for q, a, _ in train: ix.add(q, a)
def pool(q, gold, K=29, exclude=None):
    top = [i for i, _ in ix.topk(q, K + 3) if replies[i] != gold][:K]
    return top
def build(turns, n):
    Xs, ys, groups = [], [], []
    for q, g, _ in random.sample(turns, n):
        top = pool(q, g)
        if len(top) < 20: continue
        # gold vector/info (gold may not be in pool list)
        gv = cross._embed(g, cross.Wr)[None]; ginfo = rk.prep_pool([g])
        rv = np.vstack([gv, RM[top]]); ri = ginfo + [info[i] for i in top]
        F = rk.feats(q, rv, ri); y = np.zeros(len(ri), np.float32); y[0] = 1
        Xs.append(F); ys.append(y); groups.append(len(ri))
    return Xs, ys
# train the ranker on TRAIN turns (their own reply excluded from the pool), test on held-out turns
Xs, ys = build(held, 6000); rk.fit_listwise(Xs)
names = ["cross_cos", "overlap", "overlap_norm", "reply?_msgstmt", "reply?_msg?", "answer_to_q", "len", "generic", "neg*sorry", "pos*glad", "neg*glad", "pos*sorry", "bias"]
print("learned weights:", ", ".join(f"{n}={w:+.2f}" for n, w in zip(names, rk.w)))
Xt, yt = build(test, 2500)
def r1(scorefn):
    return np.mean([np.argmax(scorefn(X)) == 0 for X in Xt])
rnd = np.mean([random.randrange(len(X)) == 0 for X in Xt])
print(f"hard-negative R@1 over {len(Xt)} held-out turns (1 real + ~29 plausible):")
print(f"   random {rnd:.1%} | word overlap {r1(lambda X: X[:,1] + 1e-3*np.random.rand(len(X))):.1%} | CrossPMI only {r1(lambda X: X[:,0]):.1%} | learned ranker {r1(rk.score):.1%}")
