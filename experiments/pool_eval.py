"""Does widening the candidate pool (TF-IDF question match + CrossPMI over ALL replies) help?
Clean protocol as context_eval.py. Distractors: 'tfidf' = 29 TF-IDF neighbours, 'mixed' = 15 TF-IDF + 14 CrossPMI top.
Run: PYTHONPATH=. python experiments/pool_eval.py"""
import random, numpy as np
from ovi import dialogue as D
from ovi.retrieval import Index
from ovi.replyrank import CrossPMI, Ranker
from ovi.dialogue import _COMMON, _WORD
random.seed(0); np.random.seed(0)
train = D.dedupe(D.dailydialog("train") + D.empathetic("train"))
cross = CrossPMI().fit(train)
replies = [a for _, a, _ in train]; RM = cross.reply_matrix(replies)
rk = Ranker(cross); info = rk.prep_pool(replies)
ix = Index()
for q, a, _ in train: ix.add(q, a)
def distract(msg, gold, mode):
    t = [i for i, _ in ix.topk(msg, 40) if replies[i] != gold]
    if mode == "tfidf": return t[:29]
    c = np.argsort(-(RM @ cross.msg_vec(msg)))[:60]
    c = [i for i in c if replies[i] != gold]
    out = t[:15]
    for i in c:
        if i not in out and len(out) < 29: out.append(i)
    return out
def groups(trips, n, mode):
    Xs = []
    for hist, gold in random.sample(trips, min(n, len(trips))):
        top = distract(hist[-1], gold, mode)
        if len(top) < 20: continue
        rv = np.vstack([cross._embed(gold, cross.Wr)[None], RM[top]]); ri = rk.prep_pool([gold]) + [info[i] for i in top]
        Xs.append(rk.feats(hist[-1], rv, ri, hist))
    return Xs
trips = lambda d, e: D.triples(d, "daily") + D.triples(e, "emp")
va, te = trips("validation", "valid"), trips("test", "test")
ok = lambda r: all(rk.rdf.get(w, 0) >= 5 for w in _WORD.findall(r.lower()) if w not in _COMMON)
va, te = [t for t in va if ok(t[1])], [t for t in te if ok(t[1])]
random.seed(1); Xte = {m: groups(te, 2000, m) for m in ("tfidf", "mixed")}
def win(X, r): sc = r.score(X); return bool(np.isfinite(sc).all() and (sc[1:] >= sc[0]).sum() == 0)
for trainmode in ("tfidf", "mixed"):
    r = Ranker(cross); r.rdf = rk.rdf; r.fit_listwise(groups(va, 4000, trainmode))
    for testmode in ("tfidf", "mixed"):
        print(f"trained on {trainmode:6} pool, tested on {testmode:6} pool: R@1 {np.mean([win(X, r) for X in Xte[testmode]]):.1%}")
# raw crosspmi cosine alone on the mixed pool
print("crosspmi cos alone, mixed pool: %.1f%%" % (100*np.mean([X[0,0] > X[1:,0].max() for X in Xte["mixed"]])))
