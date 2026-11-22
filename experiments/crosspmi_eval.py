"""Run: PYTHONPATH=. python experiments/crosspmi_eval.py
(1) 1-of-50 ranking: gold reply vs 49 random replies.  (2) retrieve from ALL replies; closeness (F1) to the real reply."""
import random, time, numpy as np
from ovi import dialogue as D
from ovi.memory import STOP as ST, toks
from ovi.retrieval import Index
from ovi.replyrank import CrossPMI
random.seed(0)
train, test = D.splits(); train = D.dedupe(train); test = random.sample(test, 2000)
t = time.time(); m = CrossPMI(dim=128).fit(train); print(f"fit {time.time()-t:.0f}s on {len(train)} pairs")
replies = [a for _, a, _ in train]; RM = m.reply_matrix(replies)
cw = lambda s: {w for w in toks(s) if w not in ST and len(w) > 1}
def f1(a, b):
    a, b = cw(a), cw(b); i = len(a & b)
    return 0.0 if not a or not b or i == 0 else 2 * i / (len(a) + len(b))
# (1) 1-of-50
hits1 = hits5 = ov1 = 0
allr = [a for _, a, _ in test]
for q, g, _ in test:
    cands = [g] + random.sample(allr, 49); gm = m.reply_matrix(cands)
    sc = gm @ m.msg_vec(q); rank = int((sc > sc[0]).sum())
    hits1 += rank == 0; hits5 += rank < 5
    # overlap baseline: idf-free content-word overlap between message and reply
    o = [len(cw(q) & cw(c)) + random.random() * 1e-3 for c in cands]; ov1 += int((np.array(o) > o[0]).sum()) == 0
n = len(test)
print(f"1-of-50  R@1: CrossPMI {hits1/n:.1%} | word-overlap baseline {ov1/n:.1%} | random 2.0%     R@5: CrossPMI {hits5/n:.1%}")
# (2) retrieve from the whole reply pool vs the question-matching index
ix = Index()
for q, a, _ in train: ix.add(q, a)
f_old = f_new = f_mix = 0.0; t_new = 0
for q, g, _ in test:
    h = ix.best(q, 0.3); f_old += f1(h[0], g) if h else 0
    t0 = time.perf_counter(); sc = m.scores(q, RM); best = int(sc.argmax()); t_new += time.perf_counter() - t0
    f_new += f1(replies[best], g)
print(f"retrieval closeness F1 (overall): question-index {f_old/n:.3f} | CrossPMI over all replies {f_new/n:.3f}  ({t_new/n*1000:.1f} ms/query)")
