"""Reply quality proxy: content-word F1 between retrieved reply and the human's real reply on held-out turns.
Run: PYTHONPATH=. python experiments/dialogue_retrieval_eval.py"""
import random, re, time
from ovi import dialogue as D, dataset
from ovi.memory import STOP, toks
from ovi.retrieval import Index
random.seed(0)
train, test = D.splits()
test = random.sample(test, 3000)
def cw(s): return {w for w in toks(s) if w not in STOP and len(w) > 1}
def f1(a, b):
    a, b = cw(a), cw(b)
    if not a or not b: return 0.0
    i = len(a & b)
    return 0.0 if i == 0 else 2 * i / (len(a) + len(b))
def evaluate(name, pairs, th):
    ix = Index()
    for q, a, *_ in pairs: ix.add(q, a)
    t = time.time(); hit = 0; tot = 0.0; ms = []
    for q, gold, _ in test:
        h = ix.best(q, th)
        if h: hit += 1; tot += f1(h[0], gold)
    dt = (time.time() - t) / len(test) * 1000
    print(f"{name:22} pairs {len(pairs):7}  answered {hit/len(test):4.0%}  mean F1 when answered {tot/max(hit,1):.3f}  overall {tot/len(test):.3f}  {dt:.1f} ms/query")
corn = [(q, a, "m") for _, q, a in dataset.cornell_pairs()]
evaluate("Cornell movies (old)", corn, 0.3)
evaluate("Dialogue data (new)", D.dedupe(train), 0.3)
evaluate("both", corn + D.dedupe(train), 0.3)
# random-reply baseline for reference
tot = sum(f1(random.choice(train)[1], g) for _, g, _ in test) / len(test); print(f"random reply baseline overall F1 {tot:.3f}")
