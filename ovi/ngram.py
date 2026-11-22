"""Gradient-free language model: interpolated n-gram with absolute discounting.
'Training' = counting (seconds on CPU). Learning online = adding counts (microseconds)."""
import math, random
from collections import defaultdict


class NGram:
    def __init__(self, order=5, discount=0.75, vocab=4096):
        self.n, self.D, self.V = order, discount, vocab
        # tables[k][ctx tuple of length k] = [total, {next: count}]
        self.t = [defaultdict(lambda: [0, {}]) for _ in range(order)]

    def add(self, ids):
        n = self.n
        for i, w in enumerate(ids):
            for k in range(min(n - 1, i) + 1):
                e = self.t[k][tuple(ids[i - k:i])]
                e[0] += 1
                e[1][w] = e[1].get(w, 0) + 1

    def dist(self, ctx):
        """Full next-token distribution as dict (sparse; mass on seen tokens + uniform floor)."""
        ctx = tuple(ctx[-(self.n - 1):]) if self.n > 1 else ()
        p = {}
        base = 1.0 / self.V
        # unigram level
        e = self.t[0].get(())
        for lvl in range(len(ctx) + 1):
            c = ctx[len(ctx) - lvl:] if lvl else ()
            e = self.t[lvl].get(c)
            if e is None:
                break
            tot, nx = e
            lam = self.D * len(nx) / tot
            new = {w: max(cnt - self.D, 0) / tot for w, cnt in nx.items()}
            for w, v in p.items():
                new[w] = new.get(w, 0) + lam * v
            if lvl == 0:
                for w in new:
                    pass
            p = new if lvl else {w: v + lam * base for w, v in new.items()}
            p["_floor"] = lam * (p.get("_floor", base) if lvl else base)
        return p

    def prob(self, ctx, w):
        p = self.dist(ctx)
        return p.get(w, p.get("_floor", 1.0 / self.V))

    def loss(self, ids, start=0):
        s = 0.0
        for i in range(start, len(ids)):
            s -= math.log(max(self.prob(ids[max(0, i - self.n + 1):i], ids[i]), 1e-9))
        return s / (len(ids) - start)

    def sample(self, ctx, n, temp=0.8, stop=None, top_k=40):
        out = []
        for _ in range(n):
            p = self.dist((list(ctx) + out)[-(self.n - 1):])
            p.pop("_floor", None)
            items = sorted(p.items(), key=lambda x: -x[1])[:top_k]
            ws = [max(v, 1e-12) ** (1 / temp) for _, v in items]
            w = random.choices([k for k, _ in items], ws)[0]
            out.append(w)
            if stop is not None and w == stop:
                break
        return out
