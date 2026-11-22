"""TF-IDF retrieval over many (question -> answer) pairs. Adding a pair = microseconds."""
import math, re
from collections import defaultdict

from .memory import toks


class Index:
    def __init__(self):
        self.q, self.a = [], []
        self.inv = defaultdict(list)
        self.dead = set()          # forgotten entries: never returned

    def add(self, q, a, ws=None):
        i = len(self.q)
        ws = set(toks(q)) if ws is None else set(ws)
        self.q.append(ws); self.a.append(a)
        for w in ws:
            self.inv[w].append(i)

    def topk(self, text, k=30, min_score=0.0):
        """Top-k (answer_index, score) by question match."""
        h = self._scores(text)
        return sorted(((i, v) for i, v in h if v >= min_score), key=lambda x: -x[1])[:k]

    def best(self, text, min_score=0.45):
        ws = set(toks(text))
        return self._best_from(ws, min_score, text)

    def _scores(self, text):
        ws = set(toks(text))
        n = len(self.q) + 1
        score = defaultdict(float)
        for w in ws:
            post = self.inv.get(w)
            if not post:
                continue
            idf = math.log(n / (1 + len(post)))
            for i in post:
                if i in self.dead:
                    continue
                score[i] += idf
        if not score:
            return []
        qn = sum(math.log(n / (1 + len(self.inv[w]))) for w in ws if w in self.inv) or 1.0
        return [(i, (s / qn) / math.sqrt(max(len(self.q[i]), 1) / max(len(ws), 1)) ** 0.5) for i, s in score.items()]

    def _best_from(self, ws, min_score, text):
        n = len(self.q) + 1
        score = defaultdict(float)
        for w in ws:
            post = self.inv.get(w)
            if not post:
                continue
            idf = math.log(n / (1 + len(post)))
            for i in post:
                if i in self.dead:
                    continue
                score[i] += idf
        if not score:
            return None
        qn = sum(math.log(n / (1 + len(self.inv[w]))) for w in ws if w in self.inv) or 1.0
        top, tv = None, 0.0
        for i, s in score.items():
            # fraction of the query's weight matched, penalising long stored questions
            v = (s / qn) / math.sqrt(max(len(self.q[i]), 1) / max(len(ws), 1)) ** 0.5
            if v > tv:
                top, tv = i, v
        return (self.a[top], tv, top) if tv >= min_score else None
