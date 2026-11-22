"""TF-IDF retrieval over many (question -> answer) pairs. Adding a pair = microseconds."""
import math, re
from collections import defaultdict

W = re.compile(r"[a-z0-9']+")


class Index:
    def __init__(self):
        self.q, self.a = [], []
        self.inv = defaultdict(list)

    def add(self, q, a):
        i = len(self.q)
        ws = set(W.findall(q.lower()))
        self.q.append(ws); self.a.append(a)
        for w in ws:
            self.inv[w].append(i)

    def best(self, text, min_score=0.45):
        ws = set(W.findall(text.lower()))
        n = len(self.q) + 1
        score = defaultdict(float)
        for w in ws:
            post = self.inv.get(w)
            if not post:
                continue
            idf = math.log(n / (1 + len(post)))
            for i in post:
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
