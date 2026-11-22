"""Instant learning: a tiny retrieval memory. Learning = one dict write (microseconds)."""
import json, os, re

STOP = set("a an the is are was were be do does did of to in on at my your you i me it this that and or what who where when how why which".split())


_TOK = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def toks(t):
    return _TOK.findall(t.lower())


def norm(t):
    return " ".join(toks(t))


def keys(text):
    ws = (re.sub(r"'s?$", "", w) for w in toks(text))
    return frozenset(w for w in ws if w and w not in STOP)


class Memory:
    def __init__(self, path="data/memory.jsonl"):
        self.path, self.items, self.exact = path, [], {}
        if os.path.exists(path):
            for l in open(path):
                q, a = json.loads(l)
                self.items.append((keys(q), q, a)); self.exact[norm(q)] = a

    def learn(self, q, a, persist=True):
        self.items.append((keys(q), q, a)); self.exact[norm(q)] = a
        if persist:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps([q, a]) + "\n")

    def recall(self, q, thresh=0.55):
        if norm(q) in self.exact:
            return self.exact[norm(q)]
        k = keys(q)
        if not k:
            return None
        best, score = None, 0.0
        for kk, _, a in self.items:
            inter = len(k & kk)
            # query coverage, lightly penalised by extra stored words; need >=2 shared words unless query is 1 word
            s = 0.0 if inter < min(2, len(k)) else 0.7 * inter / len(k) + 0.3 * inter / len(k | kk)
            if s > score:
                best, score = a, s
        return best if score >= thresh else None
