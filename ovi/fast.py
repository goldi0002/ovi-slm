"""Ovi-fast: a talking, learning model with NO gradient descent and NO GPU.

  python -m ovi.fast build   # ~5 s: count n-grams from books + chat (replaces 3 min of training)
  python -m ovi.fast         # chat

Learning is instant:  /teach q => a   (memory write, ~0.1 ms)
                      /learn text     (n-gram counts, ~ms)
"""
import os, pickle, re, sys, time, zlib
import numpy as np
from tokenizers import Tokenizer
from .memory import Memory
from .ngram import NGram
from .seed_chat import PAIRS
from .retrieval import Index
from . import dataset

TOK, NG = "data/tokenizer.json", "ovi_ngram.bin"


def build(order=4):
    """Count n-grams. TinyStories (data/ts/*.txt) if present -- far better text than books -- else books."""
    import glob
    t = time.time()
    tok = Tokenizer.from_file(TOK)
    files = glob.glob("data/ts/*.txt")
    if files:
        stories = [x.strip() for f in files for x in open(f, encoding="utf-8", errors="ignore").read().split("<|endoftext|>") if len(x.strip()) > 100]
        b = [i for s_ in stories for i in tok.encode(s_ + "\n").ids]
        print(f"TinyStories: {len(stories)} stories, {len(b)/1e6:.1f}M tokens")
    else:
        order = 3
        b = np.load("data/books.npy").astype(int).tolist()
    m = NGram(order)
    m.add(b)
    save(m)
    print(f"built in {time.time()-t:.1f}s, {os.path.getsize(NG)/1e6:.1f} MB -> {NG}")


def save(m):
    plain = [{k: (v[0], v[1]) for k, v in t.items()} for t in m.t]
    open(NG, "wb").write(zlib.compress(pickle.dumps((m.n, m.D, m.V, plain), protocol=5), 1))


def load():
    n, D, V, plain = pickle.loads(zlib.decompress(open(NG, "rb").read()))
    m = NGram(n, D, V)
    for t, d in zip(m.t, plain):
        for k, v in d.items():
            t[k] = [v[0], v[1]]
    return m


class Ovi:
    def __init__(self, light=False):
        self.tok = Tokenizer.from_file(TOK)
        self.lm = load()
        if light:                       # Brain only needs the tokenizer + n-gram LM
            return
        self.mem = Memory()
        self.cb, self.movie = Index(), Index()
        if os.path.exists("ovi_pairs.bin"):
            for kind, q, a in dataset.load():
                (self.cb if kind == "c" else self.movie).add(q, a)
                if kind == "c":
                    self.mem.learn(q, a, persist=False)
        for q, a in PAIRS:  # Ovi's own seed answers take priority over dataset answers
            self.mem.learn(q, a, persist=False)

    def say(self, msg, n=40):
        r = self.mem.recall(msg)
        if r:
            return r
        for ix, th in ((self.cb, 0.7), (self.movie, 0.55)):
            hit = ix.best(msg, th)
            if hit:
                return hit[0]
        return self.gen(msg, n)

    def gen(self, msg, n=40):
        """Free text from the n-gram LM seeded with the last two tokens of msg (no memory lookup)."""
        ctx = self.tok.encode(msg).ids[-2:]
        out = self.lm.sample(ctx, n, temp=0.7)
        text = self.tok.decode(out).strip().replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        cut = max(text.rfind(". "), text.rfind("? "), text.rfind("! "))
        return (text[:cut + 1] if cut > 20 else text).strip()

    def teach(self, q, a):
        self.mem.learn(q, a)

    def learn_text(self, text):
        self.lm.add(self.tok.encode(text + "\n").ids)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        return build()
    o = Ovi()
    print("Ovi-fast is awake. (/teach q => a, /learn text, /save, /quit)")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg in ("/quit", "/exit"):
            break
        if msg.startswith("/teach ") and "=>" in msg:
            q, a = [x.strip() for x in msg[7:].split("=>", 1)]
            t = time.perf_counter(); o.teach(q, a)
            print(f"ovi> Learned it! ({(time.perf_counter()-t)*1000:.2f} ms)")
        elif msg.startswith("/learn "):
            t = time.perf_counter(); o.learn_text(msg[7:])
            print(f"ovi> Read it! ({(time.perf_counter()-t)*1000:.2f} ms)")
        elif msg == "/save":
            save(o.lm); print("ovi> Saved.")
        elif msg:
            print("ovi>", o.say(msg))
    save(o.lm)


if __name__ == "__main__":
    main()
