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

TOK, NG = "data/tokenizer.json", "ovi_ngram.bin"


def build(order=3):
    t = time.time()
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
    def __init__(self):
        self.tok = Tokenizer.from_file(TOK)
        self.lm = load()
        self.mem = Memory()
        for q, a in PAIRS:
            self.mem.learn(q, a, persist=False)
        self.mem2 = self.mem  # taught facts override seeds (later entries win on ties)

    def say(self, msg, n=40):
        r = self.mem.recall(msg)
        if r:
            return r
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
