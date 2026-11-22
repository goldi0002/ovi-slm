"""Does TinyStories give a better generator than books? Run: PYTHONPATH=. python experiments/tinystories_ngram.py"""
import time, pickle, zlib, random, numpy as np
from tokenizers import Tokenizer
from ovi.ngram import NGram
tok = Tokenizer.from_file("data/tokenizer.json")
stories = [s.strip() for s in open("data/ts/valid.txt", encoding="utf-8", errors="ignore").read().split("<|endoftext|>") if len(s.strip()) > 100]
random.seed(0); random.shuffle(stories)
test, train = stories[:150], stories[150:]
t = time.time()
enc = lambda ss: [i for s in ss for i in tok.encode(s + "\n").ids]
tr_ids, te_ids = enc(train), enc(test)
print(f"{len(train)} train stories -> {len(tr_ids)/1e6:.2f}M tokens ({time.time()-t:.0f}s tokenise); test {len(te_ids)} tokens")
books = np.load("data/books.npy").astype(int).tolist()
def size(m):
    plain = [{k: (v[0], v[1]) for k, v in tt.items()} for tt in m.t]
    return len(zlib.compress(pickle.dumps(plain, protocol=5), 1)) / 1e6
sub = te_ids[:6000]
for name, ids, order in (("books n=3", books, 3), ("stories n=3", tr_ids, 3), ("stories n=4", tr_ids, 4)):
    t = time.time(); m = NGram(order); m.add(ids); bt = time.time() - t
    l = m.loss(sub, 1)
    print(f"{name:12} build {bt:4.0f}s  held-out story loss {l:.3f} nats/token  file {size(m):.1f} MB")
    if name == "stories n=4":
        random.seed(1); ctx = tok.encode("Once upon a time").ids
        for _ in range(2): print("  sample:", tok.decode(ctx + m.sample(ctx, 60, temp=0.7)).replace("\n", " "))
