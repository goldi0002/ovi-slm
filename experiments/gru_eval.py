"""Does a tiny GRU seq2seq (CPU, <5 MB, <=12 min training) beat the numpy ranker?
Same train pairs / clean protocol as pool_eval.py (mixed pool of 29 distractors + gold; ties/NaN = loss;
validation fits the listwise ranker, test measures; real replies pass the same generic filter).
Score = length-normalised log p(reply | message). Also: GRU score as an extra feature in the listwise ranker.
Run: PYTHONPATH=. python -u experiments/gru_eval.py > experiments/gru_eval.log 2>&1"""
import random, re, time, math, collections, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from ovi import dialogue as D
from ovi.retrieval import Index
from ovi.replyrank import CrossPMI, Ranker
from ovi.dialogue import _COMMON, _WORD
torch.set_num_threads(4); torch.manual_seed(0); random.seed(0); np.random.seed(0)
BUDGET = 11 * 60          # seconds of training
VOC, EMB, HID = 5000, 96, 192
TOK = re.compile(r"[a-z0-9']+|[.,!?]")
tok = lambda s: TOK.findall(s.lower())
PAD, UNK, BOS, EOS = 0, 1, 2, 3

train = D.dedupe(D.dailydialog("train") + D.empathetic("train"))
print("train pairs", len(train))
cnt = collections.Counter(w for q, a, _ in train for w in tok(q) + tok(a))
itos = ["<pad>", "<unk>", "<s>", "</s>"] + [w for w, _ in cnt.most_common(VOC - 4)]
stoi = {w: i for i, w in enumerate(itos)}
ids = lambda s, n=30: [stoi.get(w, UNK) for w in tok(s)][:n]

class Seq2Seq(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(VOC, EMB, padding_idx=PAD)
        s.enc = nn.GRU(EMB, HID, batch_first=True)
        s.dec = nn.GRU(EMB + HID, HID, batch_first=True)
        s.proj = nn.Linear(HID, EMB); s.bias = nn.Parameter(torch.zeros(VOC)); s.drop = nn.Dropout(0.2)
    def encode(s, x, ln):
        pk = nn.utils.rnn.pack_padded_sequence(s.drop(s.emb(x)), ln.cpu(), batch_first=True, enforce_sorted=False)
        return s.enc(pk)[1][0]                                   # (B, HID)
    def step(s, y, h, c):                                         # y (B,T) tokens, h (1,B,HID) dec state, c (B,HID) context
        e = s.drop(s.emb(y)); o, h = s.dec(torch.cat([e, c[:, None].expand(-1, e.size(1), -1)], -1), h)
        return s.proj(o) @ s.emb.weight.T + s.bias, h
    def forward(s, x, xl, y):                                     # y includes BOS ... EOS; returns per-token logp of y[1:]
        c = s.encode(x, xl); lg, _ = s.step(y[:, :-1], c[None].contiguous(), c)
        return -Fn.cross_entropy(lg.transpose(1, 2), y[:, 1:], ignore_index=PAD, reduction="none")

def pad(seqs):
    L = max(len(q) for q in seqs); t = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, q in enumerate(seqs): t[i, :len(q)] = torch.tensor(q)
    return t
def batch(pairs):
    x = [ids(q) or [UNK] for q, _ in pairs]; y = [[BOS] + ids(a) + [EOS] for _, a in pairs]
    return pad(x), torch.tensor([len(q) for q in x]), pad(y)

m = Seq2Seq(); npar = sum(p.numel() for p in m.parameters())
print(f"GRU params {npar/1e6:.2f}M = {npar*4/1e6:.2f} MB fp32")
data = [(q, a) for q, a, _ in train]
trips = lambda d, e: D.triples(d, "daily") + D.triples(e, "emp")
va, te = trips("validation", "valid"), trips("test", "test")
dev = [(h[-1], r) for h, r in va[:1500]]

def evalloss():
    m.eval(); tot = n = 0
    with torch.no_grad():
        for i in range(0, len(dev), 250):
            x, xl, y = batch(dev[i:i+250]); lp = m(x, xl, y); tot -= lp.sum().item(); n += (y[:, 1:] != PAD).sum().item()
    m.train(); return tot / n

opt = torch.optim.Adam(m.parameters(), lr=3e-3); t0 = time.time(); step = 0; ep = 0; done = False
while not done:
    random.shuffle(data); ep += 1
    chunks = [data[i:i + 128 * 50] for i in range(0, len(data), 128 * 50)]
    for ch in chunks:
        ch = sorted(ch, key=lambda p: len(p[1]))                 # length-bucket for speed
        bs = [ch[i:i + 128] for i in range(0, len(ch), 128)]; random.shuffle(bs)
        for b in bs:
            frac = (time.time() - t0) / BUDGET
            if frac >= 1: done = True; break
            for g in opt.param_groups: g["lr"] = 3e-3 * max(0.05, 1 - frac)
            x, xl, y = batch(b); lp = m(x, xl, y); loss = -lp.sum() / (y[:, 1:] != PAD).sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); step += 1
            if step % 300 == 0: print(f"ep {ep} step {step} t={time.time()-t0:.0f}s train_nll {loss.item():.3f} dev_nll {evalloss():.3f}", flush=True)
        if done: break
dl = evalloss()
print(f"trained {step} steps, {ep} epochs, {time.time()-t0:.0f}s; final dev token NLL {dl:.3f} (ppl {math.exp(dl):.1f})")
sd = {k: v.half() for k, v in m.state_dict().items()}
torch.save(sd, "/tmp/gru_eval_fp16.pt")
print("fp16 state_dict file MB:", round(os.path.getsize("/tmp/gru_eval_fp16.pt") / 1e6, 2), "(fp32 above)")
m.eval()

def gru_scores(msg, cands):
    """length-normalised log p(reply|msg) per candidate"""
    with torch.no_grad():
        x = torch.tensor([ids(msg) or [UNK]]); c = m.encode(x, torch.tensor([x.size(1)]))
        ys = [[BOS] + ids(a) + [EOS] for a in cands]; y = pad(ys); K = len(cands)
        lg, _ = m.step(y[:, :-1], c[None].expand(1, K, -1).contiguous(), c.expand(K, -1))
        lp = -Fn.cross_entropy(lg.transpose(1, 2), y[:, 1:], ignore_index=PAD, reduction="none")
        return (lp.sum(1) / (y[:, 1:] != PAD).sum(1)).numpy()

# ---------------- same clean protocol as pool_eval.py ----------------
cross = CrossPMI().fit(train)
replies = [a for _, a, _ in train]; RM = cross.reply_matrix(replies)
rk = Ranker(cross); info = rk.prep_pool(replies)
ix = Index()
for q, a, _ in train: ix.add(q, a)
def distract(msg, gold):
    t = [i for i, _ in ix.topk(msg, 40) if replies[i] != gold]
    c = [i for i in np.argsort(-(RM @ cross.msg_vec(msg)))[:60] if replies[i] != gold]
    out = t[:15]
    for i in c:
        if i not in out and len(out) < 29: out.append(int(i))
    return out
def groups(tr, n):
    out = []
    for hist, gold in random.sample(tr, min(n, len(tr))):
        top = distract(hist[-1], gold)
        if len(top) < 20: continue
        rv = np.vstack([cross._embed(gold, cross.Wr)[None], RM[top]]); ri = rk.prep_pool([gold]) + [info[i] for i in top]
        X = rk.feats(hist[-1], rv, ri, hist)
        g = gru_scores(hist[-1], [gold] + [replies[i] for i in top])
        out.append((X, g))
    return out
ok = lambda r: all(rk.rdf.get(w, 0) >= 5 for w in _WORD.findall(r.lower()) if w not in _COMMON)
va, te = [t for t in va if ok(t[1])], [t for t in te if ok(t[1])]
random.seed(1); t1 = time.time()
Gva = groups(va, 4000); Gte = groups(te, 2000)
print(f"built {len(Gva)} val / {len(Gte)} test groups in {time.time()-t1:.0f}s")
def win(sc): return bool(np.isfinite(sc).all() and (sc[1:] >= sc[0]).sum() == 0)
def withg(X, g): return np.hstack([X[:, :-1], g[:, None], X[:, -1:]]).astype(np.float32)
print("chance (1/30): 3.3%")
print("GRU alone, mixed pool:           R@1 %.1f%%" % (100 * np.mean([win(g) for _, g in Gte])))
print("crosspmi cos alone:              R@1 %.1f%%" % (100 * np.mean([X[0, 0] > X[1:, 0].max() for X, _ in Gte])))
r0 = Ranker(cross); r0.rdf = rk.rdf; r0.fit_listwise([X for X, _ in Gva])
print("ranker baseline (mixed-trained): R@1 %.1f%%" % (100 * np.mean([win(r0.score(X)) for X, _ in Gte])))
r1 = Ranker(cross); r1.rdf = rk.rdf; r1.fit_listwise([withg(X, g) for X, g in Gva])
print("ranker + GRU feature:            R@1 %.1f%%" % (100 * np.mean([win(r1.score(withg(X, g))) for X, g in Gte])))
print("ranker weight on GRU feature (standardised): %.2f" % r1.w[-2])

# ---------------- free generation ----------------
def generate(msg, topk=1, maxlen=20):
    with torch.no_grad():
        x = torch.tensor([ids(msg) or [UNK]]); c = m.encode(x, torch.tensor([x.size(1)])); h = c[None].contiguous(); y = torch.tensor([[BOS]]); out = []
        for _ in range(maxlen):
            lg, h = m.step(y, h, c); lg = lg[0, -1]; lg[UNK] = -1e9; lg[PAD] = -1e9
            if topk > 1:
                v, i = lg.topk(topk); nxt = i[torch.multinomial(Fn.softmax(v, 0), 1)].item()
            else: nxt = int(lg.argmax())
            if nxt == EOS: break
            out.append(itos[nxt]); y = torch.tensor([[nxt]])
    return re.sub(r"\s+([.,!?])", r"\1", " ".join(out))
random.seed(2); ms = random.sample([h[-1] for h, _ in te], 15); lat = []
print("\n15 freely generated replies (held-out test messages):")
for q in ms:
    t = time.time(); g = generate(q); lat.append((time.time() - t) * 1000)
    print(f"  msg: {q}\n    greedy: {g}\n    top-5 : {generate(q, 5)}")
print("mean greedy generation latency: %.1f ms/reply" % np.mean(lat))
