"""Train ovi.gen (CPU, time-boxed). Run: PYTHONPATH=. python -u experiments/gen_train.py [minutes] > experiments/gen_train.log 2>&1"""
import sys, time, math, random, torch, torch.nn.functional as F
from ovi import gen as G
from experiments import gen_data as DD
torch.set_num_threads(4); torch.manual_seed(0); random.seed(0)
MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 35
OUT = sys.argv[2] if len(sys.argv) > 2 else "ovi_gen.pt"
INIT = sys.argv[3] if len(sys.argv) > 3 else None
PEAK = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3

def encode(ex):
    h, r, mem, _ = ex
    p = G.build_prompt(h, mem); y = G.enc(r)[:40] + [G.END]
    return p + y, len(p)

real = DD.real("train", "train"); syn = DD.synth(30000, 1, held=False)
train = real + syn; random.shuffle(train)
vreal = DD.real("validation", "valid"); random.Random(1).shuffle(vreal); vreal = vreal[:1500]
vsyn = DD.synth(600, 99, held=True)
print(f"train {len(real)} real + {len(syn)} synth; val {len(vreal)} real, {len(vsyn)} synth(unseen values)", flush=True)
data = [encode(e) for e in train]; vr = [encode(e) for e in vreal]; vs = [encode(e) for e in vsyn]
print("tokens", sum(len(x) for x, _ in data), flush=True)

m = G.Gen()
if INIT: m.load_state_dict(torch.load(INIT)); print('warm start', INIT, flush=True)
print(f"params {m.nparams()/1e6:.2f}M", flush=True)
opt = torch.optim.AdamW(m.parameters(), lr=PEAK, weight_decay=0.05, betas=(0.9, 0.98))

def batchify(items):
    L = max(len(x) for x, _ in items); X = torch.zeros(len(items), L, dtype=torch.long); M = torch.zeros(len(items), L)
    for i, (x, s) in enumerate(items):
        X[i, :len(x)] = torch.tensor(x); M[i, s:len(x)] = 1
    return X, M

def loss_of(items):
    X, M = batchify(items); lg = m(X[:, :-1]); t = X[:, 1:]; mk = M[:, 1:]
    l = F.cross_entropy(lg.transpose(1, 2), t, reduction="none")
    return (l * mk).sum(), mk.sum()

def val(items):
    m.eval(); tot = n = 0
    with torch.no_grad():
        for i in range(0, len(items), 100):
            a, b = loss_of(items[i:i + 100]); tot += a.item(); n += b.item()
    m.train(); return math.exp(tot / n)

def copy_acc():
    m.eval(); ok = tot = 0
    for ex in vsyn[:150]:
        if ex[3] is None: continue
        tot += 1; ok += ex[3].lower() in m.reply(ex[0], ex[2], temp=0).lower()
    m.train(); return ok / max(1, tot)

TOKB = 2200; t0 = time.time(); step = 0; last = t0; m.train()
while True:
    random.shuffle(data)
    chunks = [sorted(data[i:i + 2000], key=lambda e: len(e[0])) for i in range(0, len(data), 2000)]
    batches = []
    for ch in chunks:
        cur = []
        for e in ch:
            if cur and (len(cur) + 1) * len(e[0]) > TOKB: batches.append(cur); cur = []
            cur.append(e)
        if cur: batches.append(cur)
    random.shuffle(batches)
    for b in batches:
        frac = (time.time() - t0) / (MIN * 60)
        if frac >= 1: break
        lr = PEAK * min(1, step / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups: g["lr"] = lr
        a, n = loss_of(b); (a / n).backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad(); step += 1
        if time.time() - last > 300:
            last = time.time(); torch.save(m.state_dict(), OUT)
            print(f"step {step} {frac*100:.0f}% lr {lr:.5f} train {a.item()/n.item():.3f} val_ppl real {val(vr):.1f} synth {val(vs):.2f} copy {copy_acc():.2f}", flush=True)
    else:
        continue
    break
torch.save(m.state_dict(), OUT)
print(f"DONE step {step} val_ppl real {val(vr):.1f} synth {val(vs):.2f} copy {copy_acc():.2f}", flush=True)
