"""Chat with Ovi. Ovi learns while you talk.

Commands:
  /teach <question> => <answer>   learn a Q/A pair (trains immediately)
  /learn <text>                   learn from free text
  /learnfile <path>               learn from a text file
  /save   /quit
Ovi also keeps everything you taught it in data/learned.jsonl and replays it
during later learning so it doesn't forget.
"""
import json, os, random, sys, numpy as np, torch
from tokenizers import Tokenizer
from .model import load_compact, save_compact
from .seed_chat import PAIRS, USER, BOT, END, fmt

torch.set_num_threads(4)
CK, LOG = "ovi.pt", "data/learned.jsonl"
tok = Tokenizer.from_file("data/tokenizer.json")
model, _ = load_compact(CK)
UID, BID, EID = (tok.token_to_id(t) for t in (USER, BOT, END))
history = []


def ids(s):
    return tok.encode(s).ids


def reply(msg, temp=0.7):
    ctx = "".join(fmt(u, b) for u, b in history[-3:]) + f"{USER}{msg}\n{BOT}"
    x = torch.tensor([ids(ctx)[-model.c.ctx + 48:]])
    model.eval()
    out = model.generate(x, 48, temp=temp, top_k=30, stop=EID)[0, x.shape[1]:].tolist()
    if out and out[-1] == EID:
        out = out[:-1]
    return tok.decode(out).strip()


def load_learned():
    if not os.path.exists(LOG):
        return []
    return [json.loads(l) for l in open(LOG)]


def train_on(samples, steps=40, lr=3e-4):
    """Fine-tune on new samples, mixed with replay (old lessons + seed chat) to limit forgetting."""
    old = load_learned()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    T = model.c.ctx
    for s in range(steps):
        rows = []
        for _ in range(8):
            r = random.random()
            if r < 0.5:
                t = random.choice(samples)
            elif r < 0.75 and old:
                t = random.choice(old)
            else:
                t = fmt(*random.choice(PAIRS))
            rows.append(t if isinstance(t, str) else t["text"])
        seqs = [ids(t)[:T + 1] for t in rows]
        L = max(len(q) for q in seqs)
        x = torch.zeros(len(seqs), L - 1, dtype=torch.long)
        y = torch.full((len(seqs), L - 1), -100, dtype=torch.long)
        for i, q in enumerate(seqs):
            q = torch.tensor(q)
            x[i, :len(q) - 1] = q[:-1]; y[i, :len(q) - 1] = q[1:]
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    os.makedirs("data", exist_ok=True)
    with open(LOG, "a") as f:
        for t in samples:
            f.write(json.dumps({"text": t}) + "\n")
    save_compact(model, CK)
    return loss.item()


def teach_pair(q, a):
    return [fmt(q, a), fmt(q.capitalize() + "?", a), fmt(q.lower().rstrip("?"), a)]


def main():
    print("Ovi is awake. Talk to me. (/teach q => a, /learn text, /quit)")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg in ("/quit", "/exit"):
            break
        if msg.startswith("/teach ") and "=>" in msg:
            q, a = [p.strip() for p in msg[7:].split("=>", 1)]
            l = train_on(teach_pair(q, a) * 3, steps=40)
            print(f"ovi> Learned it! (loss {l:.2f})")
        elif msg.startswith("/learn "):
            text = msg[7:]
            l = train_on([text + "\n"] * 6, steps=30)
            print(f"ovi> Thanks, I read that. (loss {l:.2f})")
        elif msg.startswith("/learnfile "):
            body = open(msg[11:].strip(), encoding="utf-8", errors="ignore").read()
            chunks = [body[i:i + 800] for i in range(0, len(body), 800)][:300]
            l = train_on(chunks, steps=max(40, len(chunks)))
            print(f"ovi> Finished reading {len(chunks)} chunks. (loss {l:.2f})")
        else:
            r = reply(msg)
            print("ovi>", r)
            history.append((msg, r))


if __name__ == "__main__":
    main()
