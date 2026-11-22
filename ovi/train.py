"""Train Ovi: phase 1 = language from books, phase 2 = chat format (mixed with books)."""
import time, sys, numpy as np, torch
from tokenizers import Tokenizer
from .model import GPT, Config, save_compact
from .seed_chat import chat_corpus

torch.manual_seed(0)
torch.set_num_threads(4)


def batch(arr, B, T):
    ix = np.random.randint(0, len(arr) - T - 1, B)
    x = torch.from_numpy(np.stack([arr[i:i + T] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([arr[i + 1:i + T + 1] for i in ix]).astype(np.int64))
    return x, y


def run(model, opt, sources, steps, lr, B, T, label, budget=None):
    t0 = time.time()
    for s in range(steps):
        cur = lr * min(1, (s + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * s / steps)))
        for g in opt.param_groups:
            g["lr"] = cur
        r, acc, arr = np.random.rand(), 0, None
        for a, w in WEIGHTS[id(sources)]:
            acc += w
            if r <= acc:
                arr = a
                break
        x, y = batch(arr, B, T)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % 20 == 0:
            print(f"[{label}] step {s}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
        if budget and time.time() - t0 > budget:
            print(f"[{label}] time budget hit at step {s}", flush=True)
            break


WEIGHTS = {}


def main(steps1=2000, steps2=600, budget1=130, budget2=45):
    tok = Tokenizer.from_file("data/tokenizer.json")
    books = np.load("data/books.npy")
    chat = np.array(tok.encode(chat_corpus(reps=40, seed=1)).ids, dtype=np.uint16)
    cfg = Config(vocab=tok.get_vocab_size(), ctx=128, d=192, layers=4, heads=4)
    model = GPT(cfg)
    print("params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
    p1 = [books]; WEIGHTS[id(p1)] = [(books, 1.0)]
    run(model, opt, p1, steps1, 1e-3, 32, cfg.ctx, "books", budget1)
    p2 = [books, chat]; WEIGHTS[id(p2)] = [(books, 0.3), (chat, 0.7)]
    run(model, opt, p2, steps2, 5e-4, 32, cfg.ctx, "chat", budget2)
    save_compact(model, "ovi.pt")
    import os
    print("saved ovi.pt", os.path.getsize("ovi.pt") / 1e6, "MB")


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
