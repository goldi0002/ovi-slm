"""Load real conversation datasets (DailyDialog, EmpatheticDialogues, PersonaChat) into
(message -> reply) pairs. Files live in data/hf/ (downloaded from Hugging Face by the user)."""
import glob, json, re, os
import pandas as pd


def detok(s):
    s = s.replace("_comma_", ",")
    s = re.sub(r"\s+([.,!?;:%])", r"\1", s)
    s = re.sub(r"\s+'(s|t|m|re|ve|ll|d)\b", r"'\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _ok(s, lo=1, hi=32):
    n = len(s.split())
    return lo <= n <= hi and all(ord(c) < 128 for c in s) and not re.search(r"[*_]|\([^)]*\)", s)   # drop roleplay stage directions


def _chain(turns, tag, out):
    for a, b in zip(turns, turns[1:]):
        a, b = detok(a), detok(b)
        if _ok(a, 1, 40) and _ok(b, 2, 32):
            out.append((a, b, tag))


def dailydialog(split):
    path = f"data/hf/x/{split}.jsonl"
    out = []
    if not os.path.exists(path):
        return out
    for l in open(path, encoding="utf-8"):
        conv = [t["value"] for t in json.loads(l)["conversations"] if t["from"] != "system"]
        _chain(conv, "daily", out)
    return out


def empathetic(split):
    path = f"data/hf/ed/empatheticdialogues/{split}.csv"
    out = []
    if not os.path.exists(path):
        return out
    rows = {}
    for l in open(path, encoding="utf-8"):
        p = l.rstrip("\n").split(",", 7)
        if len(p) < 6 or not p[1].isdigit():
            continue
        rows.setdefault(p[0], []).append((int(p[1]), p[2], p[5]))
    for cid, r in rows.items():
        r.sort()
        _chain([u for _, _, u in r], "emp:" + r[0][1], out)
    return out


def persona():
    out = []
    for f in glob.glob("data/hf/x/*.parquet"):
        for _, row in pd.read_parquet(f).iterrows():
            for u in row["utterances"]:
                h, c = list(u["history"]), list(u["candidates"])
                if h and c:
                    _chain([h[-1], c[-1]], "persona", out)
    return out


def splits():
    """train / test pairs. Persona has only one file: hold out its last 10% as test."""
    p = persona(); cut = int(len(p) * 0.9)
    train = dailydialog("train") + dailydialog("validation") + empathetic("train") + empathetic("valid") + p[:cut]
    test = dailydialog("test") + empathetic("test") + p[cut:]
    return train, test


def dedupe(pairs, max_pairs=None):
    seen, out = set(), []
    for q, a, t in pairs:
        k = (q.lower(), a.lower())
        if k in seen:
            continue
        seen.add(k); out.append((q, a, t))
    return out[:max_pairs] if max_pairs else out
