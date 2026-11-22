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


FAMILY = r"wife|husband|girlfriend|boyfriend|fianc[eé]e?|mom|mum|mother|dad|father|son|daughter|kids?|children|brother|sister|boss|coworkers?|job|house|home|car|truck|dog|cat|puppy|pets?|family|parents|friends?|school|college|work|apartment|room|birthday|grand\w+|uncle|aunt|baby|teacher|class|roommate|neighbou?r|wedding|garden|farm|office|company|business|team|salary|boss"
CLAIMS = [
    rf"\bmy ({FAMILY})\b",                                                            # my wife / my job / my dog
    r"\bi(?: am|'m) (?:a|an) (?!bit\b|little\b|lot\b|big\b|good\b|great\b|fan\b|huge\b|very\b|real\b)\w+",   # identity: I'm a teacher
    r"\bi(?: am|'m) (?:\d+|from|in|at|on|going to|studying|working|married|single|retired)\b",
    r"\b(?:i|we) (?:have|had|own|owned|got|bought|adopted) (?:a|an|two|three|four|five|six|\d+|my|our|some|no) (?!idea\b|problem\b|time\b|question\b|choice\b|clue\b)",
    r"\bi (?:went|moved|graduated|studied|lived|worked|visited|ate|drove|played|watched|saw|met|married|divorced|joined|quit|started|retired|flew|traveled|travelled|grew)\b",
    r"\bi (?:work|live|study|play|drive|teach|volunteer|commute|cook|paint|sing|run|swim|hike|fish|ride)\b",
    r"\b(?:years? old|born in|grew up|used to live|when i was|as a kid|as a child|last (?:night|week|weekend|year|summer|month)|yesterday|this morning|my whole life)\b",
    r"\bmy name is\b|\bi'?m called\b",
    r"\b(?:i|we)(?:'ve| have) (?:been|always|never|just) \w+ed\b",
]
_CLAIM = [re.compile(c, re.I) for c in CLAIMS]


def claims_human(s):
    """True if the reply asserts personal life facts Ovi (a small program) can't truthfully claim."""
    return any(c.search(s) for c in _CLAIM)


def _ok(s, lo=1, hi=32):
    n = len(s.split())
    return lo <= n <= hi and all(ord(c) < 128 for c in s) and not re.search(r"[*_]|\([^)]*\)", s)   # drop roleplay stage directions


def _chain(turns, tag, out):
    for a, b in zip(turns, turns[1:]):
        a, b = detok(a), detok(b)
        if _ok(a, 1, 40) and _ok(b, 2, 32) and not claims_human(b):
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
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    df["utterance_idx"] = pd.to_numeric(df["utterance_idx"], errors="coerce")
    df = df.dropna(subset=["utterance_idx"])
    for cid, g in df.groupby("conv_id", sort=False):
        g = g.sort_values("utterance_idx")
        rows = list(zip(g["utterance"].astype(str), g["utterance_idx"].astype(int), g["context"].astype(str)))
        for (a, _, ctx), (b, idx, _) in zip(rows, rows[1:]):
            if idx % 2 == 0:                    # even turns = the listener answering (odd turns are personal stories)
                _chain([a, b], "emp:" + ctx, out)
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


def splits(include_persona=False):
    """train / test pairs. PersonaChat replies are off by default: they are one imagined person's
    life story ("i teach kids"), which Ovi can't truthfully say. (It has only one file: last 10% is test.)"""
    p = persona() if include_persona else []; cut = int(len(p) * 0.9)
    train = dailydialog("train") + dailydialog("validation") + empathetic("train") + empathetic("valid") + p[:cut]
    test = dailydialog("test") + empathetic("test") + p[cut:]
    return train, test


_WORD = re.compile(r"[a-z']+")
_COMMON = set("i you he she it we they me my your his her our their a an the and or but so if of to in on at for with from as is are was were be am do does did have has had not no yes oh well that this there what who how why when where which can could will would should just very really too also".split())


def generic_only(pairs, min_df=5):
    """Keep replies made of widely used words. Rare words (Kodak, arabians, Chicago...) mark scenario
    details from one particular role-play, which Ovi can't truthfully say."""
    from collections import Counter
    df = Counter()
    for _, a, _ in pairs:
        df.update(set(_WORD.findall(a.lower())))
    return [p for p in pairs if all(df[w] >= min_df for w in _WORD.findall(p[1].lower()) if w not in _COMMON)]


def dedupe(pairs, max_pairs=None):
    pairs = generic_only(pairs)
    seen, out = set(), []
    for q, a, t in pairs:
        k = (q.lower(), a.lower())
        if k in seen:
            continue
        seen.add(k); out.append((q, a, t))
    return out[:max_pairs] if max_pairs else out


def triples(split, kind):
    """(history, reply) with up to 3 previous turns. kind: 'daily' or 'emp'. history[-1] is the message being answered."""
    out = []
    if kind == "daily":
        path = f"data/hf/x/{split}.jsonl"
        if not os.path.exists(path):
            return out
        convs = [[detok(t["value"]) for t in json.loads(l)["conversations"] if t["from"] != "system"] for l in open(path, encoding="utf-8")]
        for c in convs:
            for i in range(2, len(c)):
                h, r = c[max(0, i - 3):i], c[i]
                if _ok(h[-1], 1, 40) and _ok(r, 2, 32) and not claims_human(r):
                    out.append((h, r))
    else:
        path = f"data/hf/ed/empatheticdialogues/{split}.csv"
        df = pd.read_csv(path, on_bad_lines="skip", engine="python")
        df["utterance_idx"] = pd.to_numeric(df["utterance_idx"], errors="coerce"); df = df.dropna(subset=["utterance_idx"])
        for cid, g in df.groupby("conv_id", sort=False):
            c = [detok(u) for u in g.sort_values("utterance_idx")["utterance"].astype(str)]
            for i in range(1, len(c)):                 # listener replies sit at odd list positions (even utterance_idx)
                if i % 2 == 1 and i >= 2:
                    h, r = c[max(0, i - 3):i], c[i]
                    if _ok(h[-1], 1, 40) and _ok(r, 2, 32) and not claims_human(r):
                        out.append((h, r))
    return out
