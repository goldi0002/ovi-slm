"""Build the chat dataset from data/chat (ChatterBot corpus) + data/cornell (movie lines)
into one compact file ovi_pairs.bin (zlib TSV: kind \\t question \\t answer)."""
import glob, re, zlib, collections


def clean(s):
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def chatterbot_pairs():
    pairs = []
    for f in sorted(glob.glob("data/chat/*.yml")):
        conv = []
        for line in open(f, encoding="utf-8"):
            m = re.match(r"^- - (.*)", line)
            if m:
                conv = [m.group(1)]
            else:
                m = re.match(r"^  - (.*)", line)
                if m and conv:
                    conv.append(m.group(1))
                else:
                    continue
            if len(conv) >= 2:
                pairs.append(("c", clean(conv[-2].strip("'\"")), clean(conv[-1].strip("'\""))))
    return [p for p in pairs if p[1] and p[2]]


def cornell_pairs(max_pairs=90000, max_words=22):
    rows = {}
    for line in open("data/cornell/movie_lines.txt", encoding="latin-1"):
        p = line.rstrip("\n").split(" +++$+++ ")
        if len(p) == 5:
            rows[int(p[0][1:])] = (p[1], p[2], clean(p[4]))
    out, seen = [], set()
    for i in sorted(rows):
        if i + 1 not in rows:
            continue
        (u1, m1, a), (u2, m2, b) = rows[i], rows[i + 1]
        if m1 != m2 or u1 == u2 or not a or not b:
            continue
        if not (2 <= len(a.split()) <= max_words and 2 <= len(b.split()) <= max_words):
            continue
        if not all(ord(ch) < 128 for ch in a + b) or a.isupper() or b.isupper():
            continue
        key = b.lower()
        if key in seen:  # avoid the same canned reply (e.g. "I don't know.") dominating
            continue
        seen.add(key)
        out.append(("m", a, b))
    step = max(1, len(out) // max_pairs)
    return out[::step][:max_pairs]


def build(path="ovi_pairs.bin"):
    c, m = chatterbot_pairs(), cornell_pairs()
    tsv = "\n".join("\t".join(p) for p in c + m)
    open(path, "wb").write(zlib.compress(tsv.encode(), 9))
    import os
    print(f"chatterbot pairs: {len(c)}, cornell pairs: {len(m)}, file: {os.path.getsize(path)/1e6:.2f} MB")


def load(path="ovi_pairs.bin"):
    return [tuple(l.split("\t")) for l in zlib.decompress(open(path, "rb").read()).decode().split("\n")]


if __name__ == "__main__":
    build()
