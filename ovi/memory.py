"""Instant learning: a tiny retrieval memory. Learning = one dict write (microseconds)."""
import json, os, re

STOP = set("a an the is are was were be do does did of to in on at my your you i me it this that and or what who where when how why which".split())


_TOK = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def toks(t):
    return _TOK.findall(t.lower())


def norm(t):
    return " ".join(toks(t))


def keys(text):
    ws = (re.sub(r"'s?$", "", w) for w in toks(text))
    return frozenset(w for w in ws if w and w not in STOP)


class Memory:
    def __init__(self, path="data/memory.jsonl"):
        self.path, self.items, self.exact, self.bad_lines = path, [], {}, 0
        for l in read_lines(path):
            try:
                q, a = json.loads(l)
                if not (isinstance(q, str) and isinstance(a, str)):
                    raise ValueError
            except (ValueError, TypeError):
                self.bad_lines += 1          # corrupted/truncated line: skip and count
                continue
            self.items.append((keys(q), q, a)); self.exact[norm(q)] = a

    def learn(self, q, a, persist=True):
        q, a = clean_text(q), clean_text(a)
        if not norm(q) or not a.strip():
            return
        self.items.append((keys(q), q, a)); self.exact[norm(q)] = a
        if persist:
            append_line(self.path, json.dumps([q, a]))

    def recall(self, q, thresh=0.55):
        if norm(q) in self.exact:
            return self.exact[norm(q)]
        k = keys(q)
        if not k:
            return None
        best, score = None, 0.0
        for kk, _, a in self.items:
            inter = len(k & kk)
            # query coverage, lightly penalised by extra stored words; need >=2 shared words unless query is 1 word
            s = 0.0 if inter < min(2, len(k)) else 0.7 * inter / len(k) + 0.3 * inter / len(k | kk)
            if s > score:
                best, score = a, s
        return best if score >= thresh else None


# ---------- robust file helpers (shared with brain.py) ----------
import contextlib
try:
    import fcntl
except ImportError:                       # non-POSIX: no inter-process lock, everything else still works
    fcntl = None

MAX_TEXT = 2000                           # longest question/answer/message we will process or store


def clean_text(t, limit=MAX_TEXT):
    """Make arbitrary input safe: bytes -> str, lone surrogates / bad bytes replaced, control chars dropped, length capped."""
    if t is None:
        return ""
    if isinstance(t, (bytes, bytearray)):
        t = bytes(t).decode("utf-8", "replace")
    elif not isinstance(t, str):
        t = str(t)
    t = t.encode("utf-8", "replace").decode("utf-8", "replace")
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", t)
    return t[:limit]


@contextlib.contextmanager
def file_lock(path):
    """Exclusive inter-process lock on a sidecar file (path + '.lock')."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    if fcntl is None:
        yield
        return
    with open(path + ".lock", "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def append_line(path, line):
    """Append one record under the lock. If a crash left the file without a trailing newline, start a fresh line first."""
    data = clean_text(line, limit=10 ** 9).replace("\n", " ").encode("utf-8", "replace") + b"\n"
    with file_lock(path):
        with open(path, "ab+") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")
            f.write(data)
            f.flush()


def read_lines(path):
    """Lines of a text log, decoded leniently (never raises on bad bytes). Missing file -> []."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return []
    return [l for l in raw.decode("utf-8", "replace").split("\n") if l.strip()]


def atomic_write(path, lines, backup=False):
    """Replace `path` with `lines` safely: temp file in the same dir, fsync, os.replace. Optionally keep path + '.bak'. Caller holds the lock."""
    import shutil
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    if backup and os.path.exists(path):
        shutil.copy2(path, path + ".bak.tmp")
        os.replace(path + ".bak.tmp", path + ".bak")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
