"""Ovi Brain: self-updating memory with confidence checks.

Every memory is an Entry(question, answer, source, confidence, hits, misses).
  * Answer confidence = match score x source trust x the entry's own learned confidence.
  * High -> answer.  Medium -> hedged answer.  Low -> "I'm not sure" and ask to be taught.
  * Self-updating: learns the answer to a question it abstained on, takes /right /wrong /correct
    feedback, and replays an append-only event log on startup (nothing is ever lost).
"""
import json, os, re, time
from dataclasses import dataclass, field
from .memory import norm, keys
from .retrieval import Index
from .seed_chat import PAIRS
from . import dataset
from .fast import Ovi as _Base  # tokenizer + n-gram LM (for stories)

TRUST = {"seed": 1.0, "taught": 1.0, "cb": 0.9, "learned": 0.8, "movie": 0.5}
ANSWER, HEDGE = 0.65, 0.45
QWORD = re.compile(r"^(what|who|whom|whose|where|when|why|how|which|is|are|was|were|do|does|did|can|could|will|would|should|tell|explain|define|translate|give|show|say|list|name)\b", re.I)
STORY = re.compile(r"\b(story|poem|write|continue|imagine)\b", re.I)
LOG = "data/brain.jsonl"


@dataclass
class Entry:
    q: str; a: str; src: str; conf: float = 1.0; hits: int = 0; miss: int = 0
    ts: float = field(default_factory=time.time)


@dataclass
class Answer:
    text: str; conf: float; src: str; why: str; entry: int = -1


class Brain:
    def __init__(self, log=LOG, builtin=True):
        self.log = log
        self.base = _Base() if builtin else None
        self.entries, self.index, self.exact = [], Index(), {}
        self.key = {}
        self.last = None            # last Answer given
        self.last_q = None
        self.pending = None         # question Ovi abstained on, waiting to be taught
        if builtin:
            if os.path.exists("ovi_pairs.bin"):
                for kind, q, a in dataset.load():
                    self._add(q, a, "cb" if kind == "c" else "movie")
            for q, a in PAIRS:
                self._add(q, a, "seed")
        self._replay()

    # ---------- storage ----------
    def _add(self, q, a, src, conf=1.0):
        k = (norm(q), a)
        if k in self.key:                      # duplicate: strengthen instead of re-adding
            e = self.entries[self.key[k]]
            e.conf = min(1.0, e.conf + 0.1)
            if TRUST[src] > TRUST[e.src]:
                e.src = src
            return self.key[k]
        i = len(self.entries)
        self.entries.append(Entry(q, a, src, conf)); self.key[k] = i
        self.index.add(q, a)
        self.exact.setdefault(norm(q), []).append(i)
        return i

    def _write(self, ev):
        os.makedirs(os.path.dirname(self.log), exist_ok=True)
        with open(self.log, "a") as f:
            f.write(json.dumps(ev) + "\n")

    def _replay(self):
        if not os.path.exists(self.log):
            return
        for l in open(self.log):
            ev = json.loads(l)
            if ev["e"] == "add":
                self._add(ev["q"], ev["a"], ev["src"], ev["conf"])
            elif ev["e"] == "set":
                i = self.key.get((norm(ev["q"]), ev["a"]))
                if i is not None:
                    self.entries[i].conf = ev["conf"]

    # ---------- confidence ----------
    def conf_of(self, i, score):
        e = self.entries[i]
        return min(1.0, score) * TRUST[e.src] * e.conf

    def recall(self, msg):
        n = norm(msg)
        if n in self.exact:                    # exact match: best-confidence entry wins
            i = max(self.exact[n], key=lambda j: self.conf_of(j, 1.0))
            return i, 1.0
        hit = self.index.best(msg, 0.3)
        if not hit:
            return None
        i, sc = hit[2], hit[1]
        qk, ek = keys(msg), keys(self.entries[i].q)
        if qk:
            cover = len(qk & ek) / len(qk)           # how much of what you asked did the memory actually match?
            sc *= 0.5 + 0.5 * cover
            if any(w.isdigit() and w not in ek for w in qk):   # a number you mentioned is missing: likely a different question
                sc *= 0.4
        return i, sc

    # ---------- talking ----------
    def ask(self, msg):
        msg = msg.strip()
        # 1) learn from a pending abstention: the next plain statement is the answer
        if self.pending and not msg.endswith("?") and not QWORD.match(msg) and not msg.startswith("/"):
            q, self.pending = self.pending, None
            self._add(q, msg, "learned", 0.75)
            self._write({"e": "add", "q": q, "a": msg, "src": "learned", "conf": 0.75})
            self.last_q = q
            self.last = Answer("Thanks, I'll remember that.", 1.0, "learned", "stored as unverified (conf 0.75)")
            return self.last
        self.pending = None
        self.last_q = msg
        if STORY.search(msg) and self.base:
            self.last = Answer(self.base.say(msg + " "), 0.2, "ngram", "creative mode: generated, not recalled")
            return self.last
        r = self.recall(msg)
        c = self.conf_of(*r) if r else 0.0
        if r and c >= HEDGE:
            e = self.entries[r[0]]; e.hits += 1
            text = e.a if c >= ANSWER else f"I'm not fully sure, but: {e.a}"
            self.last = Answer(text, c, e.src, f"matched '{e.q}' (score {r[1]:.2f}, src {e.src}, conf {e.conf:.2f})", r[0])
            return self.last
        self.pending = msg
        self.last = Answer("I'm not sure about that. What should I have said? (just tell me and I'll learn it)",
                           c, "none", f"best match confidence {c:.2f} < {HEDGE}")
        return self.last

    # ---------- feedback ----------
    def feedback(self, good, correction=None):
        if not self.last or self.last.entry < 0:
            return "Nothing to give feedback on."
        e = self.entries[self.last.entry]
        if good:
            e.conf = min(1.0, e.conf + 0.15)
            msg = f"Thanks! Confidence in that answer is now {e.conf:.2f}."
        else:
            e.conf *= 0.5; e.miss += 1
            msg = f"Sorry about that. I lowered my confidence to {e.conf:.2f}."
            if correction:
                self._add(self.last_q, correction, "taught", 1.0)
                self._write({"e": "add", "q": self.last_q, "a": correction, "src": "taught", "conf": 1.0})
                msg += " I learned the corrected answer."
        self._write({"e": "set", "q": e.q, "a": e.a, "conf": e.conf})
        return msg

    def teach(self, q, a):
        self._add(q, a, "taught", 1.0)
        self._write({"e": "add", "q": q, "a": a, "src": "taught", "conf": 1.0})

    def sleep(self):
        """Consolidate: bake user knowledge into the n-gram LM, compact the log, save."""
        user = [e for e in self.entries if e.src in ("taught", "learned") and e.conf >= 0.3]
        if self.base:
            for e in user:
                self.base.learn_text(f"{e.q} {e.a}")
            from .fast import save
            save(self.base.lm)
        with open(self.log, "w") as f:
            for e in user:
                f.write(json.dumps({"e": "add", "q": e.q, "a": e.a, "src": e.src, "conf": e.conf}) + "\n")
            for e in self.entries:
                if e.src in ("seed", "cb", "movie") and e.conf != 1.0:
                    f.write(json.dumps({"e": "set", "q": e.q, "a": e.a, "conf": e.conf}) + "\n")
        return f"Consolidated {len(user)} facts into long-term memory."

    def stats(self):
        from collections import Counter
        c = Counter(e.src for e in self.entries)
        return " | ".join(f"{k}: {v}" for k, v in c.items()) + f" | total {len(self.entries)}"


def main():
    b = Brain()
    print("Ovi Brain awake. /teach q => a, /right, /wrong [correct answer], /why, /sleep, /stats, /quit")
    while True:
        try:
            m = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not m:
            continue
        if m in ("/quit", "/exit"):
            break
        if m.startswith("/teach ") and "=>" in m:
            q, a = [x.strip() for x in m[7:].split("=>", 1)]
            b.teach(q, a); print("ovi> Learned it!")
        elif m == "/right":
            print("ovi>", b.feedback(True))
        elif m.startswith("/wrong"):
            print("ovi>", b.feedback(False, m[6:].strip() or None))
        elif m == "/why":
            print("ovi>", f"{b.last.why} | confidence {b.last.conf:.2f}" if b.last else "No answer yet.")
        elif m == "/sleep":
            print("ovi>", b.sleep())
        elif m == "/stats":
            print("ovi>", b.stats())
        else:
            a = b.ask(m); print(f"ovi> {a.text}   [{a.conf:.2f}]")
    b.sleep()


if __name__ == "__main__":
    main()
