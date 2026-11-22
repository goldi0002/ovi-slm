"""Ovi Brain: self-updating memory with confidence checks.

Every memory is an Entry(question, answer, source, confidence, hits, misses).
  * Answer confidence = match score x source trust x the entry's own learned confidence.
  * High -> answer.  Medium -> hedged answer.  Low -> "I'm not sure" and ask to be taught.
  * Self-updating: learns the answer to a question it abstained on, takes /right /wrong /correct
    feedback, and replays an append-only event log on startup (nothing is ever lost).
"""
import json, os, re, time
import numpy as np
from dataclasses import dataclass, field
from .memory import norm, keys, toks
from .retrieval import Index
from .seed_chat import PAIRS
from . import dataset
from . import replyrank
from .fast import Ovi as _Base  # tokenizer + n-gram LM (for stories)

TRUST = {"dlg": 0.85, "fact": 0.95, "seed": 1.0, "taught": 1.0, "cb": 0.9, "learned": 0.8, "movie": 0.5}
ANSWER, HEDGE = 0.65, 0.45
QWORD = re.compile(r"^(what|who|whom|whose|where|when|why|how|which|is|are|was|were|do|does|did|can|could|will|would|should|tell|explain|define|translate|give|show|say|list|name)\b", re.I)
STOPW = set("again today tonight yesterday tomorrow always never still already lately recently something anything nothing everything someone anyone people thing things time times this that with have what they them then than there about would could should really very just like your mine been were will from into some more".split())
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
        self._builtin, self._base = builtin, None      # tokenizer + story LM load lazily (only when needed)
        self.entries, self.index, self.exact = [], Index(), {}
        self.key = {}
        self.last = None            # last Answer given
        self.last_q = None
        self.pending = None         # question Ovi abstained on, waiting to be taught
        self.history, self.user, self.entities = [], {}, {}
        self.ranker, self.rvecs, self.rrow, self.row_entry = None, None, {}, {}
        self.use_context = True      # rank chit-chat replies using the conversation so far
        if builtin:
            nd = 0
            if os.path.exists("ovi_rank.npz"):
                self.ranker, self.rvecs = replyrank.load()
            if os.path.exists("ovi_pairs.bin"):
                for kind, q, a in dataset.load():
                    if kind == "e":
                        self.entities[q.lower()] = a
                    else:
                        i = self._add(q, a, {"c": "cb", "m": "movie", "f": "fact", "d": "dlg"}[kind])
                        if kind == "d":
                            self.rrow.setdefault(i, nd); self.row_entry[nd] = i; nd += 1
            for q, a in PAIRS:
                self._add(q, a, "seed")
        self._replay()

    @property
    def base(self):
        if self._base is None and self._builtin:
            self._base = _Base(light=True)
        return self._base

    # ---------- storage ----------
    def _add(self, q, a, src, conf=1.0):
        tk = toks(q); nq = " ".join(tk)
        k = (nq, a)
        if k in self.key:                      # duplicate: strengthen instead of re-adding
            e = self.entries[self.key[k]]
            e.conf = min(1.0, e.conf + 0.1)
            if TRUST[src] > TRUST[e.src]:
                e.src = src
            return self.key[k]
        i = len(self.entries)
        self.entries.append(Entry(q, a, src, conf)); self.key[k] = i
        self.index.add(q, a, tk)
        self.exact.setdefault(nq, []).append(i)
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

    def _rerank(self, msg, r):
        """Among the best question-matches, choose the reply that fits the CONVERSATION (not just the last message)."""
        if self.ranker is None or self.entries[r[0]].src != "dlg":
            return r
        top = self.index.topk(msg, 30)
        if not top:
            return r
        vmax = top[0][1]
        cand = [(i, v / vmax) for i, v in top if self.entries[i].src == "dlg" and i in self.rrow and v >= 0.6 * vmax][:15]
        have = {i for i, _ in cand}
        if True:               # widen the pool: replies CrossPMI finds relevant to the message
            for j in np.argsort(-(self.rvecs @ self.ranker.cross.msg_vec(msg)))[:40]:
                i = int(self.row_entry[j])
                if len(cand) >= 29: break
                if i not in have:
                    cand.append((i, 0.0)); have.add(i)
        if len(cand) < 2:
            return r
        hist = []
        if self.use_context and self.history:
            u, o = self.history[-1]; hist = [u, o]
        hist.append(msg)
        rv = self.rvecs[[self.rrow[i] for i, _ in cand]]
        info = self.ranker.prep_pool([self.entries[i].a for i, _ in cand])
        sc = self.ranker.score(self.ranker.feats(msg, rv, info, hist))
        best = cand[int(np.argmax(sc + 1.0 * np.array([v for _, v in cand])))]
        return best[0], r[1]

    # ---------- confidence ----------
    def conf_of(self, i, score):
        e = self.entries[i]
        return min(1.0, score) * TRUST[e.src] * e.conf

    def recall(self, msg):
        n = norm(msg)
        if n in self.exact:                    # exact match: best-confidence entry wins
            i = max(self.exact[n], key=lambda j: (round(self.conf_of(j, 1.0), 3), j))
            return i, 1.0
        hit = self.index.best(msg, 0.3)
        if not hit:
            return None
        i, sc = hit[2], hit[1]
        qk, ek = keys(msg), keys(self.entries[i].q)
        if qk:
            cover = len(qk & ek) / len(qk)           # how much of what you asked did the memory actually match?
            sc *= cover ** 2                           # words you asked about that the memory lacks are strong evidence of a different question
            if any(w.isdigit() and w not in ek for w in qk):   # a number you mentioned is missing: likely a different question
                sc *= 0.4
        return i, sc

    # ---------- talking ----------
    def ask(self, msg):
        msg = msg.strip()
        # 1) learn from a pending abstention: the next plain statement is the answer
        if self.pending and not msg.endswith("?") and not QWORD.match(msg) and not msg.startswith("/") \
                and not re.match(r"i('m| am| feel| think| want| went)\b", msg, re.I) and norm(msg) not in self.ACK:
            q, self.pending = self.pending, None
            self._add(q, msg, "learned", 0.75)
            self._write({"e": "add", "q": q, "a": msg, "src": "learned", "conf": 0.75})
            self.last_q = q
            self.last = Answer("Thanks, I'll remember that.", 1.0, "learned", "stored as unverified (conf 0.75)")
            return self.last
        self.pending = None
        self.last_q = msg
        if STORY.search(msg) and self.base:
            self.last = Answer(self.story(msg), 0.2, "ngram", "creative mode: generated by the story model, not recalled")
            return self.last
        r = self.recall(msg)
        if r:
            r = self._rerank(msg, r)
        isq = msg.endswith("?") or bool(QWORD.match(msg))
        if r and not isq and self.entries[r[0]].src == "movie":
            r = None
        if r and self.entries[r[0]].src == "dlg" and (norm(msg) in self.ACK or self._feeling(msg)):
            r = None                          # acknowledgements / stated feelings: use the careful rules instead                          # statements: don't answer with a random movie line, use a conversational move
        c = self.conf_of(*r) if r else 0.0
        ath, hth = ANSWER, HEDGE
        if r and self.entries[r[0]].src == "dlg":     # chit-chat replies: answer plainly, no hedging
            ath = hth = 0.5 if isq else 0.4
        if r and c >= hth:
            e = self.entries[r[0]]; e.hits += 1
            text = e.a if c >= ath else f"I'm not fully sure, but: {e.a}"
            self.last = Answer(text, c, e.src, f"matched '{e.q}' (score {r[1]:.2f}, src {e.src}, conf {e.conf:.2f})", r[0])
            return self.last
        if not (msg.endswith("?") or QWORD.match(msg)):
            self.last = Answer(self._backchannel(msg), c, "chat", "no memory matched; conversational move")
            return self.last
        self.pending = msg
        self.last = Answer("I'm not sure about that. What should I have said? (just tell me and I'll learn it)",
                           c, "none", f"best match confidence {c:.2f} < {HEDGE}")
        return self.last

    def story(self, msg, n=70):
        """Short story from the TinyStories n-gram, seeded with the topic ('a story about a dragon')."""
        m = re.search(r"\b(?:about|with|of)\s+(?:an?\s+|the\s+)?([a-z][a-z' -]{1,30})", msg.lower())
        topic = re.sub(r"\b(please|for me|now)\b.*$", "", m.group(1)).strip(" ?.!") if m else ""
        seed = f"Once upon a time, there was a {topic}." if topic else "Once upon a time, there was a little"
        ids = self.base.tok.encode(seed).ids
        out = self.base.lm.sample(ids, n, temp=0.7)
        text = re.sub(r"\s+", " ", self.base.tok.decode(ids + out).replace("\n", " ")).strip()
        cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "), text.rfind('." '))
        return (text[:cut + 1] if cut > len(seed) + 20 else text).strip()

    # ---------- conversation ----------
    GREET = {"hi", "hello", "hey", "good morning", "good evening"}
    FOLLOW = re.compile(r"^(and |what about |how about |but |so |why\b|how so|really\b|then\b)", re.I)
    PRON = re.compile(r"\b(it|that|this|they|them|he|she|there|one)\b", re.I)

    def _find_entity(self, text):
        t = " " + norm(text) + " "
        best = None
        for name, kind in self.entities.items():
            if f" {norm(name)} " in t and (best is None or len(name) > len(best[0])):
                best = (name, kind)
        return best

    def _swap_entity(self, prev, msg):
        """'and italy?' after a question about spain -> the same question about italy."""
        core = re.sub(r"^(and|what about|how about|but|so)\s+", "", msg.strip().rstrip("?!. "), flags=re.I)
        new = self._find_entity(core)
        old = self._find_entity(prev)
        if new and old and new[1] == old[1] and len(norm(core).split()) <= len(norm(new[0]).split()) + 1:
            return re.sub(re.escape(old[0]), new[0], prev, flags=re.I)
        # generic: "what is the capital of france?" + "and japan?" -> replace the question's last words
        cw, pw = core.split(), prev.rstrip("?!. ").split()
        if 1 <= len(cw) <= 2 and len(pw) >= len(cw) + 2 and not QWORD.match(core):
            return " ".join(pw[:-len(cw)] + cw) + "?"
        return None

    MOODS = {
        "sad": ("sad|down|depressed|unhappy|lonely|miserable|upset|crying|hurt", "I'm sorry you're feeling that way{n}. Do you want to tell me what happened?"),
        "tired": ("tired|exhausted|sleepy|worn out", "That sounds draining{n}. Have you been able to rest?"),
        "stress": ("stressed|anxious|worried|nervous|scared|afraid|overwhelmed", "That sounds stressful{n}. What is weighing on you the most?"),
        "angry": ("angry|mad|furious|annoyed|frustrated", "That sounds frustrating{n}. What happened?"),
        "happy": ("happy|excited|great|awesome|wonderful|amazing|proud|glad|good news", "That's wonderful{n}! What made it so good?"),
        "bored": ("bored|boring", "Shall we do something about that{n}? I can tell you a fact, a story, or we can talk about what you like."),
    }
    PAST = re.compile(r"^i (went|ate|saw|watched|played|bought|visited|met|read|made|had|got|started|finished|tried|took)\b", re.I)
    ACK = {"ok", "okay", "cool", "nice", "lol", "haha", "yes", "yeah", "yep", "no", "nope", "sure", "hmm", "wow", "thanks", "thank you", "great", "fine", "alright"}

    def _feeling(self, msg):
        n = norm(msg)
        if not re.match(r"(i|i'm|im|i am|i feel|i got|we)\b", n):
            return False
        return any(re.search(rf"\b({pat})\b", n) for pat, _ in self.MOODS.values())

    def _likes(self):
        i = self.exact.get("what do i like")
        m = re.match(r"You like (.*)\.", self.entries[max(i)].a) if i else None
        return [x for x in re.split(r", | and ", m.group(1)) if x] if m else []

    def _backchannel(self, msg):
        n = norm(msg)
        name = self.user.get("name") or ""
        nm = f", {name}" if name else ""
        # 1) acknowledgements: keep the conversation going using what we know about the user
        if n in self.ACK:
            likes = self._likes()
            if likes:
                return f"Anything else on your mind? We could talk about {likes[len(self.history) % len(likes)]}."
            return "Is there something you'd like to talk about or learn?"
        # 2) feelings
        for kind, (pat, reply) in self.MOODS.items():
            if re.search(rf"\b({pat})\b", n):
                return reply.format(n=nm)
        # 3) a known country/element mentioned: offer a relevant fact, then ask about it
        ent = self._find_entity(msg)
        if ent:
            name_, kind = ent
            r = self.recall(f"what is the capital of {name_}" if kind == "country" else f"what is the atomic number of {name_}")
            fact = self.entries[r[0]].a if r and self.conf_of(*r) >= ANSWER else None
            ask = f"Have you been to {name_}?" if kind == "country" else f"Why are you interested in {name_}?"
            return f"{fact} {ask}" if fact else ask
        # 4) things the user did
        if self.PAST.match(msg):
            return ["How was it?", "How did that go?", "Did you enjoy it?"][len(self.history) % 3]
        # 5) generic, using a topic word and rotating phrasing
        ks = [w for w in re.findall(r"[a-z']+", n) if len(w) > 3 and w not in STOPW]
        topic = ks[-1] if ks else None
        opts = ([f"What made you think about {topic}?", f"Tell me more about {topic}.", "Why do you say that?"]
                if topic else ["Go on, I'm listening.", "Tell me more.", "Interesting. Why do you say that?"])
        return opts[len(self.history) % len(opts)]

    def _backchannel_old(self, msg):
        ks = [w for w in re.findall(r"[a-z']+", msg.lower()) if len(w) > 3 and w not in STOPW]
        topic = ks[-1] if ks else None
        opts = ([f"Tell me more about {topic}.", f"What do you like about {topic}?", "Why do you say that?"]
                if topic else ["Go on, I'm listening.", "Tell me more.", "Interesting. Why do you say that?"])
        return opts[len(self.history) % len(opts)]

    def _profile(self, msg):
        """Learn facts about the user from plain statements. Returns Ovi's reply or None."""
        m = msg.strip().rstrip(".!")
        def cur(q):
            i = self.exact.get(norm(q))
            return self.entries[max(i)].a if i else ""
        r = re.match(r"(?:hi,? |hello,? )?(?:my name is|i am called|call me) ([A-Za-z]+)$", m, re.I)
        if r:
            n = r.group(1).capitalize()
            for q in ("what is my name", "who am i", "do you know my name", "what's my name"):
                self.teach(q, f"Your name is {n}.")
            self.user["name"] = n
            return f"Nice to meet you, {n}!"
        r = re.match(r"i (?:really )?(?:like|love|enjoy) (.+)$", m, re.I)
        if r and " " not in r.group(1)[:0] and len(r.group(1).split()) <= 5:
            thing = r.group(1).strip()
            old = re.match(r"You like (.*)\.", cur("what do i like"))
            items = [x for x in re.split(r", | and ", old.group(1)) if x] if old else []
            if thing.lower() not in [x.lower() for x in items]:
                items.append(thing)
            ans = "You like " + (", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else items[0]) + "."
            for q in ("what do i like", "what are my interests", "what do i enjoy"):
                self.teach(q, ans)
            return f"Got it, you like {thing}."
        r = re.match(r"i (?:live in|am from|come from|'m from) (.+)$", m, re.I)
        if r:
            place = r.group(1).strip()
            for q in ("where do i live", "where am i from", "where do i come from"):
                self.teach(q, f"You are from {place}.")
            return f"{place}! Tell me about it."
        r = re.match(r"my fav(?:ou?rite)? (\w+) is (.+)$", m, re.I)
        if r:
            k, v = r.group(1).lower(), r.group(2).strip()
            self.teach(f"what is my favorite {k}", f"Your favorite {k} is {v}.")
            return f"I'll remember that your favorite {k} is {v}."
        return None

    def converse(self, msg):
        msg = msg.strip()
        a = None
        neg = re.match(r"(?:no|nope|wrong|that'?s wrong|incorrect)\b[,.! ]*(?:it'?s |it is |the answer is |actually )?(.*)$", msg, re.I)
        if self.last and self.last.entry >= 0 and not self.pending and not msg.startswith("/"):
            if neg:                       # plain-language correction: "no, it's Max"
                fix = neg.group(1).strip().rstrip(".")
                reply = self.feedback(False, (fix[0].upper() + fix[1:] + ".") if fix else None)
                a = Answer(reply, 1.0, "feedback", "correction from conversation")
            elif self.entries[self.last.entry].src == "learned":   # no objection = implicit confirmation
                e = self.entries[self.last.entry]
                e.conf = min(1.0, e.conf + 0.1)
                self._write({"e": "set", "q": e.q, "a": e.a, "conf": e.conf})
        if a is None and not self.pending:
            reply = self._profile(msg)
            if reply:
                a = Answer(reply, 1.0, "profile", "learned a fact about you")
        if a is None:
            a = self._converse(msg)
        self.history.append((msg, a.text)); self.history = self.history[-8:]
        return a

    def _converse(self, msg):
        n = norm(msg)
        # follow-up: resolve against the previous user question and keep whichever reading is more confident
        if self.history and not self.pending and (self.FOLLOW.match(msg) or (len(n.split()) <= 3 and n not in self.GREET and n not in self.ACK)):
            prev = next((u for u, _ in reversed(self.history[-3:]) if u.endswith("?") or QWORD.match(u)), None)
            if re.match(r"(why\b|how so)", n) and self.last and self.last.entry >= 0:
                e = self.entries[self.last.entry]
                why = {"taught": "Because you told me so.", "learned": "Because you taught me that earlier.",
                       "seed": "That is how I was built to answer.", "cb": "I learned it from a conversation corpus.",
                       "movie": "I only half remember it from movie dialogue, so I might be wrong.",
                       "fact": "It comes from the facts table I was built with.",
                       "dlg": "That is how people replied in the conversations I learned from."}[e.src]
                self.last_q = msg
                return Answer(why, 0.9, "meta", "explaining the previous answer's source")
            swapped = self._swap_entity(prev, msg) if prev else None
            if swapped:
                r = self.recall(swapped)
                if r and self.conf_of(*r) >= HEDGE:
                    e = self.entries[r[0]]; e.hits += 1; c = self.conf_of(*r)
                    self.last_q = swapped
                    self.last = Answer(e.a if c >= ANSWER else f"I'm not fully sure, but: {e.a}", c, e.src,
                                       f"follow-up: '{prev}' with the topic swapped -> '{swapped}'", r[0])
                    return self.last
            if prev:
                plain = self.recall(msg)
                expanded = self.recall(f"{prev} {msg}")
                cp = self.conf_of(*plain) if plain else 0.0
                ce = self.conf_of(*expanded) if expanded else 0.0
                if plain and self.entries[plain[0]].src == "dlg":
                    cp = 0.0                  # a chit-chat line for 'and japan?' is not a real answer
                if ce >= HEDGE and ce > cp:
                    e = self.entries[expanded[0]]; e.hits += 1
                    self.last_q = f"{prev} {msg}"
                    c = ce
                    self.last = Answer(e.a if c >= ANSWER else f"I'm not fully sure, but: {e.a}", c, e.src,
                                       f"follow-up of '{prev}' matched '{e.q}' (conf {c:.2f})", expanded[0])
                    return self.last
        a = self.ask(msg)
        name = self.user.get("name") or (re.match(r"Your name is (\w+)\.", self.entries[max(self.exact["what is my name"])].a).group(1)
                                         if "what is my name" in self.exact else None)
        if name and n in self.GREET:
            a = Answer(f"Hi {name}! Good to see you. How are you?", 1.0, "profile", "greeted you by name")
            self.last = a
        return a

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
        if self._base is not None:               # only touch the story LM if it was loaded this session
            for e in user:
                self._base.learn_text(f"{e.q} {e.a}")
            from .fast import save
            save(self._base.lm)
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
            a = b.converse(m); print(f"ovi> {a.text}   [{a.conf:.2f}]")
    b.sleep()


if __name__ == "__main__":
    main()
