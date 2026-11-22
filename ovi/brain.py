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
from .memory import norm, keys, toks, clean_text, append_line, read_lines, atomic_write, file_lock
from .retrieval import Index
from .seed_chat import PAIRS
from . import dataset
from . import replyrank
from .fast import Ovi as _Base  # tokenizer + n-gram LM (for stories)
from . import chitchat as cc
from .usermem import UserMemory

TRUST = {"dlg": 0.85, "fact": 0.95, "seed": 1.0, "taught": 1.0, "cb": 0.9, "learned": 0.8, "movie": 0.5}
ANSWER, HEDGE = 0.65, 0.45
DLG_Q, DLG_S, DLG_SCORE = 0.7, 0.9, 0.75    # chit-chat line thresholds (question / statement) and minimum match score
QWORD = re.compile(r"^(what|who|whom|whose|where|when|why|how|which|is|are|was|were|do|does|did|can|could|will|would|should|tell|explain|define|translate|give|show|say|list|name)\b", re.I)
STOPW = set("again today tonight yesterday tomorrow always never still already lately recently something anything nothing everything someone anyone people thing things time times this that with have what they them then than there about would could should really very just like your mine been were will from into some more".split())
STORY = re.compile(r"\b(story|poem|write|continue|imagine)\b", re.I)
LOG = "data/brain.jsonl"
RELIABLE = ("taught", "learned", "seed", "fact")
USER_SRC = ("taught", "learned")          # only what the user told Ovi can be superseded or forgotten
NAME_QS = ("what is my name", "who am i", "do you know my name", "what's my name", "do you remember my name")
NAME_ASK = {"what is my name", "what's my name", "whats my name", "who am i", "do you know my name", "do you remember my name",
            "can you remember my name", "remember my name", "tell me my name", "what do you call me", "what did i say my name was"}
NAME_WORD = r"[^\W\d_]+(?:[-'][^\W\d_]+)?"          # one name token, any script (Zoe, Zoë, 李雷)
# words that follow "I'm" / "I am" / "this is" but are states, not names
NOT_NAMES = set("""tired happy sad fine good great ok okay alright well bad sick hungry thirsty bored busy late early sorry sure ready done back here there home new old young lost
confused curious scared afraid nervous anxious worried stressed angry mad upset lonely excited glad proud awesome wonderful amazing exhausted sleepy cold hot warm cool
free fun nice not so very just also still only really quite too going coming trying looking thinking feeling not in on at from with the a an it he she they you me
fed full ill cranky grumpy depressed overwhelmed annoyed frustrated furious miserable unhappy hurt crying single married tall short serious joking kidding wrong right
sleeping eating working studying leaving staying gone one two three yes no maybe nothing something someone anyone everything fan student teacher human robot person guy girl man woman boy kid
bot ai that this what who where when why how wondering hoping afraid unsure positive negative interested""".split())

IMPERATIVE = re.compile(r"^(?:please\s+)?(?:(?:can|could|would) you\s+(?:please\s+)?)?(?:tell|give|show)\s+me\s+(?!about\b)", re.I)
PERSONAL_Q = re.compile(r"^(?:what|who|whom|where|when|which|whose|how (?:old|many|much|long|tall))\b(?!.*\byou(?:r)?\b).*\b(?:my|mine|i|me)\b", re.I)
PERSONAL_NOT = re.compile(r"^(?:how do i|what (?:should|can|could|would) i\b|where (?:can|should) i\b|who (?:can|should) i\b)", re.I)
FUTURE_Q = re.compile(r"\b(?:will|would|going to|gonna|be)\b.*\b(?:tomorrow|tonight|next (?:week|month|year|decade)|in the future|someday|later this \w+|in (?:20[3-9]\d|\d+ (?:days|weeks|months|years)))\b", re.I)
LIVE_Q = re.compile(r"\b(?:stock|stocks|share price|bitcoin|crypto|exchange rate|weather|forecast|news|traffic|lottery|score)\b.*\b(?:today|now|tonight|tomorrow|current|currently|latest|right now|this (?:week|weekend|morning|evening))\b", re.I)
PLACE_QS = ("where do i live", "where am i from", "where do i come from")
LIKE_QS = ("what do i like", "what are my interests", "what do i enjoy")


def parse_event(line):
    """One log line -> validated event dict, or None if it is corrupted/truncated/foreign."""
    try:
        ev = json.loads(line)
    except ValueError:
        return None
    if not isinstance(ev, dict) or not isinstance(ev.get("q"), str):
        return None
    kind = ev.get("e")
    if kind == "add":
        if not isinstance(ev.get("a"), str) or ev.get("src") not in TRUST:
            return None
        if not isinstance(ev.get("conf"), (int, float)) or isinstance(ev.get("conf"), bool):
            return None
    elif kind == "set":
        if not isinstance(ev.get("a"), str) or not isinstance(ev.get("conf"), (int, float)):
            return None
    elif kind == "del":                     # tombstone: forget user entries for q (and answer a, if given)
        if ev.get("a") is not None and not isinstance(ev.get("a"), str):
            return None
    else:
        return None
    return ev


def compact_log(log=LOG, backup=True):
    """Rewrite the event log to its current state (adds with final confidence, live 'set's for built-in entries,
    no superseded/forgotten entries, no corrupt lines). Keeps log + '.bak'. Interrupt-safe: temp file + os.replace,
    under the same lock writers use. Returns {'before': n_lines, 'after': n_lines, 'bad': n_skipped}."""
    with file_lock(log):
        lines = read_lines(log)
        ents, sets, dead, bad = {}, {}, set(), 0         # ents: key -> event dict (insertion ordered)
        for l in lines:
            ev = parse_event(l)
            if ev is None:
                bad += 1; continue
            k = (norm(ev["q"]), ev.get("a"))
            if ev["e"] == "add":
                dead.discard(k)
                if k in ents:
                    o = ents[k]; o["conf"] = min(1.0, o["conf"] + 0.1)
                    if TRUST[ev["src"]] > TRUST[o["src"]]:
                        o["src"] = ev["src"]
                else:
                    ents[k] = {"e": "add", "q": ev["q"], "a": ev["a"], "src": ev["src"], "conf": ev["conf"]}
            elif ev["e"] == "set":
                if k in ents:
                    ents[k]["conf"] = ev["conf"]
                elif k not in dead:
                    sets[k] = {"e": "set", "q": ev["q"], "a": ev["a"], "conf": ev["conf"]}
            else:
                for kk in [kk for kk, o in ents.items() if kk[0] == k[0] and o["src"] in USER_SRC and (k[1] is None or kk[1] == k[1])]:
                    del ents[kk]; dead.add(kk)
        out = [json.dumps(o) for o in ents.values()] + [json.dumps(o) for o in sets.values()]
        atomic_write(log, out, backup=backup)
    return {"before": len(lines), "after": len(out), "bad": bad}


@dataclass
class Entry:
    q: str; a: str; src: str; conf: float = 1.0; hits: int = 0; miss: int = 0
    ts: float = field(default_factory=time.time)
    dead: bool = False


@dataclass
class Answer:
    text: str; conf: float; src: str; why: str; entry: int = -1


class Brain(UserMemory):
    def __init__(self, log=LOG, builtin=True):
        self.log = log
        self._builtin, self._base = builtin, None      # tokenizer + story LM load lazily (only when needed)
        self.entries, self.index, self.exact = [], Index(), {}
        self.key = {}
        self.last = None            # last Answer given
        self.last_q = None
        self.pending = None         # question Ovi abstained on, waiting to be taught
        self.history, self.user, self.entities = [], {}, {}
        self.bad_lines = 0           # corrupted log lines skipped at startup
        self.recent_fact = None      # questions touched by the last profile statement (target of 'forget that')
        self.ranker, self.rvecs, self.rrow, self.row_entry = None, None, {}, {}
        self.use_context = True      # rank chit-chat replies using the conversation so far
        self.used = set()            # backchannel / small-talk replies already given (no repeats)
        self.qctx = None             # (resolved factual question, turn) the next 'and italy?' refers to
        self.last_calc = None        # result of the previous sum ('and times 3?')
        self.focus = None            # (label, awaiting_name): the pet / person the user is talking about
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
        append_line(self.log, json.dumps(ev))

    def _kill(self, i):
        """Retire entry i (superseded or forgotten): it can no longer be matched or answered."""
        e = self.entries[i]
        if e.dead:
            return
        e.dead = True; e.conf = 0.0
        self.index.dead.add(i)
        self.key.pop((norm(e.q), e.a), None)
        lst = self.exact.get(norm(e.q), [])
        if i in lst:
            lst.remove(i)
        if not lst:
            self.exact.pop(norm(e.q), None)

    def _forget_entries(self, q, a=None, persist=True):
        """Tombstone user entries for question q (optionally only answer a). Returns how many were forgotten."""
        n = 0
        for i in list(self.exact.get(norm(q), [])):
            e = self.entries[i]
            if e.src in USER_SRC and (a is None or e.a == a):
                self._kill(i); n += 1
        if n and persist:
            self._write({"e": "del", "q": q, "a": a, "ts": time.time()})
        return n

    def _replay(self):
        self.bad_lines = 0
        for l in read_lines(self.log):
            ev = parse_event(l)
            if ev is None:
                self.bad_lines += 1; continue
            if ev["e"] == "add":
                self._add(ev["q"], ev["a"], ev["src"], ev["conf"])
            elif ev["e"] == "set":
                i = self.key.get((norm(ev["q"]), ev["a"]))
                if i is not None:
                    self.entries[i].conf = ev["conf"]
            else:
                self._forget_entries(ev["q"], ev.get("a"), persist=False)

    def compact(self, backup=True):
        """Rewrite the log to current state, keeping data/brain.jsonl.bak. See compact_log."""
        return compact_log(self.log, backup)

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
        msg = IMPERATIVE.sub("", msg.strip()) or msg      # "tell me the capital of france" -> "the capital of france"
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
        msg = clean_text(msg).strip()
        # 1) learn from a pending abstention: the next plain statement is the answer
        if self.pending and not msg.endswith("?") and not QWORD.match(msg) and not msg.startswith("/") \
                and not re.match(r"i('m| am| feel| think| want| went)\b", msg, re.I) and norm(msg) not in self.ACK \
                and not cc.is_nonanswer(norm(msg)):
            q, self.pending = self.pending, None
            mine = re.match(r"my (.+?) (is|are) (.+)$", msg.rstrip(".!"), re.I)
            if mine and re.match(r"(?:what|who|where|when|which)\b.*\bmy\b", q, re.I):      # "my best friend is Maria" -> "Your best friend is Maria."
                msg = f"Your {mine.group(1)} {mine.group(2)} {mine.group(3)}."
            i = self._add(q, msg, "learned", 0.75)
            self._write({"e": "add", "q": q, "a": msg, "src": "learned", "conf": 0.75})
            self.last_q = q
            self.last = Answer("Thanks, I'll remember that.", 1.0, "learned", "stored as unverified (conf 0.75)", i)
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
        if r and self.entries[r[0]].src in ("dlg", "cb", "movie") and not isq and self._feeling(msg):
            r = None
        if r and self.entries[r[0]].src == "dlg" and (norm(msg) in self.ACK or cc.is_ack(norm(msg)) or self._feeling(msg)):
            r = None                          # acknowledgements / stated feelings: use the careful rules instead                          # statements: don't answer with a random movie line, use a conversational move
        if r and isq and self.entries[r[0]].src == "dlg" and (
                re.search(r"\b(?:you|your|yours|u|ur)\b", msg, re.I) or len(msg.split()) < 3 or (r[1] < 0.99 and len(msg.split()) <= 6 and self.PRON.search(msg))):
            r = None                          # addressed to Ovi, or about an unresolved 'it': a corpus line would be a guess
        c = self.conf_of(*r) if r else 0.0
        unk = self._unknowable(msg, r)
        if unk:
            self.last = unk
            return unk
        ath, hth = ANSWER, HEDGE
        if r and self.entries[r[0]].src == "dlg":     # chit-chat replies: answer plainly, no hedging, but only when really close
            ath = hth = DLG_Q if isq else DLG_S
            if r[1] < DLG_SCORE:
                hth = 9.0
        elif r and self.entries[r[0]].src in ("cb", "movie"):
            hth = ath = ANSWER                        # half-remembered lines are never hedged
        if r and c >= hth:
            e = self.entries[r[0]]; e.hits += 1
            text = e.a if c >= ath else f"I'm not fully sure, but: {e.a}"
            if isq and e.src != "dlg":
                self.qctx = (msg, len(self.history))
            self.last = Answer(text, c, e.src, f"matched '{e.q}' (score {r[1]:.2f}, src {e.src}, conf {e.conf:.2f})", r[0])
            return self.last
        if not (msg.endswith("?") or QWORD.match(msg)):
            self.last = Answer(self._backchannel(msg), c, "chat", "no memory matched; conversational move")
            return self.last
        soft = self._soft_unknown(msg)
        if soft:
            self.last = Answer(soft, c, "chat", "question about Ovi/opinion/context: neutral reply, no teaching")
            return self.last
        self.pending = msg
        self.qctx = (msg, len(self.history))
        self.last = Answer("I'm not sure about that. What should I have said? (just tell me and I'll learn it)",
                           c, "none", f"best match confidence {c:.2f} < {HEDGE}")
        return self.last

    def _unknowable(self, msg, r):
        """Questions no stored answer can honestly address: the future, live data, or the user's own life when they never
        told Ovi. Only fires when nothing reliable (taught / learned / seed / fact) was matched; chit-chat lines don't count."""
        if not (msg.endswith("?") or QWORD.match(msg)):
            return None
        if r and self.entries[r[0]].src in ("taught", "learned", "seed", "fact") and self.conf_of(*r) >= HEDGE:
            return None
        if FUTURE_Q.search(msg) and not re.search(r"\byou\b", msg, re.I):
            return Answer("I don't know, I can't see the future.", 0.0, "none", "question about the future: nothing stored can answer it")
        if LIVE_Q.search(msg):
            return Answer("I don't know, I have no live information like prices, news or weather.", 0.0, "none", "needs live data")
        if PERSONAL_Q.match(msg) and not PERSONAL_NOT.match(msg):
            self.pending = msg
            return Answer("I don't know that about you yet. Tell me, and I'll remember it.", 0.0, "none",
                          "personal question with no stored answer; waiting to be told")
        return None

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
    GREET_RE = re.compile(r"(?:hi|hello|hey|hiya|yo|good (?:morning|evening|afternoon))(?: again| there| ovi| everyone| friend)?$")

    def _is_greet(self, n):
        return n in self.GREET or bool(self.GREET_RE.match(n))

    def _known_name(self):
        i = self.exact.get("what is my name")
        m = re.match(r"Your name is (.+)\.", self.entries[max(i)].a) if i else None
        return m.group(1) if m else self.user.get("name")
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

    LOSS = re.compile(r"\b(?:passed away|passed on|passes away|died|has died|is dead|funeral|lost (?:my|our|a close|her|his) (?:\w+ )?(?:mom|mother|dad|father|grand\w+|friend|dog|cat|pet|brother|sister|husband|wife|son|daughter|baby|uncle|aunt|parents?|cousin|partner|girlfriend|boyfriend|child|kid|loved one)|grieving|bereaved|(?:death|loss) of)\b", re.I)
    MISS = re.compile(r"^(?:i|we)(?:'m| am| do| still| really| just| so| truly| deeply|\s)*\bmiss(?:ed|ing)?\s+(?:my|our|him|her|them|you|home|everyone|everybody|the old|those|being|us|having)\b|\bhomesick", re.I)
    MOODS = {
        "loss": ("passed away", "I'm so sorry for your loss{n}. I'm here for you. Do you want to tell me about them?"),
        "miss": ("miss", "I understand{n}, missing people shows how much they matter to you. Do you want to tell me about them?"),
        "sad": ("sad|down|depressed|unhappy|lonely|alone|miserable|upset|crying|hurt|heartbroken|hopeless|empty", "I'm sorry you're feeling that way{n}. Do you want to tell me what happened?"),
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
        if self.MISS.search(n) or (self.LOSS.search(n) and re.match(r"(?:my|our|i|we|someone|a|his|her|their|she|he)\b", n)):
            return True
        if not re.match(r"(i|i'm|im|i am|i feel|i got|we)\b", n):
            return False
        return any(re.search(rf"\b({pat})\b", n) for k, (pat, _) in self.MOODS.items() if k not in ("loss", "miss"))

    def _likes(self):
        i = self.exact.get("what do i like")
        m = re.match(r"You like (.*)\.", self.entries[max(i)].a) if i else None
        return [x for x in re.split(r", | and ", m.group(1)) if x] if m else []

    def _backchannel(self, msg):
        n = norm(msg)
        name = self.user.get("name") or ""
        nm = f", {name}" if name else ""
        # 1) acknowledgements: keep the conversation going using what we know about the user
        if n in self.ACK or cc.is_ack(n):
            likes = self._likes()
            if likes and len(self.history) % 2 == 0:
                return f"Anything else on your mind? We could talk about {likes[len(self.history) // 2 % len(likes)]}."
            return cc._pick(["Is there something you'd like to talk about or learn?", "I'm here if you want to chat about anything.", "What else is on your mind?"],
                            self.used, "ackr", len(self.history))
        # 2) feelings
        for kind, (pat, reply) in self.MOODS.items():
            if kind == "miss":
                hit = self.MISS.search(n)
            elif kind == "loss":
                hit = self.LOSS.search(n)
            else:
                hit = re.search(rf"\b({pat})\b", n)
            if hit:
                return reply.format(n=nm)
        ev = cc.event_reply(msg, n, name, self.used)
        if ev:
            return ev
        # 3) a known country/element mentioned: offer a relevant fact, then ask about it
        ent = self._find_entity(msg)
        if ent:
            name_, kind = ent
            r = self.recall(f"what is the capital of {name_}" if kind == "country" else f"what is the atomic number of {name_}")
            fact = self.entries[r[0]].a if r and self.conf_of(*r) >= ANSWER else None
            ask = f"Have you been to {name_}?" if kind == "country" else f"Why are you interested in {name_}?"
            return f"{fact} {ask}" if fact else ask
        # 4) things the user did / plans
        if self.PAST.match(msg):
            return cc._pick(["How was it?", "How did that go?", "Did you enjoy it?"], self.used, "past", len(self.history))
        pl = cc.plan_reply(n, self.used, len(self.history))
        if pl:
            return pl
        # 5) generic: context-aware acknowledgement, never a filler about a stopword/adjective/verb
        return cc.backchannel(msg, n, self.used, len(self.history))

    def _backchannel_old(self, msg):
        ks = [w for w in re.findall(r"[a-z']+", msg.lower()) if len(w) > 3 and w not in STOPW]
        topic = ks[-1] if ks else None
        opts = ([f"Tell me more about {topic}.", f"What do you like about {topic}?", "Why do you say that?"]
                if topic else ["Go on, I'm listening.", "Tell me more.", "Interesting. Why do you say that?"])
        return opts[len(self.history) % len(opts)]

    def _profile(self, msg):
        """Learn facts about the user from plain statements. Returns Ovi's reply or None."""
        m = msg.strip().rstrip(".!")
        m = re.sub(r"^(?:(?:no|nope|actually|sorry|oops|wait|oh|well|hmm|hi|hello|hey)\b[,.! ]*)+(?=my |i |i'|call me|this is )", "", m, flags=re.I)   # 'actually my name is X'
        m = re.sub(r"^(?:(?:no|nope|sorry|oops|wait|my bad)\b[,.! ]*)*(?:i meant|i mean)\b[^.!]*[.!]\s*(?=my |i |i'|call me)", "", m, flags=re.I)   # 'sorry, I meant Tom. my name is Tom'
        m = re.sub(r"^((?:my name(?:'s| is)|i am called)) (?:actually|really|in fact) ", r"\1 ", m, flags=re.I)
        self.recent_fact = None
        def cur(q):
            i = self.exact.get(norm(q))
            return self.entries[max(i)].a if i else ""
        r = re.match(rf"(?:my name(?:'s| is)|i am called|call me|this is|i'?m|i am) ({NAME_WORD})$", m, re.I)
        if r:
            raw, lead = r.group(1), m[:r.start(1)].lower()
            weak = bool(re.search(r"(?:this is|i'?m|i am) $", lead))      # "I'm X" could be a feeling: filter
            low = raw.lower()
            ok = not weak or (low not in NOT_NAMES and not (raw[0].islower() and re.search(r"(?:ing|ed|ly)$", low)))
            if ok and norm(m) not in self.ACK:
                n = raw if (not raw.islower() and not raw.isupper()) else raw.capitalize()
                old = re.match(r"Your name is (.+)\.", cur("what is my name"))
                self._set_name(n)
                if old and old.group(1) != n:
                    return f"Got it, I'll call you {n} instead of {old.group(1)}."
                return f"Nice to meet you, {n}!"
        r = re.match(r"i (?:also |really |just )*(?:like|love|enjoy) (.+)$", m, re.I)
        thing = re.sub(r"\s+(?:too|as well|a lot|so much|very much|the most)$", "", r.group(1).strip(), flags=re.I) if r else ""
        if r and thing and len(thing.split()) <= 4 and not re.match(r"(?:how|that|it|when|what|to|being|the way|you|him|her|them|me|us|this|if|whenever|why|who|my (?:life|job)\b)", thing, re.I) \
                and not re.search(r"\b(?:today|tonight|now|but|because|and then)\b", thing, re.I):
            old = re.match(r"You like (.*)\.", cur("what do i like"))
            items = [x for x in re.split(r", | and ", old.group(1)) if x] if old else []
            if thing.lower() not in [x.lower() for x in items]:
                items.append(thing)
            ans = "You like " + (", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else items[0]) + "."
            for q in LIKE_QS:
                self.teach(q, ans, supersede=True)
            self.recent_fact = list(LIKE_QS)
            return f"Got it, you like {thing}."
        mm = self._profile_more(m)
        if mm:
            return mm
        r = re.match(r"i (?:live in|am from|come from|'m from) (.+)$", m, re.I)
        if r and len(r.group(1).split()) <= 4 and not re.search(r"\b(?:today|tonight|now|but|because|and)\b", r.group(1), re.I):
            place = r.group(1).strip()
            self._set_place(place)
            return f"{place}! Tell me about it."
        r = re.match(r"my fav(?:ou?rite)? (\w+) is (.+)$", m, re.I)
        if r:
            k, v = r.group(1).lower(), r.group(2).strip()
            self.teach(f"what is my favorite {k}", f"Your favorite {k} is {v}.", supersede=True)
            self.recent_fact = [f"what is my favorite {k}"]
            return f"I'll remember that your favorite {k} is {v}."
        return None

    FORGET = re.compile(r"^(?:please |ovi,? )*(?:forget|erase|delete|unlearn)\b[,: ]*(?:about |everything about )?(.*)$", re.I)

    def _forget(self, msg):
        """'forget that' / 'forget my name': tombstone the user's own entries. Returns Ovi's reply or None."""
        m = self.FORGET.match(msg.strip().rstrip(".!"))
        if not m:
            return None
        t = norm(m.group(1))
        t = re.sub(r"^(?:what i said about |that i |the fact that )", "", t)
        qs = None
        if t in ("that", "it", "this", "what i said", "what i just said", "what you just said", "the last thing", ""):
            if self.recent_fact:
                qs = list(self.recent_fact)
            elif self.last and self.last.entry >= 0 and not self.entries[self.last.entry].dead \
                    and self.entries[self.last.entry].src in USER_SRC:
                e = self.entries[self.last.entry]
                if self._forget_entries(e.q, e.a):
                    self.last = None
                    return "Okay, I've forgotten that."
            if not qs:
                return "I don't have anything recent of yours to forget." if t != "" else None
        elif t in ("my name", "name", "who i am", "what my name is"):
            qs = list(NAME_QS)
        elif re.match(r"(?:where i (?:live|am from|come from)|my (?:home|city|location|hometown))$", t):
            qs = list(PLACE_QS)
        elif t in ("what i like", "what i enjoy", "my interests", "my likes", "my hobbies"):
            qs = list(LIKE_QS)
        elif self._slots_for(t):
            qs = self._slots_for(t)
        elif (fav := re.match(r"my fav\w* (\w+)$", t)):
            qs = [f"what is my favorite {fav.group(1)}"]
        elif t in ("everything", "everything i told you", "everything about me", "all of it", "me"):
            qs = [e.q for e in self.entries if e.src in USER_SRC and not e.dead]
        if qs is None and t:        # free text: user entries mentioning it
            qs = [e.q for e in self.entries if e.src in USER_SRC and not e.dead and (t in norm(e.q) or t in norm(e.a))]
        n = 0
        for q in dict.fromkeys(qs or []):
            n += self._forget_entries(q)
        if n:
            if any(q in NAME_QS for q in qs):
                self.user.pop("name", None)
            self.last = None; self.recent_fact = None
            return "Okay, I've forgotten that."
        return f"I don't have anything about {t or 'that'} to forget."

    def converse(self, msg):
        msg = clean_text(msg).strip()
        if not msg:
            return Answer("I didn't catch anything. What's on your mind?", 1.0, "chat", "empty input")
        msg = re.sub(r"^(?:(?:ok|okay|so|well|anyway|anyways|hmm|um|oh|alright|btw|but)\b[, ]+)+(?=(?:what|who|where|when|how|why|which|is|are|do|does|can|could|should|tell)\b)", "", msg, flags=re.I) or msg
        a = None
        recent, self.recent_fact = self.recent_fact, None
        if not self.pending and not msg.startswith("/"):
            self.recent_fact = recent             # visible to _forget for exactly one turn
            reply = self._forget(msg)
            self.recent_fact = None
            if reply:
                a = Answer(reply, 1.0, "forget", "tombstoned in the event log")
            else:
                reply = self._profile(msg)
                if reply:
                    a = Answer(reply, 1.0, "profile", "learned a fact about you")
        if a is not None and a.src == "profile" and self.recent_fact:
            i = max(self.exact.get(norm(self.recent_fact[0]), [-1]))
            if i >= 0 and not self.entries[i].dead:       # the next "no, it's X" corrects what was just stated
                self.last = Answer(a.text, 1.0, "profile", "profile fact", i); self.last_q = self.recent_fact[0]
        if a is None:
            self._track_focus(msg)
        neg = self._correction(msg)
        if neg is not None and self.last and not self._real_correction(msg, neg):
            neg = None
        if a is None and self.last and self.last.entry >= 0 and not self.pending and not msg.startswith("/"):
            if neg is not None:           # plain-language correction: "no, it's Max"
                fix = neg.strip().rstrip(".")
                reply = self.feedback(False, fix or None)
                a = Answer(reply, 1.0, "feedback", "correction from conversation")
            elif self.entries[self.last.entry].src == "learned":   # no objection = implicit confirmation
                e = self.entries[self.last.entry]
                e.conf = min(1.0, e.conf + 0.1)
                self._write({"e": "set", "q": e.q, "a": e.a, "conf": e.conf})
        if a is None:
            a = self._multi(msg) or self._converse(msg)
        m_c = re.search(r" is (-?[\d.]+)\.$", a.text) if a.src == "calc" else None
        self.last_calc = m_c.group(1) if m_c else None
        self.history.append((msg, a.text)); self.history = self.history[-8:]
        return a

    CORR_STRONG = re.compile(r"(?:no|nope|nah|wrong|incorrect|not quite|that'?s (?:wrong|not (?:right|correct|true|it|what i said))|that is (?:wrong|not (?:right|correct|true))|you'?re wrong)\b[,.! ]*(.*)$", re.I)
    CORR_WEAK = re.compile(r"(?:(?:actually|sorry|oops|wait|hmm|well|no|nope)\b[,.! ]*)*(?:it'?s|it is|it was|i meant|i mean|make that|should be|it should be|rather)\s+(.*)$", re.I)
    CORR_NOT = re.compile(r"(?:i|you|we|they|he|she|don'?t|do not|please|thanks?|thank|why|what|how|who|where|when|can|could|would|is|are|my|your)\b", re.I)

    def _correction(self, msg):
        """'no, it's red' / 'nope that is wrong' / 'wait, I meant Bergen' -> the corrected value ('' if none given), else None."""
        m = msg.strip()
        for rx in (self.CORR_STRONG, self.CORR_WEAK):
            r = rx.match(m)
            if r:
                fix = re.sub(r"^(?:it'?s|it is|it was|the answer is|actually|i meant|i mean|make that|should be|it should be|rather)\s+", "", r.group(1).strip(), flags=re.I)
                if fix and (self.CORR_NOT.match(fix) or "?" in fix):
                    continue
                return fix
        return None

    CORR_MARK = re.compile(r"^(?:actually|i meant|i mean|make that|should be|it should be|sorry|oops|wait|my bad)\b", re.I)
    CORR_SOFT = set("today tonight so very really super too pretty quite just still always usually".split())

    def _real_correction(self, msg, fix):
        """A bare 'it's X' only corrects a reliable answer to a question, and only when X is short and not a description."""
        if self.CORR_STRONG.match(msg.strip()):
            return True
        if self.last.src not in ("taught", "learned", "seed", "fact", "profile"):
            return False
        qlike = bool(self.last_q) and (self.last_q.endswith("?") or bool(QWORD.match(self.last_q)))
        if not (self.CORR_MARK.match(msg.strip()) or qlike):
            return False
        words = norm(fix).split()
        return 0 < len(words) <= 4 and not (set(words) & self.CORR_SOFT)

    def _multi(self, msg):
        """'who am I and where do I live?' -> answer each part Ovi knows. Needs at least one part with a reliable answer."""
        if self.pending or not (msg.endswith("?") or QWORD.match(msg)):
            return None
        parts = re.split(r"\s*(?:,\s*)?\band\b\s+(?=(?:what|who|where|when|which|how|do|does|did|am|is|are)\b)", msg.rstrip("?. "), flags=re.I)
        if len(parts) < 2:
            return None
        got, miss = [], []
        for p in parts:
            p = p.strip(" ,")
            if norm(p) in NAME_ASK:
                nm = self._known_name()
                if nm:
                    got.append(f"Your name is {nm}.")
                else:
                    miss.append(p)
                continue
            r = self.recall(p + "?")
            if r and self.entries[r[0]].src in ("taught", "learned", "seed", "fact") and self.conf_of(*r) >= ANSWER:
                got.append(self.entries[r[0]].a)
            else:
                miss.append(p)
        if not got:
            return None
        text = " ".join(got) + (f" I'm not sure about the rest ({'; '.join(miss)})." if miss else "")
        self.last_q = msg
        self.last = Answer(text, 1.0, "profile", "answered a compound question part by part")
        return self.last

    def _converse(self, msg):
        n = norm(msg)
        if n in NAME_ASK and not self.pending:
            name = self._known_name()
            self.last_q = msg
            i = self.exact.get("what is my name")
            self.last = Answer(f"Your name is {name}." if name else "I don't know your name yet. What should I call you?",
                               1.0, "profile", "answered from what you told me about your name", max(i) if i and name else -1)
            return self.last
        name = self._known_name()
        if name and self._is_greet(n) and not self.pending:
            self.last = Answer(f"Hi {name}! Good to see you. How are you?", 1.0, "profile", "greeted you by name")
            return self.last
        quick = self._quick(msg, n, name)
        if quick:
            self.pending = None
            self.last_q = msg
            self.last = quick
            return quick
        # follow-up: resolve against the previous user question and keep whichever reading is more confident
        if self.history and not self.pending and (self.FOLLOW.match(msg) or (len(n.split()) <= 3 and not self._is_greet(n) and n not in self.ACK
                                                                           and not cc.is_ack(n) and n.split()[:1] != [] and n.split()[0] not in cc.ACK_WORDS)):
            prev = self.qctx[0] if self.qctx and len(self.history) - self.qctx[1] <= 6 else \
                next((u for u, _ in reversed(self.history[-3:]) if u.endswith("?") or QWORD.match(u)), None)
            if re.match(r"(?:why|how so|why is that|why so|why not|why(?:'s| is) that|why do you say that|why did you say that)$", n) and self.last and self.last.entry >= 0:
                e = self.entries[self.last.entry]
                why = {"taught": "Because you told me so.", "learned": "Because you taught me that earlier.",
                       "seed": "That is how I was built to answer.", "cb": "I learned it from a conversation corpus.",
                       "movie": "I only half remember it from movie dialogue, so I might be wrong.",
                       "fact": "It comes from the facts table I was built with.",
                       "dlg": "It seemed a natural thing to say in that moment, though I may have missed the point."}[e.src]
                self.last_q = msg
                return Answer(why, 0.9, "meta", "explaining the previous answer's source")
            swapped = self._swap_entity(prev, msg) if prev else None
            if swapped:
                r = self.recall(swapped)
                if r and self.conf_of(*r) >= HEDGE and self.entries[r[0]].src in RELIABLE:
                    e = self.entries[r[0]]; e.hits += 1; c = self.conf_of(*r)
                    self.last_q = swapped; self.qctx = (swapped, len(self.history))
                    self.last = Answer(e.a if c >= ANSWER else f"I'm not fully sure, but: {e.a}", c, e.src,
                                       f"follow-up: '{prev}' with the topic swapped -> '{swapped}'", r[0])
                    return self.last
                if self._find_entity(swapped) and not (r and self.entries[r[0]].src in ("dlg", "cb", "movie") and self.conf_of(*r) >= HEDGE):
                    self.pending = swapped; self.last_q = swapped; self.qctx = (swapped, len(self.history))
                    self.last = Answer("I'm not sure about that. What should I have said? (just tell me and I'll learn it)", 0.0, "none",
                                       f"follow-up '{swapped}' has no stored answer")
                    return self.last
            if prev:
                plain = self.recall(msg)
                expanded = self.recall(f"{prev} {msg}")
                cp = self.conf_of(*plain) if plain else 0.0
                ce = self.conf_of(*expanded) if expanded else 0.0
                if plain and self.entries[plain[0]].src == "dlg":
                    cp = 0.0                  # a chit-chat line for 'and japan?' is not a real answer
                if ce >= HEDGE and ce > cp and self.entries[expanded[0]].src in RELIABLE:
                    e = self.entries[expanded[0]]; e.hits += 1
                    self.last_q = f"{prev} {msg}"; self.qctx = (self.last_q, len(self.history))
                    c = ce
                    self.last = Answer(e.a if c >= ANSWER else f"I'm not fully sure, but: {e.a}", c, e.src,
                                       f"follow-up of '{prev}' matched '{e.q}' (conf {c:.2f})", expanded[0])
                    return self.last
        return self.ask(msg)

    def _quick(self, msg, n, name):
        """Rule-based skills tried before any retrieval: arithmetic, profile recall, small talk, advice."""
        c = cc.calc(msg)
        cont = re.sub(r"^(?:(?:and|then|now|ok|so|what about)\s+)+", "", msg.lower().strip())
        if not c and self.last_calc is not None and re.match(r"(?:times|plus|minus|divided by|multiplied by|x|\+|-|\*|/)\s*\d", cont):
            c = cc.calc(f"{self.last_calc} {cont}")
        if c:
            return Answer(c, 1.0, "calc", "arithmetic")
        t = self._recall_profile(n)
        if t:
            return Answer(t, 1.0, "profile", "answered from what you told me")
        turn = len(self.history)
        prev = self.history[-1][0] if self.history else ""
        t = cc.smalltalk(n, prev, self.used, turn, name or "")
        if t is None and (msg.rstrip().endswith("?") or QWORD.match(msg)):
            t = cc.advice(msg, n, self.used, turn)
        if t:
            return Answer(t, 1.0, "chat", "small talk rule")
        return None

    SOFT_CTX = re.compile(r"^(?:is|are|does|do|was|were|can|will|would|could|did|has|have) (?:it|that|this|they|there|he|she)\b|\b(?:better|best|worth|nice|good|bad)\b")

    def _soft_unknown(self, msg):
        """Questions about Ovi, opinions or 'it' with no referent: a neutral reply instead of asking to be taught."""
        n = norm(msg)
        if re.search(r"\b(?:you|your|yours|u|ur)\b", n):
            return "I'm not sure about that, I'm only a tiny program. What made you ask?"
        if self.SOFT_CTX.search(n):
            return "I can't really say. What do you think?"
        return None

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
                correction = clean_text(correction).strip()
                m = re.match(r"(Your [^.]*? (?:is|are)) .*\.$", e.a)
                if norm(e.q) in NAME_QS and e.src in USER_SRC:      # 'no, it's Max' after 'what is my name' fixes every name question
                    n = re.sub(r"^(?:my name is |it'?s |i am )", "", correction.rstrip("."), flags=re.I).strip().split(" ")[0].capitalize()
                    if n:
                        self._set_name(n)
                        return msg + f" I updated your name to {n}."
                pl = re.match(r"You are from ", e.a)
                if pl and e.src in USER_SRC and correction.rstrip("."):
                    place = re.sub(r"^(?:i (?:live in|am from|come from)|in|from)\s+", "", correction.rstrip("."), flags=re.I)
                    self._set_place(place)
                    return msg + f" I updated where you live to {place}."
                if correction.rstrip(".") and m and len(correction.split()) <= 6:
                    correction = f"{m.group(1)} {correction.rstrip('.')}."
                elif correction and correction[0].islower():
                    correction = correction[0].upper() + correction[1:]
                if correction and correction[-1] not in ".!?":
                    correction += "."
                if e.src in USER_SRC:                  # supersede: the wrong user-taught answer is retired, not kept beside the fix
                    self._forget_entries(e.q, e.a)
                self._add(self.last_q, correction, "taught", 1.0)
                self._write({"e": "add", "q": self.last_q, "a": correction, "src": "taught", "conf": 1.0})
                msg += " I learned the corrected answer."
        if not e.dead:
            self._write({"e": "set", "q": e.q, "a": e.a, "conf": e.conf})
        return msg

    def _set_place(self, place):
        for q in PLACE_QS:
            self.teach(q, f"You are from {place}.", supersede=True)
        self.recent_fact = list(PLACE_QS)

    def _set_name(self, n):
        for q in NAME_QS:
            self.teach(q, f"Your name is {n}.", supersede=True)
        self.user["name"] = n
        self.recent_fact = list(NAME_QS)

    def teach(self, q, a, supersede=False):
        """Store a taught fact. supersede=True retires the user's older answers to the same question (a correction)."""
        q, a = clean_text(q).strip(), clean_text(a).strip()
        if not norm(q) or not a:
            return
        if supersede:
            for i in list(self.exact.get(norm(q), [])):
                e = self.entries[i]
                if e.src in USER_SRC and e.a != a:
                    self._forget_entries(q, e.a)
        self._add(q, a, "taught", 1.0)
        self._write({"e": "add", "q": q, "a": a, "src": "taught", "conf": 1.0})

    def sleep(self):
        """Consolidate: bake user knowledge into the n-gram LM, compact the log, save."""
        user = [e for e in self.entries if e.src in USER_SRC and e.conf >= 0.3 and not e.dead]
        if self._base is not None:               # only touch the story LM if it was loaded this session
            for e in user:
                self._base.learn_text(f"{e.q} {e.a}")
            from .fast import save
            save(self._base.lm)
        lines = [json.dumps({"e": "add", "q": e.q, "a": e.a, "src": e.src, "conf": e.conf}) for e in user]
        lines += [json.dumps({"e": "set", "q": e.q, "a": e.a, "conf": e.conf})
                  for e in self.entries if e.src in ("seed", "cb", "movie") and e.conf != 1.0]
        with file_lock(self.log):
            atomic_write(self.log, lines)
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
