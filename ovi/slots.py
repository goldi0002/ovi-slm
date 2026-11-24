"""General attribute/value memory of what the user says about themselves (mixin for Brain).

Facts are stored as ordinary user ('taught') entries under internal 'zzfact <kind> ...' questions, with a ready-to-say
second-person sentence as the answer, so they persist in the event log (data/brain.jsonl), supersede by key and are
tombstoned by 'forget ...' exactly like every other user entry.  Kinds (the key after 'zzfact'):
  my <attr>             "my car is red"                     -> "Your car is red."
  age                   "I'm 25"                            -> "You are 25 years old."
  do <verb> <rest>      "I play guitar", "I run every morning"
  dur <verb>            "been playing for ten years"
  going                 "I'm going to Japan next year"
  rel <label> <verb>    "my mom makes the best lasagna", "she lives in Naples" (pronoun -> focused person/pet)
  desc <label> <text>   "she's black with white paws", "he's a beagle"
  rel <label> age       "she's 87"
Questions are answered by fuzzy matching on the slot (lemmatised verb, head noun of the attribute)."""
import re
from .memory import norm
from . import chitchat as cc
from .usermem import UserMemory

_BASES = ("play run swim cycle bike read write paint draw sing dance cook bake garden hike climb ski surf skate study practice train exercise meditate volunteer drive collect "
          "knit sew code fish camp travel teach coach jog lift learn make live work own watch").split()
_IRREG = {"ran": "run", "swam": "swim", "wrote": "write", "written": "write", "drove": "drive", "driven": "drive", "taught": "teach", "made": "make", "read": "read"}
_FORM = {}
for _b in _BASES:
    for _f in (_b, _b + "s", _b + "es", _b + "ing", _b + "ed", _b + "d", _b + _b[-1] + "ing", _b + _b[-1] + "ed", _b[:-1] + "ing", _b[:-1] + "ies", _b[:-1] + "ied"):
        _FORM.setdefault(_f, _b)
_FORM.update(_IRREG)
_HAB = "|".join(sorted(_BASES[:_BASES.index("live")], key=len, reverse=True))     # habitual activity verbs ('I play', 'I run')
_PROG = "|".join(sorted(f"{b}ing" if not b.endswith("e") else f"{b[:-1]}ing" for b in _BASES) + ["running", "swimming", "cooking"])


def lem(w):
    return _FORM.get(w.lower(), w.lower())


_TIME_END = re.compile(r"\b(?:today|tonight|yesterday|tomorrow|now|right now|later|soon|this (?:morning|evening|afternoon|week|weekend))$|\b(?:because|but|so|when|if|and then|then|although|since i)\b|[,;]")
_EPHEMERAL = set("day week night morning evening afternoon weekend year month life mind head back stomach knees knee feet legs mood boss team plan idea problem trouble fault turn".split())
_MOODISH = re.compile(r"\b(?:tired|sad|happy|angry|mad|upset|stressed|anxious|worried|scared|sick|hungry|bored|annoyed|excited|nervous|lonely|exhausted|glad|proud|jealous)\b", re.I)
_TEACH_CUE = re.compile(r"\b(?:teach you|let me teach|lemme teach|i(?:'ll| will| want to| wanna|'d like to) (?:teach|tell) you|did you know|fun fact|fyi|for your information|here(?:'s| is) (?:a|one) fact)\b", re.I)
_TEACH_ONLY = re.compile(r"^(?:(?:hey|hi|ok|okay|so|well|ovi)[, ]+)*(?:ovi[, ]+)?(?:(?:i|we) (?:want to|wanna|'?d like to|will|'?ll|can|could) |let me |lemme |can i |could i )?teach (?:you )?(?:something|a thing|a fact|you something)?(?: new)?(?: ovi)?$", re.I)
_SUPERL = re.compile(r"\b(?:\w+est|capital|president|currency|population|inventor|founder|author|language)\b", re.I)


class SlotMemory(UserMemory):
    @property
    def focus(self):
        """The pet / person being talked about; every assignment also stamps the turn so 'she' goes stale after a while."""
        return getattr(self, "_focus", None)

    @focus.setter
    def focus(self, v):
        self._focus = v
        self.focus_turn = len(getattr(self, "history", ()))

    # ---------- store ----------
    def _fq(self, key):
        return f"zzfact {key}"

    def _fset(self, key, sentence, supersede=True):
        self.teach(self._fq(key), sentence, supersede=supersede)
        self.recent_fact = [self._fq(key)]

    def _fget(self, prefix):
        """[(key words after 'zzfact ', sentence)] for live facts whose key starts with prefix (newest value per key)."""
        out, pre = [], norm(self._fq(prefix))
        for nq in getattr(self, "factq", ()):
            if nq.startswith(pre):
                i = [j for j in self.exact.get(nq, []) if not self.entries[j].dead]
                if i:
                    out.append((nq[len("zzfact "):], self.entries[max(i)].a))
        return out

    def _fresh_focus(self):
        f = getattr(self, "focus", None)
        return f[0] if f and len(self.history) - getattr(self, "focus_turn", -99) <= 6 else None

    # ---------- learning ----------
    def _profile_slots(self, m):
        """Statements about the user -> slot facts. Returns Ovi's reply or None (nothing stored / not a clear fact)."""
        if m.endswith("?"):
            return None
        if _TEACH_CUE.search(m):
            self.teach_mode = len(self.history)
            if _TEACH_ONLY.match(m):
                return "Sure, go ahead. Tell me something, like \"the capital of Spain is Madrid\", and I'll remember it."
        r = re.match(r"i(?:'m| am) (\d{1,3})(?: years? old| yo| y/?o)?$|i(?:'m| am) turn(?:ing|ed) (\d{1,3})$|i turned (\d{1,3})$|my age is (\d{1,3})$", m, re.I)
        if r:
            age = int(next(g for g in r.groups() if g))
            if 1 <= age <= 120:
                self._fset("age", f"You are {age} years old.")
                return f"Got it, you're {age}."
        r = re.match(rf"(?:and |so |also )?i (?:also |usually |often |always |still |just |really )*({_HAB}|plays?) (.+)$", m, re.I)
        if r and not _TIME_END.search(r.group(2).lower()) and len(r.group(2).split()) <= 6 and r.group(2).split()[0].lower() not in (cc._ADJ | {"late", "early", "out", "away", "home", "off", "up", "down", "again", "around", "together", "alone"}) and not re.match(r"(?:with|against|to|it|that|this|him|her|them|you|me|us|when|if|how|what|why)\b", r.group(2), re.I):
            verb, rest = lem(r.group(1)), re.sub(r"^(?:the |a |an )", "", r.group(2).strip(), flags=re.I)
            self.last_do = (verb, len(self.history))
            self._fset(f"do {verb} {norm(rest)}", f"You {r.group(1).lower()} {r.group(2).strip()}.", supersede=False)
            if verb == "play" and not re.search(r"\b(?:every|each|on|daily|weekly|often)\b", rest, re.I):
                return f"Got it, you play {rest}. How long have you been playing?"
            return f"Got it, you {r.group(1).lower()} {r.group(2).strip()}."
        r = re.match(rf"(?:i(?:'ve| have)? )?(?:(?:been|have been|am|'m) )?({_PROG}|\w+ed) (?:it |that |there |them |this )?(?:for|since) (.{{2,25}}?)(?: now| already)?$", m, re.I)
        if r and (lem(r.group(1)) in _BASES):
            verb = lem(r.group(1))
            self._fset(f"dur {verb}", f"You have been {r.group(1) if r.group(1).endswith('ing') else verb + 'ing'} {'for' if not r.group(2).lower().startswith('since') else ''} {r.group(2)}.".replace("  ", " "))
            return f"{r.group(2).strip().capitalize()}, that's a good while. Do you still enjoy it?"
        r = re.match(r"i(?:'m| am) (?:going|travell?ing|flying|moving|heading|off) to ([A-Za-z][\w ]{1,30})$", m, re.I)
        if r and not re.match(r"(?:the |a |an |be |bed|work|school|sleep|do|get|go|see|have|try|start|stop|my |your )", r.group(1), re.I):
            dest = r.group(1).strip()
            self._fset("going", f"You are going to {dest}.")
            return None            # keep the natural 'sounds like a plan' reply; the fact is stored silently
        # relations: "my mom makes the best lasagna", "my uncle plays for a local team"
        r = re.match(rf"my ({cc.RELS}) ([a-z]+s) (.+)$", m, re.I)
        if r and r.group(2).lower() not in ("is", "was", "has", "does", "always", "never", "sometimes", "usually", "really", "just", "also", "often", "keeps", "makes me", "gets", "lets") \
                and len(r.group(3).split()) <= 9 and not _TIME_END.search(r.group(3).lower()):
            lab, verb = cc.singular(r.group(1)) if r.group(1).lower() not in cc.BREEDS else cc.BREEDS[r.group(1).lower()], lem(r.group(2))
            self.focus, self.focus_turn = (lab, False), len(self.history)
            if not re.match(r"me\b", r.group(3), re.I):
                self._fset(f"rel {lab} {verb}", f"Your {lab} {r.group(2).lower()} {r.group(3).strip()}.")
            return None
        # pronoun statements about the person / pet we are talking about
        lab = self._fresh_focus()
        if lab and not _MOODISH.search(m):
            r = re.match(r"(?:she|he|it|they)(?:'s| is| was| are)? (\d{1,3})(?: years old| yo)?$", m, re.I)
            if r and 0 < int(r.group(1)) < 120:
                self._fset(f"rel {lab} age", f"Your {lab} is {r.group(1)}.")
                return f"Got it, your {lab} is {r.group(1)}."
            r = re.match(r"(?:she|he|it)\s+(?!is\b|was\b|has\b|does\b)([a-z]+s) (.+)$", m, re.I)
            if r and not _TIME_END.search(r.group(2).lower()) and len(r.group(2).split()) <= 8 and lem(r.group(1)) != r.group(1).lower():
                self._fset(f"rel {lab} {lem(r.group(1))}", f"Your {lab} {r.group(1).lower()} {r.group(2).strip()}.")
                return f"Got it, your {lab} {r.group(1).lower()} {r.group(2).strip()}."
            r = re.match(r"(?:she|he|it)(?:'s| is| was) ((?:an? )?[a-z][a-z ]{1,40})$", m, re.I)
            if r and not re.match(r"(?:so|very|really|too|just|not|kind|sort|pretty|always|never|going|being|the|my|your|a (?:bit|little|lot)|trying|getting|doing)\b", r.group(1), re.I) \
                    and len(r.group(1).split()) <= 6 and r.group(1).split()[0] not in cc._ADJ.difference({"big", "small", "little", "old", "young", "new"}):
                d = r.group(1).strip()
                self._fset(f"desc {lab} {norm(d)}", f"Your {lab} is {d}.", supersede=False)
                return None
        r = re.match(r"my ([a-z]+(?: [a-z]+){0,2}?)(?:'s)? (?:is|are|was|were) (.+)$", m, re.I)
        if r:
            x, v = r.group(1).lower(), r.group(2).strip()
            xs = x.split()
            if (not (set(xs) & _EPHEMERAL) and xs[0] not in ("fav", "favorite", "favourite", "name") and len(v.split()) <= 6
                    and not re.match(r"(?:so|very|really|too|just|not|kind|sort|pretty|always|never|still|going|being|like|about|trying|getting)\b", v, re.I)
                    and v.split()[0].lower() not in cc._ADJ and not re.match(r"(?:\w+ing|an? \w+ing)\b", v, re.I) and not _TIME_END.search(v.lower()) and not _MOODISH.search(v)):
                x = re.sub(r"'s$", "", x)
                self._fset(f"my {norm(x)}", f"Your {x} is {v}.")
                return f"Got it, your {x} is {v}."
        r = re.match(r"(?:the |a |an )?([a-z][a-z' -]{2,50}?) (?:is|are|was) (.+)$", m, re.I)
        if r and not re.match(r"(?:i|it|that|this|he|she|they|we|you|there|my|your|what|who|how|why)\b", r.group(1), re.I):
            tm = len(self.history) - getattr(self, "teach_mode", -99) <= 4
            np_, val = r.group(1).strip(), r.group(2).strip()
            factual = m.lower().startswith("the ") and _SUPERL.search(np_) and (val[:1].isupper() or val[:1].isdigit()) and len(np_.split()) <= 8
            if (tm or factual) and len(val.split()) <= 8 and not _MOODISH.search(val):
                if m.lower().startswith("the ") or not m[:1].isupper():
                    q, a = f"what is the {np_}" if not np_.lower().startswith("the ") else f"what is {np_}", f"The {np_[4:] if np_.lower().startswith('the ') else np_} is {val}."
                else:
                    q, a = f"what is {np_}", f"{np_} is {val}."
                    mo = re.match(r"(?:the )?([a-z ]+) of ([A-Z][A-Za-z ]+)$", val)
                    if mo:
                        q, a = f"what is the {mo.group(1)} of {mo.group(2)}", f"The {mo.group(1)} of {mo.group(2)} is {np_}."
                self.teach(q, a, supersede=True)
                alias = re.sub(r" (?:in the (?:world|universe|solar system)|on earth|ever|in history)$", "", q)
                if alias != q:
                    self.teach(alias, a, supersede=True)
                self.recent_fact = [q] + ([alias] if alias != q else [])
                return f"Got it, {a[0].lower() + a[1:]}"
        return None

    # ---------- recalling ----------
    @staticmethod
    def _tidy(n):
        n = re.sub(r"\b(?:what's|whats|what s)\b", "what is", n)
        n = re.sub(r"\b(?:who's|whos|who s)\b", "who is", n)
        n = re.sub(r"\b(?:where's|wheres)\b", "where is", n)
        n = re.sub(r"(?: again| exactly| now| currently| please| ovi| right now| by the way| anyway)+$", "", n.strip())
        return n

    def _attr_match(self, x):
        """Stored 'my <attr>' fact best matching the asked attribute words x (head noun must agree)."""
        want = [cc.singular(w) for w in x.split() if w not in ("favorite", "current", "main")]
        best = None
        for key, sent in self._fget("my "):
            have = [cc.singular(w) for w in key.split()[1:]]
            if not have or not want or have[-1] != want[-1]:
                continue
            sc = len(set(have) & set(want)) / len(set(have) | set(want))
            if best is None or sc > best[0]:
                best = (sc, sent)
        return best[1] if best else None

    def _recall_slots(self, n):
        n = self._tidy(n)
        if re.match(r"(?:what|who) (?:do you|have you) (?:remember|know) about me$|(?:what|who) (?:have i|did i) (?:told|tell) you(?: about me)?$|tell me what you know about me$", n):
            return self._summary()
        m = re.match(r"what (?:was|is) (?:the )?(first|last|previous) (?:thing|question|one|message)(?: that)? i (?:asked|said)(?: you)?(?: first)?$|what did i (?:ask|say) (first|last|just)$|what was my (first|last|previous) (?:question|message)$", n)
        if m:
            qs = getattr(self, "asked", [])
            w = next(g for g in m.groups() if g)
            if not qs:
                return "You haven't asked me anything yet."
            return f"You asked: \"{qs[0] if w == 'first' else qs[-1]}\""
        if re.match(r"(?:how old am i|what is my age|how old do you think i am|when was i born)$", n) and n != "when was i born":
            f = self._fget("age")
            return f[0][1] if f else None
        if re.match(r"(?:where|what (?:city|town|country|place|state|area|neighbou?rhood)) (?:do i live(?: in)?|am i from|do i come from|is my (?:home|hometown))$|where do i live in$", n):
            i = self.exact.get("where do i live") or self.exact.get("where am i from")
            return self.entries[max(i)].a if i else None
        m = re.match(r"what (?:do|did) i (like|love|enjoy|adore)(?: (?:to do|doing|the most|most|best|for fun))?$|what (?:are|were) my (?:hobbies|interests|likes|passions)$|what do i like to do(?: for fun)?$", n)
        if m:
            i = self.exact.get("what do i like")
            ans = self.entries[max(i)].a if i else None
            acts = [s for _, s in self._fget("do ")]
            if ans and m.group(1) in ("love", "adore"):
                ans = ans.replace("You like", "You love", 1)
            if "for fun" in n or "hobbies" in n or "interests" in n:
                ans = " ".join(x for x in [ans] + acts[:2] if x) or None
            return ans
        m = re.match(r"what (?:(\w+) )?(?:do|did) i (play|run|swim|cycle|bike|read|write|paint|draw|sing|dance|cook|bake|garden|hike|climb|ski|surf|skate|study|practice|train|exercise|meditate|volunteer|drive|collect|knit|sew|code|fish|camp|teach|coach|jog|lift|learn|watch|do)(?: (.+))?$", n)
        if m and not (m.group(1) in ("else", "not")):
            verb, extra = lem(m.group(2)), (m.group(3) or "")
            if verb == "do" and not extra and not m.group(1):
                job = self._slot("job")
                if job:
                    return job
                acts = self._fget("do ")
                return " ".join(s for _, s in acts[:3]) or None
            facts = [(k, s) for k, s in self._fget(f"do {verb}")] if verb != "do" else self._fget("do ")
            if extra and verb == "do":
                ew = set(norm(extra).split()) - {"for", "on", "in", "at", "the", "a", "every", "each", "during"}
                facts = [(k, s) for k, s in facts if ew and ew & set(norm(s).split())]
            elif extra:
                ew = set(norm(extra).split()) - {"for", "on", "in", "at", "the", "a"}
                facts = [(k, s) for k, s in facts if not ew or ew & set(norm(s).split())] or facts
            return " ".join(s for _, s in facts[:3]) or None
        m = re.match(r"what (?:am i|have i been) (\w+ing)(?: (.+))?$", n)
        if m and lem(m.group(1)) in _BASES:
            f = self._fget(f"do {lem(m.group(1))}")
            return " ".join(s for _, s in f[:2]) or None
        m = re.match(r"how long (?:have i|did i)(?: been)? (\w+)(?: (?:it|that|there|them))?$|how many years (?:have i|did i)(?: been)? (\w+)$", n)
        if m:
            v = lem(m.group(1) or m.group(2))
            f = self._fget(f"dur {v}")
            return f[0][1] if f else None
        m = re.match(r"where am i (?:going|traveling|travelling|heading|flying|moving)(?: to)?$|where (?:was|is) i going$", n)
        if m:
            f = self._fget("going")
            return f[0][1] if f else None
        m = re.match(r"what (?:colou?r|size|kind|type|brand|make|model|shape) (?:of \w+ )?(?:is|are) my ([a-z]+)$", n)
        if m and not re.match(rf"(?:{cc.RELS}|{cc.ANIMAL_RE})$", m.group(1)):
            return self._attr_match(m.group(1))
        m = re.match(r"what (?:is|are) my ([a-z]+(?: [a-z]+){0,2}?)(?: called| name)?$", n)
        if m:
            x = re.sub(r"'s$", "", m.group(1))
            if not re.match(rf"(?:{cc.RELS}|{cc.ANIMAL_RE})$", x):
                return self._attr_match(x)
        m = re.match(rf"how old (?:is|was) (?:my ({cc.RELS})|(she|he|it|her|him))$", n)
        if m:
            lab = cc.singular(m.group(1)) if m.group(1) else self._fresh_focus()
            f = self._fget(f"rel {lab} age") if lab else []
            return f[0][1] if f else None
        m = re.match(rf"(?:what|where|who|when|which) (?:does|did|do) (?:my ({cc.RELS})|(she|he|it|they)) (\w+)(?: [a-z ]+)?$", n)
        if m:
            lab = cc.singular(m.group(1)) if m.group(1) else self._fresh_focus()
            f = self._fget(f"rel {lab} {lem(m.group(3))}") if lab else []
            return f[0][1] if f else None
        m = re.match(rf"what (colou?r|breed|kind|sort|size) (?:of \w+ )?(?:is|are) (?:my ({cc.RELS}|{cc.ANIMAL_RE})|(she|he|it|they))$", n)
        if m:
            lab = cc.singular(m.group(2)) if m.group(2) else self._fresh_focus()
            lab = cc.BREEDS.get(lab, lab) if lab else lab
            descs = [s for _, s in self._fget(f"desc {lab}")] if lab else []
            if m.group(1) in ("color", "colour"):
                descs = [d for d in descs if re.search(r"\b(?:black|white|brown|grey|gray|orange|golden|ginger|tabby|red|blue|green|yellow|tan|cream|spotted|striped|colou?r)\b", d, re.I)]
            else:
                descs = [d for d in descs if re.search(r"\bis an? ", d)] or descs
            return descs[0] if descs else None
        return None

    def _summary(self):
        bits = self._fact_bits()
        if not bits:
            return "Not much yet. Tell me about yourself, like your name, where you live or what you like."
        return "Here's what I remember: " + "; ".join(bits[:8]) + "."

    def _fact_bits(self):
        """What Ovi knows about the user as short second-person phrases ('your dog is called Max')."""
        bits = []
        nm = self._known_name()
        if nm:
            bits.append(f"your name is {nm}")
        for _, s in self._fget("age"):
            bits.append(s[0].lower() + s[1:-1])
        i = self.exact.get("where do i live")
        if i and not self.entries[max(i)].dead:
            bits.append(self.entries[max(i)].a[0].lower() + self.entries[max(i)].a[1:-1])
        job = self._slot("job")
        if job:
            bits.append(job[0].lower() + job[1:-1])
        i = self.exact.get("what do i like")
        if i and not self.entries[max(i)].dead:
            bits.append(self.entries[max(i)].a[0].lower() + self.entries[max(i)].a[1:-1])
        for l in cc.ANIMALS:
            v = self._slot(f"pet {l}")
            if v:
                bits.append(f"your {l}{'s are' if re.search(r',| and ', v) else ' is'} called {v}")
        for prefix in ("do ", "my ", "going", "rel "):
            for _, s in self._fget(prefix):
                bits.append(s[0].lower() + s[1:-1])
        return bits

    def _fact_qs_for(self, t):
        """Slot questions to forget for 'my age' / 'what i play' / 'how old i am' / 'where i am going' (only those that exist)."""
        qs = []
        if re.match(r"(?:my age|how old i am|my birthday|what age i am)$", t):
            qs = [self._fq("age")]
        elif re.match(r"where i(?: am|'m) going$|my trip$", t):
            qs = [self._fq("going")]
        else:
            m = re.match(r"(?:what i|that i|the fact that i) (\w+)(?: .+)?$", t)
            if m and lem(m.group(1)) in _BASES:
                qs = [self._fq("dur " + lem(m.group(1)))] + [self._fq(k) for k, _ in self._fget(f"do {lem(m.group(1))}")]
                if " " in t.replace(m.group(0).split()[0] + " " + m.group(1), "", 1).strip():
                    rest = norm(t.split(m.group(1), 1)[1])
                    qs = [q for q in qs if not rest or rest in q or q.startswith(self._fq("dur"))]
            else:
                m = re.match(r"my ([a-z]+(?: [a-z]+){0,2})$", t)
                if m:
                    x = [cc.singular(w) for w in m.group(1).split()]
                    qs = [self._fq(k) for k, _ in self._fget("my ") if cc.singular(k.split()[-1]) == x[-1]]
                    qs += [self._fq(k) for k, _ in self._fget("do ") if x[-1] in k.split()]
        return [q for q in qs if self.exact.get(norm(q))]
