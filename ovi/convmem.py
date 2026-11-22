"""Conversation memory (mixin for Brain): every plain statement the user makes is kept as a sentence, and recall-style questions
("what's my cat's name again?", "what city did I say I lived in?", "what did I tell you about my sister?", "how many siblings do I have?")
are answered by finding the best-matching earlier sentence and extracting the relevant phrase.  This is the fallback behind the precise slot
rules in slots.py / usermem.py: those answer first, this only runs when they have nothing.

Storage: each sentence is an ordinary user ('taught') entry under an internal question "zzsay <ms> <n> || <slot questions it also set>" with the
sentence as the answer.  So it lives in the same append-only event log (persists across restarts), is never reachable by fuzzy retrieval, is
tombstoned by "forget everything" / "forget my cat" / "forget that" like every other user entry, and a sentence that also filled a slot dies
together with that slot ("forget my age" removes "I'm 25" here too).

Matching (no model, microseconds): lemma-ish stems + a few synonym groups (live/move/home, job/work, dad/father ...), idf weighting over the
sentences actually said, a bonus when the sentence contains the kind of thing the question word asks for (place for "what city", a number for
"how many", a name for "what's ... called", a job for "what do I do" ...), a head-noun requirement ("my puppy" must be about the puppy), pronoun
carry-over ("she is 3" belongs to the pet mentioned just before), and a small recency bonus so a later correction beats the older sentence."""
import re, time, math
from .memory import norm
from . import chitchat as cc

# ------------------------------------------------------------------ light lemmatiser + synonyms
_IRREG = {"ate": "eat", "eaten": "eat", "bought": "buy", "children": "child", "went": "go", "gone": "go", "got": "get", "took": "take", "said": "say", "told": "tell", "saw": "see",
          "gave": "give", "had": "have", "was": "be", "were": "be", "did": "do", "made": "make", "taught": "teach", "met": "meet", "drove": "drive", "wrote": "write", "ran": "run",
          "grew": "grow", "began": "begin", "sang": "sing", "swam": "swim", "flew": "fly", "thought": "think", "knew": "know", "left": "leave", "found": "find", "won": "win",
          "lost": "lose", "paid": "pay", "sold": "sell", "brought": "bring", "kept": "keep", "felt": "feel", "spent": "spend", "built": "build", "woke": "wake", "wore": "wear",
          "chose": "choose", "drank": "drink", "spoke": "speak", "mice": "mouse", "feet": "foot", "teeth": "tooth", "men": "man", "women": "woman", "people": "person"}


def stem(w):
    w = w.lower().replace("\u2019", "'")
    w = re.sub(r"'s$|'$", "", w)
    w = _IRREG.get(w, w)
    if len(w) <= 3 or w.isdigit():
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif re.search(r"(?:ss|x|ch|sh|z)es$", w):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if w.endswith("ied") and len(w) > 4:
        w = w[:-3] + "y"
    else:
        for suf in ("ing", "ers", "er", "ed"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[:-len(suf)]
                break
    if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "aeiousz":
        w = w[:-1]
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w


_GROUPS = ["live resid reside home hometown stay base move relocate from", "job work career profession occupation employ", "dad father papa pop", "mom mother mum mama mommy",
           "spouse wife husband partner married", "kid child children", "sibling sister brother", "fear afraid scared phobia terrified", "trip travel holiday vacation journey", "allergic allergy", "grandma grandmother granny nan", "grandpa grandfather", "car vehicle",
           "eat food meal", "major degree study", "favorite fav favourite", "pet animal", "puppy dog", "kitten cat", "buy purchase"]
CANON = {}
for _g in _GROUPS:
    _ws = [stem(x) for x in _g.split()]
    for _w in _ws:
        CANON.setdefault(_w, _ws[0])


# groups whose members are NOT interchangeable when they are the "my X" being asked about (my brother is not my sister)
_LOOSE_ONLY = {stem(x) for x in ("sibling", "pet", "puppy", "kitten", "major", "eat", "fear", "allergic", "favorite", "buy", "car")}
HEAD_SYN = {}
for _g in _GROUPS:
    _ws = [stem(x) for x in _g.split()]
    if _ws[0] not in _LOOSE_ONLY:
        for _w in _ws:
            HEAD_SYN.setdefault(_w, set(_ws))


for _animal in ("dog", "cat"):                       # "my dog" is answered by a sentence about a husky / corgi / puppy
    _set = {stem(_animal)} | {stem(x) for x in (("puppy", "pup", "pooch") if _animal == "dog" else ("kitten", "kitty"))} | {stem(k) for k, v in cc.BREEDS.items() if v == _animal}
    for _w in _set:
        HEAD_SYN[_w] = set(_set)


def cstem(w):
    s = stem(w)
    return CANON.get(s, s)


_TOK = re.compile(r"[A-Za-z0-9À-ɏ]+(?:['’][A-Za-z]+)?")


def _toks(t):
    return _TOK.findall(t)


# words that carry no content for matching (question frame, auxiliaries, pronouns, say-verbs)
FRAME = set("""what whats who whos whom whose where wheres when which how why many much old long often tall is are was were am be been being do does did done can could would should will have has had having
i me my mine myself we our ours you your yours the a an of in on at to for from with about that this it its they them their there here again also ever just exactly earlier before ago still now currently please
ovi say said says tell told mention mentioned mentioning remember recall thing things one ones some any kind sort type lot and or but so then too else other another yet already really actually
back first last time stuff know knew talk talked talking share shared two three four five six both all every each several few couple""".split())
NAMEISH = {"name", "names", "called", "named"}
_PRON_START = re.compile(r"(?:she|he|it|they|her|his|its|their|hers|theirs)\b", re.I)
_DETS = {"my", "the", "a", "an", "two", "three", "four", "five", "both", "all", "some", "these", "those", "our", "your"}

# ------------------------------------------------------------------ what kind of thing the question asks for
_NUMW = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|dozen|couple|few|several|a couple|a few)"
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_JOBS = "|".join(cc.OCC) if isinstance(cc.OCC, (tuple, list)) else "|".join(str(cc.OCC).split())
_JOBS = "|".join(sorted(set(_JOBS.split("|") + "plumber mechanic carpenter farmer pilot dentist banker cashier barista waiter waitress scientist researcher analyst consultant intern professor lecturer".split()), key=len, reverse=True))
CUE = {
    "name": (re.compile(r"\b(?:names?|named|called)\b", re.I), True),
    "job": (re.compile(rf"\b(?:work(?:s|ed|ing)?|job|jobs|career|profession|employ\w*|hired|teach\w*|freelanc\w*|self-employed|retired|unemployed|boss|office|shift|company|firm|business|startup|{_JOBS}|i(?:'m| am) an? [a-z]+|for a living)\b", re.I), True),
    "age": (re.compile(rf"\b\d{{1,3}}\b|\b(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|\w+teen)\b", re.I), True),
    "number": (re.compile(rf"\b\d+\b|\b{_NUMW}\b|\b(?:no|none|zero|only|both|twins?|triplets?)\b", re.I), True),
    "time": (re.compile(rf"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|{_MONTHS}|tomorrow|yesterday|tonight|today|morning|afternoon|evening|noon|midnight|weekend|week|month|year|summer|winter|spring|autumn|fall|christmas|birthday|\d{{1,2}}(?::\d\d)? ?(?:am|pm)|o'clock|\d{{1,2}}(?:st|nd|rd|th)|next|last|ago|later|soon|\d{{4}})\b", re.I), True),
    "duration": (re.compile(rf"\b(?:\d+|{_NUMW}) ?(?:years?|months?|weeks?|days?|hours?|minutes?|decades?)\b|\bsince\b|\ba while\b|\bages\b|\bforever\b", re.I), True),
    "place": (re.compile(r"\b(?:live[sd]?|living|moved|move|moving|from|grew up|born|based|stay(?:s|ing)?|resid\w+|home(?:town)?)\b|\b(?:in|from|to|near|outside|around) (?:the )?[A-Z][\w'-]+", 0), True),
    "city": (re.compile(r"\b(?:in|to|from|near|outside|around|at) (?!(?i:the|a|an|my|our|your|his|her|their|this|that|some|somewhere|anywhere|any|home|work|school|town|city|general|fact|total|time|bed|class|person|front|least|most|all|case|short|long|between|one|two)\b)[A-Za-z][\w'-]+|\b(?:called|named) [A-Z][\w'-]+|\b(?:in|to|from|near) (?:\w+ ){1,3}[A-Z][a-z]\w+", 0), True),
    "where": (re.compile(r"\b(?:in|at|from|near|to|around|by|inside|outside|on|into|across|over|behind|beside|downtown|abroad|home|here|there)\b", re.I), False),
    "instrument": (re.compile(r"\b(?:guitar|piano|violin|cello|viola|drums?|flute|ukulele|saxophone|sax|trumpet|clarinet|harp|banjo|bass|keyboard|trombone|oboe|accordion|harmonica|mandolin|synth\w*|instrument)\b", re.I), False),
    "sport": (re.compile(r"\b(?:tennis|soccer|football|basketball|baseball|hockey|golf|swim\w*|run\w*|cycl\w*|climb\w*|ski\w*|surf\w*|skat\w*|boxing|yoga|rugby|cricket|volleyball|badminton|gym|sport|sports|martial|karate|judo|rowing|sail\w*)\b", re.I), False),
    "language": (re.compile(r"\b(?:spanish|french|german|italian|japanese|mandarin|chinese|korean|portuguese|russian|arabic|hindi|latin|dutch|swedish|greek|turkish|english|language)\b", re.I), False),
    "color": (re.compile(r"\b(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|grey|gray|golden|silver|beige|teal|navy|maroon|tan|cream|colou?r)\b", re.I), False),
    "breed": (re.compile(r"\b(?:retrievers?|labs?|labradors?|poodles?|bulldogs?|terriers?|huskies|husky|shepherds?|beagles?|spaniels?|collies?|dachshunds?|chihuahuas?|pugs?|corgis?|rottweilers?|dobermans?|boxers?|greyhounds?|whippets?|siamese|persians?|tabbys?|tabbies|calicos?|gingers?|ragdolls?|bengals?|maine coons?|shorthairs?|longhairs?|mix|mutt|breed)\b", re.I), False),
    "food": (re.compile(r"\b(?:eat|ate|cook\w*|bak\w*|drink|drank|food|dinner|lunch|breakfast|brunch|snack|dessert|pizza|pasta|soup|salad|rice|curry|cake|bread|coffee|tea|sandwich|burger|sushi|tacos?|noodles?|steak|chicken|fish)\b", re.I), False),
    "subject": (re.compile(r"\b(?:study\w*|major\w*|minor\w*|degree|course|class(?:es)?|school|college|universit\w*|subject|studying|bachelor|master|phd|graduat\w*|biology|chemistry|physics|maths?|history|english|literature|economics|psychology|engineering|law|medicine|art|music|philosophy)\b", re.I), False),
    "who": (re.compile(rf"\b(?:{cc.RELS})\b|\bwith [A-Za-z]+|[a-z,] [A-Z][a-z]+", 0), False),
}
_TYPE_CUE = {"city": "city", "town": "city", "country": "city", "state": "city", "village": "city", "neighborhood": "city", "neighbourhood": "city", "area": "city", "province": "city",
             "region": "city", "place": "place", "hometown": "place", "name": "name", "age": "age", "job": "job", "career": "job", "profession": "job", "occupation": "job", "work": "job",
             "instrument": "instrument", "sport": "sport", "language": "language", "color": "color", "colour": "color", "breed": "breed", "food": "food", "dish": "food", "meal": "food",
             "drink": "food", "snack": "food", "dessert": "food", "cuisine": "food", "subject": "subject", "major": "subject", "majors": "subject", "degree": "subject", "course": "subject",
             "year": "time", "time": "time", "day": "time", "date": "time", "month": "time", "number": "number", "amount": "number"}
_WHCUE = {"how many": "number", "how much": "number", "how old": "age", "how long": "duration", "when": "time", "where": "where", "who": "who", "whos": "who"}

_NOT_RECALL = re.compile(r"\b(?:should|if|whether|supposed|suppose|might|maybe|gonna|going to|will i|would i|can i|could i|may i|need to|have to|how do i|how can i|how should i)\b|^what (?:do|did|does|would|will|can|could) i (?:think|feel|want|need|mean|wish|hope|do next|know)\b|^(?:what|how|why) about\b")
_OK_YOU = re.compile(r"\b(?:(?:tell|told|telling|said to|mention(?:ed)?(?: to)?|share[d]?(?: with)?|say to|talk(?:ed)? to|spoke to|ask(?:ed)?|show(?:ed)?) you|do you (?:still )?(?:remember|recall|know)|can you (?:still )?(?:remember|recall)|did you (?:catch|get|save|keep|note)|you (?:remember|recall|know))\b")
_WH = re.compile(r"^(?:what|who|whom|whose|where|when|which|how (?:many|much|old|long|often|tall|big|far)|(?:do|can|could|did|will|would) you (?:still )?(?:remember|recall|know)|remember|(?:did|have|had) i\b|is it true (?:that )?i)\b")
_FIRST_P = re.compile(r"\b(?:my|mine|i|me|i'm|im|i've|ive|myself|am i|we|our)\b")


def _qkind(n):
    """Recall-style question? -> normalised text, else None."""
    n = re.sub(r"^(?:(?:and|so|ok|okay|wait|hey|hmm|um|but|oh|quick question|btw|one more thing)[, ]+)+", "", n.strip())
    n = re.sub(r"\bwhat s\b", "what is", n)
    if not _WH.match(n) or not _FIRST_P.search(n):
        return None
    stripped = _OK_YOU.sub(" ", n)
    if re.search(r"\byou(?:r|rs)?\b", stripped) or _NOT_RECALL.search(n):
        return None
    return n


_NAME_QS = {"what is my name", "who am i", "do you know my name", "what's my name", "do you remember my name"}


class ConvMemory:
    # ---------- store ----------
    def _conv_sync(self):
        """Pick up zzsay entries added since the last call (also the ones replayed from the event log at startup)."""
        cv = getattr(self, "_cv", None)
        if cv is None:
            cv = self._cv = []
            self._cv_upto = 0
        for i in range(self._cv_upto, len(self.entries)):
            e = self.entries[i]
            if e.q.startswith("zzsay "):
                head, *facts = e.q.split("||")
                toks_ = _toks(e.a)
                st, raw = {cstem(w) for w in toks_}, {stem(w) for w in toks_}
                prev = next((r for r in reversed(cv) if not self.entries[r["i"]].dead), None)
                ctx, craw = set(), set()
                if prev and _PRON_START.match(e.a.strip()):     # "she is a vet": carries what the previous sentence was about
                    ctx, craw = prev["stems"] | prev["ctx"], prev["raw"] | prev["craw"]
                cv.append({"i": i, "text": e.a, "stems": st, "raw": raw, "ctx": ctx, "craw": craw, "facts": [f.strip() for f in facts], "prev": prev})
        self._cv_upto = len(self.entries)
        return cv

    def _conv_note(self, msg, ans):
        """Called once per user turn (after the reply): keep every plain statement the user made."""
        if msg.startswith("/") or ans.src in ("forget", "feedback", "calc") or msg.endswith("?") or re.match(r"(?:what|who|where|when|why|how|which|is|are|do|does|did|can|could|will|would)\b", msg, re.I):
            return
        msg = re.sub(r"^(?:(?:ok|okay|so|well|anyway|btw|oh|hey|hi|um|uh|also|and)\b[, ]+)+", "", re.sub(r"^(?:please |ovi,? )*remember(?: that| this)?[:, ]+", "", msg.strip(), flags=re.I), flags=re.I)
        facts = list(self.recent_fact or [])
        if facts and all(norm(f) in _NAME_QS for f in facts):
            return                      # "my name is Sam": the dedicated name slot answers that precisely; no sentence copy needed
        saved = []
        for s in re.split(r"(?<=[.!?])\s+", msg)[:4]:
            s = s.strip()
            w = norm(s).split()
            if len(w) < 3 or len(s) > 300 or cc.is_ack(norm(s)) or cc.is_nonanswer(norm(s)) or s.endswith("?"):
                continue
            cv = self._conv_sync()
            if cv and self.entries[cv[-1]["i"]].a == s and not self.entries[cv[-1]["i"]].dead:
                continue
            q = f"zzsay {int(time.time() * 1000)} {len(self.entries)}" + ("".join(f" || {f}" for f in facts) if facts else "")
            self._add(q, s, "taught", 1.0)
            self._write({"e": "add", "q": q, "a": s, "src": "taught", "conf": 1.0})
            saved.append(len(self.entries) - 1)
        if saved:
            self._conv_recent = saved

    # ---------- forget ----------
    def _conv_forget(self, qs=(), t="", recent=False):
        """Tombstone remembered sentences: those tied to forgotten slot questions, those mentioning every content word of t, or the latest ones."""
        cv = self._conv_sync()
        qset = {norm(q) for q in qs or [] if not norm(q).startswith("zzsay ")}
        ts = {cstem(w) for w in _toks(t) if w.lower() not in FRAME and w.lower() not in ("everything", "all")}
        n = 0
        ids = set(getattr(self, "_conv_recent", ())) if recent else set()
        for r in cv:
            e = self.entries[r["i"]]
            if e.dead:
                continue
            hit = r["i"] in ids or bool(qset and qset & {norm(f) for f in r["facts"]}) or bool(ts and ts <= r["stems"])
            if hit:
                n += self._forget_entries(e.q, e.a)
        if n and recent:
            self._conv_recent = []
        return n

    # ---------- recall ----------
    def _conv_parse(self, n):
        """Question -> dict(type cue, strict, head, focus stems, plural/aggregate flag, topic-question flag, yes/no flag)."""
        toks = [t.lower() for t in _toks(n)]
        text = " ".join(toks)
        yn = bool(re.match(r"(?:did|have|had) i\b|is it true", text)) or bool(re.match(r"(?:do|can|could|did) you (?:still )?(?:remember|recall|know) (?:that )?(?:i|my|me)\b", text) and not re.search(r"\b(?:what|who|where|when|which|how)\b", text))
        text = re.sub(r"^(?:do|can|could|did|will|would) you (?:still )?(?:remember|recall|know)\s+", "", text)
        text = re.sub(r"^remember\s+", "", text)
        toks = text.split()
        tcue, strict, typeword = None, False, None
        for k, v in _WHCUE.items():
            if text.startswith(k):
                tcue = v; strict = CUE[v][1]
        m = re.match(r"(?:what|which)\s+(?:kind of |sort of |type of )?([a-z]+)\s+(?:do|does|did|am|are|is|was|were|have|has|can)\b", text)
        if m and m.group(1) not in FRAME and m.group(1) not in ("kind", "sort", "type"):
            typeword = m.group(1)
        m2 = re.match(r"(?:what|which)\s+(?:are|were)\s+(?:the |my )?(?:(?:two|three|four|five|few|several|first|last|main|top) )*([a-z]+s)\b\s+(?:i|that i|you)\b", text)
        if m2:
            typeword = m2.group(1)
        if typeword and _TYPE_CUE.get(typeword) or typeword and _TYPE_CUE.get(stem(typeword)):
            tcue = _TYPE_CUE.get(typeword) or _TYPE_CUE.get(stem(typeword)); strict = CUE[tcue][1] if tcue in CUE else False
        elif typeword and typeword in CUE:
            tcue = typeword
        hseq_pre = []
        if typeword and not re.search(r"\bmy\b", text) and stem(typeword) in HEAD_SYN and stem(typeword) not in CUE:
            hseq_pre = [typeword]                      # "what kind of dog do I have" -> about the dog; the answer is its breed
            if re.search(r"\b(?:kind|sort|type) of\b", text):
                tcue, strict = "breed", False
        if re.match(r"what (?:do i do|am i doing for (?:a living|work))(?: for (?:a living|work|money))?$|what (?:does|did) (?:my [a-z]+(?: [a-z]+)?|she|he|they) do(?: for (?:a living|work|money))?$|what (?:is|are) (?:my|his|her)? ?(?:job|occupation|profession|line of work)", text):
            tcue, strict = "job", True
        head, hseq = None, []
        if hseq_pre:
            hseq = list(hseq_pre); typeword = None
        mm = re.search(r"\bmy ((?:[a-z']+ ){0,3}[a-z']+)", " ".join(toks))
        if mm:
            for w in mm.group(1).split():
                if w in ("is", "are", "was", "were", "do", "does", "did", "again", "and", "or", "that", "i", "have", "has", "had", "still", "now", "mentioned", "said", "told"):
                    break
                hseq.append(w)
        for w in list(hseq):
            if re.sub(r"'s?$", "", w) in NAMEISH:
                tcue, strict = "name", True
                hseq.remove(w)
            elif w in ("age", "birthday"):
                tcue, strict = "age", True
                hseq.remove(w)
        if any(w in NAMEISH for w in toks) and tcue is None:
            tcue, strict = "name", True
        hseq = [re.sub(r"'s?$", "", w) for w in hseq if w not in ("favorite", "favourite", "fav", "current", "main", "new")]
        if hseq:
            head = hseq[-1]
        about = re.search(r"\babout (?:my |the |a |an |that |those |these )?(.+)$", text)
        used = {typeword} if typeword else set()
        focus = []
        skip_k = 0
        for i, w in enumerate(toks):
            wb = re.sub(r"'s?$", "", w)
            if w in FRAME or wb in FRAME or wb in used or wb in NAMEISH or w.isdigit() or wb in ("age", "birthday") or len(wb) < 2:
                continue
            if wb in ("favorite", "favourite", "fav", "current", "main", "new", "mentioned", "said", "told"):
                continue
            focus.append(wb)
        fst = []
        for w in focus:
            c = cstem(w)
            if c not in fst and not (typeword and c == cstem(typeword)):
                fst.append(c)
        hs = stem(head) if head else None
        hset = set(HEAD_SYN.get(hs, {hs})) if head else set()
        fst = [f for f in fst if f != cstem(head or "") and f not in hset]
        multi = bool(about) or bool(re.search(r"\b(?:two|three|four|five|both|all|every|list|everything)\b", text)) or bool(typeword and typeword.endswith("s") and typeword not in ("class", "bus", "boss", "glass", "business"))
        return {"tcue": tcue, "strict": strict, "head": head, "hs": hs, "hset": hset, "focus": fst, "tword": cstem(typeword) if typeword else None, "multi": multi, "yn": yn, "raw_head": hseq,
                "plural_head": bool(head and mm and re.search(rf"\b{re.escape(head)}s\b", n))}

    def _conv_score(self, q, cv):
        """[(score, strict_ok, matched_nontype, rec, via_ctx)] for every live remembered sentence."""
        N = max(1, len(cv))
        df = {}
        for r in cv:
            for s in r["stems"]:
                df[s] = df.get(s, 0) + 1
        out = []
        for pos, r in enumerate(cv):
            e = self.entries[r["i"]]
            if e.dead:
                continue
            raw = r["text"]
            ctx = r["ctx"]
            num = den = 0.0
            matched_non = 0
            via_ctx = False
            for f in q["focus"]:
                w = 1.0 + 0.3 * math.log((N + 2) / (df.get(f, 0) + 1))
                den += w
                if f in r["stems"]:
                    num += w; matched_non += 1
                elif f in ctx:
                    num += 0.8 * w; matched_non += 1; via_ctx = True
            head_ok = True
            if q["hset"]:
                den += 1.5
                if q["hset"] & r["raw"]:
                    num += 1.5; matched_non += 1
                elif q["hset"] & r["craw"]:
                    num += 1.2; matched_non += 1; via_ctx = True
                else:
                    head_ok = False
            tc = q["tcue"]
            cue_ok = True
            if tc and tc in CUE:
                tw = 1.0
                den += tw
                hit = bool(CUE[tc][0].search(raw))
                if hit:
                    num += tw
                cue_ok = hit or not q["strict"]
            if q["tword"]:
                tw2 = 0.5 if (tc and tc in CUE) else 0.4        # the asked-for kind of thing ('grade', 'majors') appearing literally is extra evidence
                if q["tword"] in r["stems"]:
                    num += tw2; den += tw2
                elif not (tc and tc in CUE):
                    den += tw2
            if den == 0:
                continue
            sc = num / den
            if not head_ok:
                sc *= 0.4                                      # a question about "my puppy" is not answered by a sentence without the puppy
            sc += 0.04 * (pos + 1) / N                           # recency: a later correction beats the earlier sentence
            out.append((sc, cue_ok, matched_non, r, via_ctx))
        return out

    # --- turning a stored sentence into a natural second-person answer
    _VOICE = [(r"\bi'?m\b", "you're"), (r"\bi['’]?ve\b", "you've"), (r"\bi['’]?ll\b", "you'll"), (r"\bi['’]?d\b", "you'd"), (r"\bi am\b", "you are"), (r"\bi was\b", "you were"),
              (r"\bi\b", "you"), (r"\bmy\b", "your"), (r"\bmine\b", "yours"), (r"\bmyself\b", "yourself"), (r"\bme\b", "you"), (r"\bim\b", "you're"), (r"\bive\b", "you've"),
              (r"\bwe['’]re\b", "you're"), (r"\bwe['’]ve\b", "you've"), (r"\bwe['’]ll\b", "you'll"), (r"\bwe\b", "you"), (r"\bour\b", "your"), (r"\bours\b", "yours"), (r"\bus\b", "you")]
    _STARTERS = cc._FUNC | set("got had just well also today yesterday last recently started moved finally actually still been went bought adopted lately currently these those".split())

    @classmethod
    def _voice(cls, t):
        t = t.strip().strip('"').rstrip(".!? ")
        for pat, rep in cls._VOICE:
            t = re.sub(pat, rep, t, flags=re.I)
        t = re.sub(r"\byou is\b", "you are", t)
        first = t.split(" ", 1)[0]
        if first.lower() in cls._STARTERS or first.lower() in ("you", "your", "you're", "you've", "you'll", "you'd"):
            t = first.lower() + t[len(first):]
        return t

    @staticmethod
    def _clauses(s):
        parts = re.split(r"\s*(?:[,;]|\band\b|\bbut\b|\bso\b|\bbecause\b|\bthen\b|\balthough\b|\bwhile\b|\bwhich\b|\bthough\b)\s*", s)
        return [p for p in parts if p and p.strip()]

    def _conv_phrase(self, q, r, via_ctx):
        """The part of sentence r that answers q, in second person."""
        s = r["text"].strip()
        tc, head = q["tcue"], q["head"]
        if tc == "name" and head and (via_ctx or _PRON_START.match(s) or not re.search(rf"\b{re.escape(head)}", s, re.I) or re.search(r"\b(?:her|his|its|their) names?\b", s, re.I)):
            m = re.search(r"\b(?:names?(?: (?:is|was|are|were))?|called|named)\s+(?:is\s+|are\s+)?(?:the\s+)?([A-Za-z][\w'-]*(?:(?:\s*,\s*|\s+and\s+|\s*&\s*)[A-Z][\w'-]*)*)", s)
            if m and m.group(1).lower() not in cc._FUNC and m.group(1).lower() not in ("a", "an", "she", "he", "it", "they", "also", "too", "really", "just"):
                nm = re.sub(r"\b([a-z])", lambda x: x.group(1).upper(), m.group(1)) if m.group(1).islower() or " " not in m.group(1) and m.group(1)[0].islower() else m.group(1)
                nm = re.sub(r"\s*,\s*", ", ", nm)
                many = bool(re.search(r",|\band\b|&", nm))
                h = cc.singular(head) if not q["plural_head"] else head
                if many or q["plural_head"]:
                    return f"your {cc.singular(head)}s' names are {nm}"
                return f"your {h}'s name is {nm}"
        cl = self._clauses(s)
        use = s
        if len(cl) > 1 and not q["multi"] and len(s.split()) > 9 and tc not in ("number", "age", "duration"):
            best, bw = None, -1.0
            for c in cl:
                cs = {cstem(w) for w in _toks(c)}
                w = sum(1.0 for f in q["focus"] if f in cs) + (1.0 if q["hset"] & {stem(x) for x in _toks(c)} else 0.0)
                if tc in CUE and CUE[tc][0].search(c):
                    w += 1.0
                if w > bw:
                    best, bw = c, w
            allw = sum(1.0 for f in q["focus"] if f in r["stems"]) + (1.0 if q["hset"] & r["raw"] else 0.0) + (1.0 if tc in CUE and CUE[tc][0].search(s) else 0.0)
            if best is not None and bw >= allw and bw > 0:
                use = best.strip()
        v = self._voice(use)
        if tc in ("place", "where", "city"):
            v = re.sub(r"\b(in|to|from|near|at) ([a-z]{3,})\b", lambda m: m.group(0) if m.group(2) in cc._FUNC or m.group(2) in ("the", "a", "an", "my", "our", "your", "some", "this", "that", "his", "her", "their", "town", "city", "home", "work", "school", "college", "bed", "class", "person", "general", "front", "fact", "time", "case", "total", "the") else f"{m.group(1)} {m.group(2).capitalize()}", v)
        first = v.split(" ", 1)[0].lower()
        if (via_ctx or _PRON_START.match(use)) and head and re.match(r"(?:she|he|it|they|her|his|its|their)\b", v, re.I):
            sing = cc.singular(head)
            v = re.sub(r"^(?:she|he|it)\b", f"your {sing}", v, count=1, flags=re.I)
            v = re.sub(r"^they\b", f"your {sing}s", v, count=1, flags=re.I)
            v = re.sub(r"^(?:her|his|its)\b", f"your {sing}'s", v, count=1, flags=re.I)
            v = re.sub(r"^their\b", f"your {sing}s'", v, count=1, flags=re.I)
        elif (re.match(r"(?:got|had|have|bought|adopted|started|moved|went|play|plays|teach|study|work|live|lives|am|picked|found|joined|signed|took|learn|learned|learning|studying|working|living|playing|teaching|running|trying|like|love|hate|enjoy|own|owns)\b", first)
              or re.match(r"[a-z]+(?:ed|ing)\b", first) and first not in ("you", "your")) and not re.match(r"you", first):
            v = "you " + v
        return v

    def _recall_conv(self, msg, n):
        """Fallback recall from the sentences the user actually said; a natural answer, a hedged quote, or None (= nothing relevant was said)."""
        qn = _qkind(n)
        if not qn:
            return None
        hit = self.recall(msg)
        if hit and self.entries[hit[0]].src in ("taught", "learned") and self.conf_of(*hit) >= 0.45:
            return None                 # a stored answer the user gave for exactly this question wins (favorites, taught facts, corrections)
        cv = self._conv_sync()
        live = [r for r in cv if not self.entries[r["i"]].dead]
        q = self._conv_parse(qn)
        if not live:
            return None
        if not q["focus"] and not q["hset"] and not q["tcue"] and not q["tword"]:
            m = re.match(r"(?:what did i (?:just )?(?:say|tell you|mention)(?: (?:earlier|before|just now|a moment ago))?|what was (?:the )?(?:last|previous) thing i (?:said|told you|mentioned))$", qn)
            if m:
                r = live[-1]
                self._conv_recent = [r["i"]]
                return f"You said: \"{r['text'].rstrip()}\""
            return None
        scored = self._conv_score(q, live)
        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[0]
        sc, cue_ok, matched_non, r, via_ctx = top
        nontype_needed = bool(q["focus"] or q["hset"])
        good = cue_ok and sc >= 0.6 and (matched_non or not nontype_needed)
        hedge = cue_ok and sc >= 0.33 and (matched_non or not nontype_needed)
        if q["yn"]:
            if good or (hedge and matched_non):
                self._conv_recent = [r["i"]]
                return f"Yes, you said {self._conv_phrase(q, r, via_ctx)}."
            return "I don't think you've told me that." if live else None
        if not (good or hedge):
            ab = re.search(r"\bwhat (?:did|have) i (?:say|said|tell you|told you|mention(?:ed)?)(?: to you)? about ((?:my |the |a |an )?[a-z][a-z' -]{1,30})$", qn)
            if ab and not scored[0][2]:
                return f"I don't think you've told me anything about {self._voice(ab.group(1))} yet."
            return None
        picks = [top]
        if q["multi"]:
            thr = max(0.5, 0.6 * sc)
            picks = [x for x in scored if x[0] >= thr and x[1] and (x[2] or not nontype_needed)][:3]
            picks.sort(key=lambda x: x[3]["i"])
        phrases, seen = [], set()
        for x in picks:
            p = self._conv_phrase(q, x[3], x[4])
            if p.lower() not in seen:
                seen.add(p.lower()); phrases.append(p)
        self._conv_recent = [x[3]["i"] for x in picks]
        if not good:
            return f"I'm not sure, but the closest thing I remember is that you said {phrases[0]}."
        if len(phrases) == 1:
            return f"You said {phrases[0]}."
        return f"You said {phrases[0]}. " + " ".join(f"You also said {p}." for p in phrases[1:])
