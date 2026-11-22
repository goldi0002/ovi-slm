"""Generic memory of the user's pets, job and relatives (mixin for Brain). Facts are stored as ordinary user ('taught')
entries under internal 'zzslot ...' questions, so they persist in the event log, can be superseded and forgotten."""
import re
from .memory import norm
from . import chitchat as cc

_N = cc.CAPNAME
_ANY = f"(?:{cc.RELS}|{cc.ANIMAL_RE})"
_NL = rf"({_N}(?:(?:\s*,\s*and\s+|\s*,\s*|\s+and\s+|\s*&\s*){_N})*)"
_FOCUS_RE = re.compile(rf"\bmy ({_ANY})(?:'s|s)?\b", re.I)
_MULTI = re.compile(r",| and ")
_PET_STMT = re.compile(rf"(?i:(?:i|we) (?:(?:just|recently|finally|also|actually) )*(?:have|got|own|adopted|now have)) (?i:(a|an|one|two|three|four|five|\d+) )?(?:[a-z]+ )?({cc.ANIMAL_RE})\b[\s,:;-]*(?i:(?:called|named|they are called|they're called|their names are|names are|it's called|is called|are|is))?[\s,:;-]*(?:{_NL})?$")
_HAVE_NUM = re.compile(rf"(?i:i (?:have|got|own|do have)) (?i:(a|an|one|two|three|four|five|\d+))(?i: (?:of them|myself|actually|too|at home))?[\s,:;-]*(?i:(?:called|named|they are called|they're called|their names are|names are|they are|they're|it's called|it's))?[\s,:;-]*(?:{_NL})?$")
_ATTR = re.compile(rf"({_N}) (?i:is|was|looks|has) ((?i:(?:an? |the )?[a-z]+(?: and [a-z]+)?(?: [a-z]+)?))$")
_Q_WHICH = re.compile(r"^which (?:one|cat|dog|pet|of them)(?: of them)? (?:is|was) ([a-z ]{2,20})$|^who(?:'s| is) ([a-z ]{2,20})$")
_REL_STMT = re.compile(rf"(?i:my) (?i:({_ANY}))s?(?:'s)?(?i: name)?(?i: is| was)?(?i: called| named)? {_NL}$")
_PRON_STMT = re.compile(rf"(?i:(?:his|her|its|their) names? (?:is|was|are|were)|(?:he|she|it) (?:is |was )?(?:called|named)|they(?:'re| are| were) (?:called|named)|(?:it's|they're|its name is|called|named)) {_NL}$")
_BARE_NAMES = re.compile(rf"{_NL}$")
_JOB1 = re.compile(r"i(?:'m| am) (?:an? )?((?:[a-z]+ )?(?:" + "|".join(cc.OCC) + r"))(?: (?:at|for|in) (.+))?$", re.I)
_JOB2 = re.compile(r"i work (?:as (?:an? )?([a-z ]{3,30})|(?:at|for|in) ((?!the (?:morning|evening|night|afternoon|weekend)\b)[A-Za-z][\w &.'-]{1,30}))$", re.I)
_THE_OF = re.compile(r"the ([a-z]+(?: [a-z]+)?) of ([A-Za-z][A-Za-z ]*?) is (.+)$", re.I)
_COUNT = re.compile(r"i have (\d+|a|one|two|three|four|five|six) (kids?|children|brothers?|sisters?|siblings?|cousins?|sons?|daughters?)$", re.I)
_Q_REL = re.compile(rf"^(?:do you (?:remember|know) )?(?:(?:what|who|which)(?:'s| is| are| was| were|s)? (?:the )?(?:names? of )?)?my ({_ANY})s?(?:'s|s'| s)?(?: names?)?(?: called| again| is called| are called| were called)*$")
_Q_COUNT = re.compile(rf"^how many (kids|children|siblings|brothers|sisters|{cc.ANIMAL_RE}) (?:do i have|have i got|do i own)$")
_Q_JOB = re.compile(r"^(?:what do i do(?: for (?:a living|work))?|what(?:'s| is| s) my (?:job|occupation|profession|work|line of work)(?: again| now| currently)?|where do i work|what (?:job|work) do i (?:do|have)|what is my job again)$")
_Q_PETS = re.compile(rf"^(?:what|which) (?:pets|animals) do i have$|^do i have (?:a |any |an )?(?:{cc.ANIMAL_RE})$")
_Q_PRON = re.compile(r"^(?:what|who)(?:'s| is| was| are| were|s) (his|her|their|its) names?(?: again| now| called)?$|^(?:what|who) (?:is|was) (he|she|it) called$|^what (?:are|were) (they|their names) called$")


def _lab(w):
    w = cc.singular(w)
    return cc.BREEDS.get(w, w)


def _an(w):
    return "an" if w[:1].lower() in "aeiou" else "a"


class UserMemory:
    def _slot(self, slot):
        i = self.exact.get(norm(f"zzslot {slot}"))
        return self.entries[max(i)].a if i else None

    def _set_slot(self, slot, val):
        self.teach(f"zzslot {slot}", val, supersede=True)

    @staticmethod
    def _label_slot(label):
        return f"pet {label}" if label in cc.ANIMALS or label == "pet" else f"rel {label}"

    def _store_names(self, label, names, merge=False):
        slot = self._label_slot(label)
        old = cc.names_list(self._slot(slot) or "") if merge else []
        allnames = old + [x for x in names if x not in old]
        self._set_slot(slot, cc.join_names(allnames))
        self.recent_fact = [f"zzslot {slot}"]
        return allnames

    def _track_focus(self, msg):
        found = _FOCUS_RE.findall(msg)
        if found:
            self.focus = (_lab(found[-1]), False)
            self.focus_turn = len(self.history)
        elif re.search(r"\b(?:getting married|got engaged|getting engaged|my fianc)", msg, re.I):
            self.focus = ("fiancé", False)

    def _slots_for(self, t):
        """'my cats' / 'my job' / 'my sister' -> the stored slot questions to forget (only those that exist)."""
        m = re.match(rf"my ({_ANY})s?$", t)
        qs = []
        if m:
            lab = _lab(m.group(1))
            qs = [f"zzslot {self._label_slot(lab)}", f"zzslot count {lab}"]
        elif t in ("my job", "my work", "where i work", "what i do"):
            qs = ["zzslot job"]
        elif hasattr(self, "_fact_qs_for"):
            qs = self._fact_qs_for(t)
        return [q for q in qs if self.exact.get(norm(q))]

    def _profile_more(self, m):
        """Pets, jobs, relatives. Names must be capitalised words. Returns Ovi's reply or None."""
        r = _PET_STMT.match(m)
        if r:
            cnt, raw = r.group(1), r.group(2)
            animal, names = cc.BREEDS.get(cc.singular(raw), cc.singular(raw)), cc.names_list(r.group(3) or "")
            k = cc.NUMWORDS.get((cnt or "").lower()) or (int(cnt) if cnt and cnt.isdigit() else None)
            plural = bool(k and k > 1) or len(names) > 1 or (raw.lower().endswith("s") and raw.lower() != "fish" and not k)
            self.focus = (animal, False)
            if names:
                allnames = self._store_names(animal, names, merge=True)
                self._set_slot(f"count {animal}", str(max(k or 0, len(allnames))))
                return f"{cc.join_names(names)}, lovely {'names' if len(names) > 1 else 'name'}! I'll remember your {animal}{'s' if len(names) > 1 else ''}."
            if k:
                self._set_slot(f"count {animal}", str(k))
            self.focus = (animal, True)
            return f"Nice! What {'are they' if plural else 'is it'} called?"
        r = _HAVE_NUM.match(m)
        pet = self.like_animal if self.like_animal and len(self.history) - self.like_animal[1] <= 4 else (self.focus if self.focus and self.focus[0] in cc.ANIMALS else None)
        if r and pet and not _COUNT.match(m):
            animal, names = pet[0], cc.names_list(r.group(2) or "")
            k = cc.NUMWORDS.get(r.group(1).lower()) or (int(r.group(1)) if r.group(1).isdigit() else None)
            self.focus = (animal, False)
            if names:
                allnames = self._store_names(animal, names, merge=True)
                self._set_slot(f"count {animal}", str(max(k or 0, len(allnames))))
                return f"{cc.join_names(names)}, lovely {'names' if len(names) > 1 else 'name'}! I'll remember your {animal}{'s' if len(names) > 1 else ''}."
            if k:
                self._set_slot(f"count {animal}", str(k))
            self.focus = (animal, True)
            return f"Nice! What {'are they' if (k or 2) > 1 else 'is it'} called?"
        r = _ATTR.match(m)
        if r and self.focus and r.group(1) in cc.names_list(self._slot(self._label_slot(self.focus[0])) or ""):
            nm, att = r.group(1), r.group(2).strip().lower()
            self._set_slot(f"attr {nm.lower()}", att)
            return f"Got it, {nm} is {att}."
        r = _REL_STMT.match(m)
        if r:
            label, names = _lab(r.group(1)), cc.names_list(r.group(2))
            if names:
                self._store_names(label, names)
                self.focus = (label, False)
                return f"Got it, your {label} {'is called' if label in cc.ANIMALS else 'is'} {cc.join_names(names)}."
        r = _PRON_STMT.match(m) or (_BARE_NAMES.match(m) if self.focus and self.focus[1] else None)
        if r and self.focus:
            names = cc.names_list(r.group(1))
            if names:
                label = self.focus[0]
                self._store_names(label, names, merge=label in cc.ANIMALS)
                self.focus = (label, False)
                return f"{cc.join_names(names)}. I'll remember your {label}{'s' if len(names) > 1 else chr(39) + 's name'}."
        r = _JOB1.match(m) or _JOB2.match(m)
        if r:
            job, place = r.group(1), r.group(2)
            if m.lower().startswith("i work") and not m.lower().startswith("i work as"):
                job, place = None, r.group(2)
            if job:
                self._set_slot("job", f"You are {_an(job)} {job.lower()}" + (f" at {place}." if place else "."))
                self.recent_fact = ["zzslot job"]
                return f"Nice, {_an(job)} {job.lower()}! How do you like it?"
            if place:
                self._set_slot("job", f"You work at {place}.")
                self.recent_fact = ["zzslot job"]
                return f"{place}, nice! How is it working there?"
        r = _THE_OF.match(m)
        if r and len(r.group(2).split()) <= 3 and len(r.group(3).split()) <= 6:
            x, y, z = r.group(1).lower(), r.group(2).strip(), r.group(3).strip()
            self.teach(f"what is the {x} of {y}", f"The {x} of {y} is {z}.", supersede=True)
            self.recent_fact = [f"what is the {x} of {y}"]
            return f"Got it, the {x} of {y} is {z}."
        r = _COUNT.match(m)
        if r:
            k = r.group(1).lower()
            lab = cc.singular({"children": "kid", "siblings": "sibling"}.get(r.group(2).lower(), r.group(2)))
            self._set_slot(f"count {lab}", str(cc.NUMWORDS.get(k, k)))
            self.recent_fact = [f"zzslot count {lab}"]
            self.focus = (lab, False)
            return f"Nice, {k} {r.group(2).lower()}! What are they like?"
        return None

    def _recall_profile(self, n):
        """'what are my cats called?' / 'what's my job' / 'what was his name again?' -> stored answer, else None."""
        m = _Q_REL.match(n)
        if m and n.startswith(("what", "who", "which", "do you")):
            label = _lab(m.group(1))
            v = self._slot(self._label_slot(label))
            if v:
                if label in cc.ANIMALS or label == "pet":
                    return f"Your {label}{'s are' if _MULTI.search(v) else ' is'} called {v}."
                return f"Your {label} is {v}."
            k = self._slot(f"count {label}")
            if k and label in cc.ANIMALS:
                self.focus = (label, True)
                return f"You told me you have {k} {label}{'s' if k != '1' else ''}, but not their names. What are they called?"
            return None
        m = _Q_WHICH.match(n)
        if m and self.focus:
            want = (m.group(1) or m.group(2)).strip()
            for nm in cc.names_list(self._slot(self._label_slot(self.focus[0])) or ""):
                if self._slot(f"attr {nm.lower()}") == want:
                    return f"{nm} is {want}."
            return None
        m = _Q_COUNT.match(n)
        if m:
            lab = cc.singular({"children": "kid"}.get(m.group(1), m.group(1)))
            k = self._slot(f"count {lab}")
            return f"You have {k} {m.group(1)}." if k else None
        if _Q_JOB.match(n):
            return self._slot("job")
        if _Q_PETS.match(n):
            got = [(l, self._slot(f"pet {l}")) for l in cc.ANIMALS if self._slot(f"pet {l}")]
            if got:
                return "You have " + cc.join_names([f"{l}s called {v}" if _MULTI.search(v) else f"a {l} called {v}" for l, v in got]) + "."
            return None
        m = _Q_PRON.match(n)
        if m and self.focus:
            label = self.focus[0]
            v = self._slot(self._label_slot(label))
            pr = m.group(1) or "its"
            if v:
                if _MULTI.search(v):
                    return f"Their names are {v}."
                return f"{pr.capitalize() if pr in ('his', 'her', 'its') else 'Their'} name is {v}."
            self.focus = (label, True)
            return f"I don't think you've told me your {label}'s name yet. What is it?"
        return None
