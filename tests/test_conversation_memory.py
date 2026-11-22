"""Tests for conversation-memory recall (ovi/convmem.py), noun-phrase template safety, and routing before the generic fallbacks.
Run: PYTHONPATH=. python tests/test_conversation_memory.py   (temp brain logs only; data/brain.jsonl is never touched)
Phrasings here are invented for the test, not copied from any transcript."""
import json, os, re, tempfile
from ovi.brain import Brain
from ovi import chitchat as cc

fails = []; N = 0
UNKNOWN = ("don't know", "haven't told", "not told", "don't think you", "not sure what")


def tmp():
    return os.path.join(tempfile.mkdtemp(), "b.jsonl")


def new(log=None):
    return Brain(log=log or tmp())


def talk(*msgs, b=None):
    b = b or new()
    return [b.converse(m).text for m in msgs]


def check(name, got, cond):
    global N; N += 1; ok = cond(got); print("PASS" if ok else "FAIL", name, "->", got[:150])
    if not ok: fails.append(name)


def has(*words):
    return lambda t: all(w.lower() in t.lower() for w in words)


def hasnt(*words):
    return lambda t: not any(w.lower() in t.lower() for w in words)


def unknown(t):
    return any(u in t.lower() for u in UNKNOWN)


# ---- 1) single-fact recall through different question shapes
cases = [
    (("we adopted a rabbit last week, his name is Clover", "what was my rabbit called?"), has("Clover")),
    (("we adopted a rabbit last week, his name is Clover", "do you remember my rabbit's name"), has("Clover")),
    (("i grew up in Porto but these days i live in leeds", "which town do i live in now"), has("Leeds")),
    (("so my uncle repairs bicycles for a living", "what does my uncle do?"), has("bicycles")),
    (("my flat has three bedrooms", "how many bedrooms does my flat have"), has("three")),
    (("i'm allergic to peanuts", "what was i allergic to again?"), has("peanuts")),
    (("my driving test is on thursday morning", "when is my driving test"), has("thursday")),
    (("I'm saving up for a trip to Norway", "what am I saving up for"), has("Norway")),
    (("our wedding was in september", "when was our wedding?"), has("september")),
    (("my roommate studies marine biology", "what does my roommate study"), has("marine biology")),
    (("I've got a younger sister, she's called Ines", "what's my sister's name?"), has("Ines")),
    (("I drive a battered old Volvo", "what car do I drive?"), has("Volvo")),
    (("I teach sixth grade maths", "which subject do I teach?"), has("maths")),
    (("I'm learning the accordion these days", "what instrument am I learning?"), has("accordion")),
    (("my neighbour owns a small bakery", "what does my neighbour own"), has("bakery")),
    (("I volunteer at an animal shelter on weekends", "where do I volunteer?"), has("animal shelter")),
]
for msgs, cond in cases:
    check("recall: " + msgs[-1], talk(*msgs)[-1], cond)

# ---- 2) several sentences about the same thing / pronoun carry-over / answers combine
t = talk("my cousin lives in Denver", "she is a vet", "she also plays cello", "what did I tell you about my cousin?")[-1]
check("aggregate about a person", t, has("Denver", "vet"))
t = talk("we have a goldfish", "its name is Bubbles", "what is my goldfish called")[-1]
check("name given in a later sentence", t, has("Bubbles"))
t = talk("I study chemistry and also history", "what are the two things I study?")[-1]
check("list question", t, has("chemistry", "history"))

# ---- 3) later sentence beats an earlier one (a correction in plain words)
t = talk("my commute is forty minutes", "ugh, it's grim", "actually my commute is thirty minutes these days", "how long is my commute?")[-1]
check("recency wins", t, lambda x: "thirty" in x.lower() and "forty" not in x.lower())

# ---- 4) only say 'don't know' when nothing relevant was said; quote when something related was
t = talk("I like jazz", "what did I say about my boat?")[-1]
check("nothing relevant -> honest", t, lambda x: unknown(x) and "jazz" not in x)
t = talk("I like jazz", "where does my aunt work?")[-1]
check("unrelated memory is not quoted", t, lambda x: "jazz" not in x)
t = talk("my sister is a pilot", "what does my brother do?")[-1]
check("wrong person is not answered", t, lambda x: "pilot" not in x)
t = talk("I live somewhere near the coast", "what city do I live in")[-1]
check("no city said -> no invented city", t, lambda x: unknown(x) or "not sure" in x.lower())
t = talk("my sister is a pilot", "did I tell you about my sister?")[-1]
check("yes/no recall", t, has("pilot"))
t = talk("my sister is a pilot", "did I mention my brother?")[-1]
check("yes/no: not mentioned", t, lambda x: "pilot" not in x)

# ---- 5) questions about Ovi / advice are not hijacked by the memory
t = talk("I play chess", "what do you think about my chess plan?")[-1]
check("question to Ovi untouched", t, lambda x: "you said" not in x.lower())
t = talk("I play chess", "how do I get better at it?")[-1]
check("how-to goes to advice", t, lambda x: "you said" not in x.lower() and "?" in x)

# ---- 6) forgetting
b = new()
talk("my niece is called Poppy and she is five", b=b)
check("niece recalled", talk("what's my niece called?", b=b)[0], has("Poppy"))
check("forget acknowledged", talk("forget my niece", b=b)[0], has("forgotten"))
check("niece gone after forget", talk("what's my niece called?", b=b)[0], lambda x: "Poppy" not in x)
b = new()
talk("I have a fear of heights", b=b)
check("fear recalled", talk("what am I afraid of?", b=b)[0], has("heights"))
b = new()
talk("my passport expires in march", b=b)
talk("forget that", b=b)
check("'forget that' removes the last statement", talk("when does my passport expire?", b=b)[0], lambda x: "march" not in x.lower())
b = new()
talk("my boss is called Dennis", "I work in a call centre", b=b)
talk("forget everything", b=b)
check("forget everything clears sentences", talk("where do I work?", b=b)[0], lambda x: "call centre" not in x.lower())
b = new()
talk("I'm 33", b=b)
talk("forget my age", b=b)
check("forgetting a slot also forgets its sentence", talk("how old am I?", b=b)[0], lambda x: "33" not in x)

# ---- 7) persistence across restarts (event log), including tombstones
L = tmp()
b = new(L)
talk("my dentist is on Elm Street", "I keep a diary in French", b=b)
b2 = new(L)
check("restart: recalled", talk("which street is my dentist on?", b=b2)[0], has("Elm"))
check("restart: second sentence", talk("what language do I keep my diary in", b=b2)[0], has("French"))
talk("forget my dentist", b=b2)
b3 = new(L)
check("restart: tombstone holds", talk("which street is my dentist on?", b=b3)[0], lambda x: "Elm" not in x)
check("restart: other sentence survives", talk("what language do I keep my diary in", b=b3)[0], has("French"))
ev = [json.loads(l) for l in open(L)]
check("sentences are in the append-only log", str(sum(e["e"] == "add" and e["q"].startswith("zzsay ") for e in ev) >= 2), lambda x: x == "True")
check("stored sentences are not reachable by fuzzy retrieval", talk("zzsay", b=new(L))[0], lambda x: "dentist" not in x.lower())

# ---- 8) slots still answer first (precise path unchanged)
check("slot first", talk("I play the oboe", "what instrument do I play?")[-1], has("oboe"))
check("pet slot first", talk("I have a dog named Rufus", "what is my dog's name?")[-1], has("Rufus"))

# ---- 9) template safety: no fragments like 'the herself' / 'the tiny' / 'the grey'
for msg in ["she's a doctor herself", "she screamed at a tiny one yesterday", "im a tea person, earl grey", "he fixed the old one himself",
            "it was the red and the blue", "i met my cousin's new girlfriend, she's lovely"]:
    r = talk(msg)[0]
    bad = re.search(r"\b(?:the|your|a) (?:herself|himself|myself|itself|tiny|grey|gray|red|blue|lovely|one|new|old)\b[?.]", r, re.I)
    check("no broken fragment: " + msg, r, lambda x: not bad)
for p, ok in [("the herself", False), ("the tiny", False), ("the grey", False), ("your kept", False), ("the them", False), ("the running", False), ("the my", False),
              ("the weather", True), ("your boss", True), ("the presentation", True), ("your sister", True), ("the party", True), ("the meeting", True)]:
    check(f"np_ok({p!r})", str(cc.np_ok(p)), lambda x, ok=ok: (x == "True") == ok)

# ---- 10) route before the generic fallbacks
FEEL = ("feeling about all that", "I'm listening", "Is there more to it", "What else is on your mind")
check("arithmetic 'whats'", talk("whats 14 times 6")[0], has("84"))
check("arithmetic spelled", talk("what's 144 divided by 12")[0], has("12"))
check("arithmetic 'whats' plus", talk("whats 230 plus 19")[0], has("249"))
t = talk("whats the capital of bhutan")[0]
check("unknown capital: honest, not a check-in", t, lambda x: not any(f in x for f in FEEL) and ("capital" in x.lower() or "Thimphu" in x))
check("known capital still works", talk("whats the capital of japan")[0], has("Tokyo"))
for q in ["whats the best way to learn the piano", "any suggestions for a weekend trip", "how do i get better at sudoku", "tips for sleeping better", "any recs for a cozy movie night"]:
    check("think-it-through: " + q, talk(q)[0], lambda x: "?" in x and not any(f in x for f in FEEL) and "feeling" not in x.lower())
for bye in ["adios", "night night", "gn", "ttyl", "cya", "later", "peace out", "g2g", "ciao", "nighty night", "i'm off to bed", "toodles"]:
    check("goodbye: " + bye, talk(bye)[0], lambda x: re.search(r"bye|see you|talk to you|take care|come back", x, re.I) is not None)

print(f"\n{N - len(fails)}/{N} passed", ("FAILED: " + ", ".join(fails)) if fails else "")
raise SystemExit(1 if fails else 0)
