"""Tests for the general attribute/value slot memory, intent rules and slot-fill filters.
Run: PYTHONPATH=. python tests/test_recall_slots.py   (temp brain logs only; data/brain.jsonl is never touched)"""
import os, tempfile
from ovi.brain import Brain

fails = []; N = 0
UNKNOWN = "don't know that about you"


def new(log=None):
    return Brain(log=log or os.path.join(tempfile.mkdtemp(), "b.jsonl"))


def talk(*msgs, b=None):
    b = b or new()
    return [b.converse(m).text for m in msgs]


def check(name, got, cond):
    global N; N += 1; ok = cond(got); print("PASS" if ok else "FAIL", name, "->", got[:120])
    if not ok: fails.append(name)


def has(*words):
    return lambda t: all(w.lower() in t.lower() for w in words)


# ---- 1) recall of facts stated in plain sentences (varied phrasings, not only the ones from the transcripts)
cases = [
    (("I have a dog named Max", "what is my dog's name?"), has("Max")),
    (("we got a parrot called Polly", "what's my parrot called"), has("Polly")),
    (("anyway, I own a horse named Thunder", "what is my horse called?"), has("Thunder")),
    (("my cat is Luna", "what's my cat's name?"), has("Luna")),
    (("I just adopted a puppy", "his name is Rex", "who is my puppy?"), has("Rex")),
    (("I play the violin", "what instrument do I play?"), has("violin")),
    (("I play tennis on weekends", "what sport do I play?"), has("tennis")),
    (("I'm 25", "how old am I?"), has("25")),
    (("I am 41 years old", "what's my age?"), has("41")),
    (("I turned 30", "how old am I"), has("30")),
    (("I love hiking", "what do I love?"), has("hiking")),
    (("I enjoy pottery", "what are my hobbies?"), has("pottery")),
    (("I work as a nurse", "what's my job again?"), has("nurse")),
    (("I work as an electrician", "what do I do for a living?"), has("electrician")),
    (("my favorite color is blue", "what's my favourite colour?"), has("blue")),
    (("my car is red", "what colour is my car?"), has("red")),
    (("my phone number is 555 0199", "what's my phone number?"), has("555 0199")),
    (("my brother's name is Tom", "what is my brother called?"), has("Tom")),
    (("I live in Lisbon", "what city do I live in?"), has("Lisbon")),
    (("I live in Lisbon", "where do I live?"), has("Lisbon")),
    (("I run every morning", "what do I do every morning?"), has("run", "morning")),
    (("I swim on sundays", "what do I do on sundays?"), has("swim")),
    (("I play guitar", "been playing for ten years", "how long have I played?"), has("ten years")),
    (("I'm going to Japan next year", "where am I going?"), has("Japan")),
    (("my mom makes the best lasagna", "she lives in Naples", "where does my mom live?"), has("Naples")),
    (("my mom makes the best lasagna", "what does she make?"), has("lasagna")),
    (("my sister plays for a local team", "what does my sister play for?"), has("local team")),
    (("I just adopted a cat", "her name is Luna", "she's black with white paws", "what color is she?"), has("black")),
    (("last weekend I went to Colorado", "where did I go last weekend?"), has("Colorado")),
    (("omg you won't believe it, I won tickets to a game", "what did I win?"), has("tickets")),
    (("I'm Dana, remember that", "what's my name?"), has("Dana")),
    (("remember that my name is Priya", "what's my name?"), has("Priya")),
    (("I'm 25", "I love painting", "what do you remember about me?"), has("25", "painting")),
]
for turns, cond in cases:
    check(f"recall: {turns[0]} ... {turns[-1]}", talk(*turns)[-1], cond)

# the honest answer when nothing was said is still "I don't know"
check("unknown attribute stays unknown", talk("I'm 25", "what is my dog's name?")[-1], lambda t: "Max" not in t and "25" not in t)
check("no facts: how old am I", talk("how old am I?")[-1], lambda t: UNKNOWN in t)

# taught "X is Y" facts
r = talk("hey ovi, teach you something", "the tallest mountain in the world is Everest", "what is the tallest mountain?")
check("taught statement: teach cue", r[0], lambda t: "teach" in t.lower() or "tell me" in t.lower())
check("taught statement recalled", r[-1], has("Everest"))
r = talk("the largest planet is Jupiter", "what's the largest planet?")
check("taught superlative without cue", r[-1], has("Jupiter"))
r = talk("let me teach you something", "Paris is the capital of France", "what is the capital of France?")
check("taught 'X is the capital of Y'", r[-1], has("Paris"))

# ---- 2) forget / tombstone
b = new()
talk("I'm 25", "I play guitar", "my car is red", "I'm Lena", b=b)
check("forget my age", talk("forget my age", b=b)[-1], has("forgotten"))
check("age forgotten", talk("how old am I?", b=b)[-1], lambda t: "25" not in t)
check("other slots survive", talk("what instrument do I play?", b=b)[-1], has("guitar"))
check("forget what I play", talk("forget that I play guitar", b=b)[-1], has("forgotten"))
check("play forgotten", talk("what do I play?", b=b)[-1], lambda t: "guitar" not in t)
check("forget my car", talk("forget my car", b=b)[-1], has("forgotten"))
check("car forgotten", talk("what colour is my car?", b=b)[-1], lambda t: "red" not in t)

# persistence + tombstones survive a restart
log = os.path.join(tempfile.mkdtemp(), "b.jsonl")
b = new(log)
talk("I'm 29", "I play the drums", "I run every morning", b=b)
b2 = new(log)
check("restart: age", talk("how old am I?", b=b2)[-1], has("29"))
check("restart: play", talk("what do I play?", b=b2)[-1], has("drums"))
talk("forget my age", b=b2)
b3 = new(log)
check("restart: tombstone holds", talk("how old am I?", b=b3)[-1], lambda t: "29" not in t)
check("restart: other fact kept", talk("what do I do every morning?", b=b3)[-1], has("run"))
b3.compact()
check("after compaction: still forgotten", talk("how old am I?", b=new(log))[-1], lambda t: "29" not in t)
check("after compaction: kept", talk("what do I play?", b=new(log))[-1], has("drums"))

# ---- 3) intents instead of generic acknowledgements
GENERIC = ("what else is on your mind", "i'm listening", "makes sense", "is there more to it")
for m in ("bye", "lol bye", "ok bye then", "goodnight", "gotta go, see ya"):
    check(f"goodbye '{m}'", talk("hi", m)[-1], lambda t: any(w in t.lower() for w in ("bye", "see you", "take care", "talk to you")))
r = talk("tell me a joke", "another")
check("another joke", r[-1], lambda t: "another" in t.lower() and "what" not in t.lower().split("another")[0])
r = talk("tell me a fun fact", "one more")
check("another fact", r[-1], lambda t: "another" in t.lower())
check("another with no context", talk("another")[-1], lambda t: "another what" in t.lower())
for m in ("entertain me", "amuse me"):
    check(f"'{m}'", talk(m)[-1], lambda t: "?" in t and not any(g in t.lower() for g in GENERIC))
check("bored", talk("bored")[-1], lambda t: "joke" in t.lower() and "boredom" not in t)
check("wanna hear a joke", talk("wanna hear a joke?")[-1], lambda t: "teach" not in t.lower() and "should i have said" not in t.lower())
for m in ("ok that was lame", "that was so boring", "that joke was terrible"):
    check(f"that was lame '{m}'", talk("tell me a joke", m)[-1], lambda t: "humor" in t.lower() or "fair" in t.lower())
for m in ("good memory", "you have a great memory", "wow you remembered"):
    check(f"compliment '{m}'", talk(m)[-1], lambda t: "thank" in t.lower())
check("thanks variants", talk("thanks a ton for the chat")[-1], lambda t: not any(g in t.lower() for g in GENERIC))
check("you're weird", talk("you're weird")[-1], lambda t: "let you down" not in t and "went wrong" not in t)
check("u there", talk("lol u there")[-1], lambda t: "here" in t.lower())
check("wat r u doin", talk("wat r u doin")[-1], lambda t: "chatting" in t.lower())
check("do you know?", talk("do you know?")[-1], lambda t: "gregory" not in t.lower() and "know what" in t.lower())

# ---- 4) slot-fill filter: never ask about a verb / auxiliary / contraction
BAD_TOPIC = ("your kept", "your didn't", "your start", "your help", "your wouldn't", "your went", "your got", "your doesn't")
for m in ("my boss kept piling stuff on me", "my alarm didn't go off", "and then my car wouldn't start", "my team doesn't help",
          "my roommate left dishes everywhere again", "my neighbor got a new puppy and it barks all night"):
    check(f"topic filter: {m}", talk(m)[-1], lambda t: not any(x in t.lower() for x in BAD_TOPIC))
check("topic keeps the noun", talk("my boss kept piling stuff on me")[-1], lambda t: "boss" in t.lower() or "?" in t)

# ---- 5) no teach prompt on personal / advice questions
TEACH = "what should i have said"
for m in ("how do I make friends?", "what do I say to him?", "how can I deal with my stress?", "what should he do?", "what if I fail?", "how do I handle a bad boss?"):
    check(f"no teach prompt: {m}", talk(m)[-1], lambda t: TEACH not in t.lower() and UNKNOWN not in t)
check("factual unknown still asks to be taught", talk("who is miles davis")[-1], lambda t: TEACH in t.lower())

# ---- 6) smaller fixes
r = talk("I live in Toronto", "it's freezing here")
check("adjective does not overwrite the city", r[-1], lambda t: "updated where you live" not in t and "freezing here" not in t)
b = new(); talk("I live in Toronto", "it's freezing here", b=b)
check("...and the city is still Toronto", talk("where do I live?", b=b)[-1], has("Toronto"))
r = talk("I live in Toronto", "no, it's Montreal")
check("real correction still works", r[-1], lambda t: "lowered my confidence" not in t)
b = new(); talk("I live in Toronto", "no, it's Montreal", b=b)
check("...and updates the city", talk("where do I live?", b=b)[-1], has("Montreal"))
for m in ("my name is Zoe", "what's my name?", "no it's Zed"):
    pass
r = talk("my name is Zoe", "what's my name?", "no, it's Zed", "what's my name?")
check("no internal confidence text in chat", " ".join(r), lambda t: "confidence" not in t.lower())
check("name correction applied", r[-1], has("Zed"))
r = talk("I feel stupid")
check("I feel stupid -> supportive", r[-1], lambda t: "sorry" in t.lower() and "makes sense" not in t)
check("I'm not stupid right?", talk("I'm not stupid right?")[-1], lambda t: TEACH not in t.lower() and "don't think you are" in t)
check("long hard message", talk("I've had a rough week, my grandmother is in the hospital and I've been going back and forth every day and also work has been crazy")[-1],
      lambda t: "sorry" in t.lower() or "carry" in t.lower())
r = talk("I'm going to Japan next year!", "what's the capital?")
check("capital resolved from context", r[-1], has("Tokyo"))
check("bare capital with no context is not France", talk("what's the capital?")[-1], lambda t: "France" not in t)
r = talk("what's the capital of germany?", "what is a noun?", "what was the first thing I asked?")
check("what was the first thing I asked", r[-1], has("germany"))

print(f"\n{N - len(fails)}/{N} passed", "" if not fails else fails)
raise SystemExit(1 if fails else 0)
