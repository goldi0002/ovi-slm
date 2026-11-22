"""Run: PYTHONPATH=. python tests/test_conversation.py"""
import os, tempfile
from ovi.brain import Brain
log = os.path.join(tempfile.mkdtemp(), "b.jsonl")
b = Brain(log=log)
say = lambda m: b.converse(m).text
fails = []; N = 0
def check(name, got, cond):
    global N; N += 1; ok = cond(got); print(("PASS" if ok else "FAIL"), name, "->", got[:90])
    if not ok: fails.append(name)

check("greeting", say("hi"), lambda t: "Ovi" in t or "Hello" in t)
check("learn name", say("my name is Sam"), lambda t: "Sam" in t)
check("recall name", say("what is my name?"), lambda t: "Sam" in t)
check("greet by name", say("hello"), lambda t: "Sam" in t)
check("learn like 1", say("I like football"), lambda t: "football" in t)
check("learn like 2", say("I love chess"), lambda t: "chess" in t)
check("recall likes", say("what do I like?"), lambda t: "football" in t and "chess" in t)
check("learn place", say("I live in Lisbon"), lambda t: "Lisbon" in t)
check("recall place", say("where do I live?"), lambda t: "Lisbon" in t)
check("favorite", say("my favorite color is green"), lambda t: "green" in t)
check("recall favorite", say("what is my favorite color?"), lambda t: "green" in t)
check("factual", say("what is the capital of france?"), lambda t: "Paris" in t)
check("follow-up uses context", say("and japan?"), lambda t: "Tokyo" in t)
check("why explains source", say("why?"), lambda t: len(t) > 5)
check("abstains on unknown", say("what is the gdp of norway?"), lambda t: "not sure" in t.lower())
check("learn from abstention", say("About five hundred billion dollars."), lambda t: "remember" in t.lower())
check("recalls learned fact", say("what is the gdp of norway?"), lambda t: "five hundred" in t)
check("statement -> chat move", say("The weather has been strange lately."), lambda t: t.endswith(("?", ".")) and "not sure" not in t.lower())
check("implicit confirmation -> confident", say("what is the gdp of norway?"), lambda t: not t.startswith("I'm not fully sure"))
check("plain correction", say("what is my favorite color?"), lambda t: "green" in t)
check("  ...no, it's blue", say("no, it's blue"), lambda t: "learned" in t.lower())
check("  ...now says blue", say("what is my favorite color?"), lambda t: "blue" in t.lower())
# persistence across restart
b2 = Brain(log=log)
check("restart: name", b2.converse("what is my name?").text, lambda t: "Sam" in t)
check("restart: likes", b2.converse("what do I like?").text, lambda t: "chess" in t)
check("restart: learned fact", b2.converse("what is the gdp of norway?").text, lambda t: "five hundred" in t)
print(f"\n{N-len(fails)}/{N} passed", "FAILED: " + ", ".join(fails) if fails else "")
