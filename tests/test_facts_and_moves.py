"""Run: PYTHONPATH=. python tests/test_facts_and_moves.py"""
import os, tempfile
from ovi.brain import Brain
b = Brain(log=os.path.join(tempfile.mkdtemp(), "b.jsonl")); say = lambda m: b.converse(m).text
fails = []; N = 0
def check(name, got, cond):
    global N; N += 1; ok = cond(got); print(("PASS" if ok else "FAIL"), name, "->", got[:100])
    if not ok: fails.append(name)
check("capital spain", say("what is the capital of spain?"), lambda t: "Madrid" in t)
check("follow-up swap italy", say("and italy?"), lambda t: "Rome" in t)
check("follow-up swap what about", say("what about brazil?"), lambda t: "Bras" in t)
check("currency", say("what currency does japan use?"), lambda t: "yen" in t.lower())
check("population", say("how many people live in iceland?"), lambda t: "Iceland" in t and "3" in t)
check("continent", say("which continent is kenya in?"), lambda t: "Africa" in t)
check("reverse capital", say("Nairobi is the capital of which country?"), lambda t: "Kenya" in t)
check("element symbol", say("what is the chemical symbol for gold?"), lambda t: "Au" in t)
check("element follow-up", say("and silver?"), lambda t: "Ag" in t)
check("element number", say("what is the atomic number of carbon?"), lambda t: "6" in t)
check("discoverer", say("who discovered helium?"), lambda t: "Janssen" in t or "Lockyer" in t or "discovered" in t)
check("trivia", say("Who wrote '1984'?"), lambda t: "Orwell" in t)
check("capital australia (was wrong before)", say("what is the capital of australia?"), lambda t: "Canberra" in t)
check("still abstains on unknown", say("what is the gdp of norway?"), lambda t: "not sure" in t.lower())
check("mood sad", say("I feel really sad today"), lambda t: "sorry" in t.lower())
check("mood tired", say("I'm so tired"), lambda t: "rest" in t.lower() or "draining" in t.lower())
check("mood happy", say("I got great news"), lambda t: "wonderful" in t.lower())
check("entity mention", say("I want to visit portugal someday"), lambda t: "Lisbon" in t)
check("past action", say("I went to the beach yesterday"), lambda t: t.endswith("?"))
check("ack uses likes", (say("I love chess"), say("cool"))[1], lambda t: "chess" in t)
check("generic uses topic", say("the weather has been strange lately"), lambda t: t.endswith(("?", ".")))
print(f"\n{N-len(fails)}/{N} passed", "FAILED: " + ", ".join(fails) if fails else "")
