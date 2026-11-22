"""Does confidence separate 'knows' from 'doesn't know'?  Run: PYTHONPATH=. python experiments/brain_calibration.py"""
import random, re, tempfile, os
from ovi.brain import Brain, ANSWER, HEDGE
from ovi import dataset
random.seed(0)
b = Brain(log=os.path.join(tempfile.mkdtemp(), "b.jsonl"))
cb = [e for e in b.entries if e.src == "cb" and len(e.q.split()) >= 3]
known = random.sample(cb, 150)
def perturb(q):
    w = q.rstrip("?.! ").split()
    if len(w) > 4: w.pop(random.randrange(1, len(w) - 1))   # drop a word
    return " ".join(w).lower() + "?"
unknown = ["who won the 1998 world cup", "what is the boiling point of mercury", "how tall is mount kilimanjaro",
 "what is the population of brazil", "who painted the mona lisa", "when did world war two end", "what is the speed of light",
 "who is the president of mars", "how do i fix my carburetor", "what is the square root of 144", "who invented the telephone",
 "what is the tallest building in dubai", "how many moons does jupiter have", "what is the gdp of norway",
 "where is the eiffel tower located", "what is quantum chromodynamics", "how do i bake sourdough bread",
 "what year was the iphone released", "who wrote war and peace", "what is the chemical symbol for gold",
 "flurb zax quib", "xyzzy plugh", "what is the airspeed of an unladen swallow in kilometers", "translate hello into klingon",
 "how many bones are in the human foot", "what is the capital of australia", "what is the longest river in asia",
 "who discovered penicillin", "what is the melting point of tungsten", "what time does the bank open on sunday"]
def run(qs):
    return [b.ask(q) for q in qs]
for name, qs in (("exact", [e.q for e in known]), ("perturbed", [perturb(e.q) for e in known])):
    res = run(qs)
    ans = [r for r in res if r.conf >= ANSWER]; hed = [r for r in res if HEDGE <= r.conf < ANSWER]; ab = [r for r in res if r.conf < HEDGE]
    ok = sum(1 for r, e in zip(res, known) if r.conf >= HEDGE and r.text.endswith(e.a))
    print(f"KNOWN {name:9}: answered {len(ans)/len(res):.0%}  hedged {len(hed)/len(res):.0%}  abstained {len(ab)/len(res):.0%}  correct {ok/len(res):.0%}")
res = run(unknown)
ans = [r for r in res if r.conf >= ANSWER]; hed = [r for r in res if HEDGE <= r.conf < ANSWER]; ab = [r for r in res if r.conf < HEDGE]
print(f"UNKNOWN      : answered {len(ans)/len(res):.0%}  hedged {len(hed)/len(res):.0%}  abstained {len(ab)/len(res):.0%}")
for q, r in zip(unknown, res):
    if r.conf >= HEDGE: print("  wrongly answered:", q, "->", r.text[:60], f"[{r.conf:.2f}]")
