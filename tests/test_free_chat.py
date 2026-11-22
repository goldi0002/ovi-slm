"""Regression tests for free-conversation patterns. Run: PYTHONPATH=. python tests/test_free_chat.py (temp brain paths only)."""
import os, tempfile
from ovi.brain import Brain

fails = []; N = 0
TEACH = "what should i have said"


def talk(*msgs):
    b = Brain(log=os.path.join(tempfile.mkdtemp(), "b.jsonl"))
    return [b.converse(m).text for m in msgs]


def check(name, got, cond):
    global N; N += 1; ok = cond(got); print("PASS" if ok else "FAIL", name, "->", got[:110])
    if not ok: fails.append(name)


# 1) teach prompts only for substantive factual questions; a pending question does not swallow the next statement
r = talk("why not")
check("'why not' is not a teach prompt", r[-1], lambda t: TEACH not in t.lower())
r = talk("any tips?")
check("'any tips?' is not a teach prompt", r[-1], lambda t: TEACH not in t.lower())
r = talk("who is miles davis")
check("factual unknown still asks to be taught", r[-1], lambda t: TEACH in t.lower())
r = talk("who is miles davis", "you're no fun")
check("remark to Ovi is not taken as the answer", r[-1], lambda t: "remember that" not in t.lower())
r = talk("what is a quasar", "i have a test tomorrow", "what is a quasar")
check("statement about the user is not swallowed", r[1], lambda t: "remember that" not in t.lower())
check("...and the question is still unanswered", r[2], lambda t: TEACH in t.lower())
r = talk("who is miles davis", "a jazz trumpeter", "who is miles davis")
check("a real answer is still learned", r[-1], lambda t: "jazz trumpeter" in t)

# 2) recalling what the user said
r = talk("i went to the beach today", "we ate ice cream", "what did i do today?")
check("what did I do today", r[-1], lambda t: "beach" in t and "ice cream" in t)
r = talk("i cooked dinner for the first time", "what did i cook?")
check("what did I cook", r[-1], lambda t: "cooked dinner" in t)
r = talk("what's your favorite color", "mine is blue", "what is my favorite color?")
check("'mine is blue' resolves to the last favorite category", r[-1], lambda t: "Your favorite color is blue" in t)
r = talk("what's your favorite color", "mine is blue", "what's my favourite colour")
check("what's / favourite spelling", r[-1], lambda t: "blue" in t and "Your" in t)
r = talk("i like cats", "i have two, Milo and Luna", "what are my cats called?")
check("pets after 'i like cats'", r[-1], lambda t: "Milo" in t and "Luna" in t)
r = talk("i like cats", "i have two", "Tom and Jerry", "what are my cats called?")
check("pets named in a separate turn", r[-1], lambda t: "Tom" in t and "Jerry" in t)

# 3) feelings: only first-person statements get the empathy reply
for m in ("that's kinda sad", "she gets upset easy", "he is so angry all the time"):
    check(f"third person '{m}'", talk(m)[-1], lambda t: "sorry you're feeling" not in t.lower() and "draining" not in t.lower() and "weighing" not in t.lower())
for m in ("i feel sad today", "i'm so sad", "my boss makes me sad"):
    check(f"first person '{m}'", talk(m)[-1], lambda t: "sorry" in t.lower())

# 4) Ovi's own profile and advice requests
r = talk("tell me about yourself")
check("tell me about yourself", r[-1], lambda t: "Ovi" in t and "CPU" in t and "teach" in t)
for m in ("any advice", "got any advice", "any tips?", "i need some advice"):
    r = talk("i have a presentation tomorrow", m)
    check(f"advice request '{m}'", r[-1], lambda t: ("tell me" in t.lower() or "what" in t.lower()) and "leaning" not in t and TEACH not in t.lower())

print(f"\n{N - len(fails)}/{N} passed", "" if not fails else fails)
raise SystemExit(1 if fails else 0)
