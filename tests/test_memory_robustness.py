"""Run: PYTHONPATH=. python tests/test_memory_robustness.py   (temp paths only; never touches data/brain.jsonl)"""
import json, os, tempfile, threading, subprocess, sys
from ovi.brain import Brain, compact_log
from ovi.memory import Memory

fails = []; N = 0


def check(name, cond):
    global N; N += 1
    print(("PASS" if cond else "FAIL"), name)
    if not cond: fails.append(name)


def tmp():
    return os.path.join(tempfile.mkdtemp(), "brain.jsonl")


def mk(log, builtin=True):
    b = Brain(log=log, builtin=builtin); return b, (lambda m: b.converse(m).text)


# 1) corrections supersede
L = tmp(); b, say = mk(L)
say("my name is Sam"); say("actually my name is Max")
ans = say("what is my name?")
check("actually-correction answers new", "Max" in ans and "Sam" not in ans)
live = [e.a for e in b.entries if not e.dead and e.q == "what is my name" and e.src == "taught"]
check("old value not kept alive", live == ["Your name is Max."])
say("no, it's Bob")
check("'no, it's Y' updates all name questions", "Bob" in say("who am i?") and "Bob" in say("what is my name?"))
say("what is my name?"); say("no, it's Zed")
b2, say2 = mk(L)
check("correction survives restart", "Zed" in say2("what is my name?") and "Bob" not in say2("who am i?"))
b3, say3 = mk(tmp())
say3("I live in Paris"); say3("I live in Rome")
ans = say3("where do i live?")
check("place superseded", "Rome" in ans and "Paris" not in ans)

# 2) forgetting with tombstone
L = tmp(); b, say = mk(L)
say("my name is Sam"); say("forget my name")
ev = [json.loads(l) for l in open(L)]
check("tombstone event logged", any(e["e"] == "del" and e["q"] == "what is my name" for e in ev))
check("add events still in the append-only log", sum(e["e"] == "add" for e in ev) == 5)
check("forgotten after request", "Sam" not in say("what is my name?"))
b2, say2 = mk(L)
check("forgotten after restart (replay)", "Sam" not in say2("what is my name?"))
say2("my name is Ann")
check("can re-teach after forgetting", "Ann" in say2("what is my name?"))
b3, say3 = mk(tmp())
say3("my favorite color is blue"); say3("forget that")
check("'forget that' after a statement", "Your favorite color" not in say3("what is my favorite color?"))
check("forget with nothing to forget is safe", isinstance(say3("forget my pet"), str))
check("builtin entries are not forgettable", Brain(log=tmp(), builtin=False).converse("forget that").text != "")

# 3) compaction
L = tmp(); b, say = mk(L)
say("my name is Sam"); say("my name is Max"); say("forget my name"); say("I live in Oslo")
before = len(open(L).read().splitlines())
res = compact_log(L)
after = open(L).read().splitlines()
check("compaction shrinks log", len(after) < before and res["after"] == len(after))
check("backup kept", os.path.exists(L + ".bak") and len(open(L + ".bak").read().splitlines()) == before)
check("no temp file left", not os.path.exists(L + ".tmp"))
b2, say2 = mk(L)
check("compacted state equivalent", "Oslo" in say2("where do i live?") and "Max" not in say2("what is my name?"))
open(L + ".tmp", "w").write("garbage{")          # leftover of an interrupted run
compact_log(L)
check("stale tmp tolerated", Brain(log=L, builtin=False).bad_lines == 0 and not os.path.exists(L + ".tmp"))
check("Brain.compact method", Brain(log=L, builtin=False).compact()["bad"] == 0)

# 4) corrupt lines
L = tmp()
good = json.dumps({"e": "add", "q": "what is my pet", "a": "Your pet is a cat.", "src": "taught", "conf": 1.0})
with open(L, "w") as f:
    f.write(good + "\n{\"e\": \"add\", \"q\": \"trunc\n" + "not json\n" + "[1,2]\n" + '{"e":"add","q":1}\n' + '{"e":"wat","q":"x"}\n\n')
    f.write(good[:20])                       # truncated final line, no newline
b = Brain(log=L, builtin=False)
check("bad lines counted", b.bad_lines == 5 + 1)
check("good line kept", b.converse("what is my pet?").text == "Your pet is a cat.")
b.teach("what is my dog", "Rex.")
tail = open(L).read().splitlines()[-1]
check("append after truncated line starts fresh line", json.loads(tail)["q"] == "what is my dog")
with open(L, "ab") as f: f.write(b"\xff\xfe\x80 bad bytes\n")
check("non-utf8 bytes in log tolerated", Brain(log=L, builtin=False).bad_lines >= 7)
M = tmp()
with open(M, "w") as f: f.write('["what is x","y"]\n["trunc\n')
m = Memory(M)
check("Memory skips bad lines", m.bad_lines == 1 and m.recall("what is x") == "y")
check("compaction counts bad lines", compact_log(L)["bad"] >= 7)

# 5) input safety
b, say = mk(tmp(), builtin=False)
check("empty input", isinstance(say(""), str) and isinstance(say("   "), str))
check("very long input", isinstance(say("a " * 8000 + "?"), str))
L = tmp(); b, say = mk(L, builtin=False)
say("my favorite thing is " + "z" * 20000)
check("stored text is capped", all(len(l) < 6000 for l in open(L)))
check("bytes input", isinstance(b.converse(b"caf\xe9 \xff hello").text, str))
check("lone surrogate / NUL", isinstance(b.converse("hi \ud800 there\x00").text, str))
b.teach("what is x", "\ud800\x00 weird")
check("log still loads after weird teach", Brain(log=L, builtin=False).bad_lines == 0)

# concurrent writers: threads + processes, no interleaved/lost lines
L = tmp()
def work(k):
    bb = Brain(log=L, builtin=False)
    for i in range(25): bb.teach(f"what is item {k} {i}", "v" * 300 + str(i))
ts = [threading.Thread(target=work, args=(k,)) for k in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
code = ("from ovi.brain import Brain\nimport sys\nb=Brain(log=sys.argv[1],builtin=False)\n"
        "for i in range(25): b.teach('what is proc %s %d' % (sys.argv[2], i), 'p'*300)\n")
ps = [subprocess.Popen([sys.executable, "-c", code, L, str(k)], env=dict(os.environ, PYTHONPATH=os.getcwd())) for k in range(3)]
[p.wait() for p in ps]
lines = open(L).read().splitlines()
check("concurrent writers: all lines intact", len(lines) == 175 and all(json.loads(l)["e"] == "add" for l in lines))
b = Brain(log=L, builtin=False)
check("concurrent writers: nothing lost", b.bad_lines == 0 and len(b.entries) == 175)

# 6) name handling
for phrase, nm in [("I'm Sam", "Sam"), ("I am Dana", "Dana"), ("my name is Kim", "Kim"), ("call me Bo", "Bo"),
                   ("this is Lee", "Lee"), ("I'm Zoë", "Zoë"), ("我是 李雷", None), ("my name is 李雷", "李雷"), ("hi, I'm Ann", "Ann")]:
    if nm is None: continue
    b, say = mk(tmp())
    say(phrase)
    check(f"learns name from {phrase!r}", nm in say("what is my name?"))
for phrase in ["I'm tired", "I am happy", "i'm fine", "I'm hungry", "this is great", "I'm going home", "I am so sad", "i'm back", "I'm Looking"]:
    b, say = mk(tmp())
    say(phrase)
    check(f"{phrase!r} is not a name", "Your name is" not in say("what is my name?") and b._known_name() is None)
b, say = mk(tmp())
for q in ("what is my name", "who am I?", "do you remember my name?"):
    r = say(q)
    check(f"unknown name: {q!r}", "don't know" in r and "Ovi" not in r)
say("my name is Zoë")
check("who am I with history", "Zoë" in say("who am I?") and "Zoë" in say("do you remember my name?"))
check("greeting uses name", "Zoë" in say("hi again") and "Zoë" in say("hello"))
L = tmp(); b, say = mk(L); say("call me 李雷")
b2, say2 = mk(L)
check("non-ASCII name survives restart", "李雷" in say2("what is my name?") and "李雷" in say2("hey again"))

print(f"\n{N - len(fails)}/{N} passed", "" if not fails else fails)
sys.exit(1 if fails else 0)
