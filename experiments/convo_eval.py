"""Scripted multi-turn conversation eval for ovi.brain.Brain.

Run: PYTHONPATH=. python experiments/convo_eval.py [--cases experiments/convo_cases.json] [--jobs N] [-v]

Each case is a list of turns; each turn has checks (contains / not_contains / regex / not_regex / nonempty).
Every case runs on a throw-away brain log in a temp dir (data/brain.jsonl is never touched). A turn with
"restart_before": true rebuilds the Brain from the same temp log first (persistence test).
The built-in knowledge is loaded once and forked per case, so the whole run takes well under 2 minutes.
Exit code is 0 always (this is a report, not a gate); failures are listed with full transcripts.
"""
import argparse, json, multiprocessing as mp, os, re, shutil, sys, tempfile, traceback, collections

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ovi.brain import Brain

TEMPLATE = None


def run_check(chk, reply):
    t, v = chk["type"], chk["value"]
    if t == "contains":
        return v.lower() in reply.lower()
    if t == "not_contains":
        return v.lower() not in reply.lower()
    if t == "regex":
        return re.search(v, reply, re.I | re.S) is not None
    if t == "not_regex":
        return re.search(v, reply, re.I | re.S) is None
    if t == "nonempty":
        return bool(reply.strip())
    raise ValueError(t)


def run_case(case):
    """Runs in a forked child: returns (id, category, ok, transcript)."""
    tmp = tempfile.mkdtemp(prefix="ovi_convo_")
    log = os.path.join(tmp, "brain.jsonl")
    b = TEMPLATE
    b.log = log
    transcript, ok = [], True
    try:
        for turn in case["turns"]:
            if turn.get("restart_before"):
                b = Brain(log=log)
                transcript.append({"restart": True})
            try:
                reply = b.converse(turn["say"]).text
                crashed = None
            except Exception:
                reply, crashed = "", traceback.format_exc(limit=3)
            failed = [f'{c["type"]} {c["value"]!r}' for c in turn["checks"] if crashed or not run_check(c, reply)]
            if crashed:
                failed.insert(0, "CRASH")
            ok = ok and not failed
            transcript.append({"say": turn["say"], "reply": crashed or reply, "failed": failed})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return case["id"], case["category"], ok, transcript


def short(s, n=140):
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + f"...[{len(s)} chars]"


def main():
    global TEMPLATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "convo_cases.json"))
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("-v", action="store_true", help="print passing transcripts too")
    a = ap.parse_args()
    cases = json.load(open(a.cases))["cases"]
    TEMPLATE = Brain(log=os.path.join(tempfile.mkdtemp(prefix="ovi_convo_t_"), "t.jsonl"))
    if a.jobs > 1:
        with mp.get_context("fork").Pool(a.jobs, maxtasksperchild=1) as p:   # fresh fork per case => no state leaks
            res = p.map(run_case, cases, chunksize=1)
    else:
        res = [run_case(c) for c in cases]

    cats = collections.OrderedDict()
    for _, cat, ok, _ in res:
        s = cats.setdefault(cat, [0, 0]); s[1] += 1; s[0] += ok
    print(f"{'category':<14} {'pass':>5} {'total':>6} {'rate':>7}")
    for cat, (p, t) in cats.items():
        print(f"{cat:<14} {p:>5} {t:>6} {100*p/t:>6.1f}%")
    P, T = sum(v[0] for v in cats.values()), sum(v[1] for v in cats.values())
    print(f"{'TOTAL':<14} {P:>5} {T:>6} {100*P/T:>6.1f}%\n")

    for cid, cat, ok, tr in res:
        if ok and not a.v:
            continue
        print(f"{'PASS' if ok else 'FAIL'} [{cat}] {cid}")
        for t in tr:
            if t.get("restart"):
                print("    --- restart brain from same log ---"); continue
            print(f"    user: {short(t['say'])}")
            print(f"    ovi : {short(t['reply'])}")
            for f in t["failed"]:
                print(f"      x {f}")
        print()


if __name__ == "__main__":
    main()
