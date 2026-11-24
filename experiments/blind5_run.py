"""Blind set 5: the same fresh conversations through Ovi without / with the small generator, X/Y randomly assigned.
Run: PYTHONPATH=. python experiments/blind5_run.py"""
import json, os, random, tempfile
from ovi.brain import Brain
convos = json.load(open("experiments/blind5_convos.json"))
rng = random.Random(5); res = {}; mp = {}
for tag, gen in (("old", False), ("new", True)):
    out = []
    for c in convos:
        b = Brain(log=os.path.join(tempfile.mkdtemp(), "b.jsonl"), gen=gen)
        out.append({"id": c["id"], "turns": [{"user": u, "ovi": b.converse(u).text} for u in c["turns"]]})
    res[tag] = out
    json.dump(out, open(f"experiments/blind5_{tag}_transcripts.json", "w"), indent=1)
pairs, changed, total = [], 0, 0
for o, n in zip(res["old"], res["new"]):
    swap = rng.random() < 0.5; mp[str(o["id"])] = "X=new" if swap else "X=old"
    turns = []
    for to, tn in zip(o["turns"], n["turns"]):
        x, y = (tn["ovi"], to["ovi"]) if swap else (to["ovi"], tn["ovi"])
        turns.append({"user": to["user"], "X": x, "Y": y}); total += 1; changed += to["ovi"] != tn["ovi"]
    pairs.append({"id": o["id"], "turns": turns})
json.dump(pairs, open("experiments/blind5_pairs.json", "w"), indent=1); json.dump(mp, open("experiments/blind5_map.json", "w"), indent=1)
print(f"{changed}/{total} replies differ")
