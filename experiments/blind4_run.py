"""Run: PYTHONPATH=. python experiments/blind4_run.py"""
import json, os, tempfile
from ovi.brain import Brain
convos = json.load(open("experiments/blind4_convos.json"))
out, txt = [], []
for c in convos:
    b = Brain(log=os.path.join(tempfile.mkdtemp(), "b.jsonl"))
    txt.append(f"=== Conversation {c['id']} ===")
    turns = []
    for u in c["turns"]:
        r = b.converse(u).text
        turns.append({"user": u, "ovi": r})
        txt.append(f"U: {u}\nO: {r}")
    out.append({"id": c["id"], "turns": turns})
    txt.append("")
json.dump(out, open("experiments/blind4_transcripts.json", "w"), indent=1)
open("experiments/blind4_transcripts.txt", "w").write("\n".join(txt))
