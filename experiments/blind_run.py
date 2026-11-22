"""Run blind conversations through Brain on a temp log. PYTHONPATH=. python experiments/blind_run.py"""
import json, os, tempfile
from ovi.brain import Brain
here = os.path.dirname(os.path.abspath(__file__))
convos = json.load(open(os.path.join(here, "blind_convos.json")))
out = []
for i, turns in enumerate(convos):
    b = Brain(log=os.path.join(tempfile.mkdtemp(), "b.jsonl"))
    out.append({"id": i + 1, "turns": [{"user": u, "ovi": b.converse(u).text} for u in turns]})
json.dump(out, open(os.path.join(here, "blind_transcripts.json"), "w"), indent=1)
with open(os.path.join(here, "blind_transcripts.txt"), "w") as f:
    for c in out:
        f.write(f"=== {c['id']}\n")
        for k, t in enumerate(c["turns"], 1):
            f.write(f"{k}. U: {t['user']}\n   O: {t['ovi']}\n")
