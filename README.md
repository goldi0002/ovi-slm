# ovi-slm
## Reply ranking (our own CPU-trained model)
`ovi/replyrank.py`: **CrossPMI** learns which message words call for which reply words (counting + one SVD, ~5 s, no GPU),
and a small listwise ranker picks the reply that fits the *conversation so far*. Build: `python -m ovi.replyrank` (needs scipy).
Measured on unseen test conversations (`experiments/context_eval.py`): picks the real reply out of ~30 plausible ones
8.0% of the time from the message alone, **13.8% with conversation history** (chance 3.3%).

## Reply filtering
Dialogue replies are kept only if generic: no personal-life claims (family, job, age, past events), no rare scenario words,
EmpatheticDialogues listener turns only, PersonaChat replies off. Measured on held-out turns (`experiments/dialogue_retrieval_eval.py`).


### Wider candidate pool
Reply candidates now come from two places: TF-IDF question matches and CrossPMI over *all* stored replies (`replyrank.mixed_pool`). The ranker is trained on that same mixed pool. Clean test (`experiments/pool_eval.py`): a ranker trained on TF-IDF-only candidates drops 13.6% -> 10.7% on the mixed pool; trained on the mixed pool it scores 14.5%.

### Why statements are NOT yet routed to the ranker (measured)
`experiments/gate_eval.py`: the ranker's confidence cannot tell "a good reply is in the pool" from "nothing fits" -- at every gate the share of turns passing is the same with the real reply present and absent (e.g. p>=0.2: 19.6% vs 18.7%). Only at p>=0.3 does it get more precise (44.6%), but that covers 4.7% of turns. So loosening the gate would add confident-sounding wrong replies; the careful rules and "tell me more" templates stay in front.

## Conversation eval (scripted, multi-turn)
`PYTHONPATH=. python experiments/convo_eval.py` runs 151 scripted cases from `experiments/convo_cases.json` (~15 s) against the current `Brain`,
each on a throw-away temp log (never `data/brain.jsonl`; "restart" turns rebuild the Brain from the same log). Checks are contains / not_contains / regex / not_regex / nonempty.
It prints per-category pass rates and the full transcript of every failure. Baseline:

| category | pass | total | rate |
|---|---|---|---|
| name_memory | 5 | 20 | 25.0% |
| fact_recall | 19 | 20 | 95.0% |
| corrections | 2 | 14 | 14.3% |
| abstain | 20 | 23 | 87.0% |
| consistency | 12 | 13 | 92.3% |
| feelings | 15 | 17 | 88.2% |
| odd_input | 25 | 27 | 92.6% |
| persistence | 15 | 17 | 88.2% |
| **total** | **113** | **151** | **74.8%** |

Main failure patterns: "actually/no, my name is X" and "no, it's red" corrections are not recognised (old value kept); "I'm X", "I am X", "call me X"/"do you remember my name?"/"who am I?" name forms fail; "what is my name?" before being told answers "My name is Ovi."; non-ASCII names are not stored; a few unknown questions get random dialogue replies instead of "I'm not sure".
