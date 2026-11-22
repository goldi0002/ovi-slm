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
