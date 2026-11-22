# ovi-slm
## Reply ranking (our own CPU-trained model)
`ovi/replyrank.py`: **CrossPMI** learns which message words call for which reply words (counting + one SVD, ~5 s, no GPU),
and a small listwise ranker picks the reply that fits the *conversation so far*. Build: `python -m ovi.replyrank` (needs scipy).
Measured on unseen test conversations (`experiments/context_eval.py`): picks the real reply out of ~30 plausible ones
8.0% of the time from the message alone, **13.8% with conversation history** (chance 3.3%).

## Reply filtering
Dialogue replies are kept only if generic: no personal-life claims (family, job, age, past events), no rare scenario words,
EmpatheticDialogues listener turns only, PersonaChat replies off. Measured on held-out turns (`experiments/dialogue_retrieval_eval.py`).
