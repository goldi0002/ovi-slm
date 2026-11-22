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

### Tiny neural scorer (measured)
`experiments/gru_eval.py` (log: `experiments/gru_eval.log`): a GRU seq2seq (0.95M params, 3.8 MB fp32 / 1.9 MB fp16, word vocab 5000,
PyTorch CPU 4 threads) trained 11 min (7 epochs, 57k deduped DailyDialog+EmpatheticDialogues train pairs, dev ppl 52.9). Same clean protocol as
`pool_eval.py` (mixed pool of 29 distractors + gold, 2000 test turns, ties/NaN lose), score = length-normalised log p(reply|message):

| scorer | R@1 (chance 3.3%) |
|---|---|
| GRU alone | 6.4% |
| numpy listwise ranker (mixed-trained) | 14.5% |
| ranker + GRU score as extra feature (fit on validation) | 14.7% |

Verdict: at this size/budget the GRU is far below the ranker and adds only +0.2 pt in combination (within noise for n=2000, about +-0.8 pt).
Free generation (greedy, ~3 ms/reply) is fluent but generic and often off-topic ("i am sorry to hear that" for good news), so it is not shipped.
Caveat: distractors are train replies the GRU saw as targets while the gold reply is unseen. Nothing in `ovi/brain.py` or shipped models changed.

## Free-conversation fixes after the blind test (chitchat.py, usermem.py)
A blind tester scored free conversation at 0.75/2 mean with 51% wrong replies. Changes in `ovi/brain.py`, `ovi/chitchat.py`, `ovi/usermem.py`:
filler templates now need a real claim with a determiner+noun topic (never stopwords/adjectives/verbs) and rotate without repeats, with a context-aware acknowledgement pool;
chit-chat corpus lines need a much higher match score (and are dropped for questions addressed to Ovi); small-talk rules answer greetings, how-are-you, thanks, goodbye, "what about you" and
questions about Ovi from a tiny profile; teach mode no longer swallows lol/idk/ok; fact extraction is stricter (no residence overwrite from "it's hot today", no invented names) and pets/job/relatives
are remembered and recalled; follow-ups carry the last resolved question; plus an arithmetic calculator, empathy/event replies and non-committal advice replies.
Tuned only on blind conversations 1-20 (hand-scored 0/1/2: 0.69 -> 1.61 mean); 21-40 were held out. `experiments/own_convos.json` holds extra conversations written while tuning.

Second pass (all 40 blind conversations now used for tuning; `experiments/own_convos2.json` has 20 more of our own, `tests/test_free_chat.py` regression tests): teach prompts only for substantive factual questions and a pending question is no longer swallowed by an unrelated statement; recall of what the user did ("what did I do today?"), "mine is blue" resolved against the last favorite category, pets from "I like cats / I have two, Milo and Luna"; empathy replies only for first-person feelings; "tell me about yourself" and advice requests get profile / ask-for-context replies.

## General slot memory and free-chat rules (blind-scoring fixes)
`ovi/slots.py` keeps a generic attribute/value store for what the user says in plain sentences ("my car is red", "I'm 25", "I play guitar",
"I run every morning", "been playing for ten years", "I'm going to Japan", "my mom makes lasagna" / "she lives in Naples", "she's black with white paws",
taught "the tallest mountain in the world is Everest"). Facts live in the normal event log as user entries under internal `zzfact <kind> ...`
questions (never reachable by fuzzy retrieval), so they persist across restarts, are superseded by key and are tombstoned by "forget my age",
"forget that I play guitar", "forget everything". Questions ("what instrument do I play", "how old am I", "what colour is my car", "where does my mom live",
"what do you remember about me", "what was the first thing I asked") are answered by lemmatised-verb / head-noun matching on the slot, and
"what's the capital?" is resolved from the country mentioned in the last few turns.
Also in this change: intent rules (goodbye incl. "lol bye"/"thanks, bye", "another", "entertain me", "that was lame", compliments, "u there", "do you know?"),
a noun-phrase filter so Ovi never asks about "your kept" / "your didn't", no "what should I have said?" on personal/advice questions (it thinks it through
instead), supportive replies for "I feel stupid" / long hard messages, and adjective/deictic filters so "it's freezing here" is not read as a correction of the city.
Internal bookkeeping ("I lowered my confidence to ...") is shown only for the `/wrong` command, never in chat. Tests: `PYTHONPATH=. python tests/test_recall_slots.py`.

## Conversation memory, template safety and routing (blind-set 3 fixes)
`ovi/convmem.py` (mixin `ConvMemory`, now behind the slot rules): every plain statement the user makes is kept as a sentence under an internal
`zzsay <ms> <n> [|| <slot question it also filled>]` user entry in the normal event log, so it persists across restarts, is never reachable by fuzzy retrieval,
and is tombstoned by `forget everything` / `forget my niece` (all content words match) / `forget that` (the last stored or last recalled sentence) / `forget my age`
(a sentence dies with the slot it filled). "my name is X" is not copied (the name slot answers it).
Recall-style questions ("what's my cat's name again?", "what city did I say I lived in?", "what does my dad do?", "what grade do I teach?",
"how many siblings do I have?", "what did I tell you about my sister?", "do you remember my job?", "did I mention ...?") are only handled when the precise paths
(taught entries, `slots.py`, `usermem.py`) have nothing. The matcher scores every stored sentence: lemma-ish stems + a few synonym groups (live/move/home, job/work, dad/father, fear/afraid ...),
idf weighting, a bonus when the sentence holds the kind of thing the question word asks for (place for "what city", number for "how many", a name for "called", a job for "what do I do",
instrument/sport/breed/subject ... for "what <noun> ..."), a strict head-noun check ("my brother" is not answered by a sentence about a sister), pronoun carry-over ("she is a vet" belongs
to the cousin mentioned before), and a small recency bonus (a later correction wins). The relevant clause is turned into second person ("You said your cat's name is Biscuit.").
Confident -> plain answer; related but weak -> "I'm not sure, but the closest thing I remember is that you said ..."; nothing related -> the old "I don't know that about you yet".
Cost: a linear scan over the session's sentences, ~10-20 ms.
Template safety (`chitchat.np_ok`): a topic fragment ("How do you feel about the X?") must be a determiner + a real noun head: no pronouns/reflexives ("the herself"), adjectives/colours/sizes
("the tiny", "the grey"), verb forms or adverbs; otherwise the generic acknowledgement is used. Routing before the generic fallbacks: `whats/hows/whos...` count as questions, `whats 15 times 12` is
arithmetic, an unknown "the capital of X" says so honestly and offers to be taught, recommendation / how-to requests ("any recs for ...", "whats the best way to learn spanish", "how do i ...", "tips on ...")
get a think-it-through reply, and goodbyes cover adios, night night, gn, nn, ttyl, cya, later(s), peace (out), g2g, ciao, toodles, "off to bed" ...
Tests: `PYTHONPATH=. python tests/test_conversation_memory.py` (invented phrasings, multi-turn, forget, restart persistence).
