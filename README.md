# BabyMind

A from-scratch learning agent that learns **like a human baby**: it first forms concepts from raw
sensory numbers, and only then attaches the words a "parent" (you) says to those concepts.

**Hard constraints:** plain C# / .NET 8, no pretraining, no transformers, no backpropagation, no text
corpus, no ML libraries. Learning is local and online: every experience immediately changes a handful of
connections. There is no training phase. The whole mind is a few thousand numbers.

## Quick start

```bash
dotnet run --project src/BabyMind            # state is loaded from / saved to ./babymind.json
dotnet run --project src/BabyMind -- --fresh # ignore the save file, start a new baby
dotnet test                                  # unit tests
```

### Generalisation demo: "red ball" -> other red things

```
parent> show red round small     parent> say red ball
parent> show blue round small    parent> say blue ball     # "ball" now varies in colour, "red" is still one thing
parent> show red square medium   parent> say red block
parent> show red long small      parent> say red banana    # "red" has now been heard with 3 different things
parent> show red triangle big    # never seen, never named -> Baby: "red"   (generalising: "red" fits by color)
parent> show green round small   # -> Baby: "ball"  (generalising: "ball" fits by shape + size)
parent> show red round small     # -> Baby: "red ball"   (concept name + learned word order)
parent> show yellow round small  # -> "ball" ... it is a lemon:  no lemon   (it learns an exception and a new word)
parent> debug                    # "red" -> color=red,  "ball" -> shape=round, size=small
```

### Stage 3 demo: actions and two-word phrases

```
parent> show ball
parent> say more ball       # last word names the object; "more -> ball" word order is learned too
parent> show cup
parent> say cup
parent> feed more milk      # a second "more X" phrase: "more" is now a pivot word ("more _")
parent> feed more milk
parent> autofeed 2          # you now respond ONLY to two-word requests containing "milk"
parent> wait 120
parent> debug               # "when [peckish] do say:more milk" beats "say:milk"; pivots: more _
```

The baby also tries phrases it never heard, such as **"more cup"** (pivot + a known word). Whether it keeps
using a phrase is decided only by reward. Reaching is an action too: `show bottle`, `autofeed on`, `wait 30`
and a hungry baby that reaches for the bottle gets it. Reaching for a familiar object while curious
is mildly rewarding by itself.

### Stage 2 demo: needs and requests

```
parent> show ball
parent> say ball            # (Stage 1) names an object
parent> feed                # bottle + the word "milk": hunger drops (reward) and "milk" joins its vocabulary
parent> feed
parent> autofeed on         # you will now feed the baby whenever it says "milk" while hungry
parent> wait 80             # time passes; the baby chooses look / say <word> / stay quiet by itself
parent> debug               # shows the learned rule: when [peckish] do say:milk
```

At first the baby tries its known words at random. Only "milk" ever gets it fed, so its value for
`say milk` when hungry rises, and it learns that talking while sated is just wasted effort. Without `autofeed`
you play the parent by hand: `wait`, and when it says "milk" type `feed` (or `good` / `no`).

### Stage 1 demo

```
parent> show ball           # baby sees a red round small thing: "something new!" (concept born)
parent> say ball            # hears the word with the object in view -> Hebbian link, says "ball"
parent> remove
parent> show ball           # a new noisy view of the same object -> recognised, says "ball"
parent> show block          # different object -> new concept, stays quiet (no word for it)
parent> debug               # concepts, strongest word links, drive levels
parent> quit                # saves; next run the baby is still there
```

Commands: `show`, `say`, `ask`, `feed`, `wait`, `autofeed on|off|2`, `good`, `no [word]`, `remove`, `props`, `debug`, `save`, `load`, `help`, `quit`.
The baby never sees the prop names or colour words, only 8 noisy numbers. It learns only what you say.

## Long-term soak test

```bash
dotnet run -c Release --project tools/BabyMind.Soak -- --ticks 10000 --seeds 3 --quiet
```

A scripted parent (seeded) teaches 5 object classes in 7 colours for thousands of ticks (teaching-heavy for the first 1,500, then mostly waiting and feeding),
praises and corrects, and feeds when the baby requests "milk". Every 1,000 ticks it probes a *clone* of the baby. 6 colour×class combinations are never shown in teaching (held-out) to test generalisation,
"tree" is taught early and then never again, and "frog" is taught exactly once.

Results (6 seeds × 20,000 ticks, release build):

| What | Result |
|---|---|
| Resources | Saved state flat at ~28 KB, heap ~1 MB, 30 concepts, ~50 word links, 14 word meanings; the Q-table creeps from ~140 to ~250 entries and flattens. 30-35 ms per 1,000 ticks. Nothing grew without bound. |
| Needs | Requests replace waiting: 99% of feeds are the baby asking for milk (mean hunger at request 0.33). Speech while sated falls from ~55% to ~5% (the exploration floor). |
| Naming (taught combos) | 100% on every seed at every probe. Wrong words: 0%. |
| Held-out combos (e.g. "green ball", never shown) | Name correct 100% on every seed. Name **and** colour ("green ball") 83-100%; the rare miss is a neglected combination whose word-order links have decayed (it says "tree" for a yellow tree instead of "yellow tree"). |
| Neglected word ("tree", never taught again after tick 1,500) | Still named correctly 18,500 ticks later. |
| One-shot word ("frog", 1 exposure) | Still spoken at 3,000 ticks, forgotten by ~5,000 (was ~500 before the decay change). |
| Persistence | Save/load round trip reproduces probe results exactly. |

What the soak test found, and what was changed because of it:

1. **Word order was memorised per pair.** For a never-heard pair ("green ball") the baby said only one word. Now `Phrases` learns word *classes* from order (modifiers such as "red"/"green"/"more" and heads such as "ball"/"milk") and accepts any modifier before any head.
2. **Forgetting was too fast** (a single exposure lasted ~500 ticks). Base decay is now 5x slower; practised and consolidated links are still much slower again.
3. **One bad example could destroy a word's meaning.** A parent echoing a babbled "yellow milk" with a white bottle in view made colour "inconsistent" for "yellow". Established words now ignore a sharply contradicting sample unless it repeats three times in a row. The `autofeed` parent (and the soak parent) now say "milk" while feeding instead of parroting the baby's phrase.
4. **Name vs colour tie-break.** When both fit and no word order is known, the baby now prefers the more informative word ("tree" = shape + size beats "red" = colour).

Known remaining limits: a word-order link that is never heard again eventually decays away (hence the occasional single word instead of two); a thing that looks exactly like another known thing is mistaken for it (a white banana *is* the milk bottle in this world); and a feature-poor world like this one tells us little about scaling to richer perception.

## Design

```
 World ──features──▶ ConceptMap ──concept id──▶ Associations ◀── word (Teacher)
                         │                           │
                      Drives ◀── reward ── Baby ──▶ Memory ──consolidate──▶ Associations
```

| File | Role |
|---|---|
| `World.cs` | Tiny simulated world: one object (colour, shape, size) in view, turned into an 8-dim noisy vector `[r,g,b, round,square,triangle,long, size]`. |
| `ConceptMap.cs` | Online clustering (leader / ART-style incremental k-means). Nearest prototype within `Vigilance` is refined with rate `max(1/n, 0.05)`; otherwise the input **immediately becomes a new concept**, which is what makes one-example learning possible. Capped at `MaxConcepts` (2000), evicting the least-used. |
| `Associations.cs` | Concept↔word weights. Hearing a word while a concept is attended applies a Hebbian step `w += η(1-w)` with η = 0.5, so one co-occurrence gives 0.5 and the baby will already say the word. Rival words for the same concept are mildly inhibited. Slow decay `rate/(1+hits/2)`; consolidated links decay 20x slower. The same weights give production (concept→word) and comprehension (word→concept). Praise/correction are multiplicative reward updates. |
| `Memory.cs` | Episodic log whose entries fade (×0.97/tick, bounded size). Repeated `(concept, word)` patterns (3 times within a window) are *consolidated*: the link is lifted and protected from decay. Single events fade, repeated ones stay. |
| `Drives.cs` | Hunger, curiosity, attention-seeking levels that creep up; satisfying one lowers it, and the drop is the reward. Novelty satisfies curiosity, the parent speaking satisfies attention, feeding satisfies hunger. |
| `ActionSelector.cs` | Tabular RL (epsilon-greedy contextual bandit). State = hunger bucket (sated/peckish/hungry) + object in view. Actions = look, reach, `say <word or two-word phrase>`, stay quiet. `Q += α(reward - Q)`; speaking costs 0.02 *per word* (a phrase must earn more than a single word), so talking must pay off. The last action stays pending until the parent reacts (feeding/praise/correction) or the next action starts. One table entry changes per action. |
| `WordMeanings.cs` | **Generalisation.** Each word keeps an online mean/variance of the features of everything it was heard with. Features are grouped (colour / shape / size); a group is *relevant* to a word only if the word was consistent on it. "red" heard with a red ball and a red block keeps *colour* and drops shape and size, so any red thing fits; "ball" heard with red and blue balls keeps shape + size. A word needs ≥2 distinct concepts to generalise (≥3 for pivot words like `red _`/`more _`), and function-like words heard with everything end up meaning nothing. "No" stores a counter-example so a word stops matching that thing. |
| `Phrases.cs` | First grammar. Hebbian word-order links (`more→milk`) learned from phrases you say. A word followed by several different words becomes a **pivot** (`more _`), which gives the baby candidate phrases it never heard (`more cup`). Candidates are only options; rewards decide use. |
| `Baby.cs` | The agent. `Experience(sight, heardWord)` = reaction to the parent (perceive → hear/associate → remember → name). `Act(sight)` = spontaneous behaviour (look / reach / say / quiet) chosen by the ActionSelector. `Feed()`, `Praise()`, `Correct()`, `Debug()`. |
| `BabyState.cs` | JSON save/load of everything learned (versioned, atomic write; older save files still load). |
| `Teacher.cs` | Console parent interface (injectable streams, so it is testable). |

Tunable constants live in the constructors/`AssociationParams` (vigilance 0.25, learn rate 0.5, speak threshold 0.4, ...).

### Known Stage 1 limitations (by design)
- Generalisation is strict: a red ball of a different size is a *new* concept until it is also named. (A natural later improvement: let words spread weakly to nearby concepts.)
- When *you* show something, the baby still names it by a fixed rule (link ≥ 0.4). Only its *spontaneous* speech (`wait`) is reinforcement-learned.
- The request is not grounded in meaning: the baby learns "say *milk* when hungry gets food" by reward, without knowing milk is a thing.
- Only hunger and curiosity are modelled; I dropped the planned *comfort* need (it would just repeat the hunger loop).
- A word's meaning is only as good as its contrast: with 2 examples of "red" that are both small, *small* could be mistaken for part of its meaning until a big red thing is heard (this is the pivot rule's reason for asking for a third example).
- Only three feature groups exist (colour, shape, size), and a word's meaning is a region in those groups: no negation, relations or abstract words.
- Generalisation is used when *you show something* (and in `Experience` naming); spontaneous `wait` speech still draws only on concept names and phrases.
- Grammar is limited to pivot frames (`X _`); there are no categories, three-word sentences or word-order generalisation beyond that.

- Word-to-concept links ("ball" ↔ concept #1) and word-to-feature meanings ("red" ↔ colour) are learned in parallel; the concept link wins for the exact thing that was named, the feature meaning handles new things.

## Roadmap
- **Stage 1 (done):** perception, concept formation, object↔word association.
- **Stage 2 (done):** needs and requests: `ActionSelector` (look / say / stay quiet, learned by simple RL), hunger drive, saying "milk" when hungry gets fed.
- **Generalisation (done):** property and object words learned across objects; "red ball" -> other red things.
- **Stage 3 (done):** `reach` action, Hebbian word order, pivot words, productive two-word combinations.
