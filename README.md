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

Commands: `show`, `say`, `ask`, `good`, `no [word]`, `remove`, `props`, `debug`, `save`, `load`, `help`, `quit`.
The baby never sees the prop names or colour words, only 8 noisy numbers. It learns only what you say.

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
| `Drives.cs` | Hunger, curiosity, attention-seeking levels that creep up; satisfying one lowers it, and the drop is the reward. Stage 1: novelty satisfies curiosity, the parent speaking satisfies attention; hunger is inert until Stage 2. |
| `Baby.cs` | The agent. `Experience(sight, heardWord)` = perceive → hear/associate → remember → speak. Also `Praise()`, `Correct()`, `Debug()`. |
| `BabyState.cs` | JSON save/load of everything learned (versioned, atomic write). |
| `Teacher.cs` | Console parent interface (injectable streams, so it is testable). |

Tunable constants live in the constructors/`AssociationParams` (vigilance 0.25, learn rate 0.5, speak threshold 0.4, ...).

### Known Stage 1 limitations (by design)
- Generalisation is strict: a red ball of a different size is a *new* concept until it is also named. (A natural later improvement: let words spread weakly to nearby concepts.)
- The baby speaks whenever a link is above a fixed threshold. Stage 2's `ActionSelector` replaces this with learned reinforcement: speaking becomes an action that gets rewarded.

## Roadmap
- **Stage 1 (done):** perception, concept formation, object↔word association.
- **Stage 2:** needs and requests: `ActionSelector` (look / reach / say / stay quiet, learned by simple RL), hunger in the World, saying "milk" gets fed.
- **Stage 3:** actions and simple two-word combinations.
