namespace BabyMind;

/// <summary>What the baby did in response to one experience.</summary>
/// <param name="Observation">What it perceived, or null if nothing was in view.</param>
/// <param name="Said">Word it spoke, or null if it stayed quiet.</param>
/// <param name="Confidence">Strength of the link behind <paramref name="Said"/>.</param>
/// <param name="Heard">Word that was spoken to it this tick, if any.</param>
/// <param name="Evoked">When a word is heard with nothing in view: the concept it brings to mind.</param>
/// <param name="NewlyConsolidated">True if a repeated pattern became lasting knowledge this tick.</param>
public sealed record Response(Observation? Observation, string? Said, double Confidence, string? Heard,
                              (Concept Concept, double Weight)? Evoked, bool NewlyConsolidated);

/// <summary>What the baby did on its own during <see cref="Baby.Act"/>.</summary>
public sealed record ActResult(BabyAction Action, string State, string? Said, Observation? Observation);

/// <summary>
/// The learning agent. It owns the parts (ConceptMap, Associations, Drives, Memory) and runs one
/// <see cref="Experience"/> step per interaction. Everything is learned online, one experience at a time.
/// </summary>
public sealed class Baby
{
    /// <summary>Minimum link weight before the baby dares to say a word (Stage 2's ActionSelector will learn this instead).</summary>
    public const double SpeakThreshold = 0.4;

    public ConceptMap Map { get; }
    public Associations Assoc { get; }
    public Drives Drives { get; }
    public Memory Memory { get; }
    public ActionSelector Selector { get; }
    public Phrases Phrases { get; }
    public long Tick { get; private set; }

    /// <summary>Concept most recently attended to and the word the baby last said — used for praise/correction.</summary>
    public int? LastConceptId { get; private set; }
    public string? LastSaid { get; private set; }

    public Baby(ConceptMap? map = null, Associations? assoc = null, Drives? drives = null, Memory? memory = null, long tick = 0, ActionSelector? selector = null, Phrases? phrases = null)
    {
        Map = map ?? new ConceptMap(World.Dim);
        Assoc = assoc ?? new Associations();
        Drives = drives ?? new Drives();
        Memory = memory ?? new Memory();
        Selector = selector ?? new ActionSelector();
        Phrases = phrases ?? new Phrases();
        Tick = tick;
    }

    /// <summary>
    /// One moment of life. <paramref name="sight"/> is what the eyes deliver (null = nothing in view),
    /// <paramref name="heard"/> is what the parent said: one word or a two-word phrase (null = silence).
    /// In a phrase the last word is the label for what is in view; earlier words only learn word order.
    /// </summary>
    public Response Experience(SensoryInput? sight, string? heard)
    {
        Selector.Settle();           // an unrewarded earlier action just cost its effort
        Advance();
        var words = Tokenize(heard);
        heard = words.Count == 0 ? null : string.Join(' ', words);

        // 1. Perceive: form or recognise a concept from raw numbers — no word needed.
        Observation? obs = null;
        if (sight is { } s) obs = Perceive(s, out _);

        // 2. Hear: a word heard while a concept is in attention gets Hebbian-linked to it.
        bool consolidated = false;
        (Concept, double)? evoked = null;
        if (heard is not null)
        {
            string label = words[^1];
            Drives.Satisfy(DriveKind.Attention, 0.3);                 // the parent talking to me feels good
            for (int i = 0; i + 1 < words.Count; i++)
                Phrases.Hear(words[i], words[i + 1]);                 // word order: "more" -> "milk"
            if (obs is { } ob)
            {
                Assoc.Hear(ob.Concept.Id, label);
                if (Memory.Record(Tick, ob.Concept.Id, label))
                {
                    Assoc.Consolidate(ob.Concept.Id, label);          // repetition → lasting knowledge
                    consolidated = true;
                }
            }
            else if (Assoc.Comprehend(label).FirstOrDefault() is { Weight: > 0 } top && Map.Get(top.Concept) is { } c)
            {
                evoked = (c, top.Weight);                             // comprehension: word → concept
            }
        }
        else if (obs is { } ob2)
        {
            Memory.Record(Tick, ob2.Concept.Id, null);
        }

        // 3. Speak: name what I see if the link is strong enough.
        string? said = null; double conf = 0;
        if (obs is { } seen)
        {
            LastConceptId = seen.Concept.Id;
            if (Assoc.Produce(seen.Concept.Id) is { } p && p.Weight >= SpeakThreshold)
                (said, conf) = (p.Word, p.Weight);
            LastSaid = said;
        }
        return new Response(obs, said, conf, heard, evoked, consolidated);
    }

    // One tick of time passing: needs grow, memories fade, unused links decay.
    private void Advance()
    {
        Tick++;
        Drives.Tick();
        Memory.Fade(Tick);
        Assoc.Decay();
        Phrases.Decay();
    }

    // Form or recognise a concept. `reward` is the curiosity relief a novel thing gives.
    private Observation Perceive(SensoryInput sight, out double reward)
    {
        var o = Map.Observe(sight.Features, Tick);
        if (o.EvictedId is int gone) { Assoc.ForgetConcept(gone); Memory.ForgetConcept(gone); }
        reward = o.IsNew ? Drives.Satisfy(DriveKind.Curiosity, 0.3) : 0;   // novelty is rewarding
        LastConceptId = o.Concept.Id;
        return o;
    }

    private static List<string> Tokenize(string? text) =>
        text is null ? new() : text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Select(Associations.Normalize).ToList();

    /// <summary>Hunger as a coarse bucket (the RL state).</summary>
    public static string HungerBucket(double h) => h < 0.3 ? "sated" : h < 0.6 ? "peckish" : "hungry";

    /// <summary>Words the baby knows (and so can say), strongest first, capped to stay small.</summary>
    public IEnumerable<string> Vocabulary() =>
        Assoc.AllLinks.GroupBy(l => l.Word).OrderByDescending(g => g.Max(l => l.Weight)).Take(50).Select(g => g.Key);

    /// <summary>Everything the baby could say right now: known words plus two-word phrases (heard or novel pivot combinations).</summary>
    public IEnumerable<string> Utterances()
    {
        var vocab = Vocabulary().ToList();
        return vocab.Concat(Phrases.Candidates(vocab).Select(c => c.Phrase));
    }

    /// <summary>
    /// The baby acts on its own for one tick: the ActionSelector chooses to look, reach, say a word or phrase, or stay quiet,
    /// based on its needs. Rewards (feeding, praise) arrive later through <see cref="Feed"/> / <see cref="Praise"/>.
    /// </summary>
    public ActResult Act(SensoryInput? sight)
    {
        Selector.Settle();
        Advance();
        string state = $"{HungerBucket(Drives.Hunger)},{(sight is null ? "none" : "obj")}";
        var action = Selector.Choose(state, Utterances(), canLook: sight is not null);

        Observation? obs = null; string? said = null;
        switch (action.Kind)
        {
            case ActionKind.Look:
                obs = Perceive(sight!.Value, out double curiosityRelief);
                Memory.Record(Tick, obs.Value.Concept.Id, null);
                if (curiosityRelief > 0) Selector.Reward(curiosityRelief);
                break;
            case ActionKind.Reach:
                // Touching something is intrinsically rewarding while curious; the parent may also respond (e.g. hand over the bottle).
                obs = Perceive(sight!.Value, out _);
                Memory.Record(Tick, obs.Value.Concept.Id, null);
                double touch = Drives.Satisfy(DriveKind.Curiosity, 0.15);
                if (touch > 0) Selector.Reward(touch);
                break;
            case ActionKind.Say:
                said = action.Word;
                LastSaid = null;                  // a request is judged by the Selector, not as a concept name
                break;
        }
        return new ActResult(action, state, said, obs);
    }

    /// <summary>
    /// The parent feeds the baby while saying <paramref name="word"/>. Hunger drops (that drop is the reward and is
    /// credited to whatever the baby just did, e.g. saying "milk"), and the word is linked to what is in view.
    /// </summary>
    public (Response Response, double Relief) Feed(SensoryInput? sight, string word)
    {
        double relief = Drives.Satisfy(DriveKind.Hunger, 0.7);
        Selector.Reward(relief);
        return (Experience(sight, word), relief);
    }

    /// <summary>Parent praises the baby: the last spoken link and/or the last spontaneous action is reinforced.</summary>
    public bool Praise()
    {
        bool acted = Selector.Reward(0.5);            // praise for a spontaneous action (e.g. a request)
        if (LastConceptId is int c && LastSaid is not null)
        {
            Assoc.Reward(c, LastSaid, 0.3);
            acted = true;
        }
        if (acted) Drives.Satisfy(DriveKind.Attention, 0.2);
        return acted;
    }

    /// <summary>
    /// Parent says "no": the last thing said is discouraged. If <paramref name="correctWord"/> is given it is
    /// learned for the concept right away (a correction is a strong, attended example).
    /// </summary>
    public bool Correct(string? correctWord = null)
    {
        bool acted = Selector.Reward(-0.5);
        if (LastConceptId is int c && LastSaid is not null)
        {
            Assoc.Reward(c, LastSaid, -0.5);
            if (correctWord is not null)
            {
                Assoc.Hear(c, correctWord);
                if (Memory.Record(Tick, c, Associations.Normalize(correctWord)))
                    Assoc.Consolidate(c, correctWord);
            }
            LastSaid = null;
            acted = true;
        }
        return acted;
    }

    /// <summary>Multi-line report of what the baby currently knows.</summary>
    public string Debug()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"=== BabyMind @ tick {Tick} ===");
        sb.AppendLine($"Drives:   hunger {Drives.Hunger:0.00} | curiosity {Drives.Curiosity:0.00} | attention-seeking {Drives.Attention:0.00}");
        sb.AppendLine($"Concepts: {Map.Concepts.Count}   Word links: {Assoc.LinkCount}   Episodes in memory: {Memory.Episodes.Count}");
        foreach (var c in Map.Concepts.OrderBy(c => c.Id))
        {
            var words = Assoc.WordsFor(c.Id).Take(3)
                .Select(l => $"\"{l.Word}\" {l.Weight:0.00}{(l.Consolidated ? "*" : "")}");
            string w = string.Join(", ", words);
            sb.AppendLine($"  #{c.Id,-3} looks like [{World.Describe(c.Prototype)}]  seen {c.Count}x  words: {(w.Length == 0 ? "(none yet)" : w)}");
        }
        sb.AppendLine($"Learned behaviour (RL, exploring {Selector.Epsilon:0.00}):");
        var policy = Selector.Table().Where(t => Math.Abs(t.Value) > 0.01).Take(8).ToList();
        foreach (var t in policy) sb.AppendLine($"  when [{t.State}] do {t.Action,-12} value {t.Value:+0.00;-0.00}");
        if (policy.Count == 0) sb.AppendLine("  (nothing learned yet)");
        var pivots = Phrases.Pivots();
        sb.AppendLine($"Word order (pivots: {(pivots.Count == 0 ? "none" : string.Join(", ", pivots.Select(p => p + " _")))}):");
        foreach (var l in Phrases.Pairs.OrderByDescending(l => l.Weight).Take(6))
            sb.AppendLine($"  \"{l.First} {l.Second}\"  weight {l.Weight:0.00}  hits {l.Hits}");
        sb.AppendLine("Strongest word links (* = consolidated):");
        foreach (var l in Assoc.AllLinks.OrderByDescending(l => l.Weight).Take(10))
            sb.AppendLine($"  \"{l.Word}\" <-> #{l.Concept}  weight {l.Weight:0.00}  hits {l.Hits}{(l.Consolidated ? " *" : "")}");
        var cons = Memory.Patterns.Where(p => p.Value.Consolidated).Select(p => p.Key).ToList();
        sb.AppendLine($"Consolidated patterns (concept|word): {(cons.Count == 0 ? "none" : string.Join(", ", cons))}");
        return sb.ToString();
    }

    // ---- persistence ----
    public BabyState ToState() => new(BabyState.CurrentVersion, Tick, Map.ToState(), Assoc.ToState(), Memory.ToState(), Drives.ToState(), Selector.ToState(), Phrases.ToState());

    public static Baby FromState(BabyState s) =>
        new(ConceptMap.FromState(s.Map), Associations.FromState(s.Associations), Drives.FromState(s.Drives), Memory.FromState(s.Memory), s.Tick,
            s.Selector is null ? null : ActionSelector.FromState(s.Selector),
            s.Phrases is null ? null : Phrases.FromState(s.Phrases));
}
