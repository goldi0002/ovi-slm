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
    public long Tick { get; private set; }

    /// <summary>Concept most recently attended to and the word the baby last said — used for praise/correction.</summary>
    public int? LastConceptId { get; private set; }
    public string? LastSaid { get; private set; }

    public Baby(ConceptMap? map = null, Associations? assoc = null, Drives? drives = null, Memory? memory = null, long tick = 0)
    {
        Map = map ?? new ConceptMap(World.Dim);
        Assoc = assoc ?? new Associations();
        Drives = drives ?? new Drives();
        Memory = memory ?? new Memory();
        Tick = tick;
    }

    /// <summary>
    /// One moment of life. <paramref name="sight"/> is what the eyes deliver (null = nothing in view),
    /// <paramref name="heard"/> is a word spoken by the parent (null = silence).
    /// </summary>
    public Response Experience(SensoryInput? sight, string? heard)
    {
        Tick++;
        Drives.Tick();
        Memory.Fade(Tick);
        Assoc.Decay();
        if (heard is not null) heard = Associations.Normalize(heard);

        // 1. Perceive: form or recognise a concept from raw numbers — no word needed.
        Observation? obs = null;
        if (sight is { } s)
        {
            var o = Map.Observe(s.Features, Tick);
            obs = o;
            if (o.EvictedId is int gone) { Assoc.ForgetConcept(gone); Memory.ForgetConcept(gone); }
            if (o.IsNew) Drives.Satisfy(DriveKind.Curiosity, 0.3);   // novelty is rewarding
        }

        // 2. Hear: a word heard while a concept is in attention gets Hebbian-linked to it.
        bool consolidated = false;
        (Concept, double)? evoked = null;
        if (heard is not null)
        {
            Drives.Satisfy(DriveKind.Attention, 0.3);                 // the parent talking to me feels good
            if (obs is { } ob)
            {
                Assoc.Hear(ob.Concept.Id, heard);
                if (Memory.Record(Tick, ob.Concept.Id, heard))
                {
                    Assoc.Consolidate(ob.Concept.Id, heard);          // repetition → lasting knowledge
                    consolidated = true;
                }
            }
            else if (Assoc.Comprehend(heard).FirstOrDefault() is { Weight: > 0 } top && Map.Get(top.Concept) is { } c)
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

    /// <summary>Parent praises the last thing the baby said: that link is reinforced.</summary>
    public bool Praise()
    {
        if (LastConceptId is not int c || LastSaid is null) return false;
        Assoc.Reward(c, LastSaid, 0.3);
        Drives.Satisfy(DriveKind.Attention, 0.2);
        return true;
    }

    /// <summary>
    /// Parent says "no": the last spoken link is weakened. If <paramref name="correctWord"/> is given it is
    /// learned for the concept right away (a correction is a strong, attended example).
    /// </summary>
    public bool Correct(string? correctWord = null)
    {
        if (LastConceptId is not int c || LastSaid is null) return false;
        Assoc.Reward(c, LastSaid, -0.5);
        if (correctWord is not null)
        {
            Assoc.Hear(c, correctWord);
            if (Memory.Record(Tick, c, Associations.Normalize(correctWord)))
                Assoc.Consolidate(c, correctWord);
        }
        LastSaid = null;
        return true;
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
        sb.AppendLine("Strongest word links (* = consolidated):");
        foreach (var l in Assoc.AllLinks.OrderByDescending(l => l.Weight).Take(10))
            sb.AppendLine($"  \"{l.Word}\" <-> #{l.Concept}  weight {l.Weight:0.00}  hits {l.Hits}{(l.Consolidated ? " *" : "")}");
        var cons = Memory.Patterns.Where(p => p.Value.Consolidated).Select(p => p.Key).ToList();
        sb.AppendLine($"Consolidated patterns (concept|word): {(cons.Count == 0 ? "none" : string.Join(", ", cons))}");
        return sb.ToString();
    }

    // ---- persistence ----
    public BabyState ToState() => new(BabyState.CurrentVersion, Tick, Map.ToState(), Assoc.ToState(), Memory.ToState(), Drives.ToState());

    public static Baby FromState(BabyState s) =>
        new(ConceptMap.FromState(s.Map), Associations.FromState(s.Associations), Drives.FromState(s.Drives), Memory.FromState(s.Memory), s.Tick);
}
