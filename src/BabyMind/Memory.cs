namespace BabyMind;

/// <summary>One remembered moment. Its Strength fades every tick.</summary>
public sealed class Episode
{
    public long Tick { get; init; }
    public int ConceptId { get; init; }
    public string? Word { get; init; }
    public double Strength { get; set; } = 1.0;
}

/// <summary>Repetition bookkeeping for a (concept, word) pattern.</summary>
public sealed class PatternStat
{
    public int Count { get; set; }
    public long LastTick { get; set; }
    public bool Consolidated { get; set; }
}

/// <summary>
/// Episodic memory: a short, fading log of recent events. Two things happen to it:
///  - Forgetting: every episode's strength shrinks each tick and weak ones are dropped (bounded size).
///  - Consolidation: when the same (concept, word) pattern recurs enough times within a time window,
///    it is reported as consolidated so the caller can turn it into lasting knowledge
///    (see <see cref="Associations.Consolidate"/>). Single episodes fade; repeated patterns stay.
/// </summary>
public sealed class Memory
{
    private readonly List<Episode> _episodes = new();
    private readonly Dictionary<string, PatternStat> _patterns = new();

    /// <summary>Strength multiplier per tick (0.97 ≈ half-life of 23 ticks).</summary>
    public double FadeRate { get; }
    public double ForgetBelow { get; }
    public int Capacity { get; }
    /// <summary>Repetitions needed to consolidate.</summary>
    public int ConsolidateAfter { get; }
    /// <summary>If the gap between two occurrences exceeds this many ticks, the repetition count restarts.</summary>
    public int PatternWindow { get; }

    public Memory(double fadeRate = 0.97, double forgetBelow = 0.05, int capacity = 200, int consolidateAfter = 3, int patternWindow = 200)
    {
        FadeRate = fadeRate; ForgetBelow = forgetBelow; Capacity = capacity;
        ConsolidateAfter = consolidateAfter; PatternWindow = patternWindow;
    }

    public IReadOnlyList<Episode> Episodes => _episodes;
    public IEnumerable<KeyValuePair<string, PatternStat>> Patterns => _patterns;

    /// <summary>
    /// Store an episode. Returns true exactly when a (concept, word) pattern *just* became consolidated.
    /// </summary>
    public bool Record(long tick, int conceptId, string? word)
    {
        _episodes.Add(new Episode { Tick = tick, ConceptId = conceptId, Word = word });
        if (_episodes.Count > Capacity) _episodes.RemoveAt(0);
        if (word is null) return false;

        string key = Key(conceptId, word);
        if (!_patterns.TryGetValue(key, out var stat)) _patterns[key] = stat = new PatternStat();
        if (stat.Count > 0 && tick - stat.LastTick > PatternWindow) stat.Count = 0;   // too long ago: start over
        stat.Count++;
        stat.LastTick = tick;

        if (!stat.Consolidated && stat.Count >= ConsolidateAfter)
        {
            stat.Consolidated = true;
            return true;
        }
        return false;
    }

    /// <summary>One tick of fading; also drops stale unconsolidated pattern counters.</summary>
    public void Fade(long tick)
    {
        foreach (var e in _episodes) e.Strength *= FadeRate;
        _episodes.RemoveAll(e => e.Strength < ForgetBelow);
        foreach (var k in _patterns.Where(p => !p.Value.Consolidated && tick - p.Value.LastTick > PatternWindow)
                                   .Select(p => p.Key).ToList())
            _patterns.Remove(k);
    }

    /// <summary>Remembered episodes involving a concept, most recent first.</summary>
    public IEnumerable<Episode> Recall(int conceptId) =>
        _episodes.Where(e => e.ConceptId == conceptId).OrderByDescending(e => e.Tick);

    /// <summary>Forget everything about an evicted concept.</summary>
    public void ForgetConcept(int conceptId)
    {
        _episodes.RemoveAll(e => e.ConceptId == conceptId);
        foreach (var k in _patterns.Keys.Where(k => k.StartsWith(conceptId + "|")).ToList()) _patterns.Remove(k);
    }

    private static string Key(int concept, string word) => $"{concept}|{word}";

    // ---- persistence ----
    public MemoryState ToState() => new(
        _episodes.Select(e => new EpisodeState(e.Tick, e.ConceptId, e.Word, e.Strength)).ToList(),
        _patterns.ToDictionary(p => p.Key, p => new PatternState(p.Value.Count, p.Value.LastTick, p.Value.Consolidated)));

    public static Memory FromState(MemoryState s)
    {
        var m = new Memory();
        foreach (var e in s.Episodes)
            m._episodes.Add(new Episode { Tick = e.Tick, ConceptId = e.ConceptId, Word = e.Word, Strength = e.Strength });
        foreach (var (k, p) in s.Patterns)
            m._patterns[k] = new PatternStat { Count = p.Count, LastTick = p.LastTick, Consolidated = p.Consolidated };
        return m;
    }
}

public sealed record EpisodeState(long Tick, int ConceptId, string? Word, double Strength);
public sealed record PatternState(int Count, long LastTick, bool Consolidated);
public sealed record MemoryState(List<EpisodeState> Episodes, Dictionary<string, PatternState> Patterns);
