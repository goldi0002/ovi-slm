namespace BabyMind;

/// <summary>A weighted connection between a concept and a heard word.</summary>
public sealed class Link
{
    public int Concept { get; init; }
    public string Word { get; init; } = "";
    /// <summary>Connection strength in [0, 1].</summary>
    public double Weight { get; set; }
    /// <summary>Number of times this pairing was experienced together.</summary>
    public int Hits { get; set; }
    /// <summary>Lasting knowledge: promoted by Memory after the pattern repeated; decays ~20x slower.</summary>
    public bool Consolidated { get; set; }
}

/// <summary>Tunable constants of the association rules.</summary>
public sealed class AssociationParams
{
    /// <summary>Hebbian step. 0.5 means a single co-occurrence already yields weight 0.5 (one-shot word learning).</summary>
    public double LearnRate { get; set; } = 0.5;
    /// <summary>When a word is strengthened for a concept, its rival words for that concept shrink by Inhibition*LearnRate.</summary>
    public double Inhibition { get; set; } = 0.1;
    /// <summary>Per-tick forgetting. Divided by (1 + Hits/2): well-practised links fade slower.</summary>
    public double DecayRate { get; set; } = 0.0001;
    /// <summary>Decay multiplier for consolidated links.</summary>
    public double ConsolidatedDecayFactor { get; set; } = 0.05;
    /// <summary>Links weaker than this are forgotten entirely.</summary>
    public double PruneBelow { get; set; } = 0.02;
    /// <summary>Consolidation lifts a link to at least this weight.</summary>
    public double ConsolidatedFloor { get; set; } = 0.6;
}

/// <summary>
/// Concept↔word association network. One set of weights serves both directions:
///  production  (concept → word, "what do I call this?")  and
///  comprehension (word → concepts, "what does this word mean?").
/// Learning is purely local: hearing a word while a concept is active strengthens exactly that one link
/// (Hebb: "fire together, wire together"), saturating toward 1, and lightly inhibits rival words.
/// </summary>
public sealed class Associations
{
    private readonly Dictionary<int, Dictionary<string, Link>> _byConcept = new();
    private readonly AssociationParams _p;

    public Associations(AssociationParams? p = null) => _p = p ?? new AssociationParams();

    public IEnumerable<Link> AllLinks => _byConcept.Values.SelectMany(d => d.Values);
    public int LinkCount => AllLinks.Count();

    /// <summary>Hebbian co-occurrence: concept and word were active together. <paramref name="attention"/> in (0,1] scales the step.</summary>
    public void Hear(int concept, string word, double attention = 1.0)
    {
        word = Normalize(word);
        if (!_byConcept.TryGetValue(concept, out var words)) _byConcept[concept] = words = new();
        if (!words.TryGetValue(word, out var link))
            words[word] = link = new Link { Concept = concept, Word = word };

        double step = Math.Clamp(_p.LearnRate * attention, 0, 1);
        link.Weight += step * (1 - link.Weight);      // saturating growth
        link.Hits++;

        foreach (var other in words.Values)           // competition between words for one concept
            if (other != link) other.Weight *= 1 - _p.Inhibition * step;
    }

    /// <summary>
    /// Reward-based adjustment of an existing link: praise (delta &gt; 0) moves it toward 1,
    /// correction (delta &lt; 0) shrinks it proportionally. delta is a fraction in [-1, 1].
    /// </summary>
    public void Reward(int concept, string word, double delta)
    {
        var link = Find(concept, Normalize(word));
        if (link is null) return;
        link.Weight += delta > 0 ? delta * (1 - link.Weight) : delta * link.Weight;
        link.Weight = Math.Clamp(link.Weight, 0, 1);
    }

    /// <summary>Production: the strongest word for a concept, or null if it has none.</summary>
    public (string Word, double Weight)? Produce(int concept)
    {
        if (!_byConcept.TryGetValue(concept, out var words) || words.Count == 0) return null;
        var best = words.Values.MaxBy(l => l.Weight)!;
        return (best.Word, best.Weight);
    }

    /// <summary>Comprehension: concepts a word points to, strongest first.</summary>
    public IReadOnlyList<(int Concept, double Weight)> Comprehend(string word)
    {
        word = Normalize(word);
        return AllLinks.Where(l => l.Word == word)
                       .OrderByDescending(l => l.Weight)
                       .Select(l => (l.Concept, l.Weight)).ToList();
    }

    /// <summary>All words attached to a concept, strongest first.</summary>
    public IReadOnlyList<Link> WordsFor(int concept) =>
        _byConcept.TryGetValue(concept, out var w) ? w.Values.OrderByDescending(l => l.Weight).ToList() : new List<Link>();

    public Link? Find(int concept, string word) =>
        _byConcept.TryGetValue(concept, out var w) && w.TryGetValue(Normalize(word), out var l) ? l : null;

    /// <summary>Mark a pairing as lasting knowledge (called when Memory sees the pattern repeat).</summary>
    public void Consolidate(int concept, string word)
    {
        var link = Find(concept, word);
        if (link is null) return;
        link.Consolidated = true;
        link.Weight = Math.Max(link.Weight, _p.ConsolidatedFloor);
    }

    /// <summary>One tick of slow forgetting. Practised and consolidated links fade much more slowly.</summary>
    public void Decay()
    {
        foreach (var (concept, words) in _byConcept.ToList())
        {
            foreach (var link in words.Values.ToList())
            {
                double rate = _p.DecayRate / (1 + link.Hits * 0.5);
                if (link.Consolidated) rate *= _p.ConsolidatedDecayFactor;
                link.Weight *= 1 - rate;
                if (link.Weight < _p.PruneBelow) words.Remove(link.Word);
            }
            if (words.Count == 0) _byConcept.Remove(concept);
        }
    }

    /// <summary>Forget all links of a concept (used when the concept itself was evicted).</summary>
    public void ForgetConcept(int concept) => _byConcept.Remove(concept);

    public static string Normalize(string word) => word.Trim().ToLowerInvariant();

    // ---- persistence ----
    public AssociationsState ToState() =>
        new(AllLinks.Select(l => new LinkState(l.Concept, l.Word, l.Weight, l.Hits, l.Consolidated)).ToList());

    public static Associations FromState(AssociationsState s, AssociationParams? p = null)
    {
        var a = new Associations(p);
        foreach (var l in s.Links)
        {
            if (!a._byConcept.TryGetValue(l.Concept, out var words)) a._byConcept[l.Concept] = words = new();
            words[l.Word] = new Link { Concept = l.Concept, Word = l.Word, Weight = l.Weight, Hits = l.Hits, Consolidated = l.Consolidated };
        }
        return a;
    }
}

public sealed record LinkState(int Concept, string Word, double Weight, int Hits, bool Consolidated);
public sealed record AssociationsState(List<LinkState> Links);
