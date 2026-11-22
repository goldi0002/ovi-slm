namespace BabyMind;

/// <summary>A learned word-order link: <c>First</c> was heard directly before <c>Second</c>.</summary>
public sealed class PairLink
{
    public string First { get; init; } = "";
    public string Second { get; init; } = "";
    public double Weight { get; set; }
    public int Hits { get; set; }
}

/// <summary>
/// First grammar: two-word combinations, learned the way toddlers do.
///  - Hebbian word-order links: hearing "more milk" strengthens the link more→milk (same saturating rule as Associations).
///  - Pivot words: a word that has been heard in front of several different words ("more milk", "more ball")
///    becomes a pivot with an open slot: "more _". This is what lets the baby try combinations it has never heard,
///    e.g. "more cup", as soon as it knows "cup".
///  - Word classes from order: a pivot is a *modifier* ("red", "more"); a word heard as the second word of a phrase is a *head* ("ball", "milk").
///    Any modifier may then precede any head, even for a pair never heard ("green ball" after "red ball", "green cup", ...):
///    this is generalisation of word order, not memorisation of pairs.
/// Candidates are only *options*: whether the baby actually says them is decided by the ActionSelector's rewards.
/// </summary>
public sealed class Phrases
{
    private readonly Dictionary<(string, string), PairLink> _pairs = new();

    public double LearnRate { get; set; } = 0.5;
    public double DecayRate { get; set; } = 0.0001;
    public double PruneBelow { get; set; } = 0.02;
    /// <summary>Different followers needed (each link at least <see cref="MinFollowerWeight"/>) for a word to count as a pivot.</summary>
    public int PivotMinFollowers { get; set; } = 2;
    public double MinFollowerWeight { get; set; } = 0.3;

    public IEnumerable<PairLink> Pairs => _pairs.Values;

    /// <summary>The parent said <paramref name="first"/> immediately followed by <paramref name="second"/>.</summary>
    public void Hear(string first, string second)
    {
        var key = (first, second);
        if (!_pairs.TryGetValue(key, out var link))
            _pairs[key] = link = new PairLink { First = first, Second = second };
        link.Weight += LearnRate * (1 - link.Weight);
        link.Hits++;
    }

    public PairLink? Find(string first, string second) => _pairs.GetValueOrDefault((first, second));

    /// <summary>Words heard right after <paramref name="first"/>, strongest first.</summary>
    public IReadOnlyList<PairLink> Followers(string first) =>
        _pairs.Values.Where(p => p.First == first).OrderByDescending(p => p.Weight).ToList();

    /// <summary>True if the word has been heard in front of several different words ("red _", "more _").</summary>
    public bool IsModifier(string word) => Pivots().Contains(word);

    /// <summary>True if the word has been heard as the second word of a phrase ("ball" in "red ball").</summary>
    public bool IsHead(string word) => _pairs.Values.Any(p => p.Second == word && p.Weight >= MinFollowerWeight);

    /// <summary>
    /// How acceptable "first second" is, 0 if not at all. A heard pair counts with its own weight; a never-heard pair
    /// counts at <see cref="GeneralisedOrder"/> if a modifier is followed by a head (and the second word is not itself a modifier).
    /// </summary>
    public double Order(string first, string second)
    {
        double heard = Find(first, second)?.Weight ?? 0;
        if (heard >= 0.4) return heard;
        var pivots = Pivots();
        bool modifier = pivots.Contains(first), head = !pivots.Contains(second) && IsHead(second);
        return first != second && modifier && head ? GeneralisedOrder : 0;
    }

    /// <summary>Acceptability given to an order that was never heard but fits the modifier-before-head pattern.</summary>
    public double GeneralisedOrder { get; set; } = 0.4;

    /// <summary>Pivot words (open-slot frames like "more _"), the most productive first.</summary>
    public IReadOnlyList<string> Pivots() =>
        _pairs.Values.Where(p => p.Weight >= MinFollowerWeight)
               .GroupBy(p => p.First)
               .Where(g => g.Count() >= PivotMinFollowers)
               .OrderByDescending(g => g.Count())
               .Select(g => g.Key).ToList();

    /// <summary>
    /// Two-word utterances the baby could try: phrases it has heard (imitation) and novel pivot + known word
    /// combinations (productivity). <paramref name="vocabulary"/> = words it can name things with.
    /// </summary>
    public IEnumerable<(string Phrase, bool Novel)> Candidates(IEnumerable<string> vocabulary, int max = 30)
    {
        var heard = _pairs.Values.Where(p => p.Weight >= 0.4).OrderByDescending(p => p.Weight)
                          .Select(p => ($"{p.First} {p.Second}", false));
        var vocab = vocabulary.ToList();
        var novel = Pivots().SelectMany(piv => vocab.Where(w => w != piv && !_pairs.ContainsKey((piv, w)))
                                                    .Select(w => ($"{piv} {w}", true)));
        return heard.Concat(novel).Take(max).ToList();
    }

    /// <summary>One tick of slow forgetting; practised pairs fade slower.</summary>
    public void Decay()
    {
        foreach (var (key, link) in _pairs.ToList())
        {
            link.Weight *= 1 - DecayRate / (1 + link.Hits * 0.5);
            if (link.Weight < PruneBelow) _pairs.Remove(key);
        }
    }

    // ---- persistence ----
    public PhrasesState ToState() => new(_pairs.Values.Select(p => new PairState(p.First, p.Second, p.Weight, p.Hits)).ToList());

    public static Phrases FromState(PhrasesState s)
    {
        var ph = new Phrases();
        foreach (var p in s.Pairs)
            ph._pairs[(p.First, p.Second)] = new PairLink { First = p.First, Second = p.Second, Weight = p.Weight, Hits = p.Hits };
        return ph;
    }
}

public sealed record PairState(string First, string Second, double Weight, int Hits);
public sealed record PhrasesState(List<PairState> Pairs);
