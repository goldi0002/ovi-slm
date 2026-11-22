namespace BabyMind;

/// <summary>What the baby has learned about one word's perceptual meaning, from every object it was heard with.</summary>
public sealed class WordModel
{
    public string Word { get; init; } = "";
    /// <summary>How many times the word was heard with something in view.</summary>
    public int N { get; set; }
    /// <summary>Running mean of the feature vectors seen with this word.</summary>
    public float[] Mean { get; init; } = Array.Empty<float>();
    /// <summary>Running variance per feature (low = the word is consistent about that feature).</summary>
    public float[] Var { get; init; } = Array.Empty<float>();
    /// <summary>Distinct concepts the word was heard with: one concept alone can't tell a name from a property.</summary>
    public HashSet<int> Concepts { get; init; } = new();
    /// <summary>Consecutive contradicting samples ignored so far (see <see cref="WordMeanings.OutlierPatience"/>).</summary>
    public int Outliers { get; set; }
    /// <summary>Things the parent said this word does NOT apply to ("no").</summary>
    public List<float[]> Exceptions { get; init; } = new();
}

/// <summary>A word that fits a thing, with how well, and which feature groups carried the match.</summary>
public readonly record struct WordMatch(string Word, double Score, IReadOnlyList<string> Groups);

/// <summary>
/// Cross-situational word meaning, which is what lets "red" mean *red* instead of "that one red ball".
/// Every word keeps an online mean/variance of the features of the objects it was heard with. Features come in
/// groups (colour, shape, size). A group is *relevant* to a word only if the word was consistent on it:
///  - "red" heard with a red ball and a red block: colour is consistent, shape and size vary -> "red" means colour red.
///  - "ball" heard with a red ball and a blue ball: shape and size are consistent, colour varies -> any small round thing.
/// A word heard with only one concept stays object-specific (no generalisation), and function-like words
/// ("more", heard with everything) end up with no relevant group, so they match nothing.
/// Learning is online and local: one running update of one word per hearing.
/// </summary>
public sealed class WordMeanings
{
    private readonly Dictionary<string, WordModel> _models = new();
    private readonly (string Name, int Start, int Len)[] _groups;

    /// <summary>Largest standard deviation at which a feature group still counts as "consistent".</summary>
    public float Tolerance { get; set; } = 0.12f;
    /// <summary>Smallest spread used when scoring (so sensor noise doesn't make matching impossibly strict).</summary>
    public float Floor { get; set; } = 0.1f;
    /// <summary>Minimum score for a word to be said about a thing.</summary>
    public double Accept { get; set; } = 0.6;
    /// <summary>Distinct concepts needed before a word may generalise.</summary>
    public int MinConcepts { get; set; } = 2;
    public float MinRate { get; set; } = 0.05f;
    public int MaxExceptions { get; set; } = 8;
    /// <summary>
    /// An established word (≥ <see cref="EstablishedAfter"/> hearings, ≥3 distinct things) ignores a sample that sharply contradicts
    /// its meaning, e.g. a parent's slip, and only changes its mind if the contradiction repeats this many times in a row + 1.
    /// </summary>
    public int OutlierPatience { get; set; } = 2;
    public int EstablishedAfter { get; set; } = 5;
    /// <summary>Mean squared z-score (over relevant groups) beyond which a sample counts as contradicting the word.</summary>
    public double OutlierZ2 { get; set; } = 9;
    /// <summary>A thing this close to a rejected example is never matched.</summary>
    public float ExceptionRadius { get; set; } = 0.15f;

    public WordMeanings((string Name, int Start, int Len)[] groups) => _groups = groups;

    public IEnumerable<WordModel> Models => _models.Values;
    public WordModel? Get(string word) => _models.GetValueOrDefault(word);

    /// <summary>The word was heard while a thing with these features (belonging to this concept) was in view.</summary>
    public void Hear(string word, float[] features, int conceptId)
    {
        if (!_models.TryGetValue(word, out var m))
        {
            _models[word] = m = new WordModel { Word = word, Mean = (float[])features.Clone(), Var = new float[features.Length] };
            m.N = 1;
            m.Concepts.Add(conceptId);
            return;
        }
        if (m.N >= EstablishedAfter && m.Concepts.Count >= 3 && Contradicts(m, features) && m.Outliers < OutlierPatience)
        {
            m.Outliers++;                                          // probably a slip: don't let one example rewrite the meaning
            return;
        }
        m.Outliers = 0;
        m.N++;
        m.Concepts.Add(conceptId);
        float rate = Math.Max(MinRate, 1f / m.N);                 // incremental (exponentially weighted) mean and variance
        for (int i = 0; i < features.Length; i++)
        {
            float delta = features[i] - m.Mean[i];
            m.Mean[i] += rate * delta;
            m.Var[i] = (1 - rate) * (m.Var[i] + rate * delta * delta);
        }
    }

    // Does this sample clash with what the word is consistent about? (only relevant groups count)
    private bool Contradicts(WordModel m, float[] f)
    {
        var rel = Relevant(m);
        if (rel.Count == 0) return false;
        double total = 0;
        foreach (int g in rel)
        {
            var (_, start, len) = _groups[g];
            double z2 = 0;
            for (int i = start; i < start + len; i++)
            {
                double z = (f[i] - m.Mean[i]) / Math.Max(MathF.Sqrt(m.Var[i]), Floor);
                z2 += z * z;
            }
            total += z2 / len;
        }
        return total / rel.Count > OutlierZ2;
    }

    /// <summary>Names of the feature groups the word is consistent about (empty = it carries no perceptual meaning).</summary>
    public IReadOnlyList<string> RelevantGroups(string word) =>
        _models.TryGetValue(word, out var m) ? Relevant(m).Select(g => _groups[g].Name).ToList() : new List<string>();

    private List<int> Relevant(WordModel m)
    {
        var rel = new List<int>();
        for (int g = 0; g < _groups.Length; g++)
        {
            var (_, start, len) = _groups[g];
            bool consistent = true;
            for (int i = start; i < start + len; i++)
                if (MathF.Sqrt(m.Var[i]) > Tolerance) consistent = false;
            if (consistent) rel.Add(g);
        }
        return rel;
    }

    /// <summary>How well the word fits a thing, 0..1. Zero if the word has not been heard with enough different things.</summary>
    /// <param name="strict">Function-like words (pivots such as "more", which are heard with everything) must prove themselves
    /// on one more distinct thing, so a coincidence between two objects doesn't give them a meaning.</param>
    public double Score(string word, float[] f, bool strict = false)
    {
        if (!_models.TryGetValue(word, out var m) || m.Concepts.Count < MinConcepts + (strict ? 1 : 0)) return 0;
        var rel = Relevant(m);
        if (rel.Count == 0) return 0;
        foreach (var e in m.Exceptions)
            if (ConceptMap.Distance(e, f) < ExceptionRadius) return 0;

        double total = 0;
        foreach (int g in rel)
        {
            var (_, start, len) = _groups[g];
            double z2 = 0;
            for (int i = start; i < start + len; i++)
            {
                double s = Math.Max(MathF.Sqrt(m.Var[i]), Floor);
                double z = (f[i] - m.Mean[i]) / s;
                z2 += z * z;
            }
            total += z2 / len;                                    // every group counts equally
        }
        return Math.Exp(-0.5 * total / rel.Count);
    }

    /// <summary>All generalising words that fit a thing well enough, best first.</summary>
    public IReadOnlyList<WordMatch> Matches(float[] f, IReadOnlySet<string>? strictWords = null) =>
        _models.Keys.Select(w => (Word: w, Score: Score(w, f, strictWords?.Contains(w) ?? false)))
                    .Where(x => x.Score >= Accept)
                    .OrderByDescending(x => x.Score)
                    .Select(x => new WordMatch(x.Word, x.Score, RelevantGroups(x.Word)))
                    .ToList();

    /// <summary>The parent said "no, that isn't a <paramref name="word"/>": remember the counter-example.</summary>
    public void Reject(string word, float[] f)
    {
        if (!_models.TryGetValue(word, out var m)) return;
        m.Exceptions.Add((float[])f.Clone());
        if (m.Exceptions.Count > MaxExceptions) m.Exceptions.RemoveAt(0);
    }

    // ---- persistence ----
    public WordMeaningsState ToState() => new(_models.Values.Select(m =>
        new WordModelState(m.Word, m.N, (float[])m.Mean.Clone(), (float[])m.Var.Clone(), m.Concepts.OrderBy(c => c).ToList(),
                           m.Exceptions.Select(e => (float[])e.Clone()).ToList(), m.Outliers)).ToList());

    public static WordMeanings FromState(WordMeaningsState s, (string Name, int Start, int Len)[] groups)
    {
        var wm = new WordMeanings(groups);
        foreach (var m in s.Models)
            wm._models[m.Word] = new WordModel
            {
                Word = m.Word, N = m.N, Mean = m.Mean, Var = m.Var,
                Concepts = new HashSet<int>(m.Concepts), Exceptions = m.Exceptions, Outliers = m.Outliers,
            };
        return wm;
    }
}

public sealed record WordModelState(string Word, int N, float[] Mean, float[] Var, List<int> Concepts, List<float[]> Exceptions, int Outliers = 0);
public sealed record WordMeaningsState(List<WordModelState> Models);
