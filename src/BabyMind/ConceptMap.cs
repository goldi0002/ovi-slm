namespace BabyMind;

/// <summary>A prototype in sensory space: "the kind of thing that looks like this".</summary>
public sealed class Concept
{
    public int Id { get; init; }
    public float[] Prototype { get; init; } = Array.Empty<float>();
    /// <summary>How many experiences have shaped this concept.</summary>
    public int Count { get; set; }
    public long Created { get; init; }
    public long LastSeen { get; set; }
}

/// <summary>Result of presenting one input to the map.</summary>
/// <param name="Concept">The concept that won (or was just born).</param>
/// <param name="IsNew">True if this input was too unfamiliar and created a new concept (one-shot learning).</param>
/// <param name="Distance">Distance from the input to the winner's prototype *before* it was updated.</param>
/// <param name="EvictedId">Id of a rarely used concept dropped to stay under the size cap, if any.</param>
public readonly record struct Observation(Concept Concept, bool IsNew, float Distance, int? EvictedId = null);

/// <summary>
/// Online clustering (leader / ART-style incremental k-means). There is no training phase:
///  - find the nearest prototype;
///  - if it is within <see cref="Vigilance"/>, nudge it toward the input (learning rate 1/n, floored);
///  - otherwise the input is novel and becomes the prototype of a brand-new concept immediately.
/// So a single example is enough to form a concept, and the number of concepts is decided by the world,
/// not fixed in advance.
/// </summary>
public sealed class ConceptMap
{
    private readonly List<Concept> _concepts = new();
    private int _nextId = 1;

    public int Dim { get; }
    /// <summary>Max distance at which an input still counts as "the same kind of thing".</summary>
    public float Vigilance { get; }
    /// <summary>Lower bound on the prototype learning rate (keeps old concepts slowly adaptable).</summary>
    public float MinRate { get; }
    public int MaxConcepts { get; }

    public IReadOnlyList<Concept> Concepts => _concepts;

    public ConceptMap(int dim, float vigilance = 0.25f, float minRate = 0.05f, int maxConcepts = 2000)
    {
        Dim = dim; Vigilance = vigilance; MinRate = minRate; MaxConcepts = maxConcepts;
    }

    public Concept? Get(int id) => _concepts.FirstOrDefault(c => c.Id == id);

    /// <summary>Nearest concept to x without learning anything. Null if the map is empty.</summary>
    public (Concept? Concept, float Distance) Nearest(float[] x)
    {
        CheckDim(x);
        Concept? best = null; float bestD = float.MaxValue;
        foreach (var c in _concepts)
        {
            float d = Distance(c.Prototype, x);
            if (d < bestD) { bestD = d; best = c; }
        }
        return (best, bestD);
    }

    /// <summary>Perceive x: recognise it as an existing concept (and refine it) or create a new one.</summary>
    public Observation Observe(float[] x, long tick)
    {
        var (winner, dist) = Nearest(x);

        if (winner is null || dist > Vigilance)
        {
            int? evicted = _concepts.Count >= MaxConcepts ? EvictWeakest() : null;
            var fresh = new Concept { Id = _nextId++, Prototype = (float[])x.Clone(), Count = 1, Created = tick, LastSeen = tick };
            _concepts.Add(fresh);
            return new Observation(fresh, true, dist == float.MaxValue ? 0f : dist, evicted);
        }

        // Incremental mean: first updates are large, later ones settle at MinRate.
        winner.Count++;
        float rate = Math.Max(MinRate, 1f / winner.Count);
        for (int i = 0; i < Dim; i++)
            winner.Prototype[i] += rate * (x[i] - winner.Prototype[i]);
        winner.LastSeen = tick;
        return new Observation(winner, false, dist);
    }

    // Drop the concept that has been seen least (oldest on ties) to make room.
    private int EvictWeakest()
    {
        var victim = _concepts.OrderBy(c => c.Count).ThenBy(c => c.LastSeen).First();
        _concepts.Remove(victim);
        return victim.Id;
    }

    public static float Distance(float[] a, float[] b)
    {
        float s = 0;
        for (int i = 0; i < a.Length; i++) { float d = a[i] - b[i]; s += d * d; }
        return MathF.Sqrt(s);
    }

    private void CheckDim(float[] x)
    {
        if (x.Length != Dim) throw new ArgumentException($"Expected {Dim} features, got {x.Length}.");
    }

    // ---- persistence ----
    public ConceptMapState ToState() => new(Dim, Vigilance, MinRate, MaxConcepts, _nextId,
        _concepts.Select(c => new ConceptState(c.Id, (float[])c.Prototype.Clone(), c.Count, c.Created, c.LastSeen)).ToList());

    public static ConceptMap FromState(ConceptMapState s)
    {
        var map = new ConceptMap(s.Dim, s.Vigilance, s.MinRate, s.MaxConcepts) { _nextId = s.NextId };
        foreach (var c in s.Concepts)
            map._concepts.Add(new Concept { Id = c.Id, Prototype = c.Prototype, Count = c.Count, Created = c.Created, LastSeen = c.LastSeen });
        return map;
    }
}

public sealed record ConceptState(int Id, float[] Prototype, int Count, long Created, long LastSeen);
public sealed record ConceptMapState(int Dim, float Vigilance, float MinRate, int MaxConcepts, int NextId, List<ConceptState> Concepts);
