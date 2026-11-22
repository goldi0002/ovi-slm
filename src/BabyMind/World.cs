namespace BabyMind;

/// <summary>What the baby's senses deliver for one moment: a raw feature vector, nothing more.</summary>
public readonly record struct SensoryInput(float[] Features);

/// <summary>An object lying in the baby's view.</summary>
public sealed record WorldObject(string Color, string Shape, string Size);

/// <summary>
/// A tiny simulated environment. It holds at most one object in view and turns it into a noisy
/// feature vector: [r, g, b, round, square, triangle, long, size].
/// The baby never sees the names (colour, shape, "ball"); it only gets numbers.
/// (Needs such as hunger/comfort join the world in Stage 2.)
/// </summary>
public sealed class World
{
    /// <summary>Length of every sensory feature vector.</summary>
    public const int Dim = 8;

    public static readonly string[] Shapes = { "round", "square", "triangle", "long" };

    public static readonly IReadOnlyDictionary<string, float[]> Colors = new Dictionary<string, float[]>
    {
        ["red"] = new[] { 0.9f, 0.1f, 0.1f },
        ["green"] = new[] { 0.1f, 0.8f, 0.2f },
        ["blue"] = new[] { 0.1f, 0.2f, 0.9f },
        ["yellow"] = new[] { 0.95f, 0.9f, 0.1f },
        ["white"] = new[] { 0.95f, 0.95f, 0.95f },
        ["orange"] = new[] { 1.0f, 0.5f, 0.0f },
        ["purple"] = new[] { 0.6f, 0.1f, 0.7f },
    };

    public static readonly IReadOnlyDictionary<string, float> Sizes = new Dictionary<string, float>
    {
        ["small"] = 0.2f,
        ["medium"] = 0.5f,
        ["big"] = 0.8f,
    };

    /// <summary>Convenience presets for the teacher ("show ball"). Names are NOT visible to the baby.</summary>
    public static readonly IReadOnlyDictionary<string, WorldObject> Props = new Dictionary<string, WorldObject>
    {
        ["ball"] = new("red", "round", "small"),
        ["block"] = new("blue", "square", "medium"),
        ["banana"] = new("yellow", "long", "small"),
        ["cup"] = new("white", "round", "medium"),
        ["tree"] = new("green", "triangle", "big"),
    };

    private readonly Random _rng;

    /// <summary>Standard deviation of the sensor noise added to every feature.</summary>
    public float NoiseSigma { get; set; } = 0.03f;

    /// <summary>The object currently in view, or null.</summary>
    public WorldObject? Current { get; private set; }

    public World(int seed = 1) => _rng = new Random(seed);

    /// <summary>Put an object in view. Returns false (and an error message) if a property is unknown.</summary>
    public bool TryPlace(string color, string shape, string size, out string error)
    {
        error = "";
        if (!Colors.ContainsKey(color)) error = $"unknown color '{color}' (try: {string.Join(", ", Colors.Keys)})";
        else if (Array.IndexOf(Shapes, shape) < 0) error = $"unknown shape '{shape}' (try: {string.Join(", ", Shapes)})";
        else if (!Sizes.ContainsKey(size)) error = $"unknown size '{size}' (try: {string.Join(", ", Sizes.Keys)})";
        else { Current = new WorldObject(color, shape, size); return true; }
        return false;
    }

    /// <summary>Remove whatever is in view.</summary>
    public void Clear() => Current = null;

    /// <summary>Sense the current object (a fresh noisy sample each call), or null if nothing is in view.</summary>
    public SensoryInput? Sense()
    {
        if (Current is null) return null;
        var f = new float[Dim];
        var rgb = Colors[Current.Color];
        for (int i = 0; i < 3; i++) f[i] = rgb[i];
        f[3 + Array.IndexOf(Shapes, Current.Shape)] = 1f;
        f[7] = Sizes[Current.Size];
        for (int i = 0; i < Dim; i++)
            f[i] = Math.Clamp(f[i] + NoiseSigma * Gaussian(), 0f, 1f);
        return new SensoryInput(f);
    }

    /// <summary>Human-readable label of a feature vector ("red round small") — for debug output only.</summary>
    public static string Describe(float[] f)
    {
        string color = "?"; float best = float.MaxValue;
        foreach (var (name, rgb) in Colors)
        {
            float d = 0;
            for (int i = 0; i < 3; i++) d += (f[i] - rgb[i]) * (f[i] - rgb[i]);
            if (d < best) { best = d; color = name; }
        }
        int s = 0;
        for (int i = 1; i < Shapes.Length; i++) if (f[3 + i] > f[3 + s]) s = i;
        string size = Sizes.OrderBy(kv => Math.Abs(kv.Value - f[7])).First().Key;
        return $"{color} {Shapes[s]} {size}";
    }

    // Box–Muller transform: one standard-normal sample.
    private float Gaussian()
    {
        double u1 = 1.0 - _rng.NextDouble(), u2 = _rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
}
