namespace BabyMind;

public enum ActionKind { Look, Reach, Say, Quiet }

/// <summary>One thing the baby can do. Saying a word (or a two-word phrase, written "more milk") is just another action, so it is learned by reward like any other.</summary>
public readonly record struct BabyAction(ActionKind Kind, string? Word = null)
{
    /// <summary>Stable text key used in the Q-table and the save file.</summary>
    public string Key => Kind == ActionKind.Say ? $"say:{Word}" : Kind.ToString().ToLowerInvariant();
}

/// <summary>
/// Simple tabular reinforcement learning (a contextual bandit with epsilon-greedy exploration).
///  - State: a short text key such as "hungry,obj" (hunger bucket + whether something is in view).
///  - Actions: look, reach, say(word or phrase) for each utterance the baby could produce, stay quiet.
///  - Learning: after an action, Q(s,a) moves toward the reward it earned: Q += alpha * (reward - Q).
///    Speaking costs a little, so the baby only keeps talking if talking pays off.
///  - Credit: the last action stays "pending" until a reward arrives (the parent reacts) or the next
///    action starts, in which case it is settled with just its cost (nothing happened).
/// Update is local: exactly one table entry changes per action.
/// </summary>
public sealed class ActionSelector
{
    private readonly Dictionary<string, double> _q = new();
    private readonly Random _rng;
    private (string State, BabyAction Action)? _pending;

    public double Alpha { get; }
    /// <summary>Probability of trying a random action; decays toward <see cref="MinEpsilon"/>.</summary>
    public double Epsilon { get; private set; }
    public double MinEpsilon { get; }
    public double EpsilonDecay { get; }
    /// <summary>Effort cost per spoken word (a two-word phrase costs twice as much) and per reach.</summary>
    public double SayCost { get; }

    public ActionSelector(Random? rng = null, double alpha = 0.5, double epsilon = 0.3,
                          double minEpsilon = 0.05, double epsilonDecay = 0.995, double sayCost = 0.02)
    {
        _rng = rng ?? new Random();
        Alpha = alpha; Epsilon = epsilon; MinEpsilon = minEpsilon; EpsilonDecay = epsilonDecay; SayCost = sayCost;
    }

    /// <summary>Optimistic starting value of an untried action: new things are worth trying once.</summary>
    private static double Prior(BabyAction a) => a.Kind switch
    {
        ActionKind.Say => WordCount(a) > 1 ? 0.06 : 0.1,   // single words are tried before phrases
        ActionKind.Look or ActionKind.Reach => 0.05,
        _ => 0.0,
    };
    private static int WordCount(BabyAction a) => a.Word?.Count(c => c == ' ') + 1 ?? 0;
    private double Cost(BabyAction a) => a.Kind switch
    {
        ActionKind.Say => SayCost * WordCount(a),
        ActionKind.Reach => SayCost,
        _ => 0.0,
    };

    public double Value(string state, BabyAction a) => _q.TryGetValue(Slot(state, a), out var v) ? v : Prior(a);

    /// <summary>Pick an action for this state and remember it as pending, awaiting reward.</summary>
    public BabyAction Choose(string state, IEnumerable<string> utterances, bool canLook)
    {
        var options = new List<BabyAction> { new(ActionKind.Quiet) };
        if (canLook) { options.Add(new(ActionKind.Look)); options.Add(new(ActionKind.Reach)); }
        options.AddRange(utterances.Select(w => new BabyAction(ActionKind.Say, w)));

        BabyAction pick;
        if (_rng.NextDouble() < Epsilon) pick = options[_rng.Next(options.Count)];
        else
        {
            double best = options.Max(o => Value(state, o));
            var top = options.Where(o => Math.Abs(Value(state, o) - best) < 1e-9).ToList();   // random tie-break
            pick = top[_rng.Next(top.Count)];
        }
        Epsilon = Math.Max(MinEpsilon, Epsilon * EpsilonDecay);
        _pending = (state, pick);
        return pick;
    }

    /// <summary>Credit the pending action with a reward (minus its cost). Returns false if nothing was pending.</summary>
    public bool Reward(double reward)
    {
        if (_pending is not { } p) return false;
        Update(p.State, p.Action, reward - Cost(p.Action));
        _pending = null;
        return true;
    }

    /// <summary>No reward came: the pending action just cost its effort.</summary>
    public void Settle() => Reward(0);

    private void Update(string state, BabyAction a, double target)
    {
        double q = Value(state, a);
        _q[Slot(state, a)] = q + Alpha * (target - q);
    }

    private static string Slot(string state, BabyAction a) => $"{state}|{a.Key}";

    /// <summary>Learned preferences, best first (for the debug report).</summary>
    public IEnumerable<(string State, string Action, double Value)> Table() =>
        _q.Select(kv => { int i = kv.Key.IndexOf('|'); return (kv.Key[..i], kv.Key[(i + 1)..], kv.Value); })
          .OrderByDescending(t => t.Item3);

    // ---- persistence ----
    public ActionSelectorState ToState() => new(new Dictionary<string, double>(_q), Epsilon);

    public static ActionSelector FromState(ActionSelectorState s, Random? rng = null)
    {
        var a = new ActionSelector(rng, epsilon: s.Epsilon);
        foreach (var (k, v) in s.Q) a._q[k] = v;
        return a;
    }
}

public sealed record ActionSelectorState(Dictionary<string, double> Q, double Epsilon);
