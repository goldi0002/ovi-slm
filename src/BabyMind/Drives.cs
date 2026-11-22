namespace BabyMind;

public enum DriveKind { Hunger, Curiosity, Attention }

/// <summary>
/// Internal needs, each a level in [0, 1] that creeps upward over time. Satisfying a drive lowers its
/// level, and the drop *is* the reward signal (a hungry baby that gets fed is rewarded a lot).
/// Curiosity is satisfied by novel things, Attention by the parent speaking, and Hunger by being fed (Stage 2).
/// </summary>
public sealed class Drives
{
    public double Hunger { get; private set; }
    public double Curiosity { get; private set; } = 0.3;
    public double Attention { get; private set; } = 0.3;

    /// <summary>Per-tick increase of each drive.</summary>
    public double HungerGrowth { get; set; } = 0.04;
    public double CuriosityGrowth { get; set; } = 0.02;
    public double AttentionGrowth { get; set; } = 0.015;

    public double Level(DriveKind kind) => kind switch
    {
        DriveKind.Hunger => Hunger,
        DriveKind.Curiosity => Curiosity,
        _ => Attention,
    };

    /// <summary>Advance one tick: needs build up.</summary>
    public void Tick()
    {
        Hunger = Math.Min(1, Hunger + HungerGrowth);
        Curiosity = Math.Min(1, Curiosity + CuriosityGrowth);
        Attention = Math.Min(1, Attention + AttentionGrowth);
    }

    /// <summary>Lower a drive by <paramref name="amount"/>. Returns the reward: how much the level actually dropped.</summary>
    public double Satisfy(DriveKind kind, double amount)
    {
        double before = Level(kind);
        double after = Math.Max(0, before - amount);
        Set(kind, after);
        return before - after;
    }

    private void Set(DriveKind kind, double v)
    {
        switch (kind)
        {
            case DriveKind.Hunger: Hunger = v; break;
            case DriveKind.Curiosity: Curiosity = v; break;
            default: Attention = v; break;
        }
    }

    // ---- persistence ----
    public DrivesState ToState() => new(Hunger, Curiosity, Attention);

    public static Drives FromState(DrivesState s)
    {
        var d = new Drives();
        d.Hunger = s.Hunger; d.Curiosity = s.Curiosity; d.Attention = s.Attention;
        return d;
    }
}

public sealed record DrivesState(double Hunger, double Curiosity, double Attention);
