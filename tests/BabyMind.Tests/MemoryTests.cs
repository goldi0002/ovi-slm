using BabyMind;
using Xunit;

namespace BabyMind.Tests;

public class MemoryTests
{
    [Fact]
    public void Episodes_FadeAndAreForgotten()
    {
        var m = new Memory(fadeRate: 0.5, forgetBelow: 0.1);
        m.Record(1, 1, null);
        m.Fade(2);
        Assert.Equal(0.5, m.Episodes[0].Strength, 3);
        for (int i = 0; i < 5; i++) m.Fade(3 + i);
        Assert.Empty(m.Episodes);
    }

    [Fact]
    public void RepeatedPattern_ConsolidatesExactlyOnce()
    {
        var m = new Memory(consolidateAfter: 3);
        Assert.False(m.Record(1, 7, "ball"));
        Assert.False(m.Record(2, 7, "ball"));
        Assert.True(m.Record(3, 7, "ball"));
        Assert.False(m.Record(4, 7, "ball"));
    }

    [Fact]
    public void PatternSpreadTooFarApart_DoesNotConsolidate()
    {
        var m = new Memory(consolidateAfter: 3, patternWindow: 10);
        m.Record(1, 7, "ball"); m.Record(50, 7, "ball");
        Assert.False(m.Record(100, 7, "ball"));
    }

    [Fact]
    public void ConsolidatedPattern_SurvivesFading()
    {
        var m = new Memory(consolidateAfter: 2, patternWindow: 10);
        m.Record(1, 1, "ball"); m.Record(2, 1, "ball");
        for (int t = 3; t < 500; t++) m.Fade(t);
        Assert.Contains(m.Patterns, p => p.Value.Consolidated);
    }
}
