using BabyMind;
using Xunit;

namespace BabyMind.Tests;

public class PhrasesTests
{
    [Fact]
    public void HeardOnce_LinkIsStrongEnoughToImitate()
    {
        var p = new Phrases();
        p.Hear("more", "milk");
        Assert.True(p.Find("more", "milk")!.Weight >= 0.4);
        Assert.Contains(p.Candidates(new[] { "milk" }), c => c.Phrase == "more milk" && !c.Novel);
    }

    [Fact]
    public void WordOrderMatters()
    {
        var p = new Phrases();
        p.Hear("more", "milk");
        Assert.Null(p.Find("milk", "more"));
    }

    [Fact]
    public void Repetition_StrengthensAndSaturates()
    {
        var p = new Phrases();
        double prev = 0;
        for (int i = 0; i < 10; i++) { p.Hear("more", "milk"); var w = p.Find("more", "milk")!.Weight; Assert.True(w > prev && w <= 1); prev = w; }
    }

    [Fact]
    public void Pivot_NeedsSeveralDifferentFollowers()
    {
        var p = new Phrases();
        p.Hear("more", "milk");
        Assert.Empty(p.Pivots());
        p.Hear("more", "ball");
        Assert.Equal(new[] { "more" }, p.Pivots());
    }

    [Fact]
    public void Pivot_ProducesNovelCombinationsWithKnownWords()
    {
        var p = new Phrases();
        p.Hear("more", "milk"); p.Hear("more", "ball");
        var c = p.Candidates(new[] { "milk", "ball", "cup" }).ToList();
        Assert.Contains(("more cup", true), c);                 // never heard, but "more _" + known word "cup"
        Assert.DoesNotContain(c, x => x.Novel && x.Phrase is "more milk" or "more ball");
        Assert.DoesNotContain(c, x => x.Phrase == "more more");
    }

    [Fact]
    public void Decay_IsSlow_ButUnheardPairsEventuallyVanish()
    {
        var p = new Phrases();
        p.Hear("more", "milk");
        for (int i = 0; i < 50; i++) p.Decay();
        Assert.NotNull(p.Find("more", "milk"));
        for (int i = 0; i < 20000; i++) p.Decay();
        Assert.Null(p.Find("more", "milk"));
    }

    [Fact]
    public void State_RoundTrips()
    {
        var p = new Phrases();
        p.Hear("more", "milk"); p.Hear("more", "milk"); p.Hear("more", "ball");
        var q = Phrases.FromState(p.ToState());
        Assert.Equal(p.Find("more", "milk")!.Weight, q.Find("more", "milk")!.Weight);
        Assert.Equal(p.Pivots(), q.Pivots());
    }
}
