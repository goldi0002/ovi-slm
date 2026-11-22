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
        for (int i = 0; i < 100000; i++) p.Decay();
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

public class WordOrderGeneralisationTests
{
    private static Phrases Taught()
    {
        var p = new Phrases();
        p.Hear("red", "ball"); p.Hear("red", "cup");        // "red" is a modifier; ball and cup are heads
        p.Hear("green", "cup"); p.Hear("green", "block");   // "green" is a modifier too
        return p;
    }

    [Fact]
    public void ModifierBeforeHead_IsAcceptedForAPairNeverHeard()
    {
        var p = Taught();
        Assert.Null(p.Find("green", "ball"));                       // never heard...
        Assert.True(p.Order("green", "ball") >= 0.4);               // ...but fits the pattern
    }

    [Fact]
    public void WrongOrder_IsNotAccepted()
    {
        var p = Taught();
        Assert.Equal(0, p.Order("ball", "green"));                  // head before modifier
        Assert.Equal(0, p.Order("red", "green"));                   // two modifiers
        Assert.Equal(0, p.Order("ball", "cup"));                    // two heads
    }

    [Fact]
    public void UnknownWords_AreNotAccepted()
    {
        var p = Taught();
        Assert.Equal(0, p.Order("green", "frog"));                  // "frog" was never heard after a modifier
        Assert.Equal(0, p.Order("blue", "ball"));                   // "blue" was never heard in front of anything
    }

    [Fact]
    public void HeardPair_KeepsItsOwnWeight()
    {
        var p = Taught();
        p.Hear("red", "ball");
        Assert.Equal(p.Find("red", "ball")!.Weight, p.Order("red", "ball"));
    }
}

public class OneShotLongevityTests
{
    [Fact]
    public void WordHeardOnce_StillUsableAfterManyTicks_ButNotForever()
    {
        var a = new Associations();
        a.Hear(1, "frog");
        for (int i = 0; i < 2500; i++) a.Decay();
        Assert.True(a.Produce(1)!.Value.Weight >= Baby.SpeakThreshold, "a single exposure should last thousands of ticks");
        for (int i = 0; i < 100000; i++) a.Decay();
        Assert.Null(a.Produce(1));                                  // unused knowledge is eventually forgotten
    }

    [Fact]
    public void Baby_RecallsAOneShotWord_AfterWaiting()
    {
        var w = new World(1); w.TryPlace("green", "triangle", "small", out _);
        var baby = new Baby(selector: new ActionSelector(new Random(1)));
        baby.Experience(w.Sense(), "frog");
        for (int i = 0; i < 1500; i++) baby.Act(null);
        Assert.Equal("frog", baby.Experience(w.Sense(), null).Said);
    }
}
