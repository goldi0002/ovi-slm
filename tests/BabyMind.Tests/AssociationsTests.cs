using BabyMind;
using Xunit;

namespace BabyMind.Tests;

public class AssociationsTests
{
    [Fact]
    public void SingleCoOccurrence_CreatesUsableLink()
    {
        var a = new Associations();
        a.Hear(1, "ball");
        var (word, weight) = a.Produce(1)!.Value;
        Assert.Equal("ball", word);
        Assert.True(weight >= Baby.SpeakThreshold, "one example must already be strong enough to speak");
    }

    [Fact]
    public void RepeatedCoOccurrence_StrengthensMonotonicallyAndSaturatesBelowOne()
    {
        var a = new Associations();
        double prev = 0;
        for (int i = 0; i < 20; i++)
        {
            a.Hear(1, "ball");
            double w = a.Find(1, "ball")!.Weight;
            Assert.True(w > prev);
            Assert.True(w <= 1.0);
            prev = w;
        }
        Assert.True(prev > 0.99);
    }

    [Fact]
    public void OnlyTheCoActiveLinkChanges()
    {
        var a = new Associations();
        a.Hear(1, "ball");
        Assert.Null(a.Find(2, "ball"));      // other concept untouched
        Assert.Null(a.Find(1, "cup"));       // other word untouched
    }

    [Fact]
    public void RivalWords_CompeteForTheSameConcept()
    {
        var a = new Associations();
        for (int i = 0; i < 3; i++) a.Hear(1, "ball");
        double before = a.Find(1, "ball")!.Weight;
        a.Hear(1, "red");
        Assert.True(a.Find(1, "ball")!.Weight < before);
        // keep calling it "red": red overtakes ball
        for (int i = 0; i < 6; i++) a.Hear(1, "red");
        Assert.Equal("red", a.Produce(1)!.Value.Word);
    }

    [Fact]
    public void Comprehension_ReturnsStrongestConceptFirst()
    {
        var a = new Associations();
        a.Hear(1, "ball"); a.Hear(1, "ball"); a.Hear(2, "ball");
        var c = a.Comprehend("ball");
        Assert.Equal(1, c[0].Concept);
        Assert.Equal(2, c[1].Concept);
        Assert.Empty(a.Comprehend("unknown"));
    }

    [Fact]
    public void Decay_IsSlow_ButUnreinforcedLinksEventuallyVanish()
    {
        var a = new Associations();
        a.Hear(1, "ball");
        for (int i = 0; i < 50; i++) a.Decay();
        Assert.True(a.Find(1, "ball")!.Weight > 0.4, "decay must be slow");
        for (int i = 0; i < 100000; i++) a.Decay();
        Assert.Null(a.Find(1, "ball"));
    }

    [Fact]
    public void PractisedLinks_DecaySlowerThanFreshOnes()
    {
        var a = new Associations();
        for (int i = 0; i < 10; i++) a.Hear(1, "ball");
        a.Hear(2, "cup");
        double ballBefore = a.Find(1, "ball")!.Weight, cupBefore = a.Find(2, "cup")!.Weight;
        for (int i = 0; i < 500; i++) a.Decay();
        double ballRatio = a.Find(1, "ball")!.Weight / ballBefore;
        double cupRatio = a.Find(2, "cup")!.Weight / cupBefore;
        Assert.True(ballRatio > cupRatio);
    }

    [Fact]
    public void Consolidation_LiftsFloorAndSlowsDecay()
    {
        var a = new Associations();
        a.Hear(1, "ball");
        a.Hear(2, "cup");
        a.Consolidate(1, "ball");
        Assert.True(a.Find(1, "ball")!.Weight >= 0.6);
        for (int i = 0; i < 2000; i++) a.Decay();
        double cup = a.Find(2, "cup")?.Weight ?? 0;
        Assert.True(a.Find(1, "ball")!.Weight > cup);
        Assert.True(a.Find(1, "ball")!.Weight > 0.55);
    }

    [Fact]
    public void Reward_PraiseStrengthens_CorrectionWeakens()
    {
        var a = new Associations();
        a.Hear(1, "ball");
        double w = a.Find(1, "ball")!.Weight;
        a.Reward(1, "ball", +0.3);
        Assert.True(a.Find(1, "ball")!.Weight > w);
        a.Reward(1, "ball", -0.9);
        Assert.True(a.Find(1, "ball")!.Weight < Baby.SpeakThreshold);
    }

    [Fact]
    public void ForgetConcept_RemovesItsLinks()
    {
        var a = new Associations();
        a.Hear(1, "ball");
        a.ForgetConcept(1);
        Assert.Null(a.Produce(1));
    }

    [Fact]
    public void State_RoundTrips()
    {
        var a = new Associations();
        a.Hear(1, "ball"); a.Hear(1, "ball"); a.Consolidate(1, "ball");
        var b = Associations.FromState(a.ToState());
        var l = b.Find(1, "ball")!;
        Assert.Equal(a.Find(1, "ball")!.Weight, l.Weight);
        Assert.Equal(2, l.Hits);
        Assert.True(l.Consolidated);
    }
}
