using BabyMind;
using Xunit;

namespace BabyMind.Tests;

public class ConceptMapTests
{
    private static float[] V(params float[] v) => v;

    [Fact]
    public void SingleExample_FormsAConcept()
    {
        var map = new ConceptMap(2, vigilance: 0.3f);
        var o = map.Observe(V(0.9f, 0.1f), 1);
        Assert.True(o.IsNew);
        Assert.Single(map.Concepts);
        Assert.Equal(new[] { 0.9f, 0.1f }, o.Concept.Prototype);
    }

    [Fact]
    public void SimilarInput_IsRecognisedAsSameConcept_AndPrototypeMovesToMean()
    {
        var map = new ConceptMap(1, vigilance: 0.3f);
        map.Observe(V(0.5f), 1);
        var o = map.Observe(V(0.7f), 2);
        Assert.False(o.IsNew);
        Assert.Equal(0.6, o.Concept.Prototype[0], 3);   // incremental mean of 0.5 and 0.7
        Assert.Equal(2, o.Concept.Count);
        Assert.Single(map.Concepts);
    }

    [Fact]
    public void DistinctInputs_FormSeparateConcepts()
    {
        var map = new ConceptMap(2, vigilance: 0.3f);
        var a = map.Observe(V(0f, 0f), 1).Concept;
        var b = map.Observe(V(1f, 1f), 2).Concept;
        Assert.NotEqual(a.Id, b.Id);
        Assert.Equal(a.Id, map.Observe(V(0.05f, 0.0f), 3).Concept.Id);
        Assert.Equal(b.Id, map.Observe(V(0.95f, 1.0f), 4).Concept.Id);
    }

    [Fact]
    public void NoisyClusters_ConvergeToTwoConcepts()
    {
        var rng = new Random(7);
        var map = new ConceptMap(2, vigilance: 0.3f);
        for (int i = 0; i < 300; i++)
        {
            float cx = i % 2 == 0 ? 0.2f : 0.8f;
            map.Observe(V(cx + (float)(rng.NextDouble() - .5) * 0.1f, cx + (float)(rng.NextDouble() - .5) * 0.1f), i);
        }
        Assert.Equal(2, map.Concepts.Count);
        Assert.All(map.Concepts, c => Assert.True(c.Count > 100));
    }

    [Fact]
    public void Nearest_DoesNotLearn()
    {
        var map = new ConceptMap(1);
        map.Observe(V(0.5f), 1);
        map.Nearest(V(0.6f));
        Assert.Equal(0.5f, map.Concepts[0].Prototype[0]);
        Assert.Equal(1, map.Concepts[0].Count);
    }

    [Fact]
    public void CapacityCap_EvictsLeastUsedConcept()
    {
        var map = new ConceptMap(1, vigilance: 0.05f, maxConcepts: 2);
        var a = map.Observe(V(0.0f), 1).Concept;
        map.Observe(V(0.0f), 2);                       // a seen twice
        var b = map.Observe(V(0.5f), 3).Concept;       // b seen once
        var o = map.Observe(V(1.0f), 4);               // cap reached: b is weakest
        Assert.Equal(b.Id, o.EvictedId);
        Assert.Equal(2, map.Concepts.Count);
        Assert.Contains(map.Concepts, c => c.Id == a.Id);
    }

    [Fact]
    public void WrongDimension_Throws() =>
        Assert.Throws<ArgumentException>(() => new ConceptMap(3).Observe(V(1f), 1));

    [Fact]
    public void State_RoundTrips()
    {
        var map = new ConceptMap(2, 0.3f);
        map.Observe(V(0.1f, 0.2f), 1); map.Observe(V(0.9f, 0.9f), 2);
        var copy = ConceptMap.FromState(map.ToState());
        Assert.Equal(2, copy.Concepts.Count);
        Assert.Equal(map.Concepts[1].Prototype, copy.Concepts[1].Prototype);
        // ids keep counting upward after reload
        Assert.Equal(3, copy.Observe(V(0.5f, 0.0f), 3).Concept.Id);
    }
}
