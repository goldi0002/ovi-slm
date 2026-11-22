using BabyMind;
using Xunit;

namespace BabyMind.Tests;

public class WordMeaningsTests
{
    // Noise-free feature vectors: [r,g,b, round,square,triangle,long, size]
    private static float[] F(string color, string shape, string size)
    {
        var w = new World(1) { NoiseSigma = 0 };
        w.TryPlace(color, shape, size, out _);
        return w.Sense()!.Value.Features;
    }

    private static WordMeanings New() => new(World.FeatureGroups);

    [Fact]
    public void WordHeardWithOneThing_StaysSpecific_NoGeneralisation()
    {
        var m = New();
        m.Hear("red", F("red", "round", "small"), 1);
        Assert.Empty(m.Matches(F("red", "square", "medium")));
        Assert.Empty(m.Matches(F("red", "round", "small")));     // even the same thing: naming that is Associations' job
    }

    [Fact]
    public void Colour_IsLearnedFromContrastInShape()
    {
        var m = New();
        m.Hear("red", F("red", "round", "small"), 1);
        m.Hear("red", F("red", "square", "medium"), 2);
        Assert.Equal(new[] { "color" }, m.RelevantGroups("red"));
        Assert.Contains(m.Matches(F("red", "long", "big")), x => x.Word == "red");         // a never-seen red thing
        Assert.DoesNotContain(m.Matches(F("blue", "round", "small")), x => x.Word == "red");
        Assert.DoesNotContain(m.Matches(F("orange", "round", "small")), x => x.Word == "red");
    }

    [Fact]
    public void ObjectWord_IsLearnedFromContrastInColour()
    {
        var m = New();
        m.Hear("ball", F("red", "round", "small"), 1);
        m.Hear("ball", F("blue", "round", "small"), 2);
        Assert.Equal(new[] { "shape", "size" }, m.RelevantGroups("ball"));
        Assert.Contains(m.Matches(F("green", "round", "small")), x => x.Word == "ball");
        Assert.DoesNotContain(m.Matches(F("red", "square", "small")), x => x.Word == "ball");
        Assert.DoesNotContain(m.Matches(F("red", "round", "big")), x => x.Word == "ball");
    }

    [Fact]
    public void SameConceptTwice_IsNotEvidenceOfGeneralisation()
    {
        var m = New();
        m.Hear("red", F("red", "round", "small"), 1);
        m.Hear("red", F("red", "round", "small"), 1);
        Assert.Empty(m.Matches(F("red", "square", "medium")));
    }

    [Fact]
    public void FunctionWord_HeardWithEverything_HasNoMeaning()
    {
        var m = New();
        m.Hear("more", F("red", "round", "small"), 1);
        m.Hear("more", F("white", "long", "big"), 2);
        m.Hear("more", F("blue", "square", "medium"), 3);
        Assert.Empty(m.RelevantGroups("more"));
        Assert.Empty(m.Matches(F("red", "round", "small")));
    }

    [Fact]
    public void StrictWords_NeedOneMoreExample()
    {
        var m = New();
        m.Hear("red", F("red", "round", "small"), 1);
        m.Hear("red", F("red", "square", "medium"), 2);
        var strict = new HashSet<string> { "red" };
        Assert.Empty(m.Matches(F("red", "long", "big"), strict));
        m.Hear("red", F("red", "long", "small"), 3);
        Assert.Single(m.Matches(F("red", "triangle", "big"), strict));
    }

    [Fact]
    public void Rejection_StopsTheWordFromMatchingThatThing()
    {
        var m = New();
        m.Hear("ball", F("red", "round", "small"), 1);
        m.Hear("ball", F("blue", "round", "small"), 2);
        var lemon = F("yellow", "round", "small");
        Assert.Single(m.Matches(lemon));
        m.Reject("ball", lemon);
        Assert.Empty(m.Matches(lemon));
        Assert.Single(m.Matches(F("green", "round", "small")));   // other things unaffected
    }

    [Fact]
    public void State_RoundTrips()
    {
        var m = New();
        m.Hear("red", F("red", "round", "small"), 1);
        m.Hear("red", F("red", "square", "medium"), 2);
        m.Reject("red", F("orange", "round", "small"));
        var copy = WordMeanings.FromState(m.ToState(), World.FeatureGroups);
        Assert.Equal(m.RelevantGroups("red"), copy.RelevantGroups("red"));
        Assert.Equal(m.Score("red", F("red", "long", "big")), copy.Score("red", F("red", "long", "big")), 6);
        Assert.Single(copy.Get("red")!.Exceptions);
    }
}

/// <summary>The headline behaviour: "red ball" generalises to other red things (and "ball" to other balls).</summary>
public class GeneralizationTests
{
    private static Response See(Baby b, World w, string color, string shape, string size, string? say = null)
    {
        w.TryPlace(color, shape, size, out _);
        return b.Experience(w.Sense(), say);
    }

    // Teaches: red ball, blue ball, red block, red banana (all with 2-word phrases).
    private static (Baby, World) Taught()
    {
        var w = new World(3); var b = new Baby();
        See(b, w, "red", "round", "small", "red ball");
        See(b, w, "blue", "round", "small", "blue ball");
        See(b, w, "red", "square", "medium", "red block");
        See(b, w, "red", "long", "small", "red banana");
        return (b, w);
    }

    [Fact]
    public void NewRedThing_IsCalledRed()
    {
        var (b, w) = Taught();
        var r = See(b, w, "red", "triangle", "big");               // never seen, never named
        Assert.True(r.Observation!.Value.IsNew);
        Assert.Equal("red", r.Said);
        Assert.Contains("color", r.Basis);
    }

    [Fact]
    public void NewBall_IsCalledBall_EvenInANewColour()
    {
        var (b, w) = Taught();
        var r = See(b, w, "green", "round", "small");
        Assert.True(r.Observation!.Value.IsNew);
        Assert.Equal("ball", r.Said);
    }

    [Fact]
    public void KnownRedBall_IsDescribedWithBothWords_InLearnedOrder()
    {
        var (b, w) = Taught();
        Assert.Equal("red ball", See(b, w, "red", "round", "small").Said);
    }

    [Fact]
    public void RedWord_DoesNotLeakOntoOtherColours()
    {
        var (b, w) = Taught();
        Assert.Null(See(b, w, "blue", "triangle", "big").Said);
        Assert.Null(See(b, w, "green", "long", "big").Said);
    }

    [Fact]
    public void TwoExamples_AreNotEnoughForAWordUsedAsPivot_AvoidingSpuriousMeaning()
    {
        var w = new World(3); var b = new Baby();
        See(b, w, "red", "round", "small", "red ball");
        See(b, w, "red", "square", "medium", "red block");          // "red _" is a pivot with 2 things only
        Assert.Null(See(b, w, "red", "long", "big").Said);          // not yet confident
    }

    [Fact]
    public void Correction_StopsOverGeneralisation_AndTeachesTheNewWord()
    {
        var (b, w) = Taught();
        var r = See(b, w, "yellow", "round", "small");
        Assert.Equal("ball", r.Said);                               // overgeneralised: it's a lemon
        Assert.True(b.Correct("lemon"));
        Assert.Equal("lemon", See(b, w, "yellow", "round", "small").Said);
        // other balls are still balls
        Assert.Equal("ball", See(b, w, "green", "round", "small").Said);
    }

    [Fact]
    public void Debug_ShowsWhatWordsMean()
    {
        var (b, _) = Taught();
        string d = b.Debug();
        Assert.Contains("\"red\"", d);
        Assert.Contains("color=red", d);
        Assert.Contains("shape=round, size=small", d);
    }

    [Fact]
    public void Meanings_SurviveSaveLoad_AndOlderFilesStillLoad()
    {
        var (b, w) = Taught();
        string path = Path.Combine(Path.GetTempPath(), $"babymind-test-{Guid.NewGuid():N}.json");
        try
        {
            BabyState.Save(b, path);
            Assert.Equal("red", See(BabyState.Load(path), w, "red", "triangle", "big").Said);

            var v3 = b.ToState() with { Version = 3, Meanings = null };
            File.WriteAllText(path, System.Text.Json.JsonSerializer.Serialize(v3).Replace(",\"Meanings\":null", ""));
            var old = BabyState.Load(path);
            Assert.Null(See(old, w, "red", "triangle", "big").Said);  // no meanings learned yet, but loads fine
        }
        finally { File.Delete(path); }
    }
}
