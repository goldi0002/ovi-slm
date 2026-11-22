using BabyMind;
using Xunit;

namespace BabyMind.Tests;

/// <summary>Stage 3 behaviour: reaching and two-word combinations.</summary>
public class Stage3Tests
{
    private static readonly BabyAction Milk = new(ActionKind.Say, "milk");
    private static readonly BabyAction MoreMilk = new(ActionKind.Say, "more milk");

    private static Baby Bottle(World w, string say)
    {
        var baby = new Baby(selector: new ActionSelector(new Random(3)));
        var b = World.Props["bottle"];
        w.TryPlace(b.Color, b.Shape, b.Size, out _);
        baby.Experience(w.Sense(), say);
        w.Clear();
        return baby;
    }

    [Fact]
    public void PhraseHeard_LabelsObjectWithLastWordOnly_AndLearnsWordOrder()
    {
        var w = new World(1); w.TryPlace("red", "round", "small", out _);
        var baby = new Baby();
        baby.Experience(w.Sense(), "more ball");
        int c = baby.Map.Concepts[0].Id;
        Assert.NotNull(baby.Assoc.Find(c, "ball"));
        Assert.Null(baby.Assoc.Find(c, "more"));
        Assert.NotNull(baby.Phrases.Find("more", "ball"));
        Assert.Equal("ball", baby.Experience(w.Sense(), null).Said);      // one example still names it
    }

    [Fact]
    public void PivotAndKnownWord_GiveNovelPhrase_NeverHeard()
    {
        var w = new World(1);
        var baby = new Baby();
        w.TryPlace("red", "round", "small", out _); baby.Experience(w.Sense(), "more ball");
        w.TryPlace("white", "round", "medium", out _); baby.Experience(w.Sense(), "more cup");
        w.TryPlace("blue", "square", "medium", out _); baby.Experience(w.Sense(), "block");
        Assert.Contains("more block", baby.Utterances());
        Assert.Null(baby.Phrases.Find("more", "block"));
    }

    [Fact]
    public void PhraseCostsMoreThanSingleWord()
    {
        var s = new ActionSelector(new Random(1), epsilon: 0, minEpsilon: 0);
        // Untried phrases start less attractive than untried single words.
        Assert.True(s.Value("x", Milk) > s.Value("x", MoreMilk));
    }

    [Fact]
    public void WhenParentDemandsTwoWords_BabyLearnsToRequestWithAPhrase()
    {
        var w = new World(2);
        var baby = Bottle(w, "more milk");
        // learn the pivot "more _" and a second word
        w.TryPlace("red", "round", "small", out _); baby.Experience(w.Sense(), "more ball"); w.Clear();

        for (int i = 0; i < 250; i++)
        {
            var r = baby.Act(w.Sense());
            if (r.Said is { } said && said.Split(' ') is var ws && ws.Contains("milk") && ws.Length == 2 && baby.Drives.Hunger >= 0.3)
            {
                var b = World.Props["bottle"]; w.TryPlace(b.Color, b.Shape, b.Size, out _);
                baby.Feed(w.Sense(), said); w.Clear();
            }
        }
        double phrase = baby.Selector.Value("peckish,none", MoreMilk);
        double single = baby.Selector.Value("peckish,none", Milk);
        Assert.True(phrase > 0.15, $"phrase value {phrase:0.00}");
        Assert.True(phrase > single + 0.1, $"phrase {phrase:0.00} should beat single word {single:0.00}");
    }

    [Fact]
    public void Reach_AtFamiliarObjectWhileCurious_IsLearnedAsWorthwhile()
    {
        var w = new World(2); w.TryPlace("red", "round", "small", out _);
        var baby = new Baby(selector: new ActionSelector(new Random(4)));
        int reaches = 0;
        for (int i = 0; i < 120; i++)
            if (baby.Act(w.Sense()).Action.Kind == ActionKind.Reach) reaches++;
        Assert.True(reaches >= 10, $"reached {reaches} times");
        Assert.Contains(baby.Selector.Table(), t => t.Action == "reach" && t.Value > 0);
    }

    [Fact]
    public void Reach_WithNothingInView_IsNotAnOption()
    {
        var baby = new Baby(selector: new ActionSelector(new Random(1), epsilon: 1.0, minEpsilon: 1.0));
        for (int i = 0; i < 100; i++) Assert.NotEqual(ActionKind.Reach, baby.Act(null).Action.Kind);
    }

    [Fact]
    public void PhrasesAndSelector_SurviveSaveLoad_AndOldFilesLoad()
    {
        var w = new World(1);
        var baby = Bottle(w, "more milk");
        w.TryPlace("red", "round", "small", out _); baby.Experience(w.Sense(), "more ball");
        string path = Path.Combine(Path.GetTempPath(), $"babymind-test-{Guid.NewGuid():N}.json");
        try
        {
            BabyState.Save(baby, path);
            var again = BabyState.Load(path);
            Assert.Equal(new[] { "more" }, again.Phrases.Pivots());

            // a Stage 2 file (version 2, no Phrases) still loads
            var v2 = baby.ToState() with { Version = 2, Phrases = null };
            File.WriteAllText(path, System.Text.Json.JsonSerializer.Serialize(v2).Replace(",\"Phrases\":null", ""));
            Assert.Empty(BabyState.Load(path).Phrases.Pivots());
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Teacher_PhraseSession()
    {
        string path = Path.Combine(Path.GetTempPath(), $"babymind-test-{Guid.NewGuid():N}.json");
        var input = new StringReader("show ball\nsay more ball\nfeed more milk\nautofeed 2\nwait 40\ndebug\nquit\n");
        var output = new StringWriter();
        new Teacher(new Baby(selector: new ActionSelector(new Random(3))), new World(5), path, input, output).Run();
        try
        {
            string text = output.ToString();
            Assert.Contains("pivots: more _", text);
            Assert.Contains("Baby says \"more milk\"", text);
        }
        finally { File.Delete(path); }
    }
}
