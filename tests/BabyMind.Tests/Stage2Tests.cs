using BabyMind;
using Xunit;

namespace BabyMind.Tests;

/// <summary>Stage 2 behaviour: needs, requests, feeding.</summary>
public class Stage2Tests
{
    private static string Wait(Baby baby, World world, int ticks, bool autofeed, ref int feeds)
    {
        string last = "";
        for (int i = 0; i < ticks; i++)
        {
            var r = baby.Act(world.Sense());
            if (autofeed && r.Said == "milk" && baby.Drives.Hunger >= 0.3)
            {
                var b = World.Props["bottle"];
                world.TryPlace(b.Color, b.Shape, b.Size, out _);
                baby.Feed(world.Sense(), "milk"); world.Clear(); feeds++;
            }
            last = r.Action.Key;
        }
        return last;
    }

    private static Baby Taught(World world)
    {
        var baby = new Baby(selector: new ActionSelector(new Random(3)));
        var b = World.Props["bottle"];
        world.TryPlace(b.Color, b.Shape, b.Size, out _);
        baby.Feed(world.Sense(), "milk");               // parent: bottle + the word "milk"
        world.Clear();
        return baby;
    }

    [Fact]
    public void Hunger_RisesOverTime_AndFeedingGivesRewardAndRelief()
    {
        var baby = new Baby();
        for (int i = 0; i < 10; i++) baby.Act(null);
        double h = baby.Drives.Hunger;
        Assert.True(h > 0.3);
        var (_, relief) = baby.Feed(null, "milk");
        Assert.True(relief > 0.3);
        Assert.True(baby.Drives.Hunger < h);
    }

    [Fact]
    public void FeedingWithTheWord_TeachesVocabulary()
    {
        var baby = Taught(new World(1));
        Assert.Contains("milk", baby.Vocabulary());
    }

    [Fact]
    public void BabyLearnsToRequestMilkWhenHungry_ByReward()
    {
        var world = new World(2);
        var baby = Taught(world);
        int feeds = 0;
        Wait(baby, world, 150, autofeed: true, ref feeds);

        Assert.True(feeds >= 5, $"parent should have fed the baby repeatedly (fed {feeds}x)");
        // The learned policy: when hungry-ish, saying milk is the best known action.
        double req = baby.Selector.Value("peckish,none", new BabyAction(ActionKind.Say, "milk"));
        double quiet = baby.Selector.Value("peckish,none", new BabyAction(ActionKind.Quiet));
        Assert.True(req > quiet + 0.1, $"say milk ({req:0.00}) should beat quiet ({quiet:0.00}) when peckish");
    }

    [Fact]
    public void WhenSated_BabyLearnsToStayQuiet()
    {
        var world = new World(2);
        var baby = Taught(world);
        int feeds = 0;
        Wait(baby, world, 150, autofeed: true, ref feeds);
        double sayMilk = baby.Selector.Value("sated,none", new BabyAction(ActionKind.Say, "milk"));
        double quiet = baby.Selector.Value("sated,none", new BabyAction(ActionKind.Quiet));
        Assert.True(quiet > sayMilk);
    }

    [Fact]
    public void LearnedBehaviour_SurvivesSaveLoad()
    {
        var world = new World(2);
        var baby = Taught(world);
        int feeds = 0;
        Wait(baby, world, 100, autofeed: true, ref feeds);
        string path = Path.Combine(Path.GetTempPath(), $"babymind-test-{Guid.NewGuid():N}.json");
        try
        {
            BabyState.Save(baby, path);
            var again = BabyState.Load(path);
            var milk = new BabyAction(ActionKind.Say, "milk");
            Assert.Equal(baby.Selector.Value("peckish,none", milk), again.Selector.Value("peckish,none", milk));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Stage1SaveFile_StillLoads()
    {
        // A Stage 1 (version 1) file has no "Selector" property at all.
        var world = new World(1);
        world.TryPlace("red", "round", "small", out _);
        var baby = new Baby();
        baby.Experience(world.Sense(), "ball");
        var v1 = baby.ToState() with { Version = 1, Selector = null };
        string json = System.Text.Json.JsonSerializer.Serialize(v1).Replace(",\"Selector\":null", "");
        Assert.DoesNotContain("Selector", json);
        string path = Path.Combine(Path.GetTempPath(), $"babymind-test-{Guid.NewGuid():N}.json");
        File.WriteAllText(path, json);
        try
        {
            var loaded = BabyState.Load(path);
            Assert.Equal("ball", loaded.Experience(world.Sense(), null).Said);   // old knowledge intact
            Assert.NotNull(loaded.Selector);                                      // fresh selector created
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Praise_CreditsSpontaneousSpeech()
    {
        var baby = Taught(new World(1));
        BabyAction a;
        do { a = baby.Act(null).Action; if (a.Kind != ActionKind.Say) baby.Act(null); } while (a.Kind != ActionKind.Say);
        Assert.True(baby.Praise());
    }

    [Fact]
    public void Teacher_WaitAndFeed_Session()
    {
        string path = Path.Combine(Path.GetTempPath(), $"babymind-test-{Guid.NewGuid():N}.json");
        var input = new StringReader("feed\nautofeed on\nwait 60\ndebug\nquit\n");
        var output = new StringWriter();
        new Teacher(new Baby(selector: new ActionSelector(new Random(3))), new World(5), path, input, output).Run();
        try
        {
            string text = output.ToString();
            Assert.Contains("*drinks*", text);
            Assert.Contains("Parent hears the request", text);
            Assert.Contains("Learned behaviour", text);
        }
        finally { File.Delete(path); }
    }
}
