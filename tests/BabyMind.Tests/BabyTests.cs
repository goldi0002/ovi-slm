using BabyMind;
using Xunit;

namespace BabyMind.Tests;

/// <summary>End-to-end behaviour of the Stage 1 demo: red ball + the word "ball".</summary>
public class BabyTests
{
    private static World Redball(int seed = 3)
    {
        var w = new World(seed);
        Assert.True(w.TryPlace("red", "round", "small", out _));
        return w;
    }

    [Fact]
    public void LearnsNameFromOneExample_AndRecognisesLater()
    {
        var world = Redball();
        var baby = new Baby();

        baby.Experience(world.Sense(), null);              // sees it: no word yet, stays quiet
        var first = baby.Experience(world.Sense(), "ball"); // hears the name once
        Assert.Equal("ball", first.Said);

        baby.Experience(null, null);                       // object leaves
        var later = baby.Experience(world.Sense(), null);  // a new noisy view of the ball
        Assert.False(later.Observation!.Value.IsNew);
        Assert.Equal("ball", later.Said);
        Assert.Single(baby.Map.Concepts);
    }

    [Fact]
    public void DifferentObject_FormsNewConcept_AndIsNotCalledBall()
    {
        var world = Redball();
        var baby = new Baby();
        baby.Experience(world.Sense(), "ball");

        world.TryPlace("blue", "square", "medium", out _);
        var r = baby.Experience(world.Sense(), null);
        Assert.True(r.Observation!.Value.IsNew);
        Assert.Null(r.Said);
        Assert.Equal(2, baby.Map.Concepts.Count);
    }

    [Fact]
    public void WordWithoutObject_EvokesTheConcept()
    {
        var world = Redball();
        var baby = new Baby();
        baby.Experience(world.Sense(), "ball");
        var r = baby.Experience(null, "ball");
        Assert.NotNull(r.Evoked);
        Assert.Equal("red round small", World.Describe(r.Evoked!.Value.Concept.Prototype));
    }

    [Fact]
    public void Repetition_ConsolidatesIntoLastingKnowledge()
    {
        var world = Redball();
        var baby = new Baby();
        for (int i = 0; i < 3; i++) baby.Experience(world.Sense(), "ball");
        Assert.True(baby.Assoc.Find(baby.Map.Concepts[0].Id, "ball")!.Consolidated);
    }

    [Fact]
    public void Correction_ReplacesWrongWord()
    {
        var world = Redball();
        var baby = new Baby();
        baby.Experience(world.Sense(), "cup");            // wrong name learned
        baby.Experience(world.Sense(), null);             // baby says "cup"
        Assert.True(baby.Correct("ball"));
        for (int i = 0; i < 2; i++) baby.Experience(world.Sense(), "ball");
        Assert.Equal("ball", baby.Experience(world.Sense(), null).Said);
    }

    [Fact]
    public void SaveLoad_KeepsKnowledge()
    {
        var world = Redball();
        var baby = new Baby();
        baby.Experience(world.Sense(), "ball");
        string path = Path.Combine(Path.GetTempPath(), $"babymind-test-{Guid.NewGuid():N}.json");
        try
        {
            BabyState.Save(baby, path);
            var reborn = BabyState.Load(path);
            Assert.Equal(baby.Tick, reborn.Tick);
            Assert.Equal("ball", reborn.Experience(world.Sense(), null).Said);
            Assert.Single(reborn.Map.Concepts);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Teacher_ConsoleSession_Works()
    {
        string path = Path.Combine(Path.GetTempPath(), $"babymind-test-{Guid.NewGuid():N}.json");
        var input = new StringReader("show ball\nsay ball\nremove\nshow ball\ndebug\nquit\n");
        var output = new StringWriter();
        new Teacher(new Baby(), new World(5), path, input, output).Run();
        try
        {
            string text = output.ToString();
            Assert.Contains("Baby: \"ball\"", text);
            Assert.Contains("[red round small]", text);
            Assert.True(File.Exists(path));
        }
        finally { File.Delete(path); }
    }
}
