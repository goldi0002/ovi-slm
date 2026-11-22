using BabyMind;
using Xunit;

namespace BabyMind.Tests;

public class ActionSelectorTests
{
    private static ActionSelector Sel(double eps = 0) => new(new Random(1), epsilon: eps, minEpsilon: eps);
    private static readonly string[] Vocab = { "milk", "ball" };

    [Fact]
    public void RewardedAction_BecomesPreferredInThatState()
    {
        var milk = new BabyAction(ActionKind.Say, "milk");
        var t = Sel();
        for (int i = 0; i < 5; i++) { ForceChoose(t, "hungry,none", milk); t.Reward(0.7); }
        Assert.True(t.Value("hungry,none", milk) > 0.5);
        Assert.Equal(milk, t.Choose("hungry,none", Vocab, false));
    }

    // Drive the selector until it picks the wanted action (epsilon 0 + untried priors make this finite).
    private static void ForceChoose(ActionSelector s, string state, BabyAction want)
    {
        for (int i = 0; i < 100; i++)
        {
            var a = s.Choose(state, Vocab, false);
            if (a == want) return;
            s.Reward(-0.5);                              // punish others so the wanted one eventually wins
        }
        throw new Exception("could not elicit action");
    }

    [Fact]
    public void UnrewardedSpeech_CostsEffort_SoQuietWins()
    {
        var s = Sel();
        for (int i = 0; i < 40; i++) { s.Choose("sated,none", Vocab, false); s.Settle(); }
        Assert.Equal(ActionKind.Quiet, s.Choose("sated,none", Vocab, false).Kind);
    }

    [Fact]
    public void Learning_IsLocal_OneEntryPerAction()
    {
        var s = Sel();
        s.Choose("hungry,none", Vocab, false);
        s.Reward(1.0);
        Assert.Single(s.Table());
    }

    [Fact]
    public void Reward_WithoutPendingAction_ReturnsFalse() => Assert.False(Sel().Reward(1));

    [Fact]
    public void Epsilon_DecaysToFloor()
    {
        var s = new ActionSelector(new Random(1), epsilon: 0.3, minEpsilon: 0.05, epsilonDecay: 0.5);
        for (int i = 0; i < 20; i++) s.Choose("x", Vocab, false);
        Assert.Equal(0.05, s.Epsilon, 6);
    }

    [Fact]
    public void State_RoundTrips()
    {
        var s = Sel();
        s.Choose("hungry,none", Vocab, false); s.Reward(0.7);
        var copy = ActionSelector.FromState(s.ToState());
        Assert.Equal(s.Table().ToList(), copy.Table().ToList());
    }
}
