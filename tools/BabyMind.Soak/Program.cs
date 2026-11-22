// Long-term soak test for BabyMind.
// A scripted "parent" teaches, feeds, praises and corrects for thousands of ticks while we measure
// learning quality, need handling, forgetting and resource use. Probes run on a clone, so they never disturb the baby.
//
//   dotnet run -c Release --project tools/BabyMind.Soak -- [--ticks 10000] [--seeds 3] [--window 1000] [--quiet]

using System.Diagnostics;
using System.Text.Json;
using BabyMind;

int ticks = 10000, seeds = 3, window = 1000; bool quiet = false;
for (int i = 0; i < args.Length; i++)
    switch (args[i])
    {
        case "--ticks": ticks = int.Parse(args[++i]); break;
        case "--seeds": seeds = int.Parse(args[++i]); break;
        case "--window": window = int.Parse(args[++i]); break;
        case "--quiet": quiet = true; break;
    }

var results = new List<Summary>();
for (int s = 1; s <= seeds; s++)
    results.Add(new Soak(s, ticks, window, quiet).Run());

Console.WriteLine();
Console.WriteLine("================ SUMMARY ACROSS SEEDS ================");
Console.WriteLine($"{"seed",4} {"naming(seen)",13} {"name(held)",11} {"color(held)",12} {"both(held)",11} {"frog",5} {"wrong",6} {"req%",6} {"stateKB",8} {"conc",5} {"ms/1k",6}");
foreach (var r in results)
    Console.WriteLine($"{r.Seed,4} {r.NameSeen,13:P0} {r.NameHeld,11:P0} {r.ColorHeld,12:P0} {r.BothHeld,11:P0} {(r.Frog > 0 ? "yes" : "no"),5} {r.Wrong,6:P0} {r.RequestShare,6:P0} {r.StateKb,8:0} {r.Concepts,5} {r.MsPer1k,6:0}");

// ------------------------------------------------------------------------------------------------

record Cls(string Name, string Shape, string Size);
record Summary(int Seed, double NameSeen, double NameHeld, double ColorHeld, double BothHeld, double Frog, double Wrong, double RequestShare, double StateKb, int Concepts, double MsPer1k);

sealed class Soak
{
    static readonly Cls[] Classes =
    {
        new("ball", "round", "small"), new("cup", "round", "medium"), new("block", "square", "medium"),
        new("banana", "long", "small"), new("tree", "triangle", "big"),
    };
    static readonly string[] Colors = World.Colors.Keys.ToArray();
    // combinations the parent NEVER shows during teaching: they test true generalisation
    static readonly HashSet<(string, string)> HeldOut = new()
    {
        ("ball", "green"), ("banana", "blue"), ("cup", "purple"), ("block", "orange"), ("tree", "yellow"), ("ball", "purple"),
    };
    const int TreeCutoff = 1500;      // "tree" is taught early, then never again (retention test)
    const int PhaseA = 1500;          // teaching-heavy phase; afterwards mostly waiting/feeding

    readonly int _seed, _ticks, _window; readonly bool _quiet;
    readonly Random _rng; readonly World _world; readonly Baby _baby;
    int _requestFeeds, _forcedFeeds, _satedTicks, _satedSpeech;   // per-window counters
    int _totalRequest, _totalForced;
    readonly List<double> _requestHunger = new();

    public Soak(int seed, int ticks, int window, bool quiet)
    {
        _seed = seed; _ticks = ticks; _window = window; _quiet = quiet;
        _rng = new Random(seed * 7919);
        _world = new World(seed);
        _baby = new Baby(selector: new ActionSelector(new Random(seed)));
    }

    public Summary Run()
    {
        Console.WriteLine($"\n=== seed {_seed}: {_ticks} ticks ===");
        Console.WriteLine($"{"tick",6} | {"seen",5} {"held",5} {"hcol",5} {"both",5} {"wrong",5} {"frog",4} {"tree",4} | {"req/forced",10} {"chat%",5} | {"conc",4} {"links",5} {"pairs",5} {"mods",4} {"Q",4} {"epi",3} | {"KB",5} {"MB",4} {"ms/1k",5}");
        var sw = Stopwatch.StartNew();
        long lastMs = 0; int nextReport = _window; bool frogTaught = false;
        Probe? last = null; double lastKb = 0, lastMsPer1k = 0;

        while (_baby.Tick < _ticks)
        {
            if (!frogTaught && _baby.Tick >= 50) { TeachFrog(); frogTaught = true; }
            if (_rng.NextDouble() < (_baby.Tick < PhaseA ? 0.8 : 0.2)) Teach(); else WaitBlock(10);
            MaybeForceFeed();

            if (_baby.Tick >= nextReport)
            {
                lastMsPer1k = (sw.ElapsedMilliseconds - lastMs) * 1000.0 / _window; lastMs = sw.ElapsedMilliseconds;
                last = MakeProbe();
                lastKb = JsonSerializer.Serialize(_baby.ToState()).Length / 1024.0;
                Report(nextReport, last, lastKb, lastMsPer1k);
                nextReport += _window;
            }
        }

        // ---- end-of-run: persistence must not change behaviour ----
        string path = Path.Combine(Path.GetTempPath(), $"soak-{_seed}-{Guid.NewGuid():N}.json");
        BabyState.Save(_baby, path);
        var reborn = BabyState.Load(path);
        File.Delete(path);
        _misses = new();
        var a = MakeProbe(_baby); var b = MakeProbe(reborn);
        Console.WriteLine($"final probe: items where the name was missing (or, for held-out items, the colour too) ({_misses.Count / 2} of {Classes.Length * Colors.Length}):");
        foreach (var m in _misses.Take(_misses.Count / 2)) Console.WriteLine("   " + m);
        _misses = null;
        Console.WriteLine($"save/load round trip: probe before {a.NameSeen:P0}/{a.NameHeld:P0}, after reload {b.NameSeen:P0}/{b.NameHeld:P0}  {(Math.Abs(a.NameSeen - b.NameSeen) < 0.001 && Math.Abs(a.NameHeld - b.NameHeld) < 0.001 ? "OK" : "MISMATCH")}");
        Console.WriteLine($"requests vs forced feeds over whole run: {_totalRequest} requested, {_totalForced} forced; mean hunger at request {(_requestHunger.Count == 0 ? 0 : _requestHunger.Average()):0.00}");
        if (!_quiet) Console.WriteLine("\n" + _baby.Debug());

        var p = last!;
        double share = _totalRequest + _totalForced == 0 ? 0 : _totalRequest / (double)(_totalRequest + _totalForced);
        return new Summary(_seed, p.NameSeen, p.NameHeld, p.ColorHeld, p.BothHeld, FrogAtEnd(p), p.Wrong, share, lastKb, _baby.Map.Concepts.Count, lastMsPer1k);
    }

    // ---------------- the scripted parent ----------------

    void Teach()
    {
        var (cls, color) = PickCombo();
        _world.TryPlace(color, cls.Shape, cls.Size, out _);
        var seen = _baby.Experience(_world.Sense(), null);                 // show it...
        if (seen.Said is { } said && !IsRight(said, cls, color) && _rng.NextDouble() < 0.6)
            _baby.Correct(cls.Name);                                       // a corrective parent

        double u = _rng.NextDouble();
        string utter = u < 0.5 ? cls.Name : u < 0.9 ? $"{color} {cls.Name}" : color;
        var r = _baby.Experience(_world.Sense(), utter);                   // ...and name it
        if (r.Said is { } s2 && IsRight(s2, cls, color) && _rng.NextDouble() < 0.3) _baby.Praise();
        _world.Clear();
    }

    void TeachFrog()
    {   // ONE exposure of a rarely seen thing, never repeated: does it survive?
        _world.TryPlace("green", "triangle", "small", out _);
        _baby.Experience(_world.Sense(), null);
        _baby.Experience(_world.Sense(), "frog");
        _world.Clear();
    }

    void WaitBlock(int n)
    {
        if (_rng.NextDouble() < 0.3) { var (c, col) = PickCombo(); _world.TryPlace(col, c.Shape, c.Size, out _); }
        for (int i = 0; i < n; i++)
        {
            var r = _baby.Act(_world.Sense());
            double h = _baby.Drives.Hunger;
            if (h < 0.3) { _satedTicks++; if (r.Said is not null) _satedSpeech++; }
            if (r.Said is { } said && said.Split(' ').Contains("milk") && h >= 0.3)
            {
                _requestHunger.Add(h); _requestFeeds++; _totalRequest++;
                Feed("milk");                                               // parent understands the request and says "milk" (never parrots a babbled phrase)
            }
            else MaybeForceFeed();
        }
        _world.Clear();
    }

    // A parent who notices a baby that is very hungry and feeds it anyway (this is also how "milk" is first taught).
    void MaybeForceFeed()
    {
        if (_baby.Drives.Hunger < 0.8) return;
        _forcedFeeds++; _totalForced++;
        Feed("milk");
    }

    void Feed(string word)
    {
        var bottle = World.Props["bottle"];
        var keep = _world.Current;
        _world.TryPlace(bottle.Color, bottle.Shape, bottle.Size, out _);
        _baby.Feed(_world.Sense(), word);
        if (keep is null) _world.Clear(); else _world.TryPlace(keep.Color, keep.Shape, keep.Size, out _);
    }

    (Cls, string) PickCombo()
    {
        while (true)
        {
            // uneven frequencies, like real life: ball > block > banana > cup > tree
            double u = _rng.NextDouble();
            var cls = u < 0.32 ? Classes[0] : u < 0.58 ? Classes[2] : u < 0.80 ? Classes[3] : u < 0.94 ? Classes[1] : Classes[4];
            if (cls.Name == "tree" && _baby.Tick >= TreeCutoff) continue;
            string color = Colors[_rng.Next(Colors.Length)];
            if (HeldOut.Contains((cls.Name, color))) continue;
            return (cls, color);
        }
    }

    static bool IsRight(string said, Cls cls, string color) =>
        said.Split(' ').All(w => w == cls.Name || w == color);

    // ---------------- probes ----------------

    record Probe(double NameSeen, double NameHeld, double ColorHeld, double BothHeld, double Wrong, bool Frog, double Tree);

    Probe MakeProbe() => MakeProbe(_baby);
    List<string>? _misses;

    // Everything happens on a clone with its own world, so measuring never changes what is measured.
    Probe MakeProbe(Baby source)
    {
        var baby = Baby.FromState(source.ToState());
        var w = new World(_seed + 1000);
        int seenN = 0, seenName = 0, heldN = 0, heldName = 0, heldColor = 0, heldBoth = 0, wrong = 0, total = 0;
        foreach (var cls in Classes)
            foreach (var color in Colors)
            {
                if (cls.Name == "tree" && false) continue;
                w.TryPlace(color, cls.Shape, cls.Size, out _);
                var sight = w.Sense()!.Value;
                string? said = baby.Experience(sight, null).Said;
                string fits = string.Join(",", baby.Meanings.Matches(sight.Features).Select(m => $"{m.Word}:{m.Score:0.00}"));
                var words = said?.Split(' ') ?? Array.Empty<string>();
                bool held = HeldOut.Contains((cls.Name, color));
                bool nameOk = words.Contains(cls.Name), colorOk = words.Contains(color);
                if (held) { heldN++; if (nameOk) heldName++; if (colorOk) heldColor++; if (nameOk && colorOk) heldBoth++; }
                else if (cls.Name != "tree") { seenN++; if (nameOk) seenName++; }
                if (!nameOk || (held && !colorOk)) _misses?.Add($"{color} {cls.Name}{(held ? " (held-out)" : "")} -> {(said is null ? "(silent)" : "\"" + said + "\"")}   [meanings fit: {fits}]");
                total++;
                if (words.Any(x => x != cls.Name && x != color)) wrong++;
            }
        w.TryPlace("green", "triangle", "small", out _);
        bool frog = baby.Experience(w.Sense(), null).Said?.Split(' ').Contains("frog") ?? false;
        int treeOk = 0;
        foreach (var color in Colors.Where(c => !HeldOut.Contains(("tree", c))))
        {
            w.TryPlace(color, "triangle", "big", out _);
            string? ts = baby.Experience(w.Sense(), null).Said;
            if (ts?.Split(' ').Contains("tree") ?? false) treeOk++;
            else _misses?.Add($"[tree loop] {color} tree -> {(ts is null ? "(silent)" : "\"" + ts + "\"")}");
        }
        int treeN = Colors.Count(c => !HeldOut.Contains(("tree", c)));
        return new Probe(Rate(seenName, seenN), Rate(heldName, heldN), Rate(heldColor, heldN), Rate(heldBoth, heldN), Rate(wrong, total), frog, Rate(treeOk, treeN));
    }

    double FrogAtEnd(Probe p) => p.Frog ? 1 : 0;

    static double Rate(int a, int b) => b == 0 ? 0 : a / (double)b;

    void Report(int tick, Probe p, double kb, double msPer1k)
    {
        double chat = _satedTicks == 0 ? 0 : 100.0 * _satedSpeech / _satedTicks;
        long mb = GC.GetTotalMemory(true) / (1024 * 1024);
        Console.WriteLine($"{tick,6} | {p.NameSeen,5:P0} {p.NameHeld,5:P0} {p.ColorHeld,5:P0} {p.BothHeld,5:P0} {p.Wrong,5:P0} {(p.Frog ? "yes" : "no"),4} {p.Tree,4:P0} | {_requestFeeds + "/" + _forcedFeeds,10} {chat,5:0} | {_baby.Map.Concepts.Count,4} {_baby.Assoc.LinkCount,5} {_baby.Phrases.Pairs.Count(),5} {_baby.Meanings.Models.Count(),4} {_baby.Selector.Table().Count(),4} {_baby.Memory.Episodes.Count,3} | {kb,5:0} {mb,4} {msPer1k,5:0}");
        _requestFeeds = _forcedFeeds = _satedTicks = _satedSpeech = 0;
    }
}
