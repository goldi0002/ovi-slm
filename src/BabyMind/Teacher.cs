namespace BabyMind;

/// <summary>
/// Console interface where the human plays the parent: show objects, say words, praise or correct.
/// Reads from / writes to injected streams so it can be driven by tests.
/// </summary>
public sealed class Teacher
{
    private Baby _baby;
    private readonly World _world;
    private readonly string _savePath;
    private readonly TextReader _in;
    private readonly TextWriter _out;
    private int _feedMode;   // 0 off, 1 feed on any 'milk' request (or reaching for the bottle), 2 only on two-word requests
    private const string RequestWord = "milk";

    public Teacher(Baby baby, World world, string savePath, TextReader input, TextWriter output)
    {
        _baby = baby; _world = world; _savePath = savePath; _in = input; _out = output;
    }

    /// <summary>The (possibly reloaded) baby — Program saves this one on exit.</summary>
    public Baby Baby => _baby;

    public const string Help = @"
Commands (you are the parent):
  show <prop>                 put a prop in view, e.g. 'show ball'   (see 'props')
  show <color> <shape> [size] put a custom object in view, e.g. 'show red round small'
  say <word>                  speak a word while the object is in view  (this is how it learns names)
  say <word> <word>           a two-word phrase, e.g. 'say more ball'. The last word names the object; word order is learned too
  ask                         look at the object again and see what the baby says
  good                        praise what the baby just said
  no [word]                   correct the baby; optionally give the right word, e.g. 'no cup'
  feed [word [word]]          give the baby a bottle while saying a word or phrase (default 'milk', e.g. 'feed more milk')
  wait [n]                    let n ticks pass; the baby acts on its own (looks, speaks, or stays quiet)
  autofeed on|off|2           'on': you feed the baby if it says 'milk' (alone or in a phrase) or reaches for a bottle in view while
                              hungry. '2': you only respond to two-word requests like 'more milk'. 'off': you decide yourself
  remove                      take the object away
  props                       list props and the colors/shapes/sizes you can use
  debug                       print what the baby knows
  save | load                 write / read the baby's state file
  quit                        save and leave
";

    public void Run()
    {
        _out.WriteLine("BabyMind - Stage 3: actions and two-word phrases.  Type 'help' for commands.");
        _out.WriteLine(_baby.Tick == 0 ? "A brand-new baby opens its eyes." : $"The baby is back (tick {_baby.Tick}, {_baby.Map.Concepts.Count} concepts).");

        while (true)
        {
            _out.Write("parent> ");
            string? line = _in.ReadLine();
            if (line is null) { Save(); return; }                  // EOF behaves like quit
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            if (parts.Length == 0) continue;
            string cmd = parts[0].ToLowerInvariant();
            var args = parts.Skip(1).ToArray();

            switch (cmd)
            {
                case "help": case "?": _out.WriteLine(Help); break;
                case "quit": case "exit": Save(); return;
                case "debug": _out.WriteLine(_baby.Debug()); break;
                case "save": Save(); break;
                case "load": Load(); break;
                case "props": ListProps(); break;
                case "show": Show(args); break;
                case "feed": Feed(args.Length > 0 ? string.Join(' ', args.Take(2)) : RequestWord); break;
                case "wait": Wait(args); break;
                case "autofeed":
                    _feedMode = args.Length == 0 ? 0 : args[0].ToLowerInvariant() switch { "on" => 1, "2" => 2, _ => 0 };
                    _out.WriteLine($"(autofeed {(_feedMode == 0 ? "off" : _feedMode == 1 ? "on" : "on, two-word requests only")})");
                    break;
                case "remove": _world.Clear(); _out.WriteLine("(you take the object away)"); _baby.Experience(null, null); break;
                case "say": Say(args); break;
                case "ask": Respond(_baby.Experience(_world.Sense(), null)); break;
                case "good": _out.WriteLine(_baby.Praise() ? "(the baby beams - that link gets stronger)" : "(the baby hasn't said anything to praise)"); break;
                case "no": Correct(args); break;
                default: _out.WriteLine($"Unknown command '{cmd}'. Type 'help'."); break;
            }
        }
    }

    private void Show(string[] args)
    {
        string error = "";
        bool ok;
        if (args.Length == 1 && World.Props.TryGetValue(args[0].ToLowerInvariant(), out var prop))
            ok = _world.TryPlace(prop.Color, prop.Shape, prop.Size, out error);
        else if (args.Length is 2 or 3)
            ok = _world.TryPlace(args[0].ToLowerInvariant(), args[1].ToLowerInvariant(), args.Length == 3 ? args[2].ToLowerInvariant() : "medium", out error);
        else { _out.WriteLine("usage: show <prop>   or   show <color> <shape> [size]   (try 'props')"); return; }

        if (!ok) { _out.WriteLine("! " + error); return; }
        var o = _world.Current!;
        _out.WriteLine($"(you show a {o.Color} {o.Shape} {o.Size} thing)");
        Respond(_baby.Experience(_world.Sense(), null));
    }

    private void Feed(string word)
    {
        var bottle = World.Props["bottle"];
        _world.TryPlace(bottle.Color, bottle.Shape, bottle.Size, out _);
        double before = _baby.Drives.Hunger;
        _out.WriteLine($"(you hold up a bottle and say \"{word}\")");
        var (r, relief) = _baby.Feed(_world.Sense(), word);
        _world.Clear();
        _out.WriteLine(relief > 0 ? $"  Baby: *drinks* hunger {before:0.00} -> {_baby.Drives.Hunger:0.00}" : "  Baby: *not hungry* (turns away)");
        Respond(r);
    }

    private void Wait(string[] args)
    {
        int n = args.Length > 0 && int.TryParse(args[0], out var k) ? Math.Clamp(k, 1, 500) : 1;
        for (int i = 0; i < n; i++)
        {
            var r = _baby.Act(_world.Sense());
            double h = _baby.Drives.Hunger;
            string doing = r.Action.Kind switch
            {
                ActionKind.Say => $"says \"{r.Said}\"",
                ActionKind.Reach => _world.Current is null ? "reaches for nothing" : "reaches for the object",
                ActionKind.Look => r.Observation!.Value.IsNew ? "looks - something new!" : "looks at the object",
                _ => "stays quiet",
            };
            _out.WriteLine($"  [t{_baby.Tick}] hunger {h:0.00} ({Baby.HungerBucket(h)}): Baby {doing}");

            if (_feedMode > 0 && h >= 0.3 && ParentResponds(r, out string echo))
            {
                _out.WriteLine("    Parent understands the request and feeds the baby.");
                Feed(echo);
            }
        }
    }

    // The parent's rule for autofeed. `echo` is what the parent says back while feeding (parents repeat what toddlers say).
    private bool ParentResponds(ActResult r, out string echo)
    {
        echo = RequestWord;
        if (r.Action.Kind == ActionKind.Reach)
            return _feedMode == 1 && _world.Current == World.Props["bottle"];
        if (r.Said is null) return false;
        var words = r.Said.Split(' ');
        echo = r.Said;
        return words.Contains(RequestWord) && (_feedMode == 1 || words.Length == 2);
    }

    private void Say(string[] args)
    {
        if (args.Length == 0) { _out.WriteLine("usage: say <word>"); return; }
        if (args.Length > 2) _out.WriteLine("(the baby only follows two-word phrases - using the first two words)");
        string word = string.Join(' ', args.Take(2));
        _out.WriteLine($"You: \"{word}\"");
        Respond(_baby.Experience(_world.Sense(), word));
    }

    private void Correct(string[] args)
    {
        string? word = args.Length > 0 ? args[0] : null;
        if (!_baby.Correct(word)) { _out.WriteLine("(the baby hasn't said anything to correct)"); return; }
        _out.WriteLine(word is null ? "(the baby frowns - that link gets weaker)" : $"(the baby frowns, then listens: \"{word}\")");
        if (word is not null) Respond(_baby.Experience(_world.Sense(), null));
    }

    // Narrate the baby's reaction in a human-readable way.
    private void Respond(Response r)
    {
        if (r.Observation is { } o)
        {
            string what = o.IsNew ? $"it is something new! (concept #{o.Concept.Id} born)" : $"it looks familiar (concept #{o.Concept.Id})";
            _out.WriteLine($"  Baby: *looks* {what}");
        }
        else if (r.Evoked is { } ev)
            _out.WriteLine($"  Baby: *thinks of* [{World.Describe(ev.Concept.Prototype)}] (concept #{ev.Concept.Id}, link {ev.Weight:0.00})");
        else if (r.Observation is null)
            _out.WriteLine(r.Heard is null ? "  Baby: *stares around* (nothing in view)" : "  Baby: *listens* (no idea what that word means yet)");

        if (r.NewlyConsolidated) _out.WriteLine("  Baby: (this pattern keeps repeating - it sticks in long-term memory)");
        if (r.Said is not null)
            _out.WriteLine($"  Baby: \"{r.Said}\"   (confidence {r.Confidence:0.00}{(r.Basis is null ? "" : "; generalising: " + r.Basis)})");
        else if (r.Observation is not null) _out.WriteLine("  Baby: ...");
    }

    private void ListProps()
    {
        _out.WriteLine("props:  " + string.Join(", ", World.Props.Select(p => $"{p.Key} = {p.Value.Color} {p.Value.Shape} {p.Value.Size}")));
        _out.WriteLine("colors: " + string.Join(", ", World.Colors.Keys));
        _out.WriteLine("shapes: " + string.Join(", ", World.Shapes));
        _out.WriteLine("sizes:  " + string.Join(", ", World.Sizes.Keys));
        _out.WriteLine("(The baby never sees these names - only the numbers they produce. It only learns the words you say.)");
    }

    private void Save()
    {
        BabyState.Save(_baby, _savePath);
        _out.WriteLine($"(saved to {_savePath})");
    }

    private void Load()
    {
        if (!File.Exists(_savePath)) { _out.WriteLine($"(no save file at {_savePath})"); return; }
        try
        {
            _baby = BabyState.Load(_savePath);
            _out.WriteLine($"(loaded: tick {_baby.Tick}, {_baby.Map.Concepts.Count} concepts)");
        }
        catch (Exception e) when (e is InvalidDataException or System.Text.Json.JsonException)
        {
            _out.WriteLine($"(could not load: {e.Message})");
        }
    }
}
