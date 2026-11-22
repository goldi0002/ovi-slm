using BabyMind;

// Usage: BabyMind [--file path.json] [--fresh] [--seed N]
string path = "babymind.json";
bool fresh = false;
int seed = Environment.TickCount;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--file" when i + 1 < args.Length: path = args[++i]; break;
        case "--fresh": fresh = true; break;
        case "--seed" when i + 1 < args.Length: seed = int.Parse(args[++i]); break;
        default: Console.WriteLine("usage: BabyMind [--file path.json] [--fresh] [--seed N]"); return;
    }
}

// A baby that has been saved before keeps growing from where it left off.
Baby baby;
if (!fresh && File.Exists(path))
{
    try { baby = BabyState.Load(path); }
    catch (Exception e) when (e is InvalidDataException or System.Text.Json.JsonException)
    {
        Console.WriteLine($"Could not read {path} ({e.Message}); starting a new baby.");
        baby = new Baby();
    }
}
else baby = new Baby();

new Teacher(baby, new World(seed), path, Console.In, Console.Out).Run();
