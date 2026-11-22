using System.Text.Json;

namespace BabyMind;

/// <summary>Everything the baby has learned, as plain data (serialised to JSON).</summary>
public sealed record BabyState(int Version, long Tick, ConceptMapState Map, AssociationsState Associations,
                               MemoryState Memory, DrivesState Drives,
                               ActionSelectorState? Selector = null, PhrasesState? Phrases = null)
{
    public const int CurrentVersion = 3;

    private static readonly JsonSerializerOptions Options = new() { WriteIndented = true };

    /// <summary>Write the baby's state to a JSON file (atomically, via a temp file).</summary>
    public static void Save(Baby baby, string path)
    {
        string tmp = path + ".tmp";
        File.WriteAllText(tmp, JsonSerializer.Serialize(baby.ToState(), Options));
        File.Move(tmp, path, overwrite: true);
    }

    /// <summary>Load a baby from a JSON file.</summary>
    public static Baby Load(string path)
    {
        var state = JsonSerializer.Deserialize<BabyState>(File.ReadAllText(path), Options)
                    ?? throw new InvalidDataException("Empty state file.");
        if (state.Version is < 1 or > CurrentVersion)   // v1 files (Stage 1) have no Selector and still load
            throw new InvalidDataException($"Unsupported state version {state.Version}.");
        return Baby.FromState(state);
    }
}
