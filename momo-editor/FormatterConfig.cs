namespace MomoEditor;

record FormatterConfig
{
    public int LineWidth { get; init; } = 32;
    public int LinesPerPage { get; init; } = 22;
    public bool PageHeader { get; init; } = true;
    public int NumberStart { get; init; } = 1;
    public string? Title { get; init; }
}
