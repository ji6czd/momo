namespace MomoEditor;

record FormatterConfig
{
    public int LineWidth { get; init; } = 32;
    public int LinesPerPage { get; init; } = 22;
    public bool PageHeader { get; init; } = true;
    public string? Title { get; init; }
    public PageNumberStyle NumberStyle { get; init; } = PageNumberStyle.Standard;
}
