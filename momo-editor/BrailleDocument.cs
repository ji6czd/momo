namespace MomoEditor;

enum PageNumberStyle { Standard, Alternative }

record PageBreakInfo(
    string? HeaderOverride = null,
    int? NumberStart = null,
    PageNumberStyle? NumberStyle = null);

record PhysicalLine(string Content, bool LogicalEnd, PageBreakInfo? PageBreak = null);

class BrailleDocument
{
    public List<List<PhysicalLine>> Paragraphs { get; } = [];
    public FormatterConfig Config { get; set; } = new();

    public static BrailleDocument Empty() => new();

    public string GetLogicalText(int index) =>
        string.Concat(Paragraphs[index].Select(p => p.Content));

    public void SetParagraphs(IEnumerable<string> texts)
    {
        Paragraphs.Clear();
        foreach (var t in texts)
            Paragraphs.Add([new PhysicalLine(t, true)]);
    }
}
