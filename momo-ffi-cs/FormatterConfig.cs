namespace Momo;

/// <summary>ページ番号のスタイル（Rust 側 PageNumberStyle と一致させる）。</summary>
public enum PageNumberStyle { Standard = 0, Alternative = 1 }

public record FormatterConfig
{
    public int LineWidth { get; init; } = 32;
    public int LinesPerPage { get; init; } = 22;
    public bool PageHeader { get; init; } = true;
    public int NumberStart { get; init; } = 1;
    public PageNumberStyle NumberStyle { get; init; } = PageNumberStyle.Standard;
    public string? Title { get; init; }
}
