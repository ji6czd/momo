namespace Momo;

/// <summary>
/// 点字ドキュメントのセグメント。折返し前テキストと段落終端フラグを保持する。
/// </summary>
public class Segment
{
    public string Text;
    public bool ParagraphEnd;
    public string? PageBreakMarker;

    public Segment(string text, bool paragraphEnd, string? pageBreakMarker = null)
    {
        Text = text;
        ParagraphEnd = paragraphEnd;
        PageBreakMarker = pageBreakMarker;
    }
}

/// <summary>
/// 点字ドキュメントの編集モデル。セグメントのフラットなリストとして保持する。
/// 折返し・ページ分割・符号化は Rust 側（<see cref="MomoFfi"/> 経由）に委ねる。
/// </summary>
public class BrailleDocument
{
    public List<Segment> Segments { get; } = [];
    public FormatterConfig Config { get; set; } = new();

    public static BrailleDocument Empty() => new();

    public static BrailleDocument NewEmpty()
    {
        var doc = new BrailleDocument();
        doc.Segments.Add(new Segment("", true));
        return doc;
    }
}
