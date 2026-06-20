namespace MomoEditor;

/// <summary>
/// 点字ドキュメントの編集モデル。**セグメント**（強制改行・段落で区切られる単位）の
/// フラットなリストとして保持する。折返し・ページ分割・ヘッダ生成・各形式の符号化は
/// すべて Rust(momors-braille) に委ねる（<see cref="MomoFfi"/> 経由）。
/// </summary>
class Segment
{
    /// <summary>折返し前のテキスト。</summary>
    public string Text;
    /// <summary>この区切りが段落終端（true）か、段落内の強制改行（false）か。</summary>
    public bool ParagraphEnd;
    /// <summary>このセグメント直後の改ページマーカー（"====" 等）。null で改ページなし。</summary>
    public string? PageBreakMarker;

    public Segment(string text, bool paragraphEnd, string? pageBreakMarker = null)
    {
        Text = text;
        ParagraphEnd = paragraphEnd;
        PageBreakMarker = pageBreakMarker;
    }
}

class BrailleDocument
{
    /// <summary>セグメントのフラットなリスト。末尾セグメントは必ず ParagraphEnd=true。</summary>
    public List<Segment> Segments { get; } = [];
    public FormatterConfig Config { get; set; } = new();

    public static BrailleDocument Empty() => new();

    /// <summary>空の単一セグメント（カーソル位置確保用）を持つ新規ドキュメント。</summary>
    public static BrailleDocument NewEmpty()
    {
        var doc = new BrailleDocument();
        doc.Segments.Add(new Segment("", true));
        return doc;
    }
}
