namespace Momo;

/// <summary>
/// 点字ドキュメントのエントリ（テキスト段落 or 改ページ）の基底クラス。
/// テキストの結合・分割ロジックは <see cref="TextSegment"/> だけを扱えばよく、
/// 改ページ（<see cref="PageBreakEntry"/>）に誤って触れることが構造的にあり得ない。
/// </summary>
public abstract class DocEntry
{
}

/// <summary>
/// 点字ドキュメントの**セグメント**（強制改行・段落で区切られる単位）。
/// 折返し前のテキストと段落終端フラグを保持する。
/// </summary>
public sealed class TextSegment : DocEntry
{
    /// <summary>折返し前のテキスト。</summary>
    public string Text;
    /// <summary>この区切りが段落終端（true）か、段落内の強制改行（false）か。</summary>
    public bool ParagraphEnd;

    public TextSegment(string text, bool paragraphEnd)
    {
        Text = text;
        ParagraphEnd = paragraphEnd;
    }
}

/// <summary>
/// 改ページそのもの（上書き情報込み）を表す独立したエントリ。テキストを持たない。
/// どの <see cref="TextSegment"/> にも属さないため、通常のテキスト編集（結合・分割）で
/// 巻き込まれたり失われたりすることが構造的にない。
/// </summary>
public sealed class PageBreakEntry : DocEntry
{
    /// <summary>改ページマーカー（"====" 等、上書き情報込みの文字列）。</summary>
    public string Marker;

    public PageBreakEntry(string marker = "====")
    {
        Marker = marker;
    }
}

/// <summary>
/// 点字ドキュメントの編集モデル。エントリ（テキスト段落・改ページ）のフラットな
/// リストとして保持する。折返し・ページ分割・ヘッダ生成・各形式の符号化はすべて
/// Rust(momors-braille) に委ねる（<see cref="MomoFfi"/> 経由）。
/// </summary>
public class BrailleDocument
{
    /// <summary>エントリのフラットなリスト。末尾がテキストなら ParagraphEnd=true。</summary>
    public List<DocEntry> Entries { get; } = [];
    public FormatterConfig Config { get; set; } = new();

    public static BrailleDocument Empty() => new();

    /// <summary>空の単一セグメント（カーソル位置確保用）を持つ新規ドキュメント。</summary>
    public static BrailleDocument NewEmpty()
    {
        var doc = new BrailleDocument();
        doc.Entries.Add(new TextSegment("", true));
        return doc;
    }
}
