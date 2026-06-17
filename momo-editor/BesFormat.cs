namespace MomoEditor;

/// <summary>
/// BES バイナリ点字ファイル (.bes) の読み込み。
///
/// バイト列のデコードは Rust（momors-ffi の momo_bes_read）が担い、
/// ここではその結果（物理行 + logical_end + page_break フラグ）を
/// <see cref="BrailleDocument"/> の論理行リストへ組み上げる。
///
/// 改ページの強制/暗黙判定とヘッダ行の除去は Rust 側で済んでいる。
/// BES はヘッダオーバーライドや番号スタイルを持たないため、
/// 強制改ページには既定の <see cref="PageBreakInfo"/> を付ける。
/// </summary>
static class BesFormat
{
    /// <summary>BES バイト列を読み込む。DLL 不在・破損時は null。</summary>
    public static BrailleDocument? Read(byte[] bytes)
    {
        using var handle = MomoFfi.ReadBes(bytes);
        if (handle == null) return null;

        var doc = new BrailleDocument
        {
            Config = new FormatterConfig
            {
                LineWidth = handle.LineWidth,
                LinesPerPage = handle.LinesPerPage,
                PageHeader = true,
                // タイトルは読み込み時に落としている（点字→数字の逆変換が入るまで復元しない）
            },
        };

        var current = new List<PhysicalLine>();
        for (int i = 0; i < handle.Count; i++)
        {
            bool logicalEnd = handle.IsLogicalEnd(i);
            var pageBreak = handle.HasPageBreak(i) ? new PageBreakInfo() : null;
            current.Add(new PhysicalLine(handle.GetLine(i), logicalEnd, pageBreak));

            if (logicalEnd)
            {
                doc.Paragraphs.Add(current);
                current = [];
            }
        }
        // logical_end で終わらない余りがあれば1論理行として確定
        if (current.Count > 0)
            doc.Paragraphs.Add(current);

        return doc;
    }
}
