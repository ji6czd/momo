using Momo;

namespace MomoEditor;

/// <summary>
/// フォーマット済みドキュメントの平坦なビュー（表示行のリスト）。
/// Rust が描画した印刷イメージ（<see cref="MomoFfi.FormattedHandle"/>）から組み立て、
/// 表示行インデックス ⟷ (セグメントインデックス, セグメント内オフセット) の双方向
/// マッピングを提供する。
///
/// 折返し・ページ分割・ページヘッダ生成・ページ番号はすべて Rust 側で行われる。
/// </summary>
class FormattedDocumentView
{
    record FlatLine(string Content, bool IsHeader, int SegmentIndex, int CharOffsetStart, bool IsLogicalEnd);

    readonly List<FlatLine> _lines = [];

    public int PhysicalLineCount => _lines.Count;

    public string ContentAt(int flatLine) => _lines[flatLine].Content;
    public bool IsHeaderAt(int flatLine) => _lines[flatLine].IsHeader;
    public int SegmentIndexAt(int flatLine) => _lines[flatLine].SegmentIndex;

    /// <summary>表示行位置 → (セグメントインデックス, セグメント内文字オフセット)</summary>
    public (int segmentIndex, int charOffset) PhysicalToLogical(int flatLine, int cellInLine)
    {
        if (flatLine < 0 || flatLine >= _lines.Count) return (0, 0);
        var line = _lines[flatLine];
        if (line.IsHeader) return (0, 0);
        return (line.SegmentIndex, line.CharOffsetStart + Math.Max(0, cellInLine));
    }

    /// <summary>(セグメントインデックス, セグメント内文字オフセット) → 表示行位置</summary>
    public (int flatLine, int cellInLine) LogicalToPhysical(int segmentIndex, int charOffset)
    {
        int best = -1;
        for (int i = 0; i < _lines.Count; i++)
        {
            var l = _lines[i];
            if (l.IsHeader || l.SegmentIndex != segmentIndex) continue;
            if (l.CharOffsetStart <= charOffset) best = i;
        }
        if (best < 0) return (0, 0);
        var line = _lines[best];
        int cell = Math.Min(charOffset - line.CharOffsetStart, line.Content.Length);
        return (best, cell);
    }

    /// <summary>最初の非ヘッダ表示行の位置（無ければ 0）。新規時のカーソル初期位置に使う。</summary>
    public int FirstEditableLine()
    {
        for (int i = 0; i < _lines.Count; i++)
            if (!_lines[i].IsHeader) return i;
        return 0;
    }

    /// <summary>フォーマッタが空ドキュメントを返したときのフォールバック（空行1行）。</summary>
    public static FormattedDocumentView CreateEmpty(FormatterConfig config)
    {
        var view = new FormattedDocumentView();
        view._lines.Add(new FlatLine("", false, 0, 0, true));
        return view;
    }

    /// <summary>
    /// Rust が描画した印刷イメージ（ページ×表示行）から平坦なビューを組み立てる。
    /// 各行の元セグメント番号を Rust から受け取り、セグメント内オフセットを行内で復元する。
    /// </summary>
    public static FormattedDocumentView Build(MomoFfi.FormattedHandle handle)
    {
        var view = new FormattedDocumentView();
        int curSegment = -1;
        int charOffset = 0;

        for (int p = 0; p < handle.PageCount; p++)
        {
            int lineCount = handle.LineCount(p);
            for (int l = 0; l < lineCount; l++)
            {
                bool isHeader = handle.IsHeader(p, l);
                string content = handle.GetLine(p, l);
                bool isLogicalEnd = handle.IsLogicalEnd(p, l);
                int segIdx = handle.SegmentIndex(p, l);

                if (isHeader)
                {
                    view._lines.Add(new FlatLine(content, true, -1, 0, isLogicalEnd));
                    continue;
                }

                // 同じセグメントが折返しで複数行に分かれる間はオフセットを累積し、
                // セグメントが変わったらリセットする。
                if (segIdx != curSegment)
                {
                    curSegment = segIdx;
                    charOffset = 0;
                }
                view._lines.Add(new FlatLine(content, false, segIdx, charOffset, isLogicalEnd));
                charOffset += content.Length;
            }
        }
        return view;
    }
}
