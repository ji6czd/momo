namespace MomoEditor;

/// <summary>
/// フォーマット済みドキュメントの平坦なビュー（物理行のリスト）。
/// Rust が描画した印刷イメージ（<see cref="MomoFfi.FormattedHandle"/>）から組み立て、
/// 物理行インデックス ⟷ (段落インデックス, 文字オフセット) の双方向マッピングを提供する。
///
/// 折返し・ページ分割・ページヘッダ生成・ページ番号はすべて Rust 側で行われる。
/// ここではその結果を表示・カーソル制御のために保持するだけ。
/// </summary>
class FormattedDocumentView
{
    record FlatLine(string Content, bool IsHeader, int ParagraphIndex, int CharOffsetStart, bool IsLogicalEnd);

    readonly List<FlatLine> _lines = [];

    public int PhysicalLineCount => _lines.Count;

    public string ContentAt(int flatLine) => _lines[flatLine].Content;
    public bool IsHeaderAt(int flatLine) => _lines[flatLine].IsHeader;
    public int ParagraphIndexAt(int flatLine) => _lines[flatLine].ParagraphIndex;

    /// <summary>
    /// 物理行位置 → (段落インデックス, 段落内文字オフセット)
    /// </summary>
    public (int paragraphIndex, int charOffset) PhysicalToLogical(int flatLine, int cellInLine)
    {
        if (flatLine < 0 || flatLine >= _lines.Count) return (0, 0);
        var line = _lines[flatLine];
        if (line.IsHeader) return (0, 0);
        return (line.ParagraphIndex, line.CharOffsetStart + Math.Max(0, cellInLine));
    }

    /// <summary>
    /// (段落インデックス, 段落内文字オフセット) → 物理行位置
    /// </summary>
    public (int flatLine, int cellInLine) LogicalToPhysical(int paragraphIndex, int charOffset)
    {
        int best = -1;
        for (int i = 0; i < _lines.Count; i++)
        {
            var l = _lines[i];
            if (l.IsHeader || l.ParagraphIndex != paragraphIndex) continue;
            if (l.CharOffsetStart <= charOffset) best = i;
        }
        if (best < 0) return (0, 0);
        var line = _lines[best];
        int cell = Math.Min(charOffset - line.CharOffsetStart, line.Content.Length);
        return (best, cell);
    }

    /// <summary>
    /// フォーマッタが空ドキュメントを返したときに使う初期ビュー（カーソル位置確保用の空行1行）。
    /// </summary>
    public static FormattedDocumentView CreateEmpty(FormatterConfig config)
    {
        var view = new FormattedDocumentView();
        view._lines.Add(new FlatLine("", false, 0, 0, true));
        return view;
    }

    /// <summary>
    /// Rust が描画した印刷イメージ（ページ×物理行）から平坦なビューを組み立てる。
    /// </summary>
    public static FormattedDocumentView Build(MomoFfi.FormattedHandle handle)
    {
        var view = new FormattedDocumentView();
        int paraIdx = 0;
        int charOffset = 0;

        for (int p = 0; p < handle.PageCount; p++)
        {
            int lineCount = handle.LineCount(p);
            for (int l = 0; l < lineCount; l++)
            {
                bool isHeader = handle.IsHeader(p, l);
                string content = handle.GetLine(p, l);
                bool isLogicalEnd = handle.IsLogicalEnd(p, l);

                view._lines.Add(new FlatLine(
                    content, isHeader,
                    isHeader ? -1 : paraIdx,
                    isHeader ? 0 : charOffset,
                    isLogicalEnd));

                if (!isHeader)
                {
                    charOffset += content.Length;
                    if (isLogicalEnd)
                    {
                        paraIdx++;
                        charOffset = 0;
                    }
                }
            }
        }
        return view;
    }
}
