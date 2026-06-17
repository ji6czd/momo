using System.Text;

namespace MomoEditor;

/// <summary>
/// フォーマット済みドキュメントの平坦なビュー（物理行のリスト）。
/// 物理行インデックス ⟷ (段落インデックス, 文字オフセット) の双方向マッピングを提供する。
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
    /// 1論理行（段落）の物理行を差し替える。
    /// 他の段落の物理行とヘッダ行はそのまま保持する。
    /// </summary>
    public void UpdateParagraph(int paragraphIndex, MomoFfi.PhysicalLinesHandle handle)
    {
        // 該当段落のコンテンツ行を個別に削除（ヘッダ行が途中に挟まっていてもよい）。
        // 後ろから削除して先行インデックスを保ち、最小インデックスを挿入位置とする。
        int insertIdx = -1;
        for (int i = _lines.Count - 1; i >= 0; i--)
        {
            if (!_lines[i].IsHeader && _lines[i].ParagraphIndex == paragraphIndex)
            {
                insertIdx = i;
                _lines.RemoveAt(i);
            }
        }

        if (insertIdx < 0)
        {
            // 段落が空で物理行が無かった → 直前の段落の後ろに挿入
            // ヘッダ行(ParagraphIndex=-1)はコンテンツの前に来るため、
            // コンテンツ行のみを基準に挿入位置を決める
            insertIdx = 0;
            for (int i = 0; i < _lines.Count; i++)
            {
                var l = _lines[i];
                if (!l.IsHeader && l.ParagraphIndex < paragraphIndex)
                    insertIdx = i + 1;
            }
        }

        int charOffset = 0;
        int count = handle.Count;
        if (count == 0)
        {
            // 空段落はフォーマッタが行を返さないが、カーソル位置確保のため仮想空行を挿入する
            _lines.Insert(insertIdx, new FlatLine("", false, paragraphIndex, 0, true));
        }
        else
        {
            for (int i = 0; i < count; i++)
            {
                string content = handle.GetLine(i);
                bool isLogicalEnd = handle.IsLogicalEnd(i);
                _lines.Insert(insertIdx + i, new FlatLine(content, false, paragraphIndex, charOffset, isLogicalEnd));
                charOffset += content.Length;
            }
        }
    }

    /// <summary>
    /// fromIndex 以降の段落インデックスを delta だけずらす。
    /// 段落の挿入（+1）・削除（-1）時にインデックスを整合させる。
    /// </summary>
    public void ShiftParagraphIndices(int fromIndex, int delta)
    {
        for (int i = 0; i < _lines.Count; i++)
        {
            var l = _lines[i];
            if (!l.IsHeader && l.ParagraphIndex >= fromIndex)
                _lines[i] = l with { ParagraphIndex = l.ParagraphIndex + delta };
        }
    }

    /// <summary>
    /// 指定段落の物理行を削除し、それより後ろのインデックスを -1 ずらす。
    /// </summary>
    public void DeleteParagraph(int paragraphIndex)
    {
        _lines.RemoveAll(l => !l.IsHeader && l.ParagraphIndex == paragraphIndex);
        ShiftParagraphIndices(paragraphIndex + 1, -1);
    }

    /// <summary>
    /// フォーマッタが空ドキュメントを返したときに使う初期ビュー。
    /// PageHeader が有効なら1ページ目のヘッダ行、続けて段落0の仮想空行を持つ。
    /// </summary>
    public static FormattedDocumentView CreateEmpty(FormatterConfig config)
    {
        var view = new FormattedDocumentView();
        if (config.PageHeader)
            view._lines.Add(new FlatLine(MakePageHeader(config, 1), true, -1, 0, true));
        view._lines.Add(new FlatLine("", false, 0, 0, true));
        return view;
    }

    /// <summary>
    /// ヘッダ行を除いた全物理行を lines_per_page に従って再区切りし、
    /// ヘッダ行を再挿入する。編集で物理行数が変わるたびに呼ぶ。
    /// </summary>
    public void RecomputeHeaders(FormatterConfig config)
    {
        _lines.RemoveAll(l => l.IsHeader);
        if (!config.PageHeader) return;

        // ヘッダが1行を占めるため、コンテンツは (LinesPerPage - 1) 行/ページ
        int contentPerPage = Math.Max(1, config.LinesPerPage - 1);
        int page = 1;
        int contentOnPage = 0;

        _lines.Insert(0, new FlatLine(MakePageHeader(config, page), true, -1, 0, true));

        int i = 1;
        while (i < _lines.Count)
        {
            if (_lines[i].IsHeader) { i++; continue; }

            contentOnPage++;
            if (contentOnPage >= contentPerPage)
            {
                page++;
                contentOnPage = 0;
                _lines.Insert(i + 1, new FlatLine(MakePageHeader(config, page), true, -1, 0, true));
                i += 2;
            }
            else
            {
                i++;
            }
        }
    }

    private static string MakePageHeader(FormatterConfig config, int page)
    {
        string pageNumBrl = BraillePageNum(page, config.NumberStyle);
        int pageCells = pageNumBrl.Length;
        int pageStart = Math.Max(0, config.LineWidth - Math.Max(pageCells, 6));
        string title = config.Title ?? "";
        string titlePart = new string(title.Take(pageStart).ToArray());
        return titlePart + new string('⠀', pageStart - titlePart.Length) + pageNumBrl;
    }

    private static string BraillePageNum(int n, PageNumberStyle style)
    {
        char[] digits = style == PageNumberStyle.Standard
            ? ['⠚', '⠁', '⠃', '⠉', '⠙', '⠑', '⠋', '⠛', '⠓', '⠊']
            : ['⠴', '⠂', '⠆', '⠒', '⠲', '⠢', '⠖', '⠶', '⠦', '⠔'];
        var sb = new StringBuilder("⠼");
        foreach (char c in n.ToString())
            sb.Append(digits[c - '0']);
        return sb.ToString();
    }

    /// <summary>
    /// 保存済みの物理行（論理行ごとの PhysicalLine リスト）から直接ビューを組む。
    /// 再ワードラップを行わないため、ユーザーが置いた強制物理改行を保持する。
    /// ページヘッダは RecomputeHeaders で付与する（Rust フォーマッタは使わない）。
    /// </summary>
    public static FormattedDocumentView BuildFromDocument(BrailleDocument doc)
    {
        var view = new FormattedDocumentView();

        for (int p = 0; p < doc.Paragraphs.Count; p++)
        {
            var para = doc.Paragraphs[p];
            if (para.Count == 0)
            {
                // 空論理行 → 空物理行1行
                view._lines.Add(new FlatLine("", false, p, 0, true));
                continue;
            }

            int charOffset = 0;
            for (int i = 0; i < para.Count; i++)
            {
                string content = para[i].Content;
                bool isLogicalEnd = i == para.Count - 1;
                view._lines.Add(new FlatLine(content, false, p, charOffset, isLogicalEnd));
                charOffset += content.Length;
            }
        }

        view.RecomputeHeaders(doc.Config);
        return view;
    }

    public static FormattedDocumentView Build(MomoFfi.DocumentHandle handle)
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
