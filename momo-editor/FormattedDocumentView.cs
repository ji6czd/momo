using Momo;

namespace MomoEditor;

/// <summary>
/// フォーマット済みドキュメントの平坦なビュー（表示行のリスト）。
/// Rust が描画した印刷イメージ（<see cref="MomoFfi.FormattedHandle"/>）と正本
/// <see cref="BrailleDocument"/> から組み立て、表示行インデックス ⟷
/// (セグメントインデックス, セグメント内オフセット) の双方向マッピングを提供する。
///
/// 折返し・ページ分割・ページヘッダ生成・ページ番号はすべて Rust 側で行われる。
/// 改ページ（<see cref="PageBreakEntry"/>）は、Rust の出力には現れない
/// （コンテンツを持たない制御信号のため）ので、正本ドキュメントと突き合わせて
/// 専用の保護付き区切り行として挟み込む。
/// </summary>
class FormattedDocumentView
{
    // セグメント通し番号の特殊値。ヘッダ行は -1（既存）、改ページ区切り行は -2。
    // どちらも実在のセグメントには一致しないため、PhysicalToLogical/LogicalToPhysical は
    // 通常のセグメント判定ロジックだけで自然に除外できる。
    const int DividerSegmentIndex = -2;

    // EntryIndex: 区切り行(IsDividerAt)についてのみ、対応する PageBreakEntry の
    // document.Entries 内での実インデックス。それ以外の行では -1（未使用）。
    record FlatLine(string Content, bool IsHeader, int SegmentIndex, int CharOffsetStart, bool IsLogicalEnd, int PageIndex, int EntryIndex = -1);

    readonly List<FlatLine> _lines = [];

    public int PhysicalLineCount => _lines.Count;
    public int PageCount { get; private set; }

    public string ContentAt(int flatLine) => _lines[flatLine].Content;
    public bool IsHeaderAt(int flatLine) => _lines[flatLine].IsHeader;
    public bool IsDividerAt(int flatLine) => _lines[flatLine].SegmentIndex == DividerSegmentIndex;
    public int SegmentIndexAt(int flatLine) => _lines[flatLine].SegmentIndex;
    public int PageIndexAt(int flatLine) => _lines[flatLine].PageIndex;

    /// <summary>区切り行(<see cref="IsDividerAt"/>)が対応する PageBreakEntry の
    /// document.Entries 内での実インデックス。区切り行でなければ -1。</summary>
    public int EntryIndexAt(int flatLine) => _lines[flatLine].EntryIndex;

    /// <summary>指定ページの最初の表示行位置（非ヘッダ・非区切り行優先。無ければ -1）。</summary>
    public int FirstLineOfPage(int page)
    {
        int fallback = -1;
        for (int i = 0; i < _lines.Count; i++)
        {
            if (_lines[i].SegmentIndex == DividerSegmentIndex) continue;
            if (_lines[i].PageIndex != page) continue;
            if (!_lines[i].IsHeader) return i;
            if (fallback < 0) fallback = i;
        }
        return fallback;
    }

    /// <summary>(セグメントインデックス, セグメント内文字オフセット) が属するページ番号（無ければ 0）。</summary>
    public int PageAt(int segmentIndex, int charOffset)
    {
        var (flatLine, _) = LogicalToPhysical(segmentIndex, charOffset);
        return flatLine >= 0 && flatLine < _lines.Count ? _lines[flatLine].PageIndex : 0;
    }

    /// <summary>表示行位置 → (セグメントインデックス, セグメント内文字オフセット)</summary>
    public (int segmentIndex, int charOffset) PhysicalToLogical(int flatLine, int cellInLine)
    {
        if (flatLine < 0 || flatLine >= _lines.Count) return (0, 0);
        var line = _lines[flatLine];
        if (line.IsHeader || line.SegmentIndex == DividerSegmentIndex) return (0, 0);
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

    /// <summary>最初の非ヘッダ・非区切り表示行の位置（無ければ 0）。新規時のカーソル初期位置に使う。</summary>
    public int FirstEditableLine()
    {
        for (int i = 0; i < _lines.Count; i++)
            if (!_lines[i].IsHeader && _lines[i].SegmentIndex != DividerSegmentIndex) return i;
        return 0;
    }

    /// <summary>フォーマッタが空ドキュメントを返したときのフォールバック（空行1行）。</summary>
    public static FormattedDocumentView CreateEmpty(FormatterConfig config)
    {
        var view = new FormattedDocumentView();
        view._lines.Add(new FlatLine("", false, 0, 0, true, 0));
        view.PageCount = 1;
        return view;
    }

    /// <summary>
    /// Rust が描画した印刷イメージ（ページ×表示行）と正本ドキュメントから平坦なビューを組み立てる。
    /// 各行の元セグメント番号を Rust から受け取り、セグメント内オフセットを行内で復元する。
    /// 改ページ（<see cref="PageBreakEntry"/>）はコンテンツを持たない制御信号のため Rust の出力には
    /// 現れない。<paramref name="document"/> の <see cref="BrailleDocument.Entries"/> と突き合わせて、
    /// 各 <see cref="PageBreakEntry"/> の位置に専用の保護付き区切り行を挟み込む。
    /// </summary>
    public static FormattedDocumentView Build(MomoFfi.FormattedHandle handle, BrailleDocument document)
    {
        // 1. Rust の出力をそのまま平坦化する（区切り行はまだ挟み込まない）。
        var content = new List<FlatLine>();
        int curSegment = -1;
        int charOffset = 0;

        for (int p = 0; p < handle.PageCount; p++)
        {
            int lineCount = handle.LineCount(p);
            for (int l = 0; l < lineCount; l++)
            {
                bool isHeader = handle.IsHeader(p, l);
                string lineContent = handle.GetLine(p, l);
                bool isLogicalEnd = handle.IsLogicalEnd(p, l);
                int segIdx = handle.SegmentIndex(p, l);

                if (isHeader)
                {
                    content.Add(new FlatLine(lineContent, true, -1, 0, isLogicalEnd, p));
                    continue;
                }

                if (segIdx != curSegment)
                {
                    curSegment = segIdx;
                    charOffset = 0;
                }
                content.Add(new FlatLine(lineContent, false, segIdx, charOffset, isLogicalEnd, p));
                charOffset += lineContent.Length;
            }
        }

        // 2. document.Entries を順に歩きながら、TextSegment は content から対応する行を
        //    回収し、PageBreakEntry の位置には区切り行を挟み込む。
        //    Rust 側の segment_index は Text エントリの物理行（= C# の TextSegment）ごとに
        //    1 ずつ増分され、Break エントリは増分しない。document.Entries を先頭から数えた
        //    「TextSegment の通し番号」はこれと常に一致する。
        var view = new FormattedDocumentView();
        int ri = 0;
        int si = 0;
        int lastPageIndex = 0;

        void Emit(FlatLine line)
        {
            view._lines.Add(line);
            lastPageIndex = line.PageIndex;
        }

        // 先頭ページのヘッダ（あれば）は最初のエントリより前に来る。
        while (ri < content.Count && content[ri].IsHeader)
            Emit(content[ri++]);

        string dividerContent = new string('─', Math.Max(1, document.Config.LineWidth));

        for (int entryIdx = 0; entryIdx < document.Entries.Count; entryIdx++)
        {
            var entry = document.Entries[entryIdx];
            if (entry is TextSegment)
            {
                while (ri < content.Count)
                {
                    if (content[ri].IsHeader)
                    {
                        // このセグメントの折返し行が暗黙のページ分割をまたぐ場合、間に挟まる
                        // ヘッダはこのセグメントに属する（直後の行がまだ同じ si のときだけ）。
                        // 直後が別セグメントなら、そのヘッダはこのセグメントには属さない
                        // （次の Text/Break エントリの処理に委ねる）。
                        if (ri + 1 < content.Count && content[ri + 1].SegmentIndex == si)
                        {
                            Emit(content[ri++]);
                            continue;
                        }
                        break;
                    }
                    if (content[ri].SegmentIndex != si) break;
                    Emit(content[ri++]);
                }
                si++;
            }
            else
            {
                // Break はコンテンツを持たないため content 側には対応する行が無い。
                // 専用の区切り行を挟み込み、その直後に始まるページのヘッダ（あれば）も続けて回収する。
                view._lines.Add(new FlatLine(dividerContent, false, DividerSegmentIndex, 0, true, lastPageIndex, entryIdx));
                while (ri < content.Count && content[ri].IsHeader)
                    Emit(content[ri++]);
            }
        }
        // 想定外の取りこぼしがあれば安全側で末尾にそのまま流す。
        while (ri < content.Count)
            Emit(content[ri++]);

        view.PageCount = handle.PageCount;
        return view;
    }
}
