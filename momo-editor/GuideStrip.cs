using System.Drawing.Text;

namespace MomoEditor;

/// <summary>
/// 現在のカーソル行の点字セルと、その下に並べた読み（逆点訳）を描画するガイド帯。
///
/// 点字をセル幅で等間隔に並べ、各セグメントの読みを対応するセル範囲の真下に中央寄せで描く。
/// 1 物理行だけを対象にするため、折返しや横スクロールの同期問題が起きない。
/// 視覚的な学習補助であり、編集面（RichTextBox）の内容には影響しない。
/// </summary>
sealed class GuideStrip : Control
{
    string _braille = "";
    IReadOnlyList<MomoFfi.GuideSegment> _segments = [];
    int _cursorCell = -1;

    const int TopPad = 3;
    const int RowGap = 2;
    const int BotPad = 3;
    const int LeftPad = 4;

    // セル単位の等幅配置のため、パディングを含まない測定・描画フォーマットを使う。
    static readonly StringFormat CellFormat = new(StringFormat.GenericTypographic)
    {
        FormatFlags = StringFormatFlags.MeasureTrailingSpaces | StringFormatFlags.NoClip,
    };

    public GuideStrip()
    {
        DoubleBuffered = true;
        ResizeRedraw = true;
        TabStop = false;
        SetStyle(ControlStyles.Selectable, false);
        Dock = DockStyle.Bottom;
        BackColor = SystemColors.Window;
        ForeColor = SystemColors.WindowText;
        AccessibleRole = AccessibleRole.StaticText;
        // JAWS スクリプト等が UIA ツリーから名前で特定しやすいよう英語名にする。
        AccessibleName = "ReadingGuide";
    }

    /// <summary>表示する点字行・読みセグメント・カーソルセル位置を設定して再描画する。</summary>
    public void SetData(string braille, IReadOnlyList<MomoFfi.GuideSegment> segments, int cursorCell)
    {
        _braille = braille ?? "";
        _segments = segments ?? [];
        _cursorCell = cursorCell;
        // スクリーンリーダ向けに、行全体の読みを説明テキストへ反映する。
        AccessibleDescription = string.Concat(_segments.Select(s => s.Text));
        Invalidate();
    }

    // セル1個分の幅（等幅フォントの空点字セルで実測）。
    float CellWidth(Graphics g) => g.MeasureString("⠀", Font, PointF.Empty, CellFormat).Width;

    protected override void OnFontChanged(EventArgs e)
    {
        base.OnFontChanged(e);
        // 点字行 + 読み行 + 余白がちょうど収まる高さに固定する（DPI でフォントが
        // スケールすれば高さも追従する）。
        Height = TopPad + Font.Height + RowGap + Font.Height + BotPad;
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        var g = e.Graphics;
        g.Clear(BackColor);

        // 編集面との境界に上端の区切り線を引く。
        using (var pen = new Pen(SystemColors.ControlDark))
            g.DrawLine(pen, 0, 0, Width, 0);

        if (_braille.Length == 0) return;

        g.TextRenderingHint = TextRenderingHint.ClearTypeGridFit;
        float cw = CellWidth(g);
        float yBraille = TopPad;
        float yReading = TopPad + Font.Height + RowGap;

        // カーソルセルの強調（薄いハイライト矩形）。
        if (_cursorCell >= 0)
        {
            using var hl = new SolidBrush(Color.FromArgb(48, SystemColors.Highlight));
            g.FillRectangle(hl, LeftPad + _cursorCell * cw, yBraille, cw, Font.Height);
        }

        // 点字セルを 1 個ずつセル幅で等間隔に配置する。
        using (var fg = new SolidBrush(ForeColor))
            for (int i = 0; i < _braille.Length; i++)
                g.DrawString(_braille[i].ToString(), Font, fg, LeftPad + i * cw, yBraille, CellFormat);

        // 各セグメントの読みを、対応するセル範囲の下に中央寄せで描く。
        using (var rb = new SolidBrush(SystemColors.HotTrack))
            foreach (var seg in _segments)
            {
                if (seg.Text.Length == 0) continue;
                float x0 = LeftPad + seg.CellStart * cw;
                float w = Math.Max(cw, (seg.CellEnd - seg.CellStart) * cw);
                float tw = g.MeasureString(seg.Text, Font, PointF.Empty, CellFormat).Width;
                float tx = x0 + Math.Max(0, (w - tw) / 2);
                g.DrawString(seg.Text, Font, rb, tx, yReading, CellFormat);
            }
    }
}
