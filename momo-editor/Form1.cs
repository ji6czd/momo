using System.Reflection;
using System.Text;

namespace MomoEditor;

public partial class Form1 : Form
{
    private string? _filePath;
    private bool _isModified;
    private bool _suppressModified;
    private BrailleDocument _document = BrailleDocument.NewEmpty();
    private FormattedDocumentView? _view;

    // キーリマップテーブル（全モード共通）
    private readonly Dictionary<Keys, Action> _keyMap;

    // 六キー点字入力
    private bool _brailleInputMode = true;
    private readonly HashSet<Keys> _heldBrailleKeys = [];
    private int _chordDotPattern;

    private static readonly Dictionary<Keys, int> BrailleKeyBit = new()
    {
        [Keys.F] = 0, // 点1 (左人差し指)
        [Keys.D] = 1, // 点2 (左中指)
        [Keys.S] = 2, // 点3 (左薬指)
        [Keys.J] = 3, // 点4 (右人差し指)
        [Keys.K] = 4, // 点5 (右中指)
        [Keys.L] = 5, // 点6 (右薬指)
    };

    public Form1()
    {
        InitializeComponent();
        _keyMap = new()
        {
            [Keys.Control | Keys.H] = SimulateBackspace,
            [Keys.Control | Keys.M] = InsertParagraphBreak,
            [Keys.Return] = InsertParagraphBreak,
            [Keys.Shift | Keys.Return] = InsertHardBreak,   // 強制改行（段落内）
            [Keys.Control | Keys.Return] = InsertPageBreak,   // 改ページ
            [Keys.Back] = SimulateBackspace,
            [Keys.Delete] = SimulateDelete,
        };
        // 新規ドキュメントで起動
        LoadDocumentToEditor();
        IsModified = false;
        UpdateTitle();
        UpdateStatus();
        AdjustStartupSize();
    }

    /// <summary>
    /// 起動時のウィンドウ幅を、点字 1 行（LineWidth セル）が
    /// 折り返さずに収まるサイズ以上へ広げる。等幅フォントを実測するため
    /// 高 DPI 環境でも正しくスケールする。
    /// </summary>
    private void AdjustStartupSize()
    {
        int cells = _document.Config.LineWidth;
        // U+2800（点字の空セル）を LineWidth 個並べた幅を実測。
        // NoPadding で両端余白を除き、純粋な文字列幅を得る。
        int textWidth = TextRenderer.MeasureText(
            new string('⠀', cells), richTextBox.Font,
            new Size(int.MaxValue, int.MaxValue), TextFormatFlags.NoPadding).Width;
        // 縦スクロールバーと内部余白の分を加える。
        int needed = textWidth + SystemInformation.VerticalScrollBarWidth + 12;
        if (ClientSize.Width < needed)
            ClientSize = new Size(needed, ClientSize.Height);
        // 1 行分を割り込めないよう最小幅も設定。
        MinimumSize = new Size(needed + (Width - ClientSize.Width), MinimumSize.Height);
    }

    // ---- タイトル・ステータス ----

    private bool IsModified
    {
        get => _isModified;
        set { _isModified = value; UpdateTitle(); }
    }

    private void UpdateTitle()
    {
        var name = _filePath != null ? Path.GetFileName(_filePath) : "無題";
        Text = $"{(IsModified ? "* " : "")}{name} - MomoEditor";
    }

    private void UpdateStatus()
    {
        int pos = richTextBox.SelectionStart;
        int line = richTextBox.GetLineFromCharIndex(pos);
        int lineStart = richTextBox.GetFirstCharIndexFromLine(line);
        int col = pos - lineStart;
        var mode = _brailleInputMode ? "  [点字入力]" : "";
        statusLabel.Text = $"行: {line + 1}  セル: {col}{mode}";
    }

    // ---- 読みガイド ----

    // 直近に逆点訳した点字行とその結果（行内でカーソルだけ動いたときの再計算を避ける）。
    private string? _guideLineCache;
    private IReadOnlyList<MomoFfi.GuideSegment> _guideSegments = [];

    private void GuideMenuItem_Click(object? sender, EventArgs e)
    {
        guideStrip.Visible = guideMenuItem.Checked;
        UpdateGuide();
        richTextBox.Focus();
    }

    /// <summary>
    /// カーソル行の点字を逆点訳し、ガイド帯にセルと読みを表示する。
    /// 範囲選択中・ヘッダ行・フォールバック表示中は何も表示しない。
    /// </summary>
    private void UpdateGuide()
    {
        if (!guideMenuItem.Checked) return;
        // 範囲選択中はキャレット位置が定まらないので更新しない（現状維持）。
        if (richTextBox.SelectionLength != 0) return;

        if (_view == null)
        {
            ClearGuide();
            return;
        }

        int pos = richTextBox.SelectionStart;
        int flatLine = richTextBox.GetLineFromCharIndex(pos);
        if (flatLine < 0 || flatLine >= _view.PhysicalLineCount || _view.IsHeaderAt(flatLine))
        {
            ClearGuide();
            return;
        }

        string content = _view.ContentAt(flatLine);
        int col = pos - richTextBox.GetFirstCharIndexFromLine(flatLine);

        if (content != _guideLineCache)
        {
            var result = MomoFfi.BackTranslateLine(content);
            _guideSegments = result?.Segments ?? [];
            _guideLineCache = content;
        }
        guideStrip.SetData(content, _guideSegments, col);
    }

    private void ClearGuide()
    {
        _guideLineCache = "";
        _guideSegments = [];
        guideStrip.SetData("", [], -1);
    }

    // ---- フォーマット・レンダリング ----

    /// <summary>
    /// _document を Rust で整形して RichTextBox に描画する。
    /// targetSeg/targetOffset: カーソルを論理位置（セグメント+オフセット）に復元する（-1 なら先頭の編集可能行）。
    /// </summary>
    private void ReformatAndRender(int targetSeg, int targetOffset)
    {
        _suppressModified = true;
        try
        {
            using var handle = MomoFfi.RenderDocument(_document);
            if (handle == null)
            {
                // DLL 未ビルド/モデル不在: セグメントを生テキストとして表示
                _view = null;
                richTextBox.Text = LogicalFallbackText();
                richTextBox.SelectionStart = 0;
                richTextBox.SelectionLength = 0;
                return;
            }

            _view = FormattedDocumentView.Build(handle);
            if (_view.PhysicalLineCount == 0)
                _view = FormattedDocumentView.CreateEmpty(_document.Config);

            WriteViewToTextBox(targetSeg, targetOffset);

            // モデル駆動編集では TextChanged が _suppressModified で抑制されるため、
            // ここで変更フラグを立てる。targetSeg < 0 はロード/新規（LoadDocumentToEditor）で、
            // この場合は復元すべき論理カーソルが無く＝編集ではないので変更扱いしない。
            if (targetSeg >= 0) IsModified = true;
        }
        finally
        {
            _suppressModified = false;
        }
        richTextBox.Focus();
    }

    // DLL 不在時のフォールバック表示用。各セグメントを1行ずつ並べる。
    private string LogicalFallbackText() =>
        string.Join("\n", _document.Segments.Select(s => s.Text));

    // _view を RichTextBox に書き出し、行間・ヘッダ保護・カーソル復元を行う。
    // _suppressModified が true の状態で呼ぶこと。target < 0 なら先頭の編集可能行にカーソルを置く。
    private void WriteViewToTextBox(int targetSeg, int targetOffset)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < _view!.PhysicalLineCount; i++)
        {
            if (i > 0) sb.Append('\n');
            sb.Append(_view.ContentAt(i));
        }
        richTextBox.Text = sb.ToString();

        ApplyLineSpacing();
        ApplyHeaderProtection();

        if (targetSeg >= 0)
        {
            RestoreCursor(targetSeg, targetOffset);
        }
        else
        {
            int fl = _view.FirstEditableLine();
            richTextBox.SelectionStart = richTextBox.GetFirstCharIndexFromLine(fl);
            richTextBox.SelectionLength = 0;
        }
    }

    /// <summary>現在のカーソル行がヘッダ行かどうかを返す。</summary>
    private bool IsOnHeaderLine()
    {
        if (_view == null) return false;
        int flatLine = richTextBox.GetLineFromCharIndex(richTextBox.SelectionStart);
        if (flatLine < 0 || flatLine >= _view.PhysicalLineCount) return false;
        return _view.IsHeaderAt(flatLine);
    }

    /// <summary>ヘッダ行を編集不可に設定する。</summary>
    private void ApplyHeaderProtection()
    {
        if (_view == null) return;

        richTextBox.SelectAll();
        richTextBox.SelectionProtected = false;

        for (int i = 0; i < _view.PhysicalLineCount; i++)
        {
            if (!_view.IsHeaderAt(i)) continue;
            int lineStart = richTextBox.GetFirstCharIndexFromLine(i);
            int nextLineStart = i + 1 < _view.PhysicalLineCount
                ? richTextBox.GetFirstCharIndexFromLine(i + 1)
                : richTextBox.TextLength;
            richTextBox.Select(lineStart, nextLineStart - lineStart);
            richTextBox.SelectionProtected = true;
        }
    }

    /// <summary>論理カーソル位置（セグメント+オフセット）に物理カーソルを戻す。</summary>
    private void RestoreCursor(int segIdx, int charOffset)
    {
        if (_view == null) return;
        var (flatLine, cell) = _view.LogicalToPhysical(segIdx, charOffset);
        int lineStart = richTextBox.GetFirstCharIndexFromLine(flatLine);
        richTextBox.SelectionStart = lineStart + cell;
        richTextBox.SelectionLength = 0;
    }

    /// <summary>現在のカーソル位置を論理座標（セグメント+オフセット）で返す。</summary>
    private (int segmentIndex, int charOffset) GetCursor()
    {
        if (_view == null) return (0, 0);
        int pos = richTextBox.SelectionStart;
        int flatLine = richTextBox.GetLineFromCharIndex(pos);
        int lineStart = richTextBox.GetFirstCharIndexFromLine(flatLine);
        return _view.PhysicalToLogical(flatLine, pos - lineStart);
    }

    // ---- ファイル操作 ----

    private bool ConfirmDiscard()
    {
        if (!IsModified) return true;
        var result = MessageBox.Show(
            "変更が保存されていません。保存しますか？",
            "確認",
            MessageBoxButtons.YesNoCancel,
            MessageBoxIcon.Warning);
        if (result == DialogResult.Yes)
        {
            SaveMenuItem_Click(null, EventArgs.Empty);
            return !IsModified;
        }
        return result == DialogResult.No;
    }

    private void NewMenuItem_Click(object? sender, EventArgs e)
    {
        if (!ConfirmDiscard()) return;
        _document = BrailleDocument.NewEmpty();
        LoadDocumentToEditor();
        _filePath = null;
        IsModified = false;
    }

    private void OpenMenuItem_Click(object? sender, EventArgs e)
    {
        if (!ConfirmDiscard()) return;
        using var dialog = new OpenFileDialog
        {
            Filter = "Momo点字ファイル (*.mbr)|*.mbr|テキストファイル (*.txt)|*.txt|すべてのファイル (*.*)|*.*",
        };
        if (dialog.ShowDialog() != DialogResult.OK) return;
        try
        {
            // 取り込み後のファイル名と変更フラグ。点字以外（テキスト）を取り込んだ場合は
            // 拡張子を .mbr に付け替え、未保存（要保存）として扱う。
            string filePath = dialog.FileName;
            bool modified = false;

            if (FormatForPath(dialog.FileName) is int fmt)
            {
                // 点字ファイル（MBR / BES / BET）は Rust の reader で正本ドキュメントへ復元する。
                var doc = MomoFfi.ReadDocument(File.ReadAllBytes(dialog.FileName), fmt);
                if (doc == null)
                {
                    MessageBox.Show("対応していないファイル形式です。", "エラー",
                        MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
                _document = doc;
            }
            else
            {
                // 点字ファイル以外は漢字かな交じり文とみなし、1論理行ずつ点字に変換して取り込む。
                var text = File.ReadAllText(dialog.FileName, Encoding.UTF8);
                var doc = TextToBrailleDocument(text);
                if (doc == null)
                {
                    MessageBox.Show(
                        "点字変換エンジン（モデル）を読み込めないため、テキストを点字に変換できませんでした。",
                        "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
                _document = doc;
                filePath = Path.ChangeExtension(dialog.FileName, ".mbr"); // 編集中のファイル名は .mbr
                modified = true;                                       // .mbr はまだ保存されていない
            }
            if (_document.Segments.Count == 0)
                _document.Segments.Add(new Segment("", true));
            LoadDocumentToEditor();
            _filePath = filePath;
            IsModified = modified;
            richTextBox.Focus();
        }
        catch (Exception ex)
        {
            MessageBox.Show($"ファイルを開けませんでした。\n{ex.Message}", "エラー",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }

    private void SaveMenuItem_Click(object? sender, EventArgs e)
    {
        if (_filePath == null) { SaveAsMenuItem_Click(sender, e); return; }
        SaveToFile(_filePath);
    }

    private void SaveAsMenuItem_Click(object? sender, EventArgs e)
    {
        using var dialog = new SaveFileDialog
        {
            Filter = "Momo点字ファイル (*.mbr)|*.mbr|点字ファイル (*.brf)|*.brf|BASEファイル (*.bse)|*.bse",
            FileName = _filePath != null ? Path.GetFileName(_filePath) : "untitled.mbr",
        };
        if (dialog.ShowDialog() != DialogResult.OK) return;
        SaveToFile(dialog.FileName);
    }

    private void SaveToFile(string path)
    {
        try
        {
            SyncEditorToDocument();
            int? format = FormatForPath(path);
            if (format is int fmt)
            {
                // 点字形式（MBR/BES/BASE/BRF）は Rust の writer でバイト列を生成する。
                var bytes = MomoFfi.WriteDocument(_document, fmt);
                if (bytes == null)
                {
                    MessageBox.Show(
                        "点字変換エンジン（DLL）を読み込めないため、保存できませんでした。",
                        "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
                File.WriteAllBytes(path, bytes);
            }
            else
            {
                // それ以外（.txt など）はセグメントの論理テキストをそのまま書き出す。
                File.WriteAllText(path, LogicalFallbackText(), Encoding.UTF8);
            }
            _filePath = path;
            IsModified = false;
        }
        catch (Exception ex)
        {
            MessageBox.Show($"保存できませんでした。\n{ex.Message}", "エラー",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }

    // ---- ドキュメントモデルとエディタの同期 ----

    // 拡張子に対応する点字形式コード（読み書き共通）。点字形式でなければ null。
    private static int? FormatForPath(string path) =>
        Path.GetExtension(path).ToLowerInvariant() switch
        {
            ".mbr" => MomoFfi.FormatMbr,
            ".bes" => MomoFfi.FormatBes,
            ".bet" => MomoFfi.FormatBet,
            ".bse" => MomoFfi.FormatBase,
            ".brf" => MomoFfi.FormatBrf,
            _ => null,
        };

    private void SyncEditorToDocument()
    {
        // _view != null のとき _document.Segments は編集操作ごとに更新済み。
        if (_view != null) return;
        // フォールバック（DLL 不在）時は RichTextBox の各行を段落セグメントとして取り込む。
        _document.Segments.Clear();
        foreach (var t in richTextBox.Lines)
            _document.Segments.Add(new Segment(t, true));
        if (_document.Segments.Count == 0)
            _document.Segments.Add(new Segment("", true));
    }

    // ドキュメントを Rust で整形して描画する（折返し・ページ分割・ヘッダは Rust 側）。
    private void LoadDocumentToEditor() => ReformatAndRender(-1, -1);

    /// <summary>
    /// 漢字かな交じり文を 1 論理行ずつ点字へ変換してドキュメントを組み立てる。
    /// 空行は空の段落として保持する。点字変換エンジンが使えない場合は null。
    /// </summary>
    private static BrailleDocument? TextToBrailleDocument(string text)
    {
        var predictor = MomoFfi.GetPredictor();
        if (predictor == null) return null;

        var doc = new BrailleDocument();
        foreach (var raw in text.Split('\n'))
        {
            var line = raw.TrimEnd('\r');
            var braille = line.Length == 0 ? "" : predictor.ToBraille(line) ?? "";
            doc.Segments.Add(new Segment(braille, true));
        }
        if (doc.Segments.Count == 0)
            doc.Segments.Add(new Segment("", true));
        return doc;
    }

    private void ExitMenuItem_Click(object? sender, EventArgs e) => Close();

    // ---- 書式（ページ設定） ----

    private void PageSetupMenuItem_Click(object? sender, EventArgs e)
    {
        var updated = FormatterConfigDialog.Edit(this, _document.Config);
        if (updated == null || updated == _document.Config) return;

        // 設定変更を反映して再整形する。カーソルは現在の論理位置に復元する。
        var (si, off) = GetCursor();
        _document.Config = updated;
        ReformatAndRender(si, off); // 変更フラグは ReformatAndRender が立てる
    }

    // ---- ヘルプ ----

    private void AboutMenuItem_Click(object? sender, EventArgs e)
    {
        var asm = Assembly.GetExecutingAssembly();
        // バージョン・著作権・製品名はアセンブリ属性（csproj の Version/Copyright/Product）から取得する。
        var product = asm.GetCustomAttribute<AssemblyProductAttribute>()?.Product ?? "Momo Editor";
        var version = asm.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
            ?? asm.GetName().Version?.ToString()
            ?? "";
        // SourceLink などが付ける "+<commit>" ビルドメタデータは表示から落とす。
        int plus = version.IndexOf('+');
        if (plus >= 0) version = version[..plus];
        var copyright = asm.GetCustomAttribute<AssemblyCopyrightAttribute>()?.Copyright ?? "";

        var message = $"{product}\nバージョン {version}";
        if (copyright.Length > 0) message += $"\n\n{copyright}";

        MessageBox.Show(message, $"{product} について",
            MessageBoxButtons.OK, MessageBoxIcon.Information);
    }

    private void Form1_FormClosing(object? sender, FormClosingEventArgs e)
    {
        if (!ConfirmDiscard()) e.Cancel = true;
    }

    // ---- 六キー点字入力 ----

    private void BrailleInputMenuItem_Click(object? sender, EventArgs e)
    {
        _brailleInputMode = brailleInputMenuItem.Checked;
        _heldBrailleKeys.Clear();
        _chordDotPattern = 0;
        richTextBox.Focus();
        UpdateStatus();

        var message = _brailleInputMode ? "点字入力モード オン" : "点字入力モード オフ";
        richTextBox.AccessibilityObject.RaiseAutomationNotification(
            System.Windows.Forms.Automation.AutomationNotificationKind.ActionCompleted,
            System.Windows.Forms.Automation.AutomationNotificationProcessing.ImportantMostRecent,
            message
        );
    }

    // ---- 編集操作（セグメントモデルを通じて行う） ----

    private void InsertBrailleChar(int dotPattern)
    {
        var ch = (char)(0x2800 + dotPattern);
        if (_view != null)
        {
            if (IsOnHeaderLine()) return;
            var (si, off) = GetCursor();
            var seg = _document.Segments[si];
            seg.Text = seg.Text.Insert(off, ch.ToString());
            ReformatAndRender(si, off + 1);
            Earcon.PlayClick();
            return;
        }
        richTextBox.SelectedText = ch.ToString();
        richTextBox.Focus();
        Earcon.PlayClick();
    }

    // カーソル位置でセグメントを分割する。前半の終端種別と改ページマーカーを指定する。
    // 後半は元の終端種別・改ページマーカーを引き継ぐ。
    private void SplitAtCursor(bool firstParagraphEnd, string? firstPageBreak)
    {
        var (si, off) = GetCursor();
        var seg = _document.Segments[si];
        var before = seg.Text[..off];
        var after = seg.Text[off..];
        var second = new Segment(after, seg.ParagraphEnd, seg.PageBreakMarker);
        seg.Text = before;
        seg.ParagraphEnd = firstParagraphEnd;
        seg.PageBreakMarker = firstPageBreak;
        _document.Segments.Insert(si + 1, second);
        ReformatAndRender(si + 1, 0);
    }

    // Enter: 段落区切り（空行で区切られる新しい論理段落）。
    private void InsertParagraphBreak()
    {
        if (_view != null)
        {
            if (IsOnHeaderLine()) return;
            SplitAtCursor(firstParagraphEnd: true, firstPageBreak: null);
            return;
        }
        richTextBox.SelectedText = "\n";
    }

    // Shift+Enter: 強制改行（同じ論理段落のまま、その位置で必ず行を分ける）。
    private void InsertHardBreak()
    {
        if (_view != null)
        {
            if (IsOnHeaderLine()) return;
            SplitAtCursor(firstParagraphEnd: false, firstPageBreak: null);
            return;
        }
        richTextBox.SelectedText = "\n";
    }

    // Ctrl+Enter: 改ページ（カーソル位置で段落を区切り、その直後で改ページする）。
    private void InsertPageBreak()
    {
        if (_view != null)
        {
            if (IsOnHeaderLine()) return;
            SplitAtCursor(firstParagraphEnd: true, firstPageBreak: "====");
            return;
        }
        richTextBox.SelectedText = "\n";
    }

    private void SimulateBackspace()
    {
        if (_view != null)
        {
            if (IsOnHeaderLine()) return;
            if (richTextBox.SelectionLength > 0)
            {
                // 選択範囲削除（同一セグメント内に限定）
                int pos = richTextBox.SelectionStart;
                int flatLine = richTextBox.GetLineFromCharIndex(pos);
                var (si, off) = _view.PhysicalToLogical(flatLine,
                    pos - richTextBox.GetFirstCharIndexFromLine(flatLine));
                var seg = _document.Segments[si];
                int delLen = Math.Min(richTextBox.SelectionLength, seg.Text.Length - off);
                if (delLen > 0)
                {
                    seg.Text = seg.Text.Remove(off, delLen);
                    ReformatAndRender(si, off);
                }
                return;
            }
            var (csi, coff) = GetCursor();
            if (coff == 0)
            {
                // セグメント先頭: 直前のセグメントと結合（間の区切り＝改行/段落/改ページを除去）
                if (csi == 0) return;
                var prev = _document.Segments[csi - 1];
                var cur = _document.Segments[csi];
                int prevLen = prev.Text.Length;
                prev.Text += cur.Text;
                prev.ParagraphEnd = cur.ParagraphEnd;
                prev.PageBreakMarker = cur.PageBreakMarker;
                _document.Segments.RemoveAt(csi);
                ReformatAndRender(csi - 1, prevLen);
                return;
            }
            // 1文字削除
            var s = _document.Segments[csi];
            s.Text = s.Text.Remove(coff - 1, 1);
            ReformatAndRender(csi, coff - 1);
            return;
        }
        if (richTextBox.SelectionLength > 0)
            richTextBox.SelectedText = "";
        else if (richTextBox.SelectionStart > 0)
        {
            richTextBox.Select(richTextBox.SelectionStart - 1, 1);
            richTextBox.SelectedText = "";
        }
    }

    private void SimulateDelete()
    {
        if (_view != null)
        {
            if (IsOnHeaderLine()) return;
            var (si, off) = GetCursor();
            var seg = _document.Segments[si];
            if (off >= seg.Text.Length)
            {
                // セグメント末尾: 次のセグメントと結合
                if (si + 1 >= _document.Segments.Count) return;
                var next = _document.Segments[si + 1];
                seg.Text += next.Text;
                seg.ParagraphEnd = next.ParagraphEnd;
                seg.PageBreakMarker = next.PageBreakMarker;
                _document.Segments.RemoveAt(si + 1);
                ReformatAndRender(si, off);
                return;
            }
            seg.Text = seg.Text.Remove(off, 1);
            ReformatAndRender(si, off);
            return;
        }
        if (richTextBox.SelectionLength > 0)
            richTextBox.SelectedText = "";
        else if (richTextBox.SelectionStart < richTextBox.TextLength)
        {
            richTextBox.Select(richTextBox.SelectionStart, 1);
            richTextBox.SelectedText = "";
        }
    }

    private void SmartUndo()
    {
        // モデル駆動編集中は Undo スタックがないため何もしない
        if (_view != null) return;
        richTextBox.Undo();
    }

    private void SmartCut()
    {
        if (richTextBox.SelectionLength == 0) return;
        richTextBox.Copy();
        SimulateBackspace(); // 選択範囲をモデル経由で削除
    }

    /// <summary>
    /// 貼り付け（自動判定）。クリップボードが点字データならそのまま、
    /// 漢字かな交じり文なら 1 行ずつ点字へ変換して挿入する。
    /// </summary>
    private void SmartPaste()
    {
        if (!Clipboard.ContainsText()) return;
        var text = Clipboard.GetText();
        if (_view == null) { richTextBox.Paste(); return; }
        if (IsOnHeaderLine()) return;
        switch (ClassifyClipboard(text))
        {
            case ClipboardKind.Braille:
                InsertBrailleLines(SplitLines(text));
                break;
            case ClipboardKind.Text:
                InsertConvertedText(text);
                break;
                // Mixed（点字と非点字が混在）: 安全のため何もしない
        }
    }

    /// <summary>
    /// 「変換して貼り付け」。クリップボードの内容を必ず漢字かな交じり文とみなし、
    /// 1 行ずつ点字へ変換して挿入する。
    /// </summary>
    private void PasteConvertedMenuItem_Click(object? sender, EventArgs e)
    {
        if (!Clipboard.ContainsText()) return;
        var text = Clipboard.GetText();
        if (_view == null || IsOnHeaderLine()) return;
        // 点字と非点字が混在しているときは、誤変換を避けるため何もしない。
        if (ClassifyClipboard(text) == ClipboardKind.Mixed) return;
        InsertConvertedText(text);
    }

    /// <summary>漢字かな交じり文を 1 行ずつ点字へ変換し、カーソル位置に挿入する。</summary>
    private void InsertConvertedText(string text)
    {
        var predictor = MomoFfi.GetPredictor();
        if (predictor == null)
        {
            MessageBox.Show(
                "点字変換エンジン（モデル）を読み込めないため、漢字かな交じり文を点字に変換できませんでした。",
                "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }
        var lines = SplitLines(text);
        var braille = Array.ConvertAll(lines, line => line.Length == 0 ? "" : predictor.ToBraille(line) ?? "");
        InsertBrailleLines(braille);
    }

    /// <summary>点字データ（複数行）をカーソル位置に挿入する。_view != null かつ非ヘッダ行で呼ぶこと。</summary>
    private void InsertBrailleLines(string[] parts)
    {
        var (si, off) = GetCursor();
        var seg = _document.Segments[si];
        if (parts.Length == 1)
        {
            // セグメント内への単純挿入
            seg.Text = seg.Text.Insert(off, parts[0]);
            ReformatAndRender(si, off + parts[0].Length);
            return;
        }
        // 複数行: 現在のセグメントを分割しながら段落として挿入する。
        var tail = seg.Text[off..];
        var origEnd = seg.ParagraphEnd;
        var origPageBreak = seg.PageBreakMarker;
        seg.Text = seg.Text[..off] + parts[0];
        seg.ParagraphEnd = true;
        seg.PageBreakMarker = null;

        int insertAt = si + 1;
        for (int i = 1; i < parts.Length - 1; i++)
            _document.Segments.Insert(insertAt++, new Segment(parts[i], true));
        _document.Segments.Insert(insertAt, new Segment(parts[^1] + tail, origEnd, origPageBreak));
        ReformatAndRender(insertAt, parts[^1].Length);
    }

    // 改行（CRLF/CR/LF）を正規化して論理行へ分割する。
    private static string[] SplitLines(string text) =>
        text.Replace("\r\n", "\n").Replace('\r', '\n').Split('\n');

    private enum ClipboardKind
    {
        Braille, // 点字データ（U+2800 ブロック）のみ
        Text,    // 非点字（漢字かな交じり文）のみ
        Mixed,   // 点字と非点字が混在
    }

    /// <summary>
    /// クリップボード文字列を点字データ・テキスト・混在のいずれかに分類する。
    /// 空白・改行は判定対象外。点字セルと非点字文字が両方あれば混在とみなす。
    /// </summary>
    private static ClipboardKind ClassifyClipboard(string text)
    {
        bool hasBraille = false, hasOther = false;
        foreach (var ch in text)
        {
            if (ch is >= '⠀' and <= '⣿') hasBraille = true;
            else if (ch is '\r' or '\n' or '\t' or ' ' or '　') continue;
            else hasOther = true;
        }
        if (hasBraille && hasOther) return ClipboardKind.Mixed;
        return hasBraille ? ClipboardKind.Braille : ClipboardKind.Text;
    }

    // ---- キー処理 ----

    protected override void OnKeyDown(KeyEventArgs e)
    {
        // Alt 単独 / F10 でメニューバーへフォーカス。
        // 編集面の RichTextBox が Alt を内部で消費し、既定のメニュー活性化が
        // 効かないため、Form の KeyPreview 経由でここから明示的に活性化する。
        if ((e.KeyData == (Keys.Menu | Keys.Alt) || e.KeyData == Keys.F10)
            && menuStrip.Items.Count > 0)
        {
            menuStrip.Select();
            menuStrip.Items[0].Select();
            e.SuppressKeyPress = true;
            e.Handled = true;
            return;
        }

        if (_keyMap.TryGetValue(e.KeyData, out var action))
        {
            action();
            e.SuppressKeyPress = true;
            e.Handled = true;
            return;
        }

        if (_brailleInputMode)
        {
            if ((e.Modifiers & (Keys.Control | Keys.Alt)) != Keys.None)
            {
                base.OnKeyDown(e);
                return;
            }

            if (e.Modifiers == Keys.None && BrailleKeyBit.TryGetValue(e.KeyCode, out int bit))
            {
                _heldBrailleKeys.Add(e.KeyCode);
                _chordDotPattern |= 1 << bit;
                e.SuppressKeyPress = true;
                e.Handled = true;
                return;
            }

            if (e.Modifiers == Keys.None && e.KeyCode == Keys.Space)
            {
                e.SuppressKeyPress = true;
                e.Handled = true;
                return;
            }
        }
        base.OnKeyDown(e);
    }

    protected override void OnKeyPress(KeyPressEventArgs e)
    {
        if (_brailleInputMode)
        {
            e.Handled = true;
            return;
        }
        base.OnKeyPress(e);
    }

    protected override void OnKeyUp(KeyEventArgs e)
    {
        if (_brailleInputMode && (e.Modifiers & (Keys.Control | Keys.Alt)) == Keys.None)
        {
            if (BrailleKeyBit.ContainsKey(e.KeyCode))
            {
                _heldBrailleKeys.Remove(e.KeyCode);
                if (_heldBrailleKeys.Count == 0 && _chordDotPattern != 0)
                {
                    InsertBrailleChar(_chordDotPattern);
                    _chordDotPattern = 0;
                }
                e.Handled = true;
                return;
            }
            if (e.KeyCode == Keys.Space && e.Modifiers == Keys.None)
            {
                InsertBrailleChar(0); // ⠀ U+2800
                e.Handled = true;
                return;
            }
        }
        base.OnKeyUp(e);
    }

    // ---- RichTextBox イベント ----

    private void RichTextBox_SelectionChanged(object? sender, EventArgs e)
    {
        UpdateStatus();
        UpdateGuide();
    }

    private void RichTextBox_TextChanged(object? sender, EventArgs e)
    {
        if (!_suppressModified) IsModified = true;
    }
}
