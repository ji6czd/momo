using System.Text;

namespace MomoEditor;

public partial class Form1 : Form
{
    private string? _filePath;
    private bool _isModified;
    private bool _suppressModified;
    private BrailleDocument _document = BrailleDocument.Empty();
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
            [Keys.Return]           = InsertParagraphBreak,
            [Keys.Back]             = SimulateBackspace,
            [Keys.Delete]           = SimulateDelete,
        };
        // 新規ドキュメントで起動
        _document.Paragraphs.Add([new PhysicalLine("", true)]);
        LoadDocumentToEditor();
        IsModified = false;
        UpdateTitle();
        UpdateStatus();
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

    // ---- フォーマット・レンダリング ----

    /// <summary>
    /// _document をフォーマットして RichTextBox に描画する。
    /// targetParaIdx/targetCharOffset: カーソルを論理位置に復元する（-1 なら先頭）。
    /// </summary>
    private void ReformatAndRender(int targetParaIdx, int targetCharOffset)
    {
        _suppressModified = true;
        try
        {
            var logical = LogicalText();
            using var handle = MomoFfi.RenderFromParagraphs(logical, _document.Config);
            if (handle == null)
            {
                // DLL 未ビルド/モデル不在: 論理行を生テキストとして表示
                _view = null;
                richTextBox.Text = logical;
                richTextBox.SelectionStart = 0;
                richTextBox.SelectionLength = 0;
                return;
            }

            _view = FormattedDocumentView.Build(handle);
            if (_view.PhysicalLineCount == 0)
                _view = FormattedDocumentView.CreateEmpty(_document.Config);

            WriteViewToTextBox(targetParaIdx, targetCharOffset);
        }
        finally
        {
            _suppressModified = false;
        }
        richTextBox.Focus();
    }

    // 全段落の論理テキストを \n で連結して返す（Rust への受け渡し用）。
    private string LogicalText() =>
        string.Join("\n", Enumerable.Range(0, _document.Paragraphs.Count).Select(_document.GetLogicalText));

    // _view を RichTextBox に書き出し、ヘッダ保護とカーソル復元を行う。
    // _suppressModified が true の状態で呼ぶこと。target < 0 なら先頭にカーソルを置く。
    private void WriteViewToTextBox(int targetParaIdx, int targetCharOffset)
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

        if (targetParaIdx >= 0)
            RestoreCursor(targetParaIdx, targetCharOffset);
        else
        {
            richTextBox.SelectionStart = 0;
            richTextBox.SelectionLength = 0;
        }
    }

    // 編集操作後の再描画。ドキュメント全体を Rust で再フォーマットして描画し直す
    // （折返し・ページ分割・ヘッダは Rust に一元化されている）。以下は呼び出し側の
    // 意図を表す薄いラッパで、いずれもカーソルを論理位置へ復元する。
    private void ReformatOneParagraphAndRender(int paraIdx, int targetCharOffset) =>
        ReformatAndRender(paraIdx, targetCharOffset);

    private void ReformatSplitAndRender(int firstParaIdx, int secondParaIdx) =>
        ReformatAndRender(secondParaIdx, 0);

    private void ReformatInsertedParagraphsAndRender(
        int startIdx, int insertedCount, int targetParaIdx, int targetCharOffset) =>
        ReformatAndRender(targetParaIdx, targetCharOffset);

    private void ReformatMergeAndRender(int survivingIdx, int removedIdx, int targetCharOffset) =>
        ReformatAndRender(survivingIdx, targetCharOffset);

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

        // まず全体の保護を解除
        richTextBox.SelectAll();
        richTextBox.SelectionProtected = false;

        // ヘッダ行を保護
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

    /// <summary>論理カーソル位置に物理カーソルを戻す。</summary>
    private void RestoreCursor(int paraIdx, int charOffset)
    {
        if (_view == null) return;
        var (flatLine, cell) = _view.LogicalToPhysical(paraIdx, charOffset);
        int lineStart = richTextBox.GetFirstCharIndexFromLine(flatLine);
        richTextBox.SelectionStart = lineStart + cell;
        richTextBox.SelectionLength = 0;
    }

    /// <summary>現在のカーソル位置を論理座標で返す。</summary>
    private (int paragraphIndex, int charOffset) GetLogicalCursorPosition()
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
        _document = BrailleDocument.Empty();
        _document.Paragraphs.Add([new PhysicalLine("", true)]);
        LoadDocumentToEditor();
        _filePath = null;
        IsModified = false;
    }

    private void OpenMenuItem_Click(object? sender, EventArgs e)
    {
        if (!ConfirmDiscard()) return;
        using var dialog = new OpenFileDialog
        {
            Filter = "Momo点字ファイル (*.mbr)|*.mbr|点字ファイル (*.brl;*.bes)|*.brl;*.bes|テキストファイル (*.txt)|*.txt|すべてのファイル (*.*)|*.*",
        };
        if (dialog.ShowDialog() != DialogResult.OK) return;
        try
        {
            // 取り込み後のファイル名と変更フラグ。点字以外（テキスト）を取り込んだ場合は
            // 拡張子を .mbr に付け替え、未保存（要保存）として扱う。
            string filePath = dialog.FileName;
            bool modified = false;

            if (IsBesFile(dialog.FileName) || IsMbrFile(dialog.FileName))
            {
                // 点字ファイル（MBR / BES）は Rust の reader で正本ドキュメントへ復元する。
                int format = IsBesFile(dialog.FileName) ? MomoFfi.ReadBes : MomoFfi.ReadMbr;
                var doc = MomoFfi.ReadDocument(File.ReadAllBytes(dialog.FileName), format);
                if (doc == null)
                {
                    MessageBox.Show("点字ファイルを読み込めませんでした。", "エラー",
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
            if (_document.Paragraphs.Count == 0)
                _document.Paragraphs.Add([new PhysicalLine("", true)]);
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
            Filter = "Momo点字ファイル (*.mbr)|*.mbr|点字ファイル (*.brl)|*.brl|点字ESファイル (*.bes)|*.bes|テキストファイル (*.txt)|*.txt|すべてのファイル (*.*)|*.*",
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
            var logical = LogicalText();
            int? format = FormatForPath(path);
            if (format is int fmt)
            {
                // 点字形式（MBR/BES/BASE/BRL）は Rust の writer でバイト列を生成する。
                var bytes = MomoFfi.WriteFromParagraphs(logical, _document.Config, fmt);
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
                // それ以外（.txt など）は論理テキストをそのまま書き出す。
                File.WriteAllText(path, logical, Encoding.UTF8);
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

    private static bool IsMbrFile(string path) =>
        Path.GetExtension(path).Equals(".mbr", StringComparison.OrdinalIgnoreCase);

    private static bool IsBesFile(string path) =>
        Path.GetExtension(path).Equals(".bes", StringComparison.OrdinalIgnoreCase);

    // 拡張子に対応する点字書き出し形式コード。点字形式でなければ null（プレーンテキスト保存）。
    private static int? FormatForPath(string path) =>
        Path.GetExtension(path).ToLowerInvariant() switch
        {
            ".mbr" => MomoFfi.FormatMbr,
            ".bes" => MomoFfi.FormatBes,
            ".bse" => MomoFfi.FormatBase,
            ".brl" => MomoFfi.FormatBrl,
            _ => null,
        };

    private void SyncEditorToDocument()
    {
        // _view != null のとき _document.Paragraphs は編集操作ごとに更新済み
        if (_view != null) return;
        _document.SetParagraphs(richTextBox.Lines);
    }

    // ドキュメントを Rust で整形して描画する（折返し・ページ分割・ヘッダは Rust 側）。
    private void LoadDocumentToEditor() => ReformatAndRender(-1, -1);

    /// <summary>
    /// 漢字かな交じり文を 1 論理行ずつ点字へ変換してドキュメントを組み立てる。
    /// 空行は空の論理行として保持する。点字変換エンジンが使えない場合は null。
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
            doc.Paragraphs.Add([new PhysicalLine(braille, true)]);
        }
        return doc;
    }

    private void ExitMenuItem_Click(object? sender, EventArgs e) => Close();

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

    // ---- 編集操作（モデルを通じて行う） ----

    private void InsertBrailleChar(int dotPattern)
    {
        var ch = (char)(0x2800 + dotPattern);
        if (_view != null)
        {
            if (IsOnHeaderLine()) return;
            var (paraIdx, charOffset) = GetLogicalCursorPosition();
            var para = _document.GetLogicalText(paraIdx);
            _document.Paragraphs[paraIdx] = [new PhysicalLine(para.Insert(charOffset, ch.ToString()), true)];
            ReformatOneParagraphAndRender(paraIdx, charOffset + 1);
            return;
        }
        richTextBox.SelectedText = ch.ToString();
        richTextBox.Focus();
    }

    private void InsertParagraphBreak()
    {
        if (_view != null)
        {
            if (IsOnHeaderLine()) return;
            var (paraIdx, charOffset) = GetLogicalCursorPosition();
            var para = _document.GetLogicalText(paraIdx);
            _document.Paragraphs[paraIdx] = [new PhysicalLine(para[..charOffset], true)];
            _document.Paragraphs.Insert(paraIdx + 1, [new PhysicalLine(para[charOffset..], true)]);
            ReformatSplitAndRender(paraIdx, paraIdx + 1);
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
                // 選択範囲削除（同一段落内に限定）
                int pos = richTextBox.SelectionStart;
                int flatLine = richTextBox.GetLineFromCharIndex(pos);
                var (paraIdx, charOffset) = _view.PhysicalToLogical(flatLine,
                    pos - richTextBox.GetFirstCharIndexFromLine(flatLine));
                var para = _document.GetLogicalText(paraIdx);
                int delLen = Math.Min(richTextBox.SelectionLength, para.Length - charOffset);
                if (delLen > 0)
                {
                    _document.Paragraphs[paraIdx] = [new PhysicalLine(para.Remove(charOffset, delLen), true)];
                    ReformatOneParagraphAndRender(paraIdx, charOffset);
                }
                return;
            }
            var (pi, co) = GetLogicalCursorPosition();
            if (co == 0)
            {
                // 段落先頭: 前の段落と結合
                if (pi == 0) return;
                var prev = _document.GetLogicalText(pi - 1);
                var curr = _document.GetLogicalText(pi);
                _document.Paragraphs[pi - 1] = [new PhysicalLine(prev + curr, true)];
                _document.Paragraphs.RemoveAt(pi);
                ReformatMergeAndRender(pi - 1, pi, prev.Length);
                return;
            }
            // 1文字削除
            var p = _document.GetLogicalText(pi);
            _document.Paragraphs[pi] = [new PhysicalLine(p.Remove(co - 1, 1), true)];
            ReformatOneParagraphAndRender(pi, co - 1);
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
            var (paraIdx, charOffset) = GetLogicalCursorPosition();
            var para = _document.GetLogicalText(paraIdx);
            if (charOffset >= para.Length)
            {
                // 段落末尾: 次の段落と結合
                if (paraIdx + 1 >= _document.Paragraphs.Count) return;
                var next = _document.GetLogicalText(paraIdx + 1);
                _document.Paragraphs[paraIdx] = [new PhysicalLine(para + next, true)];
                _document.Paragraphs.RemoveAt(paraIdx + 1);
                ReformatMergeAndRender(paraIdx, paraIdx + 1, charOffset);
                return;
            }
            _document.Paragraphs[paraIdx] = [new PhysicalLine(para.Remove(charOffset, 1), true)];
            ReformatOneParagraphAndRender(paraIdx, charOffset);
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

    private void SmartPaste()
    {
        if (!Clipboard.ContainsText()) return;
        var text = Clipboard.GetText();
        if (_view != null)
        {
            if (IsOnHeaderLine()) return;
            // 改行で分割して段落を追加しながら挿入
            var parts = text.Split('\n');
            var (paraIdx, charOffset) = GetLogicalCursorPosition();
            var para = _document.GetLogicalText(paraIdx);
            if (parts.Length == 1)
            {
                // 1段落内への単純挿入
                _document.Paragraphs[paraIdx] = [new PhysicalLine(para.Insert(charOffset, parts[0]), true)];
                ReformatOneParagraphAndRender(paraIdx, charOffset + parts[0].Length);
            }
            else
            {
                // 複数行: 現在の段落を分割しながら挿入
                var tail = para[charOffset..];
                _document.Paragraphs[paraIdx] = [new PhysicalLine(para[..charOffset] + parts[0], true)];
                for (int i = 1; i < parts.Length - 1; i++)
                    _document.Paragraphs.Insert(paraIdx + i, [new PhysicalLine(parts[i], true)]);
                var lastIdx = paraIdx + parts.Length - 1;
                _document.Paragraphs.Insert(lastIdx, [new PhysicalLine(parts[^1] + tail, true)]);
                ReformatInsertedParagraphsAndRender(paraIdx, parts.Length - 1, lastIdx, parts[^1].Length);
            }
            return;
        }
        richTextBox.Paste();
    }

    // ---- キー処理 ----

    protected override void OnKeyDown(KeyEventArgs e)
    {
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

    private void RichTextBox_SelectionChanged(object? sender, EventArgs e) => UpdateStatus();

    private void RichTextBox_TextChanged(object? sender, EventArgs e)
    {
        if (!_suppressModified) IsModified = true;
    }
}
