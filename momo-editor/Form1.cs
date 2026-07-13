using System.Reflection;
using System.Text;
using System.Windows.Forms.Automation;
using Momo;

namespace MomoEditor;

public partial class Form1 : Form
{
    private string? _filePath;
    private bool _isModified;
    private bool _suppressModified;
    private BrailleDocument _document = BrailleDocument.NewEmpty();
    private FormattedDocumentView? _view;

    // テキスト→点字変換に使う点訳器。テーブルを切り替えたら破棄し、次の変換時に作り直す。
    private MomoFfi.BrailleTranslatorHandle? _translator;

    // 英語行に使う点字テーブル。true なら UEB Grade 2（縮約あり）、false なら Grade 1（無縮約）。
    // 日本語行は常に日本語１級。切り替えたら _translator を破棄して作り直す。
    private bool _grade2Table;

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
        // Cascadia Mono が無い環境（Windows 10 等）では、同じく点字グリフを
        // ネイティブに持つ Segoe UI Symbol（全 Windows 標準）へフォールバックする。
        // フォントリンク任せにすると環境ごとに描画フォントが変わってしまう。
        if (richTextBox.Font.Name != "Cascadia Mono")
            richTextBox.Font = new Font("Segoe UI Symbol", 28F, FontStyle.Regular, GraphicsUnit.Point);
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
        // 英語点字メニューのラベルを実データの表示名に更新する。
        InitTableMenu();

        // 新規ドキュメントで起動
        LoadDocumentToEditor();
        IsModified = false;
        UpdateTitle();
        UpdateStatus();
        AdjustStartupSize();
    }

    /// <summary>
    /// 起動時のウィンドウサイズを調整する。
    /// 幅: 点字 1 行（LineWidth セル）が折り返さずに収まるサイズ以上へ広げる。
    /// 高さ: 点字 1 ページ（ヘッダ + LinesPerPage 行）が収まるサイズへ広げる。
    /// ただし画面の作業領域は超えない。等幅フォントを実測するため
    /// 高 DPI 環境でも正しくスケールする。
    /// </summary>
    private void AdjustStartupSize()
    {
        var (textWidth, lineHeight) = MeasureBrailleMetrics(_document.Config.LineWidth);

        // 幅: 縦スクロールバーと内部余白の分を加える。
        int neededW = textWidth + SystemInformation.VerticalScrollBarWidth + 12;
        if (ClientSize.Width < neededW)
            ClientSize = new Size(neededW, ClientSize.Height);
        // 1 行分を割り込めないよう最小幅も設定。
        MinimumSize = new Size(neededW + (Width - ClientSize.Width), MinimumSize.Height);

        // 高さ: 1 ページ分の行数 + 内部余白。LinesPerPage はヘッダ行を含む値
        // （フォーマッタはヘッダ有効時に本文を LinesPerPage - 1 行にする）。
        // ウィンドウ枠・メニュー・ガイド帯・ステータスバーの高さは
        // 「現在のウィンドウ高 − 編集面の高さ」で実測する。
        int chromeH = Height - richTextBox.ClientSize.Height;
        int neededH = _document.Config.LinesPerPage * lineHeight + 8 + chromeH;
        int maxH = Screen.FromControl(this).WorkingArea.Height;
        Height = Math.Max(Height, Math.Min(neededH, maxH));
    }

    protected override void OnLoad(EventArgs e)
    {
        base.OnLoad(e);
        // ウィンドウ位置は表示直前に OS が決める（カスケード配置）ため、
        // 起動時に縦いっぱいへ広げたウィンドウが作業領域からはみ出すことがある。
        // ここで下端・右端が収まるように位置を補正する。
        var wa = Screen.FromControl(this).WorkingArea;
        if (Bottom > wa.Bottom) Top = Math.Max(wa.Top, wa.Bottom - Height);
        if (Right > wa.Right) Left = Math.Max(wa.Left, wa.Right - Width);
    }

    /// <summary>
    /// 点字 cells セル分の実描画幅と、行間（_lineSpacing 倍）適用後の
    /// 1 行の高さ（ピクセル）を返す。
    /// GDI の TextRenderer で測ると、フォントに無いグリフのフォールバックが
    /// RichEdit の実レイアウトと食い違うことがあるため（Courier New 時代に
    /// 18px/セル vs 24px/セルの実害があった）、同じフォントを持つプローブ用
    /// RichTextBox に実際に文字を置き、文字位置 API でレイアウトを測る。
    /// </summary>
    private (int width, int lineHeight) MeasureBrailleMetrics(int cells)
    {
        using var probe = new RichTextBox
        {
            Font = richTextBox.Font,
            WordWrap = false,
            ScrollBars = RichTextBoxScrollBars.None,
            Size = new Size(100, 100), // 測定はレイアウト座標なので表示サイズは無関係
        };
        _ = probe.Handle; // レイアウト計算にはハンドルが必要
        // 1 行目: cells+1 文字目の開始位置が cells セル分の右端になる。
        // 2 行目: 行頭の Y 差分が行間適用後の 1 行の高さになる。
        probe.Text = new string('⠀', cells + 1) + "\n⠀";
        ApplyLineSpacing(probe);
        var origin = probe.GetPositionFromCharIndex(0);
        int width = probe.GetPositionFromCharIndex(cells).X - origin.X;
        int lineHeight = probe.GetPositionFromCharIndex(probe.GetFirstCharIndexFromLine(1)).Y - origin.Y;
        if (width > 0 && lineHeight > 0) return (width, lineHeight);
        // 万一レイアウトを取れなかったときは従来の GDI 測定にフォールバック。
        var size = TextRenderer.MeasureText(
            new string('⠀', cells), richTextBox.Font,
            new Size(int.MaxValue, int.MaxValue), TextFormatFlags.NoPadding);
        return (size.Width, (int)Math.Round(size.Height * _lineSpacing));
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

    // ---- 読みガイド・読み上げガイド ----

    // 直近に逆点訳した点字行とその結果（行内でカーソルだけ動いたときの再計算を避ける）。
    private string? _guideLineCache;
    private IReadOnlyList<MomoFfi.GuideSegment> _guideSegments = [];

    // 直近に読み上げた表示行。同じ行で繰り返し発話しないための追跡。
    private int _announcedLine = -1;

    private void GuideMenuItem_Click(object? sender, EventArgs e)
    {
        guideStrip.Visible = guideMenuItem.Checked;
        UpdateGuide();
        richTextBox.Focus();
    }

    private void SpeechGuideMenuItem_Click(object? sender, EventArgs e)
    {
        richTextBox.Focus();
        Announce(speechGuideMenuItem.Checked ? "読み上げガイド オン" : "読み上げガイド オフ",
            AutomationNotificationProcessing.ImportantMostRecent);
    }

    /// <summary>
    /// カーソル位置の変化に合わせて、読みガイド帯（視覚）と読み上げガイド（UIA 通知）を
    /// 更新する。逆点訳キャッシュは両者で共有するため、ガイド帯の表示状態とは無関係に
    /// 更新する。範囲選択中・フォールバック表示中は何もしない。
    /// </summary>
    private void UpdateGuide()
    {
        // 範囲選択中はキャレット位置が定まらないので更新しない（現状維持）。
        if (richTextBox.SelectionLength != 0) return;

        if (_view == null)
        {
            ClearGuide();
            return;
        }

        int pos = richTextBox.SelectionStart;
        int flatLine = richTextBox.GetLineFromCharIndex(pos);
        if (flatLine < 0 || flatLine >= _view.PhysicalLineCount)
        {
            ClearGuide();
            return;
        }

        bool isHeader = _view.IsHeaderAt(flatLine);
        string content = _view.ContentAt(flatLine);
        int col = pos - richTextBox.GetFirstCharIndexFromLine(flatLine);

        // 逆点訳（行内容単位のキャッシュ）。ヘッダ行は点字ではないので対象外。
        if (!isHeader && content != _guideLineCache)
        {
            var result = MomoFfi.BackTranslateLine(content);
            _guideSegments = result?.Segments ?? [];
            _guideLineCache = content;
        }

        // ガイド帯のデータは表示 OFF でも更新する。GuideStrip の AccessibleDescription に
        // 行全体の読みが入るため、スクリーンリーダー（JAWS スクリプト等）が
        // いつでも UIA 経由で現在行の読みを取得できる。
        if (isHeader) guideStrip.SetData("", [], -1);
        else guideStrip.SetData(content, _guideSegments, col);

        AnnounceCursorMove(flatLine, isHeader, content);
    }

    /// <summary>
    /// カーソル移動に応じた読み上げ。行が変わったときだけ行全体の読みを UIA 通知で
    /// 発話する。行内の左右移動はスクリーンリーダー自身のセル読み（ドット構成）に任せる。
    /// 再整形中（_suppressModified）や読み上げオフのときは位置の追跡だけ行い発話しない。
    /// </summary>
    private void AnnounceCursorMove(int flatLine, bool isHeader, string content)
    {
        bool lineChanged = flatLine != _announcedLine;
        _announcedLine = flatLine;

        if (!lineChanged || _suppressModified || !speechGuideMenuItem.Checked) return;

        string reading;
        if (isHeader) reading = content.Trim();
        else if (content.Length == 0) reading = "空行";
        else reading = string.Concat(_guideSegments.Select(s => s.Text));
        Announce(reading);
    }

    /// <summary>
    /// スクリーンリーダーへ UIA 通知で発話を依頼する。既定の MostRecent は、
    /// 矢印キー連打などで未発話の通知が溜まったとき最新のものだけを発話させる。
    /// </summary>
    private void Announce(string text,
        AutomationNotificationProcessing processing = AutomationNotificationProcessing.MostRecent)
    {
        if (text.Length == 0) return;
        richTextBox.AccessibilityObject.RaiseAutomationNotification(
            AutomationNotificationKind.Other, processing, text);
    }

    private void ClearGuide()
    {
        _guideLineCache = "";
        _guideSegments = [];
        _announcedLine = -1;
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
    /// テキストを点字へ変換する。選択中のテーブルで点訳器を組み立て、日本語行は
    /// 予測器（漢字かな交じり文→かな）を経由する。エンジンが使えないときは null。
    /// </summary>
    private Func<string, string>? GetTranslateLine()
    {
        var predictor = MomoFfi.GetPredictor();
        if (predictor == null) return null;

        _translator ??= MomoFfi.CreateTranslator(
            MomoFfi.TableJapaneseGrade1,
            _grade2Table ? MomoFfi.TableEnglishUebGrade2 : MomoFfi.TableEnglishUebGrade1);
        var translator = _translator;
        if (translator == null) return null;

        return line => line.Length == 0 ? "" : translator.ToBraille(line, predictor) ?? "";
    }

    /// <summary>
    /// 漢字かな交じり文を 1 論理行ずつ点字へ変換してドキュメントを組み立てる。
    /// 空行は空の段落として保持する。点字変換エンジンが使えない場合は null。
    /// </summary>
    private BrailleDocument? TextToBrailleDocument(string text)
    {
        var translate = GetTranslateLine();
        if (translate == null) return null;

        var doc = new BrailleDocument();
        foreach (var raw in text.Split('\n'))
            doc.Segments.Add(new Segment(translate(raw.TrimEnd('\r')), true));
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

        Announce(_brailleInputMode ? "点字入力モード オン" : "点字入力モード オフ",
            AutomationNotificationProcessing.ImportantMostRecent);
    }

    // ---- 英語点字テーブルの切り替え ----
    //
    // 行ごとの言語判定は Rust 側の点訳器が行う。日本語を含む行は常に日本語１級で点訳され、
    // ここで選べるのは英字だけの行に使う UEB のグレード（Grade 1 = 無縮約 / Grade 2 = 縮約）。
    // 点字ドキュメント（点字セル）そのものには影響せず、テキストを変換して取り込む／
    // 貼り付けるときにのみ効く。

    // メニューのラベルはアクセラレータ記法（&1/&2）を含むため、読み上げにはこちらの
    // 素の表示名を使う。
    private string _grade1Name = "UEB English (Grade 1)";
    private string _grade2Name = "UEB English (Grade 2)";

    /// <summary>
    /// 英語点字テーブルメニューのラベルを組み込みテーブルの表示名に更新する。
    /// 先頭に "&amp;1" / "&amp;2" を付け、キーボードだけで選べるようにする。
    /// </summary>
    private void InitTableMenu()
    {
        var tables = MomoFfi.EmbeddedTables();
        _grade1Name = TableDisplayName(tables, MomoFfi.TableEnglishUebGrade1, _grade1Name);
        _grade2Name = TableDisplayName(tables, MomoFfi.TableEnglishUebGrade2, _grade2Name);
        // 表示名中の & はメニューのアクセラレータ記法と衝突するのでエスケープする。
        tableGrade1MenuItem.Text = $"&1 {_grade1Name.Replace("&", "&&")}";
        tableGrade2MenuItem.Text = $"&2 {_grade2Name.Replace("&", "&&")}";
    }

    // DLL やテーブルが無い環境でもメニューは出すため、引けないときは既定のラベルを使う。
    private static string TableDisplayName(
        IReadOnlyList<MomoFfi.TableInfo> tables, string name, string fallback) =>
        tables.FirstOrDefault(t => t.Name == name)?.DisplayName ?? fallback;

    private void TableGrade1MenuItem_Click(object? sender, EventArgs e) => SelectTable(grade2: false);

    private void TableGrade2MenuItem_Click(object? sender, EventArgs e) => SelectTable(grade2: true);

    /// <summary>英語点字テーブルを切り替える。二択のチェック状態を排他的に更新する。</summary>
    private void SelectTable(bool grade2)
    {
        tableGrade1MenuItem.Checked = !grade2;
        tableGrade2MenuItem.Checked = grade2;
        if (grade2 != _grade2Table)
        {
            _grade2Table = grade2;
            _translator?.Dispose();
            _translator = null;   // 次の変換時に新しいテーブルで作り直す
        }
        richTextBox.Focus();

        Announce($"英語点字 {(grade2 ? _grade2Name : _grade1Name)}",
            AutomationNotificationProcessing.ImportantMostRecent);
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
        var translate = GetTranslateLine();
        if (translate == null)
        {
            MessageBox.Show(
                "点字変換エンジン（モデル）を読み込めないため、漢字かな交じり文を点字に変換できませんでした。",
                "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }
        InsertBrailleLines(Array.ConvertAll(SplitLines(text), line => translate(line)));
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

    private const int WM_SYSCOMMAND = 0x0112;
    private const int SC_KEYMENU = 0xF100;

    // メニューバーの Alt/F10 活性化はフレームワーク任せにせず自前で行う。
    // 実測(tmp/AltMenuTest)により、MenuStrip 既定のメニューモードは
    // Win32 フォーカスを移さず、スクリーンリーダーへの通知が不安定なことが
    // 分かったため、実フォーカスを移す ActivateMenuBar() 方式を採る。
    // その代償として、既定の活性化(SC_KEYMENU)の抑止と、Windows 標準の
    // キー挙動(解放時活性化・他キー割込みで取消・再 Alt で解除)の再現を
    // ここで自前実装している。

    /// <summary>Alt 単独押下中(他のキーが挟まっていない)なら true。解放時のメニュー活性化に使う。</summary>
    private bool _altMenuPending;

    protected override void WndProc(ref Message m)
    {
        // Alt / F10 キーアップ由来の既定のメニュー活性化 (SC_KEYMENU) を無効化する。
        // 活性化は OnKeyDown/OnKeyUp で自前で行っているため不要であり、
        // Alt+Tab でこのウィンドウに切り替えた直後の Alt 解放(KeyDown を
        // 受け取っていない KeyUp)でもメニューが活性化されてしまうのを防ぐ。
        // lParam != 0(Alt+Space のシステムメニュー等)は既定処理に任せる。
        if (m.Msg == WM_SYSCOMMAND && ((int)m.WParam & 0xFFF0) == SC_KEYMENU
            && m.LParam == IntPtr.Zero)
        {
            return;
        }
        base.WndProc(ref m);
    }

    private void ActivateMenuBar()
    {
        menuStrip.Select();
        menuStrip.Items[0].Select();
    }

    protected override void OnKeyDown(KeyEventArgs e)
    {
        // メニューバーがアクティブな状態での Alt はフォーカス解除(Windows 標準では
        // 押下時に解除される)。手動の Select() による活性化はフレームワークの
        // メニューモードに入らず既定の Alt 解除が働かないため、ここで自前で行う。
        if (e.KeyData == (Keys.Menu | Keys.Alt) && menuStrip.ContainsFocus)
        {
            _altMenuPending = false;
            richTextBox.Focus();
            e.SuppressKeyPress = true;
            e.Handled = true;
            return;
        }

        // Alt 単独は Windows 標準に合わせて「解放時」にメニューバーを活性化する
        // (OnKeyUp 参照)。押下時はフラグを立てるだけにし、解放までに他のキーが
        // 挟まったら取り消す(Alt+Tab や Alt+ショートカットでは活性化しない)。
        _altMenuPending = e.KeyData == (Keys.Menu | Keys.Alt) && menuStrip.Items.Count > 0;

        // F10 でメニューバーへフォーカス(Windows 標準では押下時に活性化)。
        // 既定の F10 活性化も SC_KEYMENU 経由のため上の WndProc で抑止されて
        // おり、ここで明示的に活性化する必要がある。
        if (e.KeyData == Keys.F10 && menuStrip.Items.Count > 0)
        {
            ActivateMenuBar();
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
        // Alt 単独の押下→解放でメニューバーへフォーカス(Windows 標準の挙動)。
        // フラグは OnKeyDown で管理しており、間に他のキーが挟まった場合や
        // Alt+Tab 直後(押下をこのウィンドウが受けていない)では立っていない。
        if (e.KeyCode == Keys.Menu && _altMenuPending)
        {
            _altMenuPending = false;
            ActivateMenuBar();
            e.Handled = true;
            return;
        }

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

    protected override void OnDeactivate(EventArgs e)
    {
        // Alt を押したままフォーカスが他ウィンドウへ移った場合、
        // 戻ってきた後の Alt 解放でメニューが活性化しないように取り消す。
        _altMenuPending = false;
        base.OnDeactivate(e);
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
