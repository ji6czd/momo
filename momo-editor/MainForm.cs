using System.Reflection;
using System.Text;
using System.Windows.Forms.Automation;
using Momo;

namespace MomoEditor;

public partial class MainForm : Form
{
    private string? _filePath;
    private bool _isModified;
    private bool _suppressModified;
    private BrailleDocument _document = BrailleDocument.NewEmpty();
    private FormattedDocumentView? _view;
    private readonly AppSettings _settings = AppSettings.Load();

    // AdjustStartupSize の幅計算に足す安全マージン（ピクセル）。プローブ測定と
    // 実際の描画幅が環境依存でわずかに食い違うケースへの経験的な余裕代。
    private const int WidthSafetyMargin = 12;

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

    public MainForm(string? initialPath = null)
    {
        InitializeComponent();
        // SixBraille HLF はインストーラでシステムインストールされる想定だが、インストーラを
        // 経由しない実行（開発時の dotnet run 等）では無いことがある。その場合は
        // Cascadia Mono（Windows 11 標準搭載の等幅コーディングフォント）へフォールバックする。
        // フォントリンク任せにすると環境ごとに描画フォントが変わってしまう。
        if (richTextBox.Font.Name != "SixBraille HLF")
            richTextBox.Font = new Font("Cascadia Mono", 28F, FontStyle.Regular, GraphicsUnit.Point);
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
        RebuildRecentFilesMenu();

        // 新規ドキュメントで起動
        LoadDocumentToEditor();
        IsModified = false;
        UpdateTitle();
        UpdateStatus();
        AdjustStartupSize();

        // コマンドライン引数（Explorer の「プログラムから開く」を含む）でファイルが
        // 指定されていれば、新規ドキュメントの代わりにそれを開く。起動直後で未編集の
        // ため破棄確認は不要。存在しないパスは黙って無視して空ドキュメントのままにする。
        if (initialPath != null && File.Exists(initialPath))
            OpenPath(initialPath);
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

        // 幅: 縦スクロールバーと内部余白の分に加え、環境によって実測値と実際の描画幅が
        // わずかに食い違うことがあるための安全マージンを加える。
        int neededW = textWidth + SystemInformation.VerticalScrollBarWidth + 12 + WidthSafetyMargin;
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
        bool isDivider = _view.IsDividerAt(flatLine);
        string content = _view.ContentAt(flatLine);
        int col = pos - richTextBox.GetFirstCharIndexFromLine(flatLine);

        // 逆点訳（行内容単位のキャッシュ）。ヘッダ行・改ページ区切り行は点字ではないので対象外。
        if (!isHeader && !isDivider && content != _guideLineCache)
        {
            var result = MomoFfi.BackTranslateLine(content);
            _guideSegments = result?.Segments ?? [];
            _guideLineCache = content;
        }

        // ガイド帯のデータは表示 OFF でも更新する。GuideStrip の AccessibleDescription に
        // 行全体の読みが入るため、スクリーンリーダー（JAWS スクリプト等）が
        // いつでも UIA 経由で現在行の読みを取得できる。
        if (isHeader || isDivider) guideStrip.SetData("", [], -1);
        else guideStrip.SetData(content, _guideSegments, col);

        AnnounceCursorMove(flatLine, isHeader, isDivider, content);
    }

    /// <summary>
    /// カーソル移動に応じた読み上げ。行が変わったときだけ行全体の読みを UIA 通知で
    /// 発話する。行内の左右移動はスクリーンリーダー自身のセル読み（ドット構成）に任せる。
    /// 再整形中（_suppressModified）や読み上げオフのときは位置の追跡だけ行い発話しない。
    /// </summary>
    private void AnnounceCursorMove(int flatLine, bool isHeader, bool isDivider, string content)
    {
        bool lineChanged = flatLine != _announcedLine;
        _announcedLine = flatLine;

        if (!lineChanged || _suppressModified || !speechGuideMenuItem.Checked) return;

        string reading;
        if (isDivider) reading = "改ページ";
        else if (isHeader) reading = content.Trim();
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
    /// _document を Rust で整形して描画する（Undo/Redo スタック管理込み）。
    /// targetSeg &lt; 0（ロード/新規）なら前の文書の Undo/Redo 履歴を破棄。
    /// それ以外（実際の編集）は、直前の状態（_lastSnapshot）を Undo スタックへ積んでから
    /// 描画し、描画後の状態で _lastSnapshot を更新し直す。
    /// </summary>
    private void ReformatAndRender(int targetSeg, int targetOffset)
    {
        if (targetSeg < 0)
        {
            _undoStack.Clear();
            _redoStack.Clear();
        }
        else if (_lastSnapshot != null)
        {
            PushCapped(_undoStack, _lastSnapshot);
            _redoStack.Clear();
        }

        RenderDocumentToEditor(targetSeg, targetOffset);

        if (_view != null)
        {
            var (si, off) = GetCursor();
            _lastSnapshot = new EditSnapshot(CloneDocument(_document), si, off);
        }
        else
        {
            _lastSnapshot = null; // フォールバック表示中はモデル駆動 Undo の対象外
        }
    }

    /// <summary>
    /// _document を Rust で整形して RichTextBox に描画する（描画のみ、Undo/Redo スタックには
    /// 触れない）。ReformatAndRender と RestoreSnapshot（MainForm.Undo.cs）の双方から呼ばれる。
    /// targetSeg/targetOffset: カーソルを論理位置（セグメント+オフセット）に復元する（-1 なら先頭の編集可能行）。
    /// </summary>
    private void RenderDocumentToEditor(int targetSeg, int targetOffset)
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

            _view = FormattedDocumentView.Build(handle, _document);
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
        string.Join("\n", _document.Entries.OfType<TextSegment>().Select(s => s.Text));

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

        // RichTextBox 自前のネイティブ Undo バッファを毎回クリアする。ContextMenuStrip
        // 未設定のため右クリックでネイティブの「元に戻す」を含むコンテキストメニューが
        // 出せる状態になっており、それを操作すると _document を介さず表示だけが戻って
        // しまい、モデルと表示がズレる。モデル駆動描画のたびにクリアしておけば、その
        // 項目は常に無効化され実害が起きない（フォールバックモードはここを通らないため
        // 既存の richTextBox.Undo() は影響を受けない）。
        richTextBox.ClearUndo();

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

    /// <summary>現在のカーソル行がヘッダ行または改ページ区切り行（保護行）かどうかを返す。
    /// どちらも論理カーソル位置（セグメント+オフセット）が定まらないため、他の編集コマンドは
    /// この上では何もしない。</summary>
    private bool IsOnProtectedLine()
    {
        if (_view == null) return false;
        int flatLine = richTextBox.GetLineFromCharIndex(richTextBox.SelectionStart);
        if (flatLine < 0 || flatLine >= _view.PhysicalLineCount) return false;
        return _view.IsHeaderAt(flatLine) || _view.IsDividerAt(flatLine);
    }

    /// <summary>ヘッダ行・改ページ区切り行を編集不可に設定する。</summary>
    private void ApplyHeaderProtection()
    {
        if (_view == null) return;

        richTextBox.SelectAll();
        richTextBox.SelectionProtected = false;

        for (int i = 0; i < _view.PhysicalLineCount; i++)
        {
            if (!_view.IsHeaderAt(i) && !_view.IsDividerAt(i)) continue;
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
        OpenPath(dialog.FileName);
    }

    /// <summary>「最近使ったファイル」サブメニューを現在の一覧から組み立て直す。</summary>
    private void RebuildRecentFilesMenu()
    {
        recentFilesMenuItem.DropDownItems.Clear();
        if (_settings.RecentFiles.Count == 0)
        {
            recentFilesMenuItem.DropDownItems.Add(new ToolStripMenuItem("(なし)") { Enabled = false });
            return;
        }
        foreach (var path in _settings.RecentFiles)
        {
            var item = new ToolStripMenuItem(Path.GetFileName(path));
            item.Click += (_, _) => OpenRecentFile(path);
            recentFilesMenuItem.DropDownItems.Add(item);
        }
    }

    /// <summary>「最近使ったファイル」の項目から開く。既に存在しなければ一覧から取り除く。</summary>
    private void OpenRecentFile(string path)
    {
        if (!File.Exists(path))
        {
            MessageBox.Show($"ファイルが見つかりません。\n{path}", "エラー",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
            _settings.RecentFiles.Remove(path);
            _settings.Save();
            RebuildRecentFilesMenu();
            return;
        }
        if (!ConfirmDiscard()) return;
        OpenPath(path);
    }

    /// <summary>
    /// 指定パスのファイルを開いてエディタに取り込む。ダイアログを介さないコア処理で、
    /// メニューの「開く」とコマンドライン引数起動の双方から呼ばれる。
    /// 呼び出し前の破棄確認（ConfirmDiscard）は呼び出し側の責務。
    /// </summary>
    private void OpenPath(string path)
    {
        try
        {
            // 取り込み後のファイル名と変更フラグ。点字以外（テキスト）を取り込んだ場合は
            // 拡張子を .mbr に付け替え、未保存（要保存）として扱う。
            string filePath = path;
            bool modified = false;

            if (FormatForPath(path) is int fmt)
            {
                // 点字ファイル（MBR / BES / BET）は Rust の reader で正本ドキュメントへ復元する。
                var doc = MomoFfi.ReadDocument(File.ReadAllBytes(path), fmt);
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
                var text = File.ReadAllText(path, Encoding.UTF8);
                var doc = TextToBrailleDocument(text);
                if (doc == null)
                {
                    MessageBox.Show(
                        "点字変換エンジン（モデル）を読み込めないため、テキストを点字に変換できませんでした。",
                        "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
                _document = doc;
                filePath = Path.ChangeExtension(path, ".mbr"); // 編集中のファイル名は .mbr
                modified = true;                               // .mbr はまだ保存されていない
            }
            if (_document.Entries.Count == 0)
                _document.Entries.Add(new TextSegment("", true));
            LoadDocumentToEditor();
            _filePath = filePath;
            IsModified = modified;
            // 実ファイルを読み込んだ場合のみ最近使ったファイルに記録する。
            // 漢字かな交じり文からの取り込み（.mbr への拡張子付け替え）は
            // まだ何も書き出していないパスなので対象外（保存時に SaveToFile 側で拾われる）。
            if (!modified) { _settings.AddRecentFile(filePath); RebuildRecentFilesMenu(); }
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
            _settings.AddRecentFile(path);
            RebuildRecentFilesMenu();
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
        _document.Entries.Clear();
        foreach (var t in richTextBox.Lines)
            _document.Entries.Add(new TextSegment(t, true));
        if (_document.Entries.Count == 0)
            _document.Entries.Add(new TextSegment("", true));
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
            doc.Entries.Add(new TextSegment(translate(raw.TrimEnd('\r')), true));
        if (doc.Entries.Count == 0)
            doc.Entries.Add(new TextSegment("", true));
        return doc;
    }

    private void ExitMenuItem_Click(object? sender, EventArgs e) => Close();

    // ---- ページ設定（文書全体の既定値） ----

    private void PageSetupMenuItem_Click(object? sender, EventArgs e)
    {
        var updated = FormatterConfigDialog.Edit(this, _document.Config);
        if (updated == null || updated == _document.Config) return;

        // 設定変更を反映して再整形する。カーソルは現在の論理位置に復元する。
        var (si, off) = GetCursor();
        _document.Config = updated;
        ReformatAndRender(si, off); // 変更フラグは ReformatAndRender が立てる
    }

    // ---- ページ行設定（今表示されているページ以降に適用） ----

    private void PageSectionMenuItem_Click(object? sender, EventArgs e)
    {
        if (_view == null)
        {
            MessageBox.Show("この機能を使うには点訳エンジン（DLL）が必要です。", "ページ行設定",
                MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }
        // ヘッダ行の上では論理カーソル位置（セグメント+オフセット）が定まらないため、
        // 他の編集コマンドと同様にヘッダ行上では何もしない。
        if (IsOnProtectedLine()) return;

        var (cursorSeg, cursorOff) = GetCursor();
        int page = _view.PageAt(cursorSeg, cursorOff);

        if (page == 0)
        {
            // 1ページ目には上書きを付ける先の物理行が無い（直前の物理行が存在しない）ため、
            // 文書全体の既定値（DocumentConfig）を直接編集する。
            var cfg = _document.Config;
            var current = new PageSectionSettings(cfg.PageHeader, cfg.Title, cfg.NumberStart, cfg.NumberStyle);
            var updated = PageSectionDialog.Edit(this, current, pageDisplayNumber: 1);
            if (updated == null) return;

            _document.Config = cfg with
            {
                PageHeader = updated.PageHeader,
                Title = updated.Title,
                NumberStart = updated.NumberStart,
                NumberStyle = updated.NumberStyle,
            };
            ReformatAndRender(cursorSeg, cursorOff);
            return;
        }

        // ダイアログの初期値は「このページで今実際に効いている設定」（継続的状態の解決結果）。
        using var handle = MomoFfi.RenderDocument(_document);
        if (handle == null) return;
        var effective = new PageSectionSettings(
            handle.PageHeaderEnabled(page),
            handle.PageTitle(page),
            handle.PageNumber(page),
            handle.PageNumberStyleAt(page));

        var edited = PageSectionDialog.Edit(this, effective, pageDisplayNumber: page + 1);
        if (edited == null) return;

        int firstLine = _view.FirstLineOfPage(page);
        var (segIdx, charOff) = _view.PhysicalToLogical(firstLine, 0);
        if (segIdx <= 0) return; // 想定外（2ページ目以降のはずが先頭セグメントに解決された）
        string marker = BuildPageSectionMarker(edited);
        int entryIdx = EntryIndexOfSegment(segIdx);

        if (charOff == 0)
        {
            // ページの先頭が既存のセグメント境界と一致する（強制／暗黙どちらの改ページでもよい）。
            // 直前に既存の PageBreakEntry（強制改ページ）があればその Marker を直接書き換える。
            // 無ければ（暗黙の改ページ）新規に PageBreakEntry を挿入する。
            if (entryIdx > 0 && _document.Entries[entryIdx - 1] is PageBreakEntry existing)
                existing.Marker = marker;
            else
                _document.Entries.Insert(entryIdx, new PageBreakEntry(marker));
            ReformatAndRender(cursorSeg, cursorOff);
        }
        else
        {
            // ページの先頭が折返し途中（既存のセグメント境界ではない）。その位置でテキストを
            // 分割し（段落終端ではなく強制的な改ページ分割として扱う）、分割点に
            // PageBreakEntry を挿入する。
            SplitAt(segIdx, charOff, firstParagraphEnd: false);
            int newEntryIdx = EntryIndexOfSegment(segIdx + 1);
            _document.Entries.Insert(newEntryIdx, new PageBreakEntry(marker));
            ReformatAndRender(segIdx + 1, 0);
        }
    }

    private static string BuildPageSectionMarker(PageSectionSettings s)
    {
        var style = s.NumberStyle == PageNumberStyle.Alternative ? "alt" : "standard";
        var show = s.PageHeader ? "true" : "false";
        return $"==== start={s.NumberStart} style={style} show_header={show} header={s.Title ?? ""}";
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

    private void MainForm_FormClosing(object? sender, FormClosingEventArgs e)
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
            if (IsOnProtectedLine()) return;
            var (si, off) = GetCursor();
            var seg = (TextSegment)_document.Entries[EntryIndexOfSegment(si)];
            seg.Text = seg.Text.Insert(off, ch.ToString());
            ReformatAndRender(si, off + 1);
            Earcon.PlayClick();
            return;
        }
        richTextBox.SelectedText = ch.ToString();
        richTextBox.Focus();
        Earcon.PlayClick();
    }

    /// <summary>
    /// テキストセグメント通し番号(si、Rust の segment_index と一致)から
    /// <see cref="_document"/>.Entries 内の実インデックスを引く。TextSegment だけを数えるため、
    /// PageBreakEntry の位置を返すことは構造的にない。
    /// </summary>
    private int EntryIndexOfSegment(int si)
    {
        int count = 0;
        for (int i = 0; i < _document.Entries.Count; i++)
        {
            if (_document.Entries[i] is TextSegment)
            {
                if (count == si) return i;
                count++;
            }
        }
        return -1;
    }

    // カーソル位置でセグメントを分割する。前半の終端種別を指定する。後半は元の終端種別を引き継ぐ。
    private void SplitAtCursor(bool firstParagraphEnd)
    {
        var (si, off) = GetCursor();
        SplitAt(si, off, firstParagraphEnd);
    }

    // 論理位置 (si, off) でセグメントを分割する。前半の終端種別を指定する。後半は元の終端種別を
    // 引き継ぐ。カーソル位置以外（ページ行設定の分割点計算など）から呼ぶ場合はこちらを直接使う。
    private void SplitAt(int si, int off, bool firstParagraphEnd)
    {
        int idx = EntryIndexOfSegment(si);
        var seg = (TextSegment)_document.Entries[idx];
        var before = seg.Text[..off];
        var after = seg.Text[off..];
        var second = new TextSegment(after, seg.ParagraphEnd);
        seg.Text = before;
        seg.ParagraphEnd = firstParagraphEnd;
        _document.Entries.Insert(idx + 1, second);
        ReformatAndRender(si + 1, 0);
    }

    // Enter: 段落区切り（空行で区切られる新しい論理段落）。
    private void InsertParagraphBreak()
    {
        if (_view != null)
        {
            if (IsOnProtectedLine()) return;
            SplitAtCursor(firstParagraphEnd: true);
            return;
        }
        richTextBox.SelectedText = "\n";
    }

    // Shift+Enter: 強制改行（同じ論理段落のまま、その位置で必ず行を分ける）。
    private void InsertHardBreak()
    {
        if (_view != null)
        {
            if (IsOnProtectedLine()) return;
            SplitAtCursor(firstParagraphEnd: false);
            return;
        }
        richTextBox.SelectedText = "\n";
    }

    // Ctrl+Enter: 改ページ。カーソルがセグメント境界（行頭/行末）にあれば、テキストは一切
    // 分割せず PageBreakEntry を挿入するだけ（余計な空段落を作らない）。境界外なら先に分割
    // してから挿入する。文書先頭では改ページを作れない（先頭が Break になる退化ケースを
    // そもそも発生させないため）。
    private void InsertPageBreak()
    {
        if (_view != null)
        {
            if (IsOnProtectedLine()) return;
            var (si, off) = GetCursor();
            int idx = EntryIndexOfSegment(si);
            var seg = (TextSegment)_document.Entries[idx];
            if (off == 0)
            {
                if (idx == 0) return; // 文書先頭では無効化
                _document.Entries.Insert(idx, new PageBreakEntry());
                ReformatAndRender(si, 0);
            }
            else if (off == seg.Text.Length)
            {
                _document.Entries.Insert(idx + 1, new PageBreakEntry());
                ReformatAndRender(si, off);
            }
            else
            {
                SplitAt(si, off, firstParagraphEnd: true);
                int newIdx = EntryIndexOfSegment(si + 1);
                _document.Entries.Insert(newIdx, new PageBreakEntry());
                ReformatAndRender(si + 1, 0);
            }
            return;
        }
        richTextBox.SelectedText = "\n";
    }

    /// <summary>
    /// カーソルが改ページ区切り行そのものの上にあれば、その PageBreakEntry を確認なしで
    /// 1回の操作で削除する（Backspace/Delete どちらでも同じ挙動）。区切りは見える・選択できる
    /// 独立したオブジェクトとして扱う設計のため、矢印キーで区切り行そのものに移動した状態からも
    /// 直接削除できる必要がある（隣接セグメント端からの削除だけでは、区切り行に乗った状態からは
    /// 削除できない）。区切り行上でなければ何もせず false を返す。
    /// </summary>
    private bool TryDeleteDividerAtCursor()
    {
        if (_view == null) return false;
        if (richTextBox.SelectionLength > 0) return false;   // 選択がある場合は DeleteSelection() に委ねる
        int flatLine = richTextBox.GetLineFromCharIndex(richTextBox.SelectionStart);
        if (flatLine < 0 || flatLine >= _view.PhysicalLineCount || !_view.IsDividerAt(flatLine)) return false;
        int entryIdx = _view.EntryIndexAt(flatLine);
        if (entryIdx < 0) return false;

        // 削除前に、区切りより前にある TextSegment の個数を数えておく。これは
        // 「区切りの直後にセグメントがあればその si」に一致する（si はエントリ順の
        // TextSegment 通し番号のため）。
        int precedingCount = 0;
        for (int i = 0; i < entryIdx; i++)
            if (_document.Entries[i] is TextSegment) precedingCount++;

        _document.Entries.RemoveAt(entryIdx);

        // カーソルは削除した区切りの直後のセグメント先頭へ。直後に無ければ直前のセグメント末尾へ。
        if (EntryIndexOfSegment(precedingCount) >= 0)
        {
            ReformatAndRender(precedingCount, 0);
        }
        else
        {
            int lastSi = precedingCount - 1;
            var lastSeg = (TextSegment)_document.Entries[EntryIndexOfSegment(lastSi)];
            ReformatAndRender(lastSi, lastSeg.Text.Length);
        }
        return true;
    }

    /// <summary>選択範囲（複数行にまたがるものを含む）がヘッダ行または改ページ区切り行の
    /// いずれかに触れているかどうかを返す。選択が無ければ（SelectionLength == 0）false。</summary>
    private bool SelectionTouchesProtectedLine()
    {
        if (_view == null || richTextBox.SelectionLength == 0) return false;
        int start = richTextBox.SelectionStart;
        int end = start + richTextBox.SelectionLength;
        int startLine = richTextBox.GetLineFromCharIndex(start);
        int endLine = richTextBox.GetLineFromCharIndex(end);
        for (int i = startLine; i <= endLine && i < _view.PhysicalLineCount; i++)
            if (_view.IsHeaderAt(i) || _view.IsDividerAt(i)) return true;
        return false;
    }

    /// <summary>選択範囲（複数セグメントにまたがるものを含む）を削除する。選択が保護行
    /// （ヘッダ行・改ページ区切り行）のいずれかにかかっている場合は何もせず false を返す
    /// （それらは1回の確定操作でのみ削除できる構造的に保護された行のため）。
    /// 呼び出し前に richTextBox.SelectionLength > 0 かつ _view != null であることを
    /// 確認しておくこと。成功時は ReformatAndRender まで完了させて true を返す。</summary>
    private bool DeleteSelection()
    {
        if (SelectionTouchesProtectedLine()) return false;

        int start = richTextBox.SelectionStart;
        int end = start + richTextBox.SelectionLength;
        int startLine = richTextBox.GetLineFromCharIndex(start);
        int endLine = richTextBox.GetLineFromCharIndex(end);

        var (siStart, offStart) = _view!.PhysicalToLogical(startLine, start - richTextBox.GetFirstCharIndexFromLine(startLine));
        var (siEnd, offEnd) = _view.PhysicalToLogical(endLine, end - richTextBox.GetFirstCharIndexFromLine(endLine));

        int idxStart = EntryIndexOfSegment(siStart);
        int idxEnd = EntryIndexOfSegment(siEnd);
        var first = (TextSegment)_document.Entries[idxStart];
        var last = (TextSegment)_document.Entries[idxEnd];

        offStart = Math.Min(offStart, first.Text.Length);
        offEnd = Math.Min(offEnd, last.Text.Length);

        first.Text = first.Text[..offStart] + last.Text[offEnd..];
        first.ParagraphEnd = last.ParagraphEnd;
        if (idxEnd > idxStart) _document.Entries.RemoveRange(idxStart + 1, idxEnd - idxStart);

        ReformatAndRender(siStart, offStart);
        return true;
    }

    private void SimulateBackspace()
    {
        if (_view != null)
        {
            if (TryDeleteDividerAtCursor()) return;
            if (IsOnProtectedLine()) return;
            if (richTextBox.SelectionLength > 0)
            {
                DeleteSelection();
                return;
            }
            var (csi, coff) = GetCursor();
            if (coff == 0)
            {
                int idx = EntryIndexOfSegment(csi);
                if (idx > 0 && _document.Entries[idx - 1] is PageBreakEntry)
                {
                    // セグメント先頭が改ページに隣接: 区切りエントリ自体を1回の操作で削除する
                    // （確認なし）。前後のテキストは自動結合しない（結合したければもう一度
                    // Backspace を押せば、以下の通常の結合ロジックがそのまま効く）。
                    _document.Entries.RemoveAt(idx - 1);
                    ReformatAndRender(csi, 0);
                    return;
                }
                // セグメント先頭: 直前のセグメントと結合
                if (csi == 0) return;
                var prev = (TextSegment)_document.Entries[idx - 1];
                var cur = (TextSegment)_document.Entries[idx];
                int prevLen = prev.Text.Length;
                prev.Text += cur.Text;
                prev.ParagraphEnd = cur.ParagraphEnd;
                _document.Entries.RemoveAt(idx);
                ReformatAndRender(csi - 1, prevLen);
                return;
            }
            // 1文字削除
            var s = (TextSegment)_document.Entries[EntryIndexOfSegment(csi)];
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
            if (TryDeleteDividerAtCursor()) return;
            if (IsOnProtectedLine()) return;
            if (richTextBox.SelectionLength > 0)
            {
                DeleteSelection();
                return;
            }
            var (si, off) = GetCursor();
            int idx = EntryIndexOfSegment(si);
            var seg = (TextSegment)_document.Entries[idx];
            if (off >= seg.Text.Length)
            {
                if (idx + 1 < _document.Entries.Count && _document.Entries[idx + 1] is PageBreakEntry)
                {
                    // セグメント末尾が改ページに隣接: 区切りエントリ自体を1回の操作で削除する
                    // （確認なし）。前後のテキストは自動結合しない。
                    _document.Entries.RemoveAt(idx + 1);
                    ReformatAndRender(si, off);
                    return;
                }
                // セグメント末尾: 次のセグメントと結合
                if (idx + 1 >= _document.Entries.Count) return;
                var next = (TextSegment)_document.Entries[idx + 1];
                seg.Text += next.Text;
                seg.ParagraphEnd = next.ParagraphEnd;
                _document.Entries.RemoveAt(idx + 1);
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

    private void SmartCut()
    {
        if (richTextBox.SelectionLength == 0) return;
        if (SelectionTouchesProtectedLine()) return; // 保護行にかかる範囲は切り取らない（コピーもしない）
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
        if (IsOnProtectedLine()) return;
        if (richTextBox.SelectionLength > 0 && !DeleteSelection()) return;
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
        if (_view == null || IsOnProtectedLine()) return;
        if (richTextBox.SelectionLength > 0 && !DeleteSelection()) return;
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

    /// <summary>点字データ（複数行）をカーソル位置に挿入する。_view != null かつ非保護行で呼ぶこと。</summary>
    private void InsertBrailleLines(string[] parts)
    {
        var (si, off) = GetCursor();
        int idx = EntryIndexOfSegment(si);
        var seg = (TextSegment)_document.Entries[idx];
        if (parts.Length == 1)
        {
            // セグメント内への単純挿入
            seg.Text = seg.Text.Insert(off, parts[0]);
            ReformatAndRender(si, off + parts[0].Length);
            return;
        }
        // 複数行: 現在のセグメントを分割しながら段落として挿入する。改ページを跨いだ分割は
        // 発生しない（PageBreakEntry は独立したエントリで、テキスト分割の対象にならない）。
        var tail = seg.Text[off..];
        var origEnd = seg.ParagraphEnd;
        seg.Text = seg.Text[..off] + parts[0];
        seg.ParagraphEnd = true;

        int insertAt = idx + 1;
        for (int i = 1; i < parts.Length - 1; i++)
            _document.Entries.Insert(insertAt++, new TextSegment(parts[i], true));
        _document.Entries.Insert(insertAt, new TextSegment(parts[^1] + tail, origEnd));
        ReformatAndRender(si + parts.Length - 1, parts[^1].Length);
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
