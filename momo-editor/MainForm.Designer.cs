namespace MomoEditor;

partial class MainForm
{
    private System.ComponentModel.IContainer components = null;

    protected override void Dispose(bool disposing)
    {
        if (disposing && (components != null))
            components.Dispose();
        base.Dispose(disposing);
    }

    private void InitializeComponent()
    {
        menuStrip = new MenuStrip();
        fileMenu = new ToolStripMenuItem();
        newMenuItem = new ToolStripMenuItem();
        openMenuItem = new ToolStripMenuItem();
        recentFilesMenuItem = new ToolStripMenuItem();
        saveMenuItem = new ToolStripMenuItem();
        saveAsMenuItem = new ToolStripMenuItem();
        pageSetupMenuItem = new ToolStripMenuItem();
        exitMenuItem = new ToolStripMenuItem();
        editMenu = new ToolStripMenuItem();
        undoMenuItem = new ToolStripMenuItem();
        redoMenuItem = new ToolStripMenuItem();
        cutMenuItem = new ToolStripMenuItem();
        copyMenuItem = new ToolStripMenuItem();
        pasteMenuItem = new ToolStripMenuItem();
        pasteConvertedMenuItem = new ToolStripMenuItem();
        hardBreakMenuItem = new ToolStripMenuItem();
        pageBreakMenuItem = new ToolStripMenuItem();
        selectAllMenuItem = new ToolStripMenuItem();
        brailleInputMenuItem = new ToolStripMenuItem();
        tableMenu = new ToolStripMenuItem();
        tableGrade1MenuItem = new ToolStripMenuItem();
        tableGrade2MenuItem = new ToolStripMenuItem();
        formatMenu = new ToolStripMenuItem();
        pageSectionMenuItem = new ToolStripMenuItem();
        viewMenu = new ToolStripMenuItem();
        guideMenuItem = new ToolStripMenuItem();
        speechGuideMenuItem = new ToolStripMenuItem();
        helpMenu = new ToolStripMenuItem();
        aboutMenuItem = new ToolStripMenuItem();
        richTextBox = new RichTextBox();
        guideStrip = new GuideStrip();
        statusStrip = new StatusStrip();
        statusLabel = new ToolStripStatusLabel();

        menuStrip.SuspendLayout();
        statusStrip.SuspendLayout();
        SuspendLayout();

        // menuStrip
        menuStrip.Items.AddRange(new ToolStripItem[] { fileMenu, editMenu, formatMenu, viewMenu, helpMenu });
        menuStrip.TabStop = false;

        // fileMenu
        fileMenu.Text = "ファイル(&F)";
        fileMenu.DropDownItems.AddRange(new ToolStripItem[] {
            newMenuItem,
            openMenuItem,
            recentFilesMenuItem,
            new ToolStripSeparator(),
            saveMenuItem,
            saveAsMenuItem,
            new ToolStripSeparator(),
            pageSetupMenuItem,
            new ToolStripSeparator(),
            exitMenuItem,
        });

        newMenuItem.Text = "新規(&N)";
        newMenuItem.ShortcutKeys = Keys.Control | Keys.N;
        newMenuItem.Click += NewMenuItem_Click;

        openMenuItem.Text = "開く(&O)...";
        openMenuItem.ShortcutKeys = Keys.Control | Keys.O;
        openMenuItem.Click += OpenMenuItem_Click;

        // 一覧は起動時・追加のたびに RebuildRecentFilesMenu() が動的に組み立てる。
        recentFilesMenuItem.Text = "最近使ったファイル(&R)";

        saveMenuItem.Text = "保存(&S)";
        saveMenuItem.ShortcutKeys = Keys.Control | Keys.S;
        saveMenuItem.Click += SaveMenuItem_Click;

        saveAsMenuItem.Text = "名前を付けて保存(&A)...";
        saveAsMenuItem.ShortcutKeys = Keys.Control | Keys.Shift | Keys.S;
        saveAsMenuItem.Click += SaveAsMenuItem_Click;

        // 文書全体の既定値（1行の文字数・1ページの行数・ページヘッダー・タイトル等）。
        // ページ単位の設定は書式(&O)メニューの「ページ行設定」で行う。
        pageSetupMenuItem.Text = "ページ設定(&G)...";
        pageSetupMenuItem.Click += PageSetupMenuItem_Click;

        exitMenuItem.Text = "終了(&X)";
        exitMenuItem.Click += ExitMenuItem_Click;

        // editMenu
        editMenu.Text = "編集(&E)";
        editMenu.DropDownItems.AddRange(new ToolStripItem[] {
            undoMenuItem,
            redoMenuItem,
            new ToolStripSeparator(),
            cutMenuItem,
            copyMenuItem,
            pasteMenuItem,
            pasteConvertedMenuItem,
            new ToolStripSeparator(),
            hardBreakMenuItem,
            pageBreakMenuItem,
            new ToolStripSeparator(),
            selectAllMenuItem,
            brailleInputMenuItem,
            new ToolStripSeparator(),
            tableMenu,
        });

        undoMenuItem.Text = "元に戻す(&U)";
        undoMenuItem.ShortcutKeys = Keys.Control | Keys.Z;
        undoMenuItem.Click += (_, _) => SmartUndo();

        redoMenuItem.Text = "やり直し(&Y)";
        redoMenuItem.ShortcutKeys = Keys.Control | Keys.Y;
        redoMenuItem.Click += (_, _) => SmartRedo();

        // Undo/Redo スタックが空のときはメニューをグレーアウトする。フォールバックモード
        // （_view == null）ではモデル側スタックが常に空なので、RichTextBox 標準の
        // CanUndo/CanRedo で判定を分岐する。
        editMenu.DropDownOpening += (_, _) =>
        {
            undoMenuItem.Enabled = _view == null ? richTextBox.CanUndo : _undoStack.Count > 0;
            redoMenuItem.Enabled = _view == null ? richTextBox.CanRedo : _redoStack.Count > 0;
        };

        cutMenuItem.Text = "切り取り(&T)";
        cutMenuItem.ShortcutKeys = Keys.Control | Keys.X;
        cutMenuItem.Click += (_, _) => SmartCut();

        copyMenuItem.Text = "コピー(&C)";
        copyMenuItem.ShortcutKeys = Keys.Control | Keys.C;
        copyMenuItem.Click += (_, _) => richTextBox.Copy();

        pasteMenuItem.Text = "貼り付け(&P)";
        pasteMenuItem.ShortcutKeys = Keys.Control | Keys.V;
        pasteMenuItem.Click += (_, _) => SmartPaste();

        pasteConvertedMenuItem.Text = "変換して貼り付け(&R)";
        pasteConvertedMenuItem.ShortcutKeys = Keys.Control | Keys.Shift | Keys.V;
        pasteConvertedMenuItem.Click += PasteConvertedMenuItem_Click;

        // 強制改行・改ページの挿入。実際のキー処理は OnKeyDown 側で行うため、
        // ここではショートカット表示のみ付ける（Enter 系は ShortcutKeys に設定できないため）。
        hardBreakMenuItem.Text = "強制改行(&L)";
        hardBreakMenuItem.ShortcutKeyDisplayString = "Shift+Enter";
        hardBreakMenuItem.Click += (_, _) => InsertHardBreak();

        pageBreakMenuItem.Text = "改ページ(&K)";
        pageBreakMenuItem.ShortcutKeyDisplayString = "Ctrl+Enter";
        pageBreakMenuItem.Click += (_, _) => InsertPageBreak();

        selectAllMenuItem.Text = "すべて選択(&A)";
        selectAllMenuItem.ShortcutKeys = Keys.Control | Keys.A;
        selectAllMenuItem.Click += (_, _) => richTextBox.SelectAll();

        brailleInputMenuItem.Text = "点字入力モード(&B)";
        brailleInputMenuItem.ShortcutKeys = Keys.Control | Keys.B;
        brailleInputMenuItem.CheckOnClick = true;
        brailleInputMenuItem.Checked = true;
        brailleInputMenuItem.Visible = false;
        brailleInputMenuItem.Click += BrailleInputMenuItem_Click;

        // 英語点字: 英字だけの行に使う UEB のグレードを切り替える（日本語行は常に日本語１級）。
        // 表示名（ラベル）は実行時に FFI から取得して設定する（MainForm コンストラクタ）。
        tableMenu.Text = "英語点字(&E)";
        tableMenu.DropDownItems.AddRange(new ToolStripItem[] {
            tableGrade1MenuItem,
            tableGrade2MenuItem,
        });

        tableGrade1MenuItem.Text = "&1 UEB English (Grade 1)";
        tableGrade1MenuItem.Checked = true; // 既定は Grade 1（無縮約）
        tableGrade1MenuItem.Click += TableGrade1MenuItem_Click;

        tableGrade2MenuItem.Text = "&2 UEB English (Grade 2)";
        tableGrade2MenuItem.Click += TableGrade2MenuItem_Click;

        // formatMenu
        formatMenu.Text = "書式(&O)";
        formatMenu.DropDownItems.AddRange(new ToolStripItem[] { pageSectionMenuItem });

        // 今表示されているページ以降に適用するヘッダー表示有無・タイトル・番号・番号スタイル。
        pageSectionMenuItem.Text = "ページ行設定(&G)...";
        pageSectionMenuItem.Click += PageSectionMenuItem_Click;

        // viewMenu
        viewMenu.Text = "表示(&V)";
        viewMenu.DropDownItems.AddRange(new ToolStripItem[] { guideMenuItem, speechGuideMenuItem });

        guideMenuItem.Text = "読みガイド(&G)";
        guideMenuItem.ShortcutKeys = Keys.Control | Keys.G;
        guideMenuItem.CheckOnClick = true;
        guideMenuItem.Checked = true;
        guideMenuItem.Click += GuideMenuItem_Click;

        // 読み上げガイド: カーソル移動に合わせて逆点訳の読みを UIA 通知で
        // スクリーンリーダーに発話させる（視覚の読みガイドとは独立に動く）。
        speechGuideMenuItem.Text = "読み上げガイド(&S)";
        speechGuideMenuItem.ShortcutKeys = Keys.Control | Keys.Shift | Keys.G;
        speechGuideMenuItem.CheckOnClick = true;
        speechGuideMenuItem.Checked = true;
        speechGuideMenuItem.Click += SpeechGuideMenuItem_Click;

        // helpMenu
        helpMenu.Text = "ヘルプ(&H)";
        helpMenu.DropDownItems.AddRange(new ToolStripItem[] { aboutMenuItem });

        aboutMenuItem.Text = "Momo Editor について(&A)...";
        aboutMenuItem.Click += AboutMenuItem_Click;

        // richTextBox
        richTextBox.Dock = DockStyle.Fill;
        // 点字グリフ（U+2800〜）を256/256完全カバーし、点字グリフ同士は1500unit等幅
        // （全体はプロポーショナルフォント）。OS/2 の Unicode Range も Braille Patterns を
        // 宣言済みなのでフォントリンクによる差し替えが起きない。インストーラで
        // システムインストールされる（momo-installer/fonts/）。無い環境（インストーラを
        // 経由しない実行等）へのフォールバックは MainForm コンストラクタで行う。
        richTextBox.Font = new Font("DejaVu Sans", 28F, FontStyle.Regular, GraphicsUnit.Point);
        richTextBox.ScrollBars = RichTextBoxScrollBars.Both;
        richTextBox.WordWrap = false;
        richTextBox.AcceptsTab = true;
        richTextBox.AccessibleName = "テキスト編集エリア";
        richTextBox.SelectionChanged += RichTextBox_SelectionChanged;
        richTextBox.TextChanged += RichTextBox_TextChanged;

        // guideStrip（編集面の下に読みガイドを表示する帯）
        guideStrip.Dock = DockStyle.Bottom;
        guideStrip.Font = new Font("MS Gothic", 16F, FontStyle.Regular, GraphicsUnit.Point);
        guideStrip.AccessibleName = "ReadingGuide"; // GuideStrip コンストラクタと同名（UIA 特定用の英語名）

        // statusStrip
        statusLabel.Text = "行: 1  セル: 0";
        statusLabel.Spring = true;
        statusLabel.TextAlign = ContentAlignment.MiddleLeft;
        statusStrip.Items.Add(statusLabel);

        // Form
        KeyPreview = true;
        AutoScaleMode = AutoScaleMode.Font;
        ClientSize = new Size(800, 480);
        // 追加順で下端のドッキングが決まる: statusStrip を最下段、guideStrip をその上に置く。
        Controls.Add(richTextBox);
        Controls.Add(statusStrip);
        Controls.Add(guideStrip);
        Controls.Add(menuStrip);
        MainMenuStrip = menuStrip;
        Text = "MomoEditor";
        FormClosing += MainForm_FormClosing;

        menuStrip.ResumeLayout(false);
        menuStrip.PerformLayout();
        statusStrip.ResumeLayout(false);
        statusStrip.PerformLayout();
        ResumeLayout(false);
        PerformLayout();
    }

    private MenuStrip menuStrip;
    private ToolStripMenuItem fileMenu;
    private ToolStripMenuItem newMenuItem;
    private ToolStripMenuItem openMenuItem;
    private ToolStripMenuItem recentFilesMenuItem;
    private ToolStripMenuItem saveMenuItem;
    private ToolStripMenuItem saveAsMenuItem;
    private ToolStripMenuItem pageSetupMenuItem;
    private ToolStripMenuItem exitMenuItem;
    private ToolStripMenuItem editMenu;
    private ToolStripMenuItem undoMenuItem;
    private ToolStripMenuItem redoMenuItem;
    private ToolStripMenuItem cutMenuItem;
    private ToolStripMenuItem copyMenuItem;
    private ToolStripMenuItem pasteMenuItem;
    private ToolStripMenuItem pasteConvertedMenuItem;
    private ToolStripMenuItem hardBreakMenuItem;
    private ToolStripMenuItem pageBreakMenuItem;
    private ToolStripMenuItem selectAllMenuItem;
    private ToolStripMenuItem formatMenu;
    private ToolStripMenuItem pageSectionMenuItem;
    private ToolStripMenuItem viewMenu;
    private ToolStripMenuItem guideMenuItem;
    private ToolStripMenuItem speechGuideMenuItem;
    private RichTextBox richTextBox;
    private GuideStrip guideStrip;
    private StatusStrip statusStrip;
    private ToolStripStatusLabel statusLabel;
    private ToolStripMenuItem brailleInputMenuItem;
    private ToolStripMenuItem tableMenu;
    private ToolStripMenuItem tableGrade1MenuItem;
    private ToolStripMenuItem tableGrade2MenuItem;
    private ToolStripMenuItem helpMenu;
    private ToolStripMenuItem aboutMenuItem;
}
