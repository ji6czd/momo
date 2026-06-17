namespace MomoEditor;

partial class Form1
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
        saveMenuItem = new ToolStripMenuItem();
        saveAsMenuItem = new ToolStripMenuItem();
        exitMenuItem = new ToolStripMenuItem();
        editMenu = new ToolStripMenuItem();
        undoMenuItem = new ToolStripMenuItem();
        cutMenuItem = new ToolStripMenuItem();
        copyMenuItem = new ToolStripMenuItem();
        pasteMenuItem = new ToolStripMenuItem();
        selectAllMenuItem = new ToolStripMenuItem();
        brailleInputMenuItem = new ToolStripMenuItem();
        richTextBox = new RichTextBox();
        statusStrip = new StatusStrip();
        statusLabel = new ToolStripStatusLabel();

        menuStrip.SuspendLayout();
        statusStrip.SuspendLayout();
        SuspendLayout();

        // menuStrip
        menuStrip.Items.AddRange(new ToolStripItem[] { fileMenu, editMenu });
        menuStrip.TabStop = false;

        // fileMenu
        fileMenu.Text = "ファイル(&F)";
        fileMenu.DropDownItems.AddRange(new ToolStripItem[] {
            newMenuItem,
            openMenuItem,
            new ToolStripSeparator(),
            saveMenuItem,
            saveAsMenuItem,
            new ToolStripSeparator(),
            exitMenuItem,
        });

        newMenuItem.Text = "新規(&N)";
        newMenuItem.ShortcutKeys = Keys.Control | Keys.N;
        newMenuItem.Click += NewMenuItem_Click;

        openMenuItem.Text = "開く(&O)...";
        openMenuItem.ShortcutKeys = Keys.Control | Keys.O;
        openMenuItem.Click += OpenMenuItem_Click;

        saveMenuItem.Text = "保存(&S)";
        saveMenuItem.ShortcutKeys = Keys.Control | Keys.S;
        saveMenuItem.Click += SaveMenuItem_Click;

        saveAsMenuItem.Text = "名前を付けて保存(&A)...";
        saveAsMenuItem.ShortcutKeys = Keys.Control | Keys.Shift | Keys.S;
        saveAsMenuItem.Click += SaveAsMenuItem_Click;

        exitMenuItem.Text = "終了(&X)";
        exitMenuItem.Click += ExitMenuItem_Click;

        // editMenu
        editMenu.Text = "編集(&E)";
        editMenu.DropDownItems.AddRange(new ToolStripItem[] {
            undoMenuItem,
            new ToolStripSeparator(),
            cutMenuItem,
            copyMenuItem,
            pasteMenuItem,
            new ToolStripSeparator(),
            selectAllMenuItem,
            brailleInputMenuItem,
        });

        undoMenuItem.Text = "元に戻す(&U)";
        undoMenuItem.ShortcutKeys = Keys.Control | Keys.Z;
        undoMenuItem.Click += (_, _) => SmartUndo();

        cutMenuItem.Text = "切り取り(&T)";
        cutMenuItem.ShortcutKeys = Keys.Control | Keys.X;
        cutMenuItem.Click += (_, _) => SmartCut();

        copyMenuItem.Text = "コピー(&C)";
        copyMenuItem.ShortcutKeys = Keys.Control | Keys.C;
        copyMenuItem.Click += (_, _) => richTextBox.Copy();

        pasteMenuItem.Text = "貼り付け(&P)";
        pasteMenuItem.ShortcutKeys = Keys.Control | Keys.V;
        pasteMenuItem.Click += (_, _) => SmartPaste();

        selectAllMenuItem.Text = "すべて選択(&A)";
        selectAllMenuItem.ShortcutKeys = Keys.Control | Keys.A;
        selectAllMenuItem.Click += (_, _) => richTextBox.SelectAll();

        brailleInputMenuItem.Text = "点字入力モード(&B)";
        brailleInputMenuItem.ShortcutKeys = Keys.Control | Keys.B;
        brailleInputMenuItem.CheckOnClick = true;
        brailleInputMenuItem.Checked = true;
        brailleInputMenuItem.Visible = false;
        brailleInputMenuItem.Click += BrailleInputMenuItem_Click;

        // richTextBox
        richTextBox.Dock = DockStyle.Fill;
        richTextBox.Font = new Font("Courier New", 13F, FontStyle.Regular, GraphicsUnit.Point);
        richTextBox.ScrollBars = RichTextBoxScrollBars.Both;
        richTextBox.WordWrap = false;
        richTextBox.AcceptsTab = true;
        richTextBox.AccessibleName = "テキスト編集エリア";
        richTextBox.SelectionChanged += RichTextBox_SelectionChanged;
        richTextBox.TextChanged += RichTextBox_TextChanged;

        // statusStrip
        statusLabel.Text = "行: 1  セル: 0";
        statusLabel.Spring = true;
        statusLabel.TextAlign = ContentAlignment.MiddleLeft;
        statusStrip.Items.Add(statusLabel);

        // Form
        KeyPreview = true;
        AutoScaleMode = AutoScaleMode.Font;
        ClientSize = new Size(800, 480);
        Controls.Add(richTextBox);
        Controls.Add(statusStrip);
        Controls.Add(menuStrip);
        MainMenuStrip = menuStrip;
        Text = "MomoEditor";
        FormClosing += Form1_FormClosing;

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
    private ToolStripMenuItem saveMenuItem;
    private ToolStripMenuItem saveAsMenuItem;
    private ToolStripMenuItem exitMenuItem;
    private ToolStripMenuItem editMenu;
    private ToolStripMenuItem undoMenuItem;
    private ToolStripMenuItem cutMenuItem;
    private ToolStripMenuItem copyMenuItem;
    private ToolStripMenuItem pasteMenuItem;
    private ToolStripMenuItem selectAllMenuItem;
    private RichTextBox richTextBox;
    private StatusStrip statusStrip;
    private ToolStripStatusLabel statusLabel;
    private ToolStripMenuItem brailleInputMenuItem;
}
