namespace MomoEditor;

/// <summary>
/// 点字ドキュメントの整形設定（1 行の文字数・1 ページの行数・ページヘッダー・
/// 開始ページ番号・タイトル）を編集するダイアログ。
/// スクリーンリーダー利用を考慮し、各入力欄にラベルを関連付けてタブ順を整える。
/// </summary>
sealed class FormatterConfigDialog : Form
{
    private readonly NumericUpDown _lineWidth = new();
    private readonly NumericUpDown _linesPerPage = new();
    private readonly NumericUpDown _numberStart = new();
    private readonly CheckBox _pageHeader = new();
    private readonly TextBox _title = new();

    /// <summary>OK で閉じたときの編集結果。元の設定値を基に変更箇所だけ反映する。</summary>
    public FormatterConfig Result { get; private set; }

    /// <summary>
    /// 現在の設定でダイアログを開く。OK なら編集後の設定、キャンセルなら null を返す。
    /// </summary>
    public static FormatterConfig? Edit(IWin32Window owner, FormatterConfig current)
    {
        using var dialog = new FormatterConfigDialog(current);
        return dialog.ShowDialog(owner) == DialogResult.OK ? dialog.Result : null;
    }

    private FormatterConfigDialog(FormatterConfig config)
    {
        Result = config;
        BuildUi();
        LoadFrom(config);
    }

    private void BuildUi()
    {
        Text = "ページ設定";
        FormBorderStyle = FormBorderStyle.FixedDialog;
        StartPosition = FormStartPosition.CenterParent;
        MinimizeBox = false;
        MaximizeBox = false;
        ShowInTaskbar = false;
        AutoScaleMode = AutoScaleMode.Font;
        AutoSize = true;
        AutoSizeMode = AutoSizeMode.GrowAndShrink;
        Padding = new Padding(12);

        var layout = new TableLayoutPanel
        {
            ColumnCount = 2,
            AutoSize = true,
            AutoSizeMode = AutoSizeMode.GrowAndShrink,
            Dock = DockStyle.Fill,
        };
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));

        // 1 行の文字数
        _lineWidth.Minimum = 1;
        _lineWidth.Maximum = 100;
        _lineWidth.AccessibleName = "1 行の文字数";
        AddRow(layout, "1 行の文字数(&W):", _lineWidth);

        // 1 ページの行数
        _linesPerPage.Minimum = 1;
        _linesPerPage.Maximum = 100;
        _linesPerPage.AccessibleName = "1 ページの行数";
        AddRow(layout, "1 ページの行数(&L):", _linesPerPage);

        // 開始ページ番号
        _numberStart.Minimum = 1;
        _numberStart.Maximum = 9999;
        _numberStart.AccessibleName = "開始ページ番号";
        AddRow(layout, "開始ページ番号(&P):", _numberStart);

        // タイトル
        _title.Width = 200;
        _title.AccessibleName = "タイトル";
        AddRow(layout, "タイトル(&T):", _title);

        // ページヘッダーを付ける（チェックボックスは 2 列にまたがって配置）
        _pageHeader.Text = "ページヘッダーを付ける(&H)";
        _pageHeader.AutoSize = true;
        _pageHeader.Margin = new Padding(3, 6, 3, 6);
        int row = layout.RowCount;
        layout.RowCount = row + 1;
        layout.Controls.Add(_pageHeader, 0, row);
        layout.SetColumnSpan(_pageHeader, 2);

        // OK / キャンセル
        var okButton = new Button
        {
            Text = "OK",
            DialogResult = DialogResult.OK,
            AutoSize = true,
        };
        okButton.Click += (_, _) => Save();
        var cancelButton = new Button
        {
            Text = "キャンセル",
            DialogResult = DialogResult.Cancel,
            AutoSize = true,
        };

        var buttons = new FlowLayoutPanel
        {
            FlowDirection = FlowDirection.RightToLeft,
            Dock = DockStyle.Bottom,
            AutoSize = true,
            AutoSizeMode = AutoSizeMode.GrowAndShrink,
            Padding = new Padding(0, 8, 0, 0),
        };
        // RightToLeft フローでは「先に追加したコントロールが一番右」に並ぶため、
        // 見た目は OK（左）・キャンセル（右）になる。一方タブ順は TabIndex で決まるので、
        // 見た目と一致するよう OK を先（0）、キャンセルを後（1）に明示する。
        buttons.Controls.Add(cancelButton);
        buttons.Controls.Add(okButton);
        okButton.TabIndex = 0;
        cancelButton.TabIndex = 1;

        var root = new TableLayoutPanel
        {
            ColumnCount = 1,
            AutoSize = true,
            AutoSizeMode = AutoSizeMode.GrowAndShrink,
            Dock = DockStyle.Fill,
        };
        root.Controls.Add(layout, 0, 0);
        root.Controls.Add(buttons, 0, 1);
        Controls.Add(root);

        AcceptButton = okButton;
        CancelButton = cancelButton;
    }

    // ラベル＋入力欄の 1 行を追加する。ラベルを入力欄の直前のタブ順に置くことで、
    // スクリーンリーダーが入力欄の名前としてラベル文言を読み上げる。
    private static void AddRow(TableLayoutPanel layout, string labelText, Control input)
    {
        var label = new Label
        {
            Text = labelText,
            AutoSize = true,
            Anchor = AnchorStyles.Left,
            Margin = new Padding(3, 6, 6, 6),
        };
        input.Anchor = AnchorStyles.Left;
        input.Margin = new Padding(3, 3, 3, 3);

        int row = layout.RowCount;
        layout.RowCount = row + 1;
        layout.Controls.Add(label, 0, row);
        layout.Controls.Add(input, 1, row);
    }

    private void LoadFrom(FormatterConfig config)
    {
        _lineWidth.Value = Math.Clamp(config.LineWidth, (int)_lineWidth.Minimum, (int)_lineWidth.Maximum);
        _linesPerPage.Value = Math.Clamp(config.LinesPerPage, (int)_linesPerPage.Minimum, (int)_linesPerPage.Maximum);
        _numberStart.Value = Math.Clamp(config.NumberStart, (int)_numberStart.Minimum, (int)_numberStart.Maximum);
        _pageHeader.Checked = config.PageHeader;
        _title.Text = config.Title ?? "";
    }

    private void Save()
    {
        var title = _title.Text.Trim();
        Result = Result with
        {
            LineWidth = (int)_lineWidth.Value,
            LinesPerPage = (int)_linesPerPage.Value,
            NumberStart = (int)_numberStart.Value,
            PageHeader = _pageHeader.Checked,
            Title = title.Length > 0 ? title : null,
        };
    }
}
