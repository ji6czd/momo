using Momo;

namespace MomoEditor;

/// <summary>
/// 「ページ行設定」ダイアログの結果。今表示されているページ以降に適用する、
/// ヘッダー表示有無・タイトル・開始ページ番号・ページ番号スタイル。
/// </summary>
sealed record PageSectionSettings(bool PageHeader, string? Title, int NumberStart, PageNumberStyle NumberStyle);

/// <summary>
/// 「今表示されているページ以降」に適用するページヘッダー関連設定を編集するダイアログ。
/// <see cref="FormatterConfigDialog"/>（文書全体の既定値）とほぼ同じ項目を持つが、
/// 1 行の文字数・1 ページの行数は文書全体にしか設定できないため対象外。
/// スクリーンリーダー利用を考慮し、何ページ目以降に効くかを冒頭のラベルで明示する。
/// </summary>
sealed class PageSectionDialog : Form
{
    private readonly NumericUpDown _numberStart = new();
    private readonly ComboBox _numberStyle = new() { DropDownStyle = ComboBoxStyle.DropDownList };
    private readonly CheckBox _pageHeader = new();
    private readonly TextBox _title = new();

    /// <summary>OK で閉じたときの編集結果。元の設定値を基に変更箇所だけ反映する。</summary>
    public PageSectionSettings Result { get; private set; }

    /// <summary>
    /// 現在の実効設定でダイアログを開く。<paramref name="pageDisplayNumber"/> は
    /// 冒頭の説明ラベルに表示する1始まりの表示用ページ番号。
    /// OK なら編集後の設定、キャンセルなら null を返す。
    /// </summary>
    public static PageSectionSettings? Edit(IWin32Window owner, PageSectionSettings current, int pageDisplayNumber)
    {
        using var dialog = new PageSectionDialog(current, pageDisplayNumber);
        return dialog.ShowDialog(owner) == DialogResult.OK ? dialog.Result : null;
    }

    private PageSectionDialog(PageSectionSettings config, int pageDisplayNumber)
    {
        Result = config;
        BuildUi(pageDisplayNumber);
        LoadFrom(config);
    }

    private void BuildUi(int pageDisplayNumber)
    {
        Text = "ページ行設定";
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

        // 何ページ目以降に適用されるかを明示する説明ラベル（スクリーンリーダーでも読み上げられる）。
        var scopeLabel = new Label
        {
            Text = $"現在表示中の {pageDisplayNumber} ページ目以降に適用されます。",
            AutoSize = true,
            Anchor = AnchorStyles.Left,
            Margin = new Padding(3, 3, 3, 9),
        };
        layout.RowCount = 1;
        layout.Controls.Add(scopeLabel, 0, 0);
        layout.SetColumnSpan(scopeLabel, 2);

        // 開始ページ番号
        _numberStart.Minimum = 1;
        _numberStart.Maximum = 9999;
        _numberStart.AccessibleName = "開始ページ番号";
        AddRow(layout, "開始ページ番号(&P):", _numberStart);

        // ページ番号スタイル
        _numberStyle.Items.Add("標準");
        _numberStyle.Items.Add("代替（下がり数字）");
        _numberStyle.AccessibleName = "ページ番号スタイル";
        AddRow(layout, "ページ番号スタイル(&Y):", _numberStyle);

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

    private void LoadFrom(PageSectionSettings config)
    {
        _numberStart.Value = Math.Clamp(config.NumberStart, (int)_numberStart.Minimum, (int)_numberStart.Maximum);
        _numberStyle.SelectedIndex = config.NumberStyle == PageNumberStyle.Alternative ? 1 : 0;
        _title.Text = config.Title ?? "";
        _pageHeader.Checked = config.PageHeader;
    }

    private void Save()
    {
        var title = _title.Text.Trim();
        Result = new PageSectionSettings(
            _pageHeader.Checked,
            title.Length > 0 ? title : null,
            (int)_numberStart.Value,
            _numberStyle.SelectedIndex == 1 ? PageNumberStyle.Alternative : PageNumberStyle.Standard);
    }
}
