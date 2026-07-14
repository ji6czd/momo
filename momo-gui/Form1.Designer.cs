namespace Momo;

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
        lblInput = new Label();
        txtInput = new TextBox();
        btnBrowseInput = new Button();
        lblOutput = new Label();
        txtOutput = new TextBox();
        btnBrowseOutput = new Button();
        lblOutputFormat = new Label();
        cmbOutputFormat = new ComboBox();
        chkEnglishGrade2 = new CheckBox();
        grpModel = new GroupBox();
        rdoSmall = new RadioButton();
        rdoMedium = new RadioButton();
        rdoLarge = new RadioButton();
        grpFormat = new GroupBox();
        lblLineWidth = new Label();
        numLineWidth = new NumericUpDown();
        lblLinesPerPage = new Label();
        numLinesPerPage = new NumericUpDown();
        lblTitle = new Label();
        txtTitle = new TextBox();
        btnOk = new Button();
        btnCancel = new Button();

        grpModel.SuspendLayout();
        grpFormat.SuspendLayout();
        ((System.ComponentModel.ISupportInitialize)numLineWidth).BeginInit();
        ((System.ComponentModel.ISupportInitialize)numLinesPerPage).BeginInit();
        SuspendLayout();

        // lblInput
        lblInput.AutoSize = true;
        lblInput.Location = new Point(12, 15);
        lblInput.Text = "入力ファイル:";

        // txtInput
        txtInput.Location = new Point(104, 12);
        txtInput.Size = new Size(316, 23);

        // btnBrowseInput
        btnBrowseInput.Location = new Point(426, 11);
        btnBrowseInput.Size = new Size(88, 25);
        btnBrowseInput.Text = "参照...";
        btnBrowseInput.Click += btnBrowseInput_Click;

        // lblOutput
        lblOutput.AutoSize = true;
        lblOutput.Location = new Point(12, 47);
        lblOutput.Text = "出力ファイル:";

        // txtOutput
        txtOutput.Location = new Point(104, 44);
        txtOutput.Size = new Size(316, 23);

        // btnBrowseOutput
        btnBrowseOutput.Location = new Point(426, 43);
        btnBrowseOutput.Size = new Size(88, 25);
        btnBrowseOutput.Text = "参照...";
        btnBrowseOutput.Click += btnBrowseOutput_Click;

        // lblOutputFormat
        lblOutputFormat.AutoSize = true;
        lblOutputFormat.Location = new Point(12, 79);
        lblOutputFormat.Text = "出力形式:";

        // cmbOutputFormat
        cmbOutputFormat.DropDownStyle = ComboBoxStyle.DropDownList;
        cmbOutputFormat.Location = new Point(104, 76);
        cmbOutputFormat.Size = new Size(316, 23);
        foreach (var f in OutputFormats)
            cmbOutputFormat.Items.Add(f.DisplayName);
        cmbOutputFormat.SelectedIndex = 1; // BASE
        cmbOutputFormat.SelectedIndexChanged += cmbOutputFormat_SelectedIndexChanged;

        // chkEnglishGrade2
        // 英語（UEB）の級を選ぶ。オフなら 1 級（無縮約）、オンなら 2 級（縮約あり）。
        chkEnglishGrade2.AutoSize = true;
        chkEnglishGrade2.Location = new Point(12, 110);
        chkEnglishGrade2.Text = "英語2級を使う(&2)";

        // grpModel
        grpModel.Location = new Point(12, 136);
        grpModel.Size = new Size(200, 54);
        grpModel.Text = "モデル";
        grpModel.Controls.Add(rdoSmall);
        grpModel.Controls.Add(rdoMedium);
        grpModel.Controls.Add(rdoLarge);

        // rdoSmall
        rdoSmall.AutoSize = true;
        rdoSmall.Location = new Point(10, 22);
        rdoSmall.Text = "Small";

        // rdoMedium
        rdoMedium.AutoSize = true;
        rdoMedium.Location = new Point(72, 22);
        rdoMedium.Text = "Medium";

        // rdoLarge
        rdoLarge.AutoSize = true;
        rdoLarge.Checked = true;
        rdoLarge.Location = new Point(148, 22);
        rdoLarge.Text = "Large";

        // grpFormat
        grpFormat.Location = new Point(220, 136);
        grpFormat.Size = new Size(294, 88);
        grpFormat.Text = "フォーマット";
        grpFormat.Controls.Add(lblLineWidth);
        grpFormat.Controls.Add(numLineWidth);
        grpFormat.Controls.Add(lblLinesPerPage);
        grpFormat.Controls.Add(numLinesPerPage);
        grpFormat.Controls.Add(lblTitle);
        grpFormat.Controls.Add(txtTitle);

        // lblLineWidth
        lblLineWidth.AutoSize = true;
        lblLineWidth.Location = new Point(8, 24);
        lblLineWidth.Text = "1行のマス数:";

        // numLineWidth
        numLineWidth.Location = new Point(80, 22);
        numLineWidth.Size = new Size(54, 23);
        numLineWidth.Minimum = 1;
        numLineWidth.Maximum = 100;
        numLineWidth.Value = 32;

        // lblLinesPerPage
        lblLinesPerPage.AutoSize = true;
        lblLinesPerPage.Location = new Point(144, 24);
        lblLinesPerPage.Text = "1ページの行数:";

        // numLinesPerPage
        numLinesPerPage.Location = new Point(232, 22);
        numLinesPerPage.Size = new Size(54, 23);
        numLinesPerPage.Minimum = 1;
        numLinesPerPage.Maximum = 100;
        numLinesPerPage.Value = 22;

        // lblTitle
        lblTitle.AutoSize = true;
        lblTitle.Location = new Point(8, 56);
        lblTitle.Text = "タイトル:";

        // txtTitle
        txtTitle.Location = new Point(62, 53);
        txtTitle.Size = new Size(222, 23);

        // btnOk
        btnOk.Location = new Point(346, 236);
        btnOk.Size = new Size(80, 28);
        btnOk.Text = "OK";
        btnOk.Click += btnOk_Click;

        // btnCancel
        btnCancel.DialogResult = DialogResult.Cancel;
        btnCancel.Location = new Point(434, 236);
        btnCancel.Size = new Size(80, 28);
        btnCancel.Text = "キャンセル";
        btnCancel.Click += (_, _) => Close();

        grpModel.ResumeLayout(false);
        grpModel.PerformLayout();
        grpFormat.ResumeLayout(false);
        grpFormat.PerformLayout();
        ((System.ComponentModel.ISupportInitialize)numLineWidth).EndInit();
        ((System.ComponentModel.ISupportInitialize)numLinesPerPage).EndInit();

        // Form
        AcceptButton = btnOk;
        CancelButton = btnCancel;
        ClientSize = new Size(526, 276);
        Controls.Add(lblInput);
        Controls.Add(txtInput);
        Controls.Add(btnBrowseInput);
        Controls.Add(lblOutput);
        Controls.Add(txtOutput);
        Controls.Add(btnBrowseOutput);
        Controls.Add(lblOutputFormat);
        Controls.Add(cmbOutputFormat);
        Controls.Add(chkEnglishGrade2);
        Controls.Add(grpModel);
        Controls.Add(grpFormat);
        Controls.Add(btnOk);
        Controls.Add(btnCancel);
        FormBorderStyle = FormBorderStyle.FixedDialog;
        MaximizeBox = false;
        MinimizeBox = false;
        StartPosition = FormStartPosition.CenterScreen;
        Text = "Momo 点字変換";

        ResumeLayout(false);
        PerformLayout();
    }

    private Label lblInput;
    private TextBox txtInput;
    private Button btnBrowseInput;
    private Label lblOutput;
    private TextBox txtOutput;
    private Button btnBrowseOutput;
    private Label lblOutputFormat;
    private ComboBox cmbOutputFormat;
    private CheckBox chkEnglishGrade2;
    private GroupBox grpModel;
    private RadioButton rdoSmall;
    private RadioButton rdoMedium;
    private RadioButton rdoLarge;
    private GroupBox grpFormat;
    private Label lblLineWidth;
    private NumericUpDown numLineWidth;
    private Label lblLinesPerPage;
    private NumericUpDown numLinesPerPage;
    private Label lblTitle;
    private TextBox txtTitle;
    private Button btnOk;
    private Button btnCancel;
}
