namespace Momo;

public partial class MainForm : Form
{
    public MainForm()
    {
        InitializeComponent();
    }

    private void btnBrowseInput_Click(object sender, EventArgs e)
    {
        using var dlg = new OpenFileDialog
        {
            Filter = "テキストファイル (*.txt)|*.txt|すべてのファイル (*.*)|*.*",
            Title = "入力ファイルを選択"
        };
        if (dlg.ShowDialog() == DialogResult.OK)
            txtInput.Text = dlg.FileName;
    }

    private string SelectedOutputExtension => cmbOutputFormat.SelectedIndex switch
    {
        0 => ".txt",
        2 => ".brf",
        _ => ".bse"
    };

    private void cmbOutputFormat_SelectedIndexChanged(object sender, EventArgs e)
    {
        if (!string.IsNullOrEmpty(txtOutput.Text))
            txtOutput.Text = Path.ChangeExtension(txtOutput.Text, SelectedOutputExtension);
    }

    private void btnBrowseOutput_Click(object sender, EventArgs e)
    {
        var filter = cmbOutputFormat.SelectedIndex switch
        {
            0 => "テキストファイル (*.txt)|*.txt",
            2 => "フォーマット済み点字ファイル (*.brf)|*.brf",
            _ => "BASE ファイル (*.bse)|*.bse"
        };
        using var dlg = new SaveFileDialog
        {
            Filter = filter + "|すべてのファイル (*.*)|*.*",
            Title = "出力ファイルを選択",
            DefaultExt = SelectedOutputExtension.TrimStart('.'),
            FileName = Path.GetFileNameWithoutExtension(txtInput.Text)
        };
        if (dlg.ShowDialog() == DialogResult.OK)
            txtOutput.Text = dlg.FileName;
    }

    private async void btnOk_Click(object sender, EventArgs e)
    {
        if (string.IsNullOrWhiteSpace(txtInput.Text))
        {
            MessageBox.Show("入力ファイルを指定してください。", "エラー",
                MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }
        if (string.IsNullOrWhiteSpace(txtOutput.Text))
        {
            MessageBox.Show("出力ファイルを指定してください。", "エラー",
                MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }

        var momoExe = FindMomoExe();
        if (momoExe is null)
        {
            MessageBox.Show("momo.exe が見つかりません。\nアプリケーションと同じフォルダか PATH 上に置いてください。",
                "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        btnOk.Enabled = false;
        Cursor = Cursors.WaitCursor;
        try
        {
            var model = rdoSmall.Checked ? "small" : rdoMedium.Checked ? "medium" : "large";

            var psi = new System.Diagnostics.ProcessStartInfo
            {
                FileName = momoExe,
                WorkingDirectory = Path.GetDirectoryName(momoExe)!,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardError = true
            };
            // インストール直後など現セッションに未反映の場合に備え、
            // レジストリから直接読んで子プロセスへ注入する
            var datasetDir =
                Environment.GetEnvironmentVariable("MOMO_DATASET_DIR") ??
                Environment.GetEnvironmentVariable("MOMO_DATASET_DIR", EnvironmentVariableTarget.Machine) ??
                Environment.GetEnvironmentVariable("MOMO_DATASET_DIR", EnvironmentVariableTarget.User);
            if (datasetDir != null)
                psi.Environment["MOMO_DATASET_DIR"] = datasetDir;
            psi.ArgumentList.Add("--braille");
            psi.ArgumentList.Add("--model");
            psi.ArgumentList.Add(model);
            psi.ArgumentList.Add("--input");
            psi.ArgumentList.Add(txtInput.Text);
            psi.ArgumentList.Add("--output");
            psi.ArgumentList.Add(txtOutput.Text);
            psi.ArgumentList.Add("--line-width");
            psi.ArgumentList.Add(((int)numLineWidth.Value).ToString());
            psi.ArgumentList.Add("--lines-per-page");
            psi.ArgumentList.Add(((int)numLinesPerPage.Value).ToString());
            if (!string.IsNullOrWhiteSpace(txtTitle.Text))
            {
                psi.ArgumentList.Add("--title");
                psi.ArgumentList.Add(txtTitle.Text);
            }

            using var process = System.Diagnostics.Process.Start(psi)!;
            var stderr = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            if (process.ExitCode == 0)
                MessageBox.Show("変換が完了しました。", "完了",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
            else
                MessageBox.Show($"変換中にエラーが発生しました。\n{stderr}", "エラー",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            btnOk.Enabled = true;
            Cursor = Cursors.Default;
        }
    }

    private static string? FindMomoExe()
    {
        var appDir = AppContext.BaseDirectory;
        var candidate = Path.Combine(appDir, "momo.exe");
        if (File.Exists(candidate)) return candidate;

        foreach (var dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(';'))
        {
            var path = Path.Combine(dir.Trim(), "momo.exe");
            if (File.Exists(path)) return path;
        }
        return null;
    }
}
