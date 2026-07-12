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
        0 => ".mbr",
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
            0 => "MOMO文書ファイル (*.mbr)|*.mbr",
            _ => "BASEファイル (*.bse)|*.bse",
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

        var modelFile = rdoSmall.Checked ? "basic_data_4.mbm"
            : rdoMedium.Checked ? "basic_data_5.mbm"
            : "basic_data_7.mbm";
        var modelPath = FindDataFile(modelFile);
        if (modelPath is null)
        {
            MessageBox.Show(
                $"{modelFile} が見つかりません。\nmomors_ffi.dll と同じフォルダか MOMO_DATASET_DIR に置いてください。",
                "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        btnOk.Enabled = false;
        Cursor = Cursors.WaitCursor;
        try
        {
            string input = txtInput.Text, output = txtOutput.Text;
            int format = cmbOutputFormat.SelectedIndex switch
            {
                0 => MomoFfi.FormatMbr,
                _ => MomoFfi.FormatBase,
            };
            var config = new FormatterConfig
            {
                LineWidth = (int)numLineWidth.Value,
                LinesPerPage = (int)numLinesPerPage.Value,
                PageHeader = true,
            };
            string title = txtTitle.Text;
            bool englishGrade2 = chkEnglishGrade2.Checked;

            var error = await Task.Run(
                () => RunConversion(modelPath, englishGrade2, input, output, format, config, title));

            if (error is null)
                MessageBox.Show("変換が完了しました。", "完了",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
            else
                MessageBox.Show($"変換中にエラーが発生しました。\n{error}", "エラー",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            btnOk.Enabled = true;
            Cursor = Cursors.Default;
        }
    }

    private static string? FindDataFile(string name)
    {
        var dirs = new[]
        {
            Environment.GetEnvironmentVariable("MOMO_DATASET_DIR")
                ?? Environment.GetEnvironmentVariable("MOMO_DATASET_DIR", EnvironmentVariableTarget.Machine)
                ?? Environment.GetEnvironmentVariable("MOMO_DATASET_DIR", EnvironmentVariableTarget.User),
            AppContext.BaseDirectory,
        };
        foreach (var dir in dirs)
        {
            if (dir is null) continue;
            var path = Path.Combine(dir, name);
            if (File.Exists(path)) return path;
        }
        return null;
    }

    /// <summary>
    /// テキストファイルを 1 行ずつ点訳し、指定形式で書き出す。成功なら null、失敗ならメッセージ。
    /// 日本語は１級固定、英語は <paramref name="englishGrade2"/> で UEB の級を切り替える。
    /// </summary>
    private static string? RunConversion(
        string modelPath, bool englishGrade2, string inputFile, string outputFile,
        int format, FormatterConfig config, string title)
    {
        using var predictor = MomoFfi.CreatePredictor(modelPath);
        if (predictor is null)
            return $"モデルの読み込みに失敗しました: {Path.GetFileName(modelPath)}";

        using var translator = MomoFfi.CreateTranslator(
            MomoFfi.TableJapaneseGrade1,
            englishGrade2 ? MomoFfi.TableUebEnglishGrade2 : MomoFfi.TableUebEnglishGrade1);
        if (translator is null)
            return "点訳器の初期化に失敗しました";

        string text;
        try { text = File.ReadAllText(inputFile); }
        catch (Exception ex) { return $"ファイル読み込みエラー: {ex.Message}"; }

        // 点訳器が行ごとに日本語／英語を判定する。日本語行だけ予測器（漢字かな交じり文→かな）を通る。
        string? Translate(string line) =>
            line.Length == 0 ? "" : translator.ToBraille(line, predictor);

        string? brailleTitle = null;
        if (!string.IsNullOrWhiteSpace(title))
        {
            brailleTitle = Translate(title);
            if (brailleTitle is null) return "タイトル点字変換エラーが発生しました";
        }

        var doc = new BrailleDocument { Config = config with { Title = brailleTitle } };
        foreach (var line in text.Replace("\r\n", "\n").Replace("\r", "\n").Split('\n'))
        {
            var braille = Translate(line);
            if (braille is null) return "点字変換エラーが発生しました";
            doc.Segments.Add(new Segment(braille, paragraphEnd: true));
        }
        if (doc.Segments.Count == 0)
            doc.Segments.Add(new Segment("", paragraphEnd: true));

        // 折返し・ページ分割・ヘッダ生成・符号化はすべて Rust(momors-braille) 側が行う。
        var bytes = MomoFfi.WriteDocument(doc, format);
        if (bytes is null) return "ドキュメント生成に失敗しました";

        try { File.WriteAllBytes(outputFile, bytes); }
        catch (Exception ex) { return $"ファイル書き込みエラー: {ex.Message}"; }
        return null;
    }
}
