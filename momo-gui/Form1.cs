namespace Momo;

public partial class MainForm : Form
{
    /// <summary>出力形式の選択肢。表示名・拡張子・ダイアログフィルタ・FFI 形式コードを1件にまとめる。</summary>
    /// <remarks>
    /// .brf は NABCC 大文字・NABCC 小文字・Unicode 点字の3件がある（拡張子は同じ）。
    /// Braille ASCII の規格は大文字のみだが、大文字のまま点字ディスプレイで読むと全セルに
    /// dot7 が付くため、その用途では小文字を選ぶ。Unicode は NABCC を使わず点字セルを
    /// UTF-8 のまま書く。
    /// .brl（ページの書式を持たない点字テキスト。1段落1行で、見出し・折返し・改ページなし）も
    /// 同じ3件を持つ。
    /// </remarks>
    private sealed record OutputFormatItem(string DisplayName, string Extension, string Filter, int FfiCode);

    private static readonly OutputFormatItem[] OutputFormats =
    [
        new("MOMO文書ファイル (.mbr)", ".mbr",
            "MOMO文書ファイル (*.mbr)|*.mbr", MomoFfi.FormatMbr),
        new("BASEファイル (.bse)", ".bse",
            "BASEファイル (*.bse)|*.bse", MomoFfi.FormatBase),
        new("点字テキストファイル 大文字NABCC (.brf)", ".brf",
            "点字テキストファイル (*.brf)|*.brf", MomoFfi.FormatBrf),
        new("点字テキストファイル 小文字NABCC (.brf)", ".brf",
            "点字テキストファイル (*.brf)|*.brf", MomoFfi.FormatBrfLower),
        new("点字テキストファイル Unicode点字 (.brf)", ".brf",
            "点字テキストファイル (*.brf)|*.brf", MomoFfi.FormatBrfUnicode),
        new("書式なし点字テキスト 大文字NABCC (.brl)", ".brl",
            "書式なし点字テキスト (*.brl)|*.brl", MomoFfi.FormatBrl),
        new("書式なし点字テキスト 小文字NABCC (.brl)", ".brl",
            "書式なし点字テキスト (*.brl)|*.brl", MomoFfi.FormatBrlLower),
        new("書式なし点字テキスト Unicode点字 (.brl)", ".brl",
            "書式なし点字テキスト (*.brl)|*.brl", MomoFfi.FormatBrlUnicode),
    ];

    private OutputFormatItem SelectedOutputFormat => OutputFormats[cmbOutputFormat.SelectedIndex];

    public MainForm()
    {
        InitializeComponent();
        // BERT は DLL が BERT 付きで、モデル（momo-bert/）が置いてあるときだけ選べる。
        rdoBert.Enabled = FindBertDir() is not null;
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

    private void cmbOutputFormat_SelectedIndexChanged(object sender, EventArgs e)
    {
        if (!string.IsNullOrEmpty(txtOutput.Text))
            txtOutput.Text = Path.ChangeExtension(txtOutput.Text, SelectedOutputFormat.Extension);
    }

    private void btnBrowseOutput_Click(object sender, EventArgs e)
    {
        var format = SelectedOutputFormat;
        using var dlg = new SaveFileDialog
        {
            Filter = format.Filter + "|すべてのファイル (*.*)|*.*",
            Title = "出力ファイルを選択",
            DefaultExt = format.Extension.TrimStart('.'),
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

        // BERT は Large の読みモデルに BERT 読みモードを足して使う（CLI の --bert と同じ）。
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
        string? bertDir = null;
        if (rdoBert.Checked)
        {
            bertDir = FindBertDir();
            if (bertDir is null)
            {
                MessageBox.Show(
                    $"BERT モデル（{MomoFfi.BertDirName} フォルダ）が見つかりません。\n" +
                    ".mbm と同じフォルダの下に置いてください。",
                    "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
        }

        btnOk.Enabled = false;
        Cursor = Cursors.WaitCursor;
        try
        {
            string input = txtInput.Text, output = txtOutput.Text;
            int format = SelectedOutputFormat.FfiCode;
            var config = new FormatterConfig
            {
                LineWidth = (int)numLineWidth.Value,
                LinesPerPage = (int)numLinesPerPage.Value,
                PageHeader = true,
            };
            string title = txtTitle.Text;
            bool englishGrade2 = chkEnglishGrade2.Checked;

            var error = await Task.Run(
                () => RunConversion(modelPath, bertDir, englishGrade2, input, output, format, config, title));

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

    /// <summary>モデルを探す場所。MOMO_DATASET_DIR 環境変数、次に実行ファイルと同じフォルダ。</summary>
    private static IEnumerable<string> DataDirs()
    {
        var env = Environment.GetEnvironmentVariable("MOMO_DATASET_DIR")
            ?? Environment.GetEnvironmentVariable("MOMO_DATASET_DIR", EnvironmentVariableTarget.Machine)
            ?? Environment.GetEnvironmentVariable("MOMO_DATASET_DIR", EnvironmentVariableTarget.User);
        if (env is not null) yield return env;
        yield return AppContext.BaseDirectory;
    }

    private static string? FindDataFile(string name) =>
        DataDirs().Select(dir => Path.Combine(dir, name)).FirstOrDefault(File.Exists);

    /// <summary>BERT が使えるなら、そのモデルのフォルダ（.mbm を探す場所の下の momo-bert）。</summary>
    private static string? FindBertDir() =>
        DataDirs().Select(dir => Path.Combine(dir, MomoFfi.BertDirName))
            .FirstOrDefault(MomoFfi.IsBertAvailable);

    /// <summary>
    /// テキストファイルを 1 行ずつ点訳し、指定形式で書き出す。成功なら null、失敗ならメッセージ。
    /// 日本語は１級固定、英語は <paramref name="englishGrade2"/> で UEB の級を切り替える。
    /// <paramref name="bertDir"/> を渡すと、読みに BERT 読みモードを使う。
    /// </summary>
    private static string? RunConversion(
        string modelPath, string? bertDir, bool englishGrade2, string inputFile, string outputFile,
        int format, FormatterConfig config, string title)
    {
        using var predictor = MomoFfi.CreatePredictor(modelPath);
        if (predictor is null)
            return $"モデルの読み込みに失敗しました: {Path.GetFileName(modelPath)}";
        if (bertDir is not null && !predictor.SetBert(bertDir))
            return $"BERT モデルの読み込みに失敗しました: {bertDir}";

        using var translator = MomoFfi.CreateTranslator(
            MomoFfi.TableJapaneseGrade1,
            englishGrade2 ? MomoFfi.TableEnglishUebGrade2 : MomoFfi.TableEnglishUebGrade1);
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
            doc.Entries.Add(new TextSegment(braille, paragraphEnd: true));
        }
        if (doc.Entries.Count == 0)
            doc.Entries.Add(new TextSegment("", paragraphEnd: true));

        // 折返し・ページ分割・ヘッダ生成・符号化はすべて Rust(momors-braille) 側が行う。
        var bytes = MomoFfi.WriteDocument(doc, format);
        if (bytes is null) return "ドキュメント生成に失敗しました";

        try { File.WriteAllBytes(outputFile, bytes); }
        catch (Exception ex) { return $"ファイル書き込みエラー: {ex.Message}"; }
        return null;
    }
}
