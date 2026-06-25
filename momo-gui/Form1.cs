namespace Momo;

using System.Runtime.InteropServices;

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
        1 => ".bse",
        _ => ".brf"
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
            1 => "BASEファイル (*.bse)|*.bse",
            _ => "点字テキスト (*.brf)|*.brf",
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
            int format = cmbOutputFormat.SelectedIndex switch { 0 => 0, 1 => 2, _ => 3 };
            int lw = (int)numLineWidth.Value, lpp = (int)numLinesPerPage.Value;
            string title = txtTitle.Text;

            var error = await Task.Run(() => RunConversion(modelPath, input, output, format, lw, lpp, title));

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

    private static string? RunConversion(
        string modelPath, string inputFile, string outputFile,
        int format, int lineWidth, int linesPerPage, string title)
    {
        var predictor = MomoNative.momo_predictor_new_w(modelPath);
        if (predictor == 0)
            return $"モデルの読み込みに失敗しました: {Path.GetFileName(modelPath)}";
        try
        {
            string text;
            try { text = File.ReadAllText(inputFile); }
            catch (Exception ex) { return $"ファイル読み込みエラー: {ex.Message}"; }

            var lines = text.Replace("\r\n", "\n").Replace("\r", "\n").Split('\n');

            return ConvertToBraille(predictor, lines, outputFile, lineWidth, linesPerPage, title, format);
        }
        finally
        {
            MomoNative.momo_predictor_free(predictor);
        }
    }

    private static string? ConvertToBraille(
        nint predictor, string[] lines, string outputFile,
        int lineWidth, int linesPerPage, string title, int format)
    {
        var brailleLines = new string[lines.Length];
        for (int i = 0; i < lines.Length; i++)
        {
            if (lines[i].Length == 0) { brailleLines[i] = ""; continue; }
            var brl = PredictText(predictor, lines[i], kana: false);
            if (brl is null) return "点字変換エラーが発生しました";
            brailleLines[i] = brl;
        }

        string? titleBraille = null;
        if (!string.IsNullOrWhiteSpace(title))
        {
            titleBraille = PredictText(predictor, title, kana: false);
            if (titleBraille is null) return "タイトル点字変換エラーが発生しました";
        }

        var paragraphs = string.Join("\n", brailleLines);
        var bytes = MomoNative.momo_doc_write_from_paragraphs(
            paragraphs, lineWidth, linesPerPage, true, 0, titleBraille, format);
        if (bytes == 0) return "ドキュメント生成に失敗しました";
        try
        {
            int len = MomoNative.momo_bytes_len(bytes);
            var buf = new byte[len];
            MomoNative.momo_bytes_copy(bytes, buf);
            try { File.WriteAllBytes(outputFile, buf); }
            catch (Exception ex) { return $"ファイル書き込みエラー: {ex.Message}"; }
        }
        finally { MomoNative.momo_bytes_free(bytes); }
        return null;
    }

    private static string? PredictText(nint predictor, string text, bool kana)
    {
        var pred = MomoNative.momo_predict_w(predictor, text);
        if (pred == 0) return null;
        try
        {
            int size = kana
                ? MomoNative.momo_prediction_kana_w(pred, nint.Zero, 0)
                : MomoNative.momo_prediction_braille_w(pred, nint.Zero, 0);
            if (size <= 1) return "";
            var buf = Marshal.AllocHGlobal(size * 2);
            try
            {
                if (kana) _ = MomoNative.momo_prediction_kana_w(pred, buf, size);
                else _ = MomoNative.momo_prediction_braille_w(pred, buf, size);
                return Marshal.PtrToStringUni(buf, size - 1);
            }
            finally { Marshal.FreeHGlobal(buf); }
        }
        finally { MomoNative.momo_prediction_free(pred); }
    }
}

internal static class MomoNative
{
    private const string Dll = "momors_ffi";

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
    public static extern nint momo_predictor_new_w([MarshalAs(UnmanagedType.LPWStr)] string model_path);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
    public static extern void momo_predictor_free(nint handle);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
    public static extern nint momo_predict_w(nint handle, [MarshalAs(UnmanagedType.LPWStr)] string src_text);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
    public static extern void momo_prediction_free(nint handle);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
    public static extern int momo_prediction_kana_w(nint handle, nint buf, int buf_len);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
    public static extern int momo_prediction_braille_w(nint handle, nint buf, int buf_len);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
    public static extern nint momo_doc_write_from_paragraphs(
        [MarshalAs(UnmanagedType.LPWStr)] string paragraphs,
        int line_width,
        int lines_per_page,
        [MarshalAs(UnmanagedType.U1)] bool page_header,
        int number_start,
        [MarshalAs(UnmanagedType.LPWStr)] string? title,
        int format);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
    public static extern int momo_bytes_len(nint b);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
    public static extern void momo_bytes_copy(nint b, [Out] byte[] out_buf);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
    public static extern void momo_bytes_free(nint b);
}
