using System.Runtime.InteropServices;

namespace MomoEditor;

/// <summary>
/// momors-ffi (momors_ffi.dll) への P/Invoke ラッパ。
///
/// 点字ドキュメントの読み書き・折返し・ページ分割・ヘッダ生成・各形式の符号化は
/// すべて Rust 側（momors-braille）に集約されている。エディタは論理段落＋設定だけを
/// 保持し、ここを通して Rust にデータ生成を委ねる。
/// </summary>
static class MomoFfi
{
    const string DllName = "momors_ffi";

    // 書き出し形式コード（Rust 側 momo_doc_write* と一致させる）。
    public const int FormatMbr = 0;
    public const int FormatBes = 1;
    public const int FormatBase = 2;
    public const int FormatBrl = 3;

    // 読み込み形式コード。
    public const int ReadAuto = 0;
    public const int ReadMbr = 1;
    public const int ReadBes = 2;

    // ---- 予測器（漢字かな交じり文 → 点字） ----

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_predictor_new_w([MarshalAs(UnmanagedType.LPWStr)] string modelPath);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_predictor_free(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_predict_w(nint predictor, [MarshalAs(UnmanagedType.LPWStr)] string srcText);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_prediction_free(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_prediction_braille_w(nint handle, nint buf, int bufLen);

    // ---- ドキュメント読み込み ----

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_doc_read(byte[] bytes, int len, int format);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_doc_free(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_doc_line_width(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_doc_lines_per_page(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    static extern bool momo_doc_page_header(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_doc_number_style(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_doc_title_w(nint handle, nint buf, int bufLen);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_doc_paragraph_count(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_doc_logical_text_w(nint handle, int para, nint buf, int bufLen);

    // ---- 描画（印刷イメージ） ----

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_doc_render_from_paragraphs(
        [MarshalAs(UnmanagedType.LPWStr)] string paragraphs,
        int lineWidth,
        int linesPerPage,
        [MarshalAs(UnmanagedType.I1)] bool pageHeader,
        int numberStyle,
        [MarshalAs(UnmanagedType.LPWStr)] string? title);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_formatted_free(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_formatted_page_count(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_formatted_line_count(nint handle, int page);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_formatted_line_w(nint handle, int page, int line, nint buf, int bufLen);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    static extern bool momo_formatted_line_is_header(nint handle, int page, int line);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    static extern bool momo_formatted_line_logical_end(nint handle, int page, int line);

    // ---- 書き出し（バイト列） ----

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_doc_write_from_paragraphs(
        [MarshalAs(UnmanagedType.LPWStr)] string paragraphs,
        int lineWidth,
        int linesPerPage,
        [MarshalAs(UnmanagedType.I1)] bool pageHeader,
        int numberStyle,
        [MarshalAs(UnmanagedType.LPWStr)] string? title,
        int format);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_bytes_len(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_bytes_copy(nint handle, [Out] byte[] outBuf);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_bytes_free(nint handle);

    // ---- 文字列取得ヘルパ ----

    delegate int BufWriter(nint buf, int bufLen);

    static string ReadString(BufWriter fn)
    {
        int needed = fn(nint.Zero, 0);
        if (needed <= 1) return "";
        nint buf = Marshal.AllocHGlobal(needed * 2);
        try
        {
            fn(buf, needed);
            return Marshal.PtrToStringUni(buf, needed - 1) ?? "";
        }
        finally { Marshal.FreeHGlobal(buf); }
    }

    static int NumberStyleInt(PageNumberStyle s) =>
        s == PageNumberStyle.Alternative ? 1 : 0;

    // ---- 描画ハンドル（印刷イメージ） ----

    public sealed class FormattedHandle : IDisposable
    {
        nint _ptr;
        internal FormattedHandle(nint ptr) => _ptr = ptr;

        public void Dispose()
        {
            if (_ptr != nint.Zero)
            {
                momo_formatted_free(_ptr);
                _ptr = nint.Zero;
            }
        }

        public int PageCount => momo_formatted_page_count(_ptr);
        public int LineCount(int page) => momo_formatted_line_count(_ptr, page);
        public string GetLine(int page, int line) =>
            ReadString((b, l) => momo_formatted_line_w(_ptr, page, line, b, l));
        public bool IsHeader(int page, int line) => momo_formatted_line_is_header(_ptr, page, line);
        public bool IsLogicalEnd(int page, int line) => momo_formatted_line_logical_end(_ptr, page, line);
    }

    public sealed class PredictorHandle : IDisposable
    {
        nint _ptr;
        internal PredictorHandle(nint ptr) => _ptr = ptr;

        public void Dispose()
        {
            if (_ptr != nint.Zero)
            {
                momo_predictor_free(_ptr);
                _ptr = nint.Zero;
            }
        }

        /// <summary>漢字かな交じり文を日本語点字に変換する。失敗時は null。</summary>
        public string? ToBraille(string text)
        {
            var pred = momo_predict_w(_ptr, text);
            if (pred == nint.Zero) return null;
            try
            {
                return ReadString((b, l) => momo_prediction_braille_w(pred, b, l));
            }
            finally { momo_prediction_free(pred); }
        }
    }

    // ---- 公開 API ----

    static bool? _dllAvailable;

    /// <summary>
    /// バイト列をドキュメントへ読み込み、論理段落＋設定を持つ <see cref="BrailleDocument"/> へ展開する。
    /// 失敗（破損・DLL 不在・不正データ）なら null。
    /// </summary>
    public static BrailleDocument? ReadDocument(byte[] bytes, int format)
    {
        if (_dllAvailable == false) return null;
        try
        {
            var ptr = momo_doc_read(bytes, bytes.Length, format);
            _dllAvailable = true;
            if (ptr == nint.Zero) return null;
            try
            {
                var title = ReadString((b, l) => momo_doc_title_w(ptr, b, l));
                var doc = new BrailleDocument
                {
                    Config = new FormatterConfig
                    {
                        LineWidth = momo_doc_line_width(ptr),
                        LinesPerPage = momo_doc_lines_per_page(ptr),
                        PageHeader = momo_doc_page_header(ptr),
                        NumberStyle = momo_doc_number_style(ptr) == 1
                            ? PageNumberStyle.Alternative
                            : PageNumberStyle.Standard,
                        Title = title.Length > 0 ? title : null,
                    },
                };
                int pc = momo_doc_paragraph_count(ptr);
                for (int p = 0; p < pc; p++)
                {
                    var text = ReadString((b, l) => momo_doc_logical_text_w(ptr, p, b, l));
                    doc.Paragraphs.Add([new PhysicalLine(text, true)]);
                }
                if (doc.Paragraphs.Count == 0)
                    doc.Paragraphs.Add([new PhysicalLine("", true)]);
                return doc;
            }
            finally { momo_doc_free(ptr); }
        }
        catch (DllNotFoundException) { _dllAvailable = false; return null; }
        catch (EntryPointNotFoundException) { _dllAvailable = false; return null; }
    }

    /// <summary>
    /// 論理段落（\n 区切り）を折返し・ページ分割して印刷イメージを返す。
    /// DLL 不在なら null。
    /// </summary>
    public static FormattedHandle? RenderFromParagraphs(string paragraphs, FormatterConfig cfg)
    {
        if (_dllAvailable == false) return null;
        try
        {
            var ptr = momo_doc_render_from_paragraphs(
                paragraphs, cfg.LineWidth, cfg.LinesPerPage, cfg.PageHeader,
                NumberStyleInt(cfg.NumberStyle), cfg.Title);
            _dllAvailable = true;
            return ptr == nint.Zero ? null : new FormattedHandle(ptr);
        }
        catch (DllNotFoundException) { _dllAvailable = false; return null; }
        catch (EntryPointNotFoundException) { _dllAvailable = false; return null; }
    }

    /// <summary>
    /// 論理段落（\n 区切り）を折返した上で指定形式のバイト列へ書き出す。
    /// DLL 不在・形式不正なら null。
    /// </summary>
    public static byte[]? WriteFromParagraphs(string paragraphs, FormatterConfig cfg, int format)
    {
        if (_dllAvailable == false) return null;
        try
        {
            var ptr = momo_doc_write_from_paragraphs(
                paragraphs, cfg.LineWidth, cfg.LinesPerPage, cfg.PageHeader,
                NumberStyleInt(cfg.NumberStyle), cfg.Title, format);
            _dllAvailable = true;
            if (ptr == nint.Zero) return null;
            try
            {
                int len = momo_bytes_len(ptr);
                if (len < 0) return null;
                var arr = new byte[len];
                if (len > 0) momo_bytes_copy(ptr, arr);
                return arr;
            }
            finally { momo_bytes_free(ptr); }
        }
        catch (DllNotFoundException) { _dllAvailable = false; return null; }
        catch (EntryPointNotFoundException) { _dllAvailable = false; return null; }
    }

    // 予測器はアプリ全体で 1 つだけ使い回す（モデル読み込みが重いため）。
    static PredictorHandle? _predictor;
    static bool _predictorLoaded;

    /// <summary>
    /// 漢字かな交じり文を点字に変換するための予測器を返す。
    /// モデルが見つからない／DLL 不在なら null。一度だけ読み込んで以降キャッシュする。
    /// </summary>
    public static PredictorHandle? GetPredictor()
    {
        if (_predictorLoaded) return _predictor;
        _predictorLoaded = true;
        if (_dllAvailable == false) return null;

        var modelPath = FindModelPath();
        if (modelPath == null) return null;

        try
        {
            var ptr = momo_predictor_new_w(modelPath);
            _dllAvailable = true;
            _predictor = ptr == nint.Zero ? null : new PredictorHandle(ptr);
        }
        catch (DllNotFoundException) { _dllAvailable = false; }
        catch (EntryPointNotFoundException) { _dllAvailable = false; }
        return _predictor;
    }

    // モデルファイル (.mbm) を探す。large→medium→small の順に優先する。
    // 探索場所は MOMO_DATASET_DIR 環境変数、次に実行ファイルと同じディレクトリ。
    static string? FindModelPath()
    {
        string[] models = ["basic_data_7.mbm", "basic_data_5.mbm", "basic_data_4.mbm"];
        var dirs = new List<string>();
        var env = Environment.GetEnvironmentVariable("MOMO_DATASET_DIR");
        if (!string.IsNullOrEmpty(env)) dirs.Add(env);
        dirs.Add(AppContext.BaseDirectory);

        foreach (var dir in dirs)
            foreach (var model in models)
            {
                var path = Path.Combine(dir, model);
                if (File.Exists(path)) return path;
            }
        return null;
    }
}
