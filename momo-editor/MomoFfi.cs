using System.Runtime.InteropServices;

namespace MomoEditor;

/// <summary>
/// momors-ffi (momors_ffi.dll) への P/Invoke ラッパ。
///
/// 点字ドキュメントの読み書き・折返し・ページ分割・ヘッダ生成・各形式の符号化は
/// すべて Rust 側（momors-braille）に集約されている。エディタはセグメント列＋設定を
/// 保持し、ここを通して Rust にデータ生成を委ねる（ビルダー経由）。
/// </summary>
static class MomoFfi
{
    const string DllName = "momors_ffi";

    // 書き出し形式コード（Rust 側 momo_doc_write と一致させる）。
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
    static extern int momo_doc_number_start(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_doc_title_w(nint handle, nint buf, int bufLen);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_doc_paragraph_count(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_doc_line_count(nint handle, int para);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_doc_line_w(nint handle, int para, int line, nint buf, int bufLen);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    static extern bool momo_doc_line_logical_end(nint handle, int para, int line);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_doc_line_page_break_w(nint handle, int para, int line, nint buf, int bufLen);

    // ---- ビルダー（保存・描画用にドキュメントを組み立てる） ----

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_doc_builder_new(
        int lineWidth, int linesPerPage,
        [MarshalAs(UnmanagedType.I1)] bool pageHeader,
        int numberStart,
        [MarshalAs(UnmanagedType.LPWStr)] string? title);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_doc_builder_add_line(
        nint builder,
        [MarshalAs(UnmanagedType.LPWStr)] string content,
        [MarshalAs(UnmanagedType.I1)] bool logicalEnd,
        [MarshalAs(UnmanagedType.LPWStr)] string? pageBreak);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_doc_builder_build(nint builder);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_doc_builder_free(nint builder);

    // ---- 描画（印刷イメージ）と書き出し ----

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_doc_render(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_doc_write(nint handle, int format);

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

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_formatted_line_segment_index(nint handle, int page, int line);

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
        public int SegmentIndex(int page, int line) => momo_formatted_line_segment_index(_ptr, page, line);
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

    // ドキュメント（セグメント列）から Rust のビルダーでドキュメントハンドルを組む。
    // 成功時はハンドル、DLL 不在・失敗時は nint.Zero。呼び出し側が momo_doc_free する。
    static nint BuildDocHandle(BrailleDocument doc)
    {
        var cfg = doc.Config;
        var builder = momo_doc_builder_new(
            cfg.LineWidth, cfg.LinesPerPage, cfg.PageHeader,
            cfg.NumberStart, cfg.Title);
        if (builder == nint.Zero) return nint.Zero;
        foreach (var seg in doc.Segments)
            momo_doc_builder_add_line(builder, seg.Text, seg.ParagraphEnd, seg.PageBreakMarker);
        return momo_doc_builder_build(builder);
    }

    /// <summary>
    /// バイト列をドキュメントへ読み込み、セグメント列＋設定を持つ <see cref="BrailleDocument"/> へ展開する。
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
                        NumberStart = momo_doc_number_start(ptr),
                        Title = title.Length > 0 ? title : null,
                    },
                };
                int pc = momo_doc_paragraph_count(ptr);
                for (int p = 0; p < pc; p++)
                {
                    int lc = momo_doc_line_count(ptr, p);
                    for (int l = 0; l < lc; l++)
                    {
                        string text = ReadString((b, len) => momo_doc_line_w(ptr, p, l, b, len));
                        bool paragraphEnd = momo_doc_line_logical_end(ptr, p, l);
                        string marker = ReadString((b, len) => momo_doc_line_page_break_w(ptr, p, l, b, len));
                        doc.Segments.Add(new Segment(text, paragraphEnd, marker.Length > 0 ? marker : null));
                    }
                }
                if (doc.Segments.Count == 0)
                    doc.Segments.Add(new Segment("", true));
                return doc;
            }
            finally { momo_doc_free(ptr); }
        }
        catch (DllNotFoundException) { _dllAvailable = false; return null; }
        catch (EntryPointNotFoundException) { _dllAvailable = false; return null; }
    }

    /// <summary>
    /// ドキュメントを折返し・ページ分割して印刷イメージを返す。DLL 不在なら null。
    /// </summary>
    public static FormattedHandle? RenderDocument(BrailleDocument doc)
    {
        if (_dllAvailable == false) return null;
        try
        {
            var h = BuildDocHandle(doc);
            _dllAvailable = true;
            if (h == nint.Zero) return null;
            try
            {
                var f = momo_doc_render(h);
                return f == nint.Zero ? null : new FormattedHandle(f);
            }
            finally { momo_doc_free(h); }
        }
        catch (DllNotFoundException) { _dllAvailable = false; return null; }
        catch (EntryPointNotFoundException) { _dllAvailable = false; return null; }
    }

    /// <summary>
    /// ドキュメントを指定形式のバイト列へ書き出す。DLL 不在・形式不正なら null。
    /// </summary>
    public static byte[]? WriteDocument(BrailleDocument doc, int format)
    {
        if (_dllAvailable == false) return null;
        try
        {
            var h = BuildDocHandle(doc);
            _dllAvailable = true;
            if (h == nint.Zero) return null;
            try
            {
                var bytesPtr = momo_doc_write(h, format);
                if (bytesPtr == nint.Zero) return null;
                try
                {
                    int len = momo_bytes_len(bytesPtr);
                    if (len < 0) return null;
                    var arr = new byte[len];
                    if (len > 0) momo_bytes_copy(bytesPtr, arr);
                    return arr;
                }
                finally { momo_bytes_free(bytesPtr); }
            }
            finally { momo_doc_free(h); }
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
