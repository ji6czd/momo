using System.Runtime.InteropServices;

namespace MomoEditor;

static class MomoFfi
{
    const string DllName = "momors_ffi";

    // ---- P/Invoke ----

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_predictor_new_w(
        [MarshalAs(UnmanagedType.LPWStr)] string modelPath);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_predictor_free(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_predict_w(
        nint predictor,
        [MarshalAs(UnmanagedType.LPWStr)] string srcText);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_prediction_free(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_prediction_braille_w(nint handle, nint buf, int bufLen);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_document_format_w(
        [MarshalAs(UnmanagedType.LPWStr)] string paragraphsText,
        int lineWidth,
        int linesPerPage,
        [MarshalAs(UnmanagedType.I1)] bool pageHeader,
        [MarshalAs(UnmanagedType.LPWStr)] string? title);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_document_free(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_document_page_count(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_document_line_count(nint handle, int page);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_document_line_w(nint handle, int page, int line, nint buf, int bufLen);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    static extern bool momo_document_line_is_header(nint handle, int page, int line);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    static extern bool momo_document_line_logical_end(nint handle, int page, int line);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_format_logical_line_w(
        [MarshalAs(UnmanagedType.LPWStr)] string text,
        int lineWidth);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_format_suffix_w(
        [MarshalAs(UnmanagedType.LPWStr)] string text,
        int lineWidth,
        int firstLineRemaining);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_physical_lines_free(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_physical_lines_count(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_physical_lines_get_w(nint handle, int index, nint buf, int bufLen);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    static extern bool momo_physical_lines_logical_end(nint handle, int index);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_bes_read(byte[] bytes, int len);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_bes_free(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_bes_line_width(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_bes_lines_per_page(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_bes_line_count(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_bes_line_get_w(nint handle, int index, nint buf, int bufLen);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    static extern bool momo_bes_line_logical_end(nint handle, int index);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    static extern bool momo_bes_line_page_break(nint handle, int index);

    // ---- Safe wrapper ----

    public sealed class DocumentHandle : IDisposable
    {
        nint _ptr;

        internal DocumentHandle(nint ptr) => _ptr = ptr;

        public void Dispose()
        {
            if (_ptr != nint.Zero)
            {
                momo_document_free(_ptr);
                _ptr = nint.Zero;
            }
        }

        public int PageCount => momo_document_page_count(_ptr);

        public int LineCount(int page) => momo_document_line_count(_ptr, page);

        public string GetLine(int page, int line)
        {
            int needed = momo_document_line_w(_ptr, page, line, nint.Zero, 0);
            if (needed <= 1) return "";
            nint buf = Marshal.AllocHGlobal(needed * 2);
            try
            {
                momo_document_line_w(_ptr, page, line, buf, needed);
                return Marshal.PtrToStringUni(buf, needed - 1) ?? "";
            }
            finally { Marshal.FreeHGlobal(buf); }
        }

        public bool IsHeader(int page, int line) =>
            momo_document_line_is_header(_ptr, page, line);

        public bool IsLogicalEnd(int page, int line) =>
            momo_document_line_logical_end(_ptr, page, line);
    }

    public sealed class PhysicalLinesHandle : IDisposable
    {
        nint _ptr;

        internal PhysicalLinesHandle(nint ptr) => _ptr = ptr;

        public void Dispose()
        {
            if (_ptr != nint.Zero)
            {
                momo_physical_lines_free(_ptr);
                _ptr = nint.Zero;
            }
        }

        public int Count => momo_physical_lines_count(_ptr);

        public string GetLine(int index)
        {
            int needed = momo_physical_lines_get_w(_ptr, index, nint.Zero, 0);
            if (needed <= 1) return "";
            nint buf = Marshal.AllocHGlobal(needed * 2);
            try
            {
                momo_physical_lines_get_w(_ptr, index, buf, needed);
                return Marshal.PtrToStringUni(buf, needed - 1) ?? "";
            }
            finally { Marshal.FreeHGlobal(buf); }
        }

        public bool IsLogicalEnd(int index) => momo_physical_lines_logical_end(_ptr, index);
    }

    public sealed class BesDocumentHandle : IDisposable
    {
        nint _ptr;

        internal BesDocumentHandle(nint ptr) => _ptr = ptr;

        public void Dispose()
        {
            if (_ptr != nint.Zero)
            {
                momo_bes_free(_ptr);
                _ptr = nint.Zero;
            }
        }

        public int LineWidth => momo_bes_line_width(_ptr);
        public int LinesPerPage => momo_bes_lines_per_page(_ptr);
        public int Count => momo_bes_line_count(_ptr);

        public string GetLine(int index)
        {
            int needed = momo_bes_line_get_w(_ptr, index, nint.Zero, 0);
            if (needed <= 1) return "";
            nint buf = Marshal.AllocHGlobal(needed * 2);
            try
            {
                momo_bes_line_get_w(_ptr, index, buf, needed);
                return Marshal.PtrToStringUni(buf, needed - 1) ?? "";
            }
            finally { Marshal.FreeHGlobal(buf); }
        }

        public bool IsLogicalEnd(int index) => momo_bes_line_logical_end(_ptr, index);
        public bool HasPageBreak(int index) => momo_bes_line_page_break(_ptr, index);
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

        /// <summary>
        /// 漢字かな交じり文を日本語点字に変換する。失敗時は null。
        /// </summary>
        public string? ToBraille(string text)
        {
            var pred = momo_predict_w(_ptr, text);
            if (pred == nint.Zero) return null;
            try
            {
                int needed = momo_prediction_braille_w(pred, nint.Zero, 0);
                if (needed <= 1) return "";
                nint buf = Marshal.AllocHGlobal(needed * 2);
                try
                {
                    momo_prediction_braille_w(pred, buf, needed);
                    return Marshal.PtrToStringUni(buf, needed - 1) ?? "";
                }
                finally { Marshal.FreeHGlobal(buf); }
            }
            finally { momo_prediction_free(pred); }
        }
    }

    // ---- Factory ----

    static bool? _dllAvailable;

    public static DocumentHandle? FormatDocument(BrailleDocument doc)
    {
        if (_dllAvailable == false) return null;
        try
        {
            var text = string.Join("\n",
            Enumerable.Range(0, doc.Paragraphs.Count).Select(i => doc.GetLogicalText(i)));
            var ptr = momo_document_format_w(
                text,
                doc.Config.LineWidth,
                doc.Config.LinesPerPage,
                doc.Config.PageHeader,
                doc.Config.Title);
            _dllAvailable = true;
            return ptr == nint.Zero ? null : new DocumentHandle(ptr);
        }
        catch (DllNotFoundException) { _dllAvailable = false; return null; }
        catch (EntryPointNotFoundException) { _dllAvailable = false; return null; }
    }

    public static PhysicalLinesHandle? FormatLogicalLine(string text, int lineWidth)
    {
        if (_dllAvailable == false) return null;
        try
        {
            var ptr = momo_format_logical_line_w(text, lineWidth);
            _dllAvailable = true;
            return ptr == nint.Zero ? null : new PhysicalLinesHandle(ptr);
        }
        catch (DllNotFoundException) { _dllAvailable = false; return null; }
        catch (EntryPointNotFoundException) { _dllAvailable = false; return null; }
    }

    public static PhysicalLinesHandle? FormatSuffix(string text, int lineWidth, int firstLineRemaining)
    {
        if (_dllAvailable == false) return null;
        try
        {
            var ptr = momo_format_suffix_w(text, lineWidth, firstLineRemaining);
            _dllAvailable = true;
            return ptr == nint.Zero ? null : new PhysicalLinesHandle(ptr);
        }
        catch (DllNotFoundException) { _dllAvailable = false; return null; }
        catch (EntryPointNotFoundException) { _dllAvailable = false; return null; }
    }

    // BES バイナリを読み込む。失敗（マジック不一致・破損・DLL 不在）なら null。
    public static BesDocumentHandle? ReadBes(byte[] bytes)
    {
        if (_dllAvailable == false) return null;
        try
        {
            var ptr = momo_bes_read(bytes, bytes.Length);
            _dllAvailable = true;
            return ptr == nint.Zero ? null : new BesDocumentHandle(ptr);
        }
        catch (DllNotFoundException) { _dllAvailable = false; return null; }
        catch (EntryPointNotFoundException) { _dllAvailable = false; return null; }
    }

    // 予測器はアプリ全体で 1 つだけ使い回す（モデル読み込みが重いため）。
    static PredictorHandle? _predictor;
    static bool _predictorLoaded;

    // 漢字かな交じり文を点字に変換するための予測器を返す。
    // モデルが見つからない／DLL 不在なら null。一度だけ読み込んで以降キャッシュする。
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
