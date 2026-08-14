using System.Runtime.InteropServices;

namespace Momo;

/// <summary>
/// momors-ffi (momors_ffi.dll) への P/Invoke ラッパ。
///
/// 点字ドキュメントの読み書き・折返し・ページ分割・ヘッダ生成・各形式の符号化は
/// すべて Rust 側（momors-braille）に集約されている。クライアントはセグメント列＋設定を
/// 保持し、ここを通して Rust にデータ生成を委ねる（ビルダー経由）。
///
/// 点訳は「点訳器（<see cref="BrailleTranslatorHandle"/>）＋予測器（<see cref="PredictorHandle"/>）」で行う。
/// 点訳器が行ごとに言語を判定し、日本語行だけ予測器（漢字かな交じり文→かな）を使う。
/// 結果は原文／読み／点字の3層と、その間の索引対応（<see cref="BrailleLine"/>）として返る。
/// </summary>
public static class MomoFfi
{
    const string DllName = "momors_ffi";

    // 形式コード（読み込み・書き出しで共通。Rust 側 momo_doc_read / momo_doc_write と一致させる）。
    public const int FormatMbr  = 0;   // 読み書き両対応
    public const int FormatBes  = 1;   // 読み書き両対応
    public const int FormatBet  = 2;   // 読み込みのみ
    public const int FormatBase = 3;   // 書き出しのみ
    public const int FormatBrf  = 4;   // 書き出しのみ（NABCC 大文字。Braille ASCII の規格どおり）
    public const int FormatBrfLower = 5; // 書き出しのみ（NABCC 小文字。点字ディスプレイで直接読む用）

    // 読み込み時の別名（可読性のため）。
    public const int ReadMbr = FormatMbr;
    public const int ReadBes = FormatBes;
    public const int ReadBet = FormatBet;

    // 組み込みテーブル名（TOML の [metadata].name と一致させる）。
    public const string TableJapaneseGrade1 = "japanese_grade1";
    public const string TableJapaneseNoConversion = "japanese_no_conversion";
    public const string TableEnglishUebGrade2 = "english_ueb_grade2";
    public const string TableEnglishUebGrade1 = "english_ueb_grade1";

    // テーブル未定義文字の扱い（Rust 側 UnknownCharPolicy と一致させる）。
    public const int PolicySpace = 0;
    public const int PolicyPassThrough = 1;

    // ---- 予測器（漢字かな交じり文 → かな） ----

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_predictor_new_w([MarshalAs(UnmanagedType.LPWStr)] string modelPath);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_predictor_free(nint handle);

    // ---- 変換テーブル ----

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_table_embedded_count();

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_table_embedded_name_w(int idx, nint buf, int bufLen);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_table_embedded_displayname_w(int idx, nint buf, int bufLen);

    // ---- 言語別変換器（点訳器の構築部品。ハンドルは点訳器に消費される） ----

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_japanese_translator_from_file_w(
        [MarshalAs(UnmanagedType.LPWStr)] string path);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_japanese_translator_set_unknown_char_policy(nint handle, int policy);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_japanese_translator_free(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_english_translator_from_file_w(
        [MarshalAs(UnmanagedType.LPWStr)] string path);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_english_translator_set_unknown_char_policy(nint handle, int policy);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_english_translator_free(nint handle);

    // ---- 点訳器（入口） ----

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_braille_translator_from_embedded();

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_braille_translator_from_names_w(
        [MarshalAs(UnmanagedType.LPWStr)] string japaneseName,
        [MarshalAs(UnmanagedType.LPWStr)] string? englishName);

    // 日本語・英語変換器のハンドルを消費する（呼び出し後、両ハンドルは無効）。
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_braille_translator_new(nint japanese, nint english);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_braille_translator_translate_w(
        nint handle, [MarshalAs(UnmanagedType.LPWStr)] string line, nint predictor);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_braille_translator_translate_japanese_w(
        nint handle, [MarshalAs(UnmanagedType.LPWStr)] string line, nint predictor);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_braille_translator_translate_english_w(
        nint handle, [MarshalAs(UnmanagedType.LPWStr)] string line);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_braille_translator_free(nint handle);

    // ---- 点訳結果（原文／読み／点字の3層） ----

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_braille_result_language(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_braille_result_source_text_w(nint handle, nint buf, int bufLen);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_braille_result_reading_text_w(nint handle, nint buf, int bufLen);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_braille_result_braille_text_w(nint handle, nint buf, int bufLen);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_braille_result_source_char_count(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_braille_result_reading_char_count(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_braille_result_braille_char_count(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_braille_result_reading_to_source(nint handle, [Out] int[] outIdx);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_braille_result_braille_to_source(nint handle, [Out] int[] outIdx);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_braille_result_source_to_reading(
        nint handle, [Out] int[] rowPtr, [Out] int[] colIdx);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_braille_result_source_to_braille(
        nint handle, [Out] int[] rowPtr, [Out] int[] colIdx);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    static extern bool momo_braille_result_has_prediction(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_braille_result_free(nint handle);

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
    static extern int momo_doc_number_style(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_doc_title_w(nint handle, nint buf, int bufLen);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_doc_entry_count(nint handle);

    /// <summary>0=テキスト段落, 1=改ページ, 無効なら -1。</summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_doc_entry_kind(nint handle, int entry);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_doc_line_count(nint handle, int entry);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_doc_line_w(nint handle, int entry, int line, nint buf, int bufLen);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    static extern bool momo_doc_line_logical_end(nint handle, int entry, int line);

    /// <summary>改ページエントリのマーカー文字列。テキスト段落エントリ/無効なら -1。</summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_doc_entry_page_break_w(nint handle, int entry, nint buf, int bufLen);

    // ---- ビルダー（保存・描画用にドキュメントを組み立てる） ----

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_doc_builder_new(
        int lineWidth, int linesPerPage,
        [MarshalAs(UnmanagedType.I1)] bool pageHeader,
        int numberStart,
        int numberStyle,
        [MarshalAs(UnmanagedType.LPWStr)] string? title);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_doc_builder_add_line(
        nint builder,
        [MarshalAs(UnmanagedType.LPWStr)] string content,
        [MarshalAs(UnmanagedType.I1)] bool logicalEnd);

    /// <summary>marker: NULL/空でプレーンな改ページ。</summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_doc_builder_add_page_break(
        nint builder,
        [MarshalAs(UnmanagedType.LPWStr)] string? marker);

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
    [return: MarshalAs(UnmanagedType.I1)]
    static extern bool momo_formatted_page_header_enabled(nint handle, int page);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_formatted_page_title_w(nint handle, int page, nint buf, int bufLen);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_formatted_page_number(nint handle, int page);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_formatted_page_number_style(nint handle, int page);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_bytes_len(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_bytes_copy(nint handle, [Out] byte[] outBuf);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_bytes_free(nint handle);

    // ---- 逆点訳（点字 → かな表層） ----

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_back_translator_new();

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_back_translator_free(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern nint momo_back_translate_w(nint translator, [MarshalAs(UnmanagedType.LPWStr)] string braille);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_back_trans_free(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_back_trans_text_w(nint handle, nint buf, int bufLen);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_back_trans_segment_count(nint handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern void momo_back_trans_cell_bounds(nint handle, [Out] int[] outStart, [Out] int[] outEnd);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    static extern int momo_back_trans_segment_text_w(nint handle, int idx, nint buf, int bufLen);

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

    // ---- 点訳の型 ----

    /// <summary>点訳経路。行ごとに点訳器が判定する。</summary>
    public enum BrailleLanguage { Japanese = 0, English = 1 }

    /// <summary>
    /// 1 行の点訳結果。原文・読み・点字の3層と、その間の索引対応を持つ。
    /// 索引はすべてコードポイント単位（UTF-16 ではない）。
    /// 英語行では読み＝原文（恒等）で、<see cref="HasPrediction"/> は false。
    /// </summary>
    /// <param name="Language">この行がどちらの経路で点訳されたか。</param>
    /// <param name="Source">原文。</param>
    /// <param name="Reading">読み（日本語＝かな、英語＝原文）。</param>
    /// <param name="Braille">点字。</param>
    /// <param name="ReadingToSource">読みの各文字が原文の何文字目に由来するか。</param>
    /// <param name="BrailleToSource">点字の各セルが原文の何文字目に由来するか。</param>
    /// <param name="SourceToReading">原文の各文字が読みのどの文字群になったか（0個以上）。</param>
    /// <param name="SourceToBraille">原文の各文字が点字のどのセル群になったか（0個以上）。</param>
    /// <param name="HasPrediction">日本語予測（かな・確信度）が付随するか。</param>
    public sealed record BrailleLine(
        BrailleLanguage Language,
        string Source,
        string Reading,
        string Braille,
        IReadOnlyList<int> ReadingToSource,
        IReadOnlyList<int> BrailleToSource,
        IReadOnlyList<IReadOnlyList<int>> SourceToReading,
        IReadOnlyList<IReadOnlyList<int>> SourceToBraille,
        bool HasPrediction);

    /// <summary>予測器（.mbm モデル）。読み込みが重いため使い回すこと。</summary>
    public sealed class PredictorHandle : IDisposable
    {
        internal nint Ptr;
        internal PredictorHandle(nint ptr) => Ptr = ptr;

        public void Dispose()
        {
            if (Ptr != nint.Zero)
            {
                momo_predictor_free(Ptr);
                Ptr = nint.Zero;
            }
        }
    }

    /// <summary>
    /// 点訳器。行ごとに日本語／英語を判定し、日本語行だけ予測器を使う。
    /// 生成は <see cref="CreateTranslator(string, string?)"/> ほかのファクトリから行う。
    /// </summary>
    public sealed class BrailleTranslatorHandle : IDisposable
    {
        nint _ptr;
        internal BrailleTranslatorHandle(nint ptr) => _ptr = ptr;

        public void Dispose()
        {
            if (_ptr != nint.Zero)
            {
                momo_braille_translator_free(_ptr);
                _ptr = nint.Zero;
            }
        }

        /// <summary>1 行を言語判定して点訳する。失敗時は null。</summary>
        public BrailleLine? Translate(string line, PredictorHandle predictor) =>
            Materialize(momo_braille_translator_translate_w(_ptr, line, predictor.Ptr));

        /// <summary>1 行を必ず日本語として点訳する（英字は外字符＋無縮約）。失敗時は null。</summary>
        public BrailleLine? TranslateJapanese(string line, PredictorHandle predictor) =>
            Materialize(momo_braille_translator_translate_japanese_w(_ptr, line, predictor.Ptr));

        /// <summary>
        /// 1 行を必ず英語（UEB）として点訳する。予測器は不要。
        /// 英語エンジンを持たない点訳器では null。
        /// </summary>
        public BrailleLine? TranslateEnglish(string line) =>
            Materialize(momo_braille_translator_translate_english_w(_ptr, line));

        /// <summary>点字テキストだけが要るときの簡便版。失敗時は null。</summary>
        public string? ToBraille(string line, PredictorHandle predictor) =>
            Translate(line, predictor)?.Braille;
    }

    // 点訳結果ハンドルを C# 側のレコードへ写し取り、ハンドルは解放する。
    // 呼び出し側にネイティブハンドルを持たせないことで、寿命管理を1か所に閉じ込める。
    static BrailleLine? Materialize(nint h)
    {
        if (h == nint.Zero) return null;
        try
        {
            var language = (BrailleLanguage)momo_braille_result_language(h);
            string source  = ReadString((b, l) => momo_braille_result_source_text_w(h, b, l));
            string reading = ReadString((b, l) => momo_braille_result_reading_text_w(h, b, l));
            string braille = ReadString((b, l) => momo_braille_result_braille_text_w(h, b, l));

            int srcCount = Math.Max(0, momo_braille_result_source_char_count(h));
            int readCount = Math.Max(0, momo_braille_result_reading_char_count(h));
            int brlCount = Math.Max(0, momo_braille_result_braille_char_count(h));

            var readingToSource = new int[readCount];
            if (readCount > 0) momo_braille_result_reading_to_source(h, readingToSource);
            var brailleToSource = new int[brlCount];
            if (brlCount > 0) momo_braille_result_braille_to_source(h, brailleToSource);

            var sourceToReading = ReadCsr(srcCount, readCount,
                (rows, cols) => momo_braille_result_source_to_reading(h, rows, cols));
            var sourceToBraille = ReadCsr(srcCount, brlCount,
                (rows, cols) => momo_braille_result_source_to_braille(h, rows, cols));

            return new BrailleLine(
                language, source, reading, braille,
                readingToSource, brailleToSource,
                sourceToReading, sourceToBraille,
                momo_braille_result_has_prediction(h));
        }
        finally { momo_braille_result_free(h); }
    }

    // Rust 側の CSR（row_ptr: 行数+1 要素、col_idx: 非ゼロ要素数）を行ごとのリストへ展開する。
    static IReadOnlyList<IReadOnlyList<int>> ReadCsr(int rowCount, int colCount, Action<int[], int[]> fill)
    {
        if (rowCount <= 0) return [];
        var rowPtr = new int[rowCount + 1];
        var colIdx = new int[colCount];
        fill(rowPtr, colIdx);

        var rows = new List<IReadOnlyList<int>>(rowCount);
        for (int i = 0; i < rowCount; i++)
        {
            int start = rowPtr[i];
            int end = rowPtr[i + 1];
            if (start < 0 || end < start || end > colCount) { rows.Add([]); continue; }
            rows.Add(colIdx[start..end]);
        }
        return rows;
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

        // ページ単位で実効しているヘッダ関連の継続的状態（セクションごとの上書きの解決結果）。
        // 「今表示されているページ以降に適用する設定」ダイアログの初期値に使う。
        public bool PageHeaderEnabled(int page) => momo_formatted_page_header_enabled(_ptr, page);
        public string? PageTitle(int page)
        {
            var title = ReadString((b, l) => momo_formatted_page_title_w(_ptr, page, b, l));
            return title.Length > 0 ? title : null;
        }
        public int PageNumber(int page) => momo_formatted_page_number(_ptr, page);
        public PageNumberStyle PageNumberStyleAt(int page) =>
            (PageNumberStyle)Math.Max(0, momo_formatted_page_number_style(_ptr, page));
    }

    // ---- 公開 API ----

    static bool? _dllAvailable;

    static nint BuildDocHandle(BrailleDocument doc)
    {
        var cfg = doc.Config;
        var builder = momo_doc_builder_new(
            cfg.LineWidth, cfg.LinesPerPage, cfg.PageHeader,
            cfg.NumberStart, (int)cfg.NumberStyle, cfg.Title);
        if (builder == nint.Zero) return nint.Zero;
        foreach (var entry in doc.Entries)
        {
            if (entry is TextSegment seg)
                momo_doc_builder_add_line(builder, seg.Text, seg.ParagraphEnd);
            else if (entry is PageBreakEntry pb)
                momo_doc_builder_add_page_break(builder, pb.Marker);
        }
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
                        NumberStyle = (PageNumberStyle)Math.Max(0, momo_doc_number_style(ptr)),
                        Title = title.Length > 0 ? title : null,
                    },
                };
                int ec = momo_doc_entry_count(ptr);
                for (int e = 0; e < ec; e++)
                {
                    if (momo_doc_entry_kind(ptr, e) == 1)
                    {
                        int entry = e;
                        var marker = ReadString((b, len) => momo_doc_entry_page_break_w(ptr, entry, b, len));
                        doc.Entries.Add(new PageBreakEntry(marker.Length > 0 ? marker : "===="));
                        continue;
                    }
                    int lc = momo_doc_line_count(ptr, e);
                    for (int l = 0; l < lc; l++)
                    {
                        int entry = e, line = l;
                        string text = ReadString((b, len) => momo_doc_line_w(ptr, entry, line, b, len));
                        bool paragraphEnd = momo_doc_line_logical_end(ptr, entry, line);
                        doc.Entries.Add(new TextSegment(text, paragraphEnd));
                    }
                }
                if (doc.Entries.Count == 0)
                    doc.Entries.Add(new TextSegment("", true));
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

    // ---- テーブル一覧 ----

    /// <summary>組み込み変換テーブルの識別名と表示名。</summary>
    public sealed record TableInfo(string Name, string DisplayName);

    /// <summary>
    /// 組み込みテーブルを列挙する（テーブル選択メニュー用）。DLL 不在なら空。
    /// </summary>
    public static IReadOnlyList<TableInfo> EmbeddedTables()
    {
        if (_dllAvailable == false) return [];
        try
        {
            int count = momo_table_embedded_count();
            _dllAvailable = true;
            var list = new List<TableInfo>(Math.Max(0, count));
            for (int i = 0; i < count; i++)
            {
                int idx = i;
                var name = ReadString((b, l) => momo_table_embedded_name_w(idx, b, l));
                var display = ReadString((b, l) => momo_table_embedded_displayname_w(idx, b, l));
                if (name.Length > 0)
                    list.Add(new TableInfo(name, display.Length > 0 ? display : name));
            }
            return list;
        }
        catch (DllNotFoundException) { _dllAvailable = false; return []; }
        catch (EntryPointNotFoundException) { _dllAvailable = false; return []; }
    }

    // ---- 点訳器の生成 ----

    /// <summary>
    /// 組み込みテーブルを名前で指定して点訳器を作る。
    /// <paramref name="englishTable"/> が null なら英語エンジンを持たず、英字も日本語テーブルで
    /// 点訳する（無変換テーブルなどの用途）。名前が無効・DLL 不在なら null。
    /// </summary>
    public static BrailleTranslatorHandle? CreateTranslator(
        string japaneseTable = TableJapaneseGrade1,
        string? englishTable = TableEnglishUebGrade2)
    {
        if (_dllAvailable == false) return null;
        try
        {
            var ptr = momo_braille_translator_from_names_w(japaneseTable, englishTable);
            _dllAvailable = true;
            return ptr == nint.Zero ? null : new BrailleTranslatorHandle(ptr);
        }
        catch (DllNotFoundException) { _dllAvailable = false; return null; }
        catch (EntryPointNotFoundException) { _dllAvailable = false; return null; }
    }

    /// <summary>
    /// 既定の組み込みテーブル（日本語１級 + UEB grade 2）で点訳器を作る。DLL 不在なら null。
    /// </summary>
    public static BrailleTranslatorHandle? CreateDefaultTranslator()
    {
        if (_dllAvailable == false) return null;
        try
        {
            var ptr = momo_braille_translator_from_embedded();
            _dllAvailable = true;
            return ptr == nint.Zero ? null : new BrailleTranslatorHandle(ptr);
        }
        catch (DllNotFoundException) { _dllAvailable = false; return null; }
        catch (EntryPointNotFoundException) { _dllAvailable = false; return null; }
    }

    /// <summary>
    /// TOML ファイルから点訳器を作る（データセットのテーブルを差し替えたいとき）。
    /// <paramref name="englishTomlPath"/> が null なら英語エンジンなし。
    /// unknownCharPolicy はテーブル未定義文字の扱い（<see cref="PolicySpace"/> / <see cref="PolicyPassThrough"/>）。
    /// 失敗・DLL 不在なら null。
    /// </summary>
    public static BrailleTranslatorHandle? CreateTranslatorFromFiles(
        string japaneseTomlPath,
        string? englishTomlPath = null,
        int unknownCharPolicy = PolicySpace)
    {
        if (_dllAvailable == false) return null;

        nint jp = nint.Zero, en = nint.Zero;
        try
        {
            jp = momo_japanese_translator_from_file_w(japaneseTomlPath);
            _dllAvailable = true;
            if (jp == nint.Zero) return null;
            momo_japanese_translator_set_unknown_char_policy(jp, unknownCharPolicy);

            if (englishTomlPath != null)
            {
                en = momo_english_translator_from_file_w(englishTomlPath);
                if (en == nint.Zero) { momo_japanese_translator_free(jp); return null; }
                momo_english_translator_set_unknown_char_policy(en, unknownCharPolicy);
            }

            // 成功／失敗にかかわらず両ハンドルは Rust 側に消費される。以後 free してはならない。
            var ptr = momo_braille_translator_new(jp, en);
            return ptr == nint.Zero ? null : new BrailleTranslatorHandle(ptr);
        }
        catch (DllNotFoundException) { _dllAvailable = false; return null; }
        catch (EntryPointNotFoundException) { _dllAvailable = false; return null; }
    }

    // ---- 予測器（プロセス内で使い回す） ----

    // 予測器はアプリ全体で 1 つだけ使い回す（モデル読み込みが重いため）。
    static PredictorHandle? _predictor;
    static bool _predictorLoaded;

    /// <summary>
    /// 漢字かな交じり文の読みを推定する予測器を返す。日本語行の点訳に必要。
    /// モデルが見つからない／DLL 不在なら null。一度だけ読み込んで以降キャッシュする。
    /// </summary>
    public static PredictorHandle? GetPredictor()
    {
        if (_predictorLoaded) return _predictor;
        _predictorLoaded = true;

        var modelPath = FindModelPath();
        if (modelPath == null) return null;
        _predictor = CreatePredictor(modelPath);
        return _predictor;
    }

    /// <summary>
    /// .mbm モデルのパスを指定して予測器を作る（モデルを利用者に選ばせる UI 向け）。
    /// 呼び出し側が Dispose すること。読み込み失敗・DLL 不在なら null。
    /// </summary>
    public static PredictorHandle? CreatePredictor(string modelPath)
    {
        if (_dllAvailable == false) return null;
        try
        {
            var ptr = momo_predictor_new_w(modelPath);
            _dllAvailable = true;
            return ptr == nint.Zero ? null : new PredictorHandle(ptr);
        }
        catch (DllNotFoundException) { _dllAvailable = false; return null; }
        catch (EntryPointNotFoundException) { _dllAvailable = false; return null; }
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

    // ---- 逆点訳（ガイド表示用）公開 API ----

    /// <summary>逆点訳の 1 セグメント。点字セル範囲 [CellStart, CellEnd) とその読み。</summary>
    public sealed record GuideSegment(int CellStart, int CellEnd, string Text);

    /// <summary>1 行の逆点訳結果。全文と、セル範囲ごとの読みセグメント列。</summary>
    public sealed record GuideLine(string Text, IReadOnlyList<GuideSegment> Segments);

    static nint _backTranslator;
    static bool _backTranslatorLoaded;

    static nint GetBackTranslator()
    {
        if (_backTranslatorLoaded) return _backTranslator;
        _backTranslatorLoaded = true;
        if (_dllAvailable == false) return nint.Zero;
        try
        {
            _backTranslator = momo_back_translator_new();
            _dllAvailable = true;
        }
        catch (DllNotFoundException) { _dllAvailable = false; }
        catch (EntryPointNotFoundException) { _dllAvailable = false; }
        return _backTranslator;
    }

    /// <summary>
    /// 点字 1 行を逆点訳し、全文とセル範囲ごとの読みセグメントを返す。
    /// 空行は空の結果、DLL 不在・失敗時は null。
    /// </summary>
    public static GuideLine? BackTranslateLine(string braille)
    {
        if (string.IsNullOrEmpty(braille)) return new GuideLine("", []);
        var translator = GetBackTranslator();
        if (translator == nint.Zero) return null;

        nint h;
        try { h = momo_back_translate_w(translator, braille); }
        catch (DllNotFoundException) { _dllAvailable = false; return null; }
        catch (EntryPointNotFoundException) { _dllAvailable = false; return null; }
        if (h == nint.Zero) return null;
        try
        {
            string text = ReadString((b, l) => momo_back_trans_text_w(h, b, l));
            int n = momo_back_trans_segment_count(h);
            var segs = new List<GuideSegment>(Math.Max(0, n));
            if (n > 0)
            {
                var start = new int[n];
                var end = new int[n];
                momo_back_trans_cell_bounds(h, start, end);
                for (int i = 0; i < n; i++)
                {
                    int idx = i;
                    string segText = ReadString((b, l) => momo_back_trans_segment_text_w(h, idx, b, l));
                    segs.Add(new GuideSegment(start[i], end[i], segText));
                }
            }
            return new GuideLine(text, segs);
        }
        finally { momo_back_trans_free(h); }
    }
}
