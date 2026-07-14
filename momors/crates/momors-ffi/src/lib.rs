//! Windows DLL 向け C FFI。
//!
//! # 関数一覧
//!
//! momors-core / momors-braille の公開型を C へ忠実に写す。各ハンドルは対応する
//! `momo_*_free` で解放する。文字列の戻り値は必要バッファサイズ（null 終端含む）で、
//! `buf_len` 不足時は書かずにサイズだけ返す。パスは UTF-16（`_w`）。
//!
//! ## 予測器（momors-core::Predictor）
//! ```c
//! MomoPredictor momo_predictor_new_w(const uint16_t* model_path_utf16); // .mbm
//! void          momo_predictor_free(MomoPredictor);
//! ```
//!
//! ## 変換テーブル（momors-braille::Table。日本語・英語 両スキーマを同型で保持）
//! ```c
//! int32_t   momo_table_embedded_count();
//! int32_t   momo_table_embedded_name_w(int32_t idx, uint16_t* buf, int32_t len);
//! int32_t   momo_table_embedded_displayname_w(int32_t idx, uint16_t* buf, int32_t len);
//! MomoTable momo_table_from_embedded_name_w(const uint16_t* name);   // [metadata].name と照合
//! MomoTable momo_table_from_file_w(const uint16_t* path);            // 日本語スキーマ
//! MomoTable momo_table_from_ueb_file_w(const uint16_t* path);        // 英語(UEB)スキーマ
//! int32_t   momo_table_name_w(MomoTable, uint16_t* buf, int32_t len);
//! int32_t   momo_table_displayname_w(MomoTable, uint16_t* buf, int32_t len);
//! void      momo_table_free(MomoTable);
//! ```
//!
//! ## 言語別変換器（JapaneseTranslator / EnglishTranslator）
//! `_new` はテーブルを**消費**する。policy: 0=Space, 1=PassThrough。
//! ```c
//! MomoJapaneseTranslator momo_japanese_translator_new(MomoTable /*consumed*/);
//! MomoJapaneseTranslator momo_japanese_translator_from_embedded_name_w(const uint16_t* name);
//! MomoJapaneseTranslator momo_japanese_translator_from_file_w(const uint16_t* path);
//! void momo_japanese_translator_set_unknown_char_policy(MomoJapaneseTranslator, int32_t policy);
//! void momo_japanese_translator_free(MomoJapaneseTranslator);
//! // English 版も同名（momo_english_translator_*）。from_file は UEB スキーマ。
//! ```
//!
//! ## 点訳器（BrailleTranslator）＝入口
//! 行単位で日本語／英語を振り分ける。`_new`/`_japanese_only` はハンドルを**消費**する。
//! english を持たない点訳器は英語行も日本語テーブルで点訳（no conversion 用途）。
//! ```c
//! MomoBrailleTranslator momo_braille_translator_from_embedded();
//! MomoBrailleTranslator momo_braille_translator_new(MomoJapaneseTranslator /*consumed*/, MomoEnglishTranslator /*consumed, NULL可*/);
//! MomoBrailleTranslator momo_braille_translator_japanese_only(MomoJapaneseTranslator /*consumed*/);
//! MomoBrailleTranslator momo_braille_translator_from_names_w(const uint16_t* jp_name, const uint16_t* en_name /*NULL可*/);
//! MomoBrailleResult momo_braille_translator_translate_w         (MomoBrailleTranslator, const uint16_t* line, MomoPredictor);
//! MomoBrailleResult momo_braille_translator_translate_japanese_w(MomoBrailleTranslator, const uint16_t* line, MomoPredictor);
//! MomoBrailleResult momo_braille_translator_translate_english_w (MomoBrailleTranslator, const uint16_t* line); // 英語エンジン無→NULL
//! void momo_braille_translator_free(MomoBrailleTranslator);
//! ```
//!
//! ## 点訳結果（BrailleResult）＝3層 source/reading/braille
//! 英語行では reading=原文（恒等）。索引はコードポイント単位。
//! momors-braille の点訳器は 読み(かな)↔点字 の 2層しか持たない（漢字列を扱わない）。
//! 原文↔読み は予測器が持ち、**この FFI が両者を合成して 3層に組み立てる**。
//! ```c
//! int32_t momo_braille_result_language(MomoBrailleResult);       // 0=JP, 1=EN
//! int32_t momo_braille_result_source_text_w (MomoBrailleResult, uint16_t* buf, int32_t len);
//! int32_t momo_braille_result_reading_text_w(MomoBrailleResult, uint16_t* buf, int32_t len);
//! int32_t momo_braille_result_braille_text_w(MomoBrailleResult, uint16_t* buf, int32_t len);
//! int32_t momo_braille_result_source_char_count (MomoBrailleResult);
//! int32_t momo_braille_result_reading_char_count(MomoBrailleResult);
//! int32_t momo_braille_result_braille_char_count(MomoBrailleResult);
//! void momo_braille_result_reading_to_source(MomoBrailleResult, int32_t* out);  // reading_char_count 要素
//! void momo_braille_result_braille_to_source(MomoBrailleResult, int32_t* out);  // braille_char_count 要素
//! void momo_braille_result_source_to_reading(MomoBrailleResult, int32_t* row_ptr, int32_t* col_idx); // CSR
//! void momo_braille_result_source_to_braille(MomoBrailleResult, int32_t* row_ptr, int32_t* col_idx); // CSR
//! bool momo_braille_result_has_prediction(MomoBrailleResult);
//! void momo_braille_result_free(MomoBrailleResult);
//! ```
//!
//! ## 点字ドキュメント（正本）の読み書き・描画
//! すべての形式（MBR / BES / BASE / BrailleText）の読み書きは Rust 側に集約される。
//! 改ページ情報は MBR の `====` マーカー文字列として不透明に授受する。
//! 詳細はソース下部の各関数ドキュメントを参照。
//! ```c
//! // 読込: バイト列 -> ドキュメント（format: 0=MBR,1=BES,2=BET）
//! MomoDoc momo_doc_read(const uint8_t* bytes, int32_t len, int32_t format);
//! void    momo_doc_free(MomoDoc);
//! // 設定・段落・物理行の getter（momo_doc_line_width / _paragraph_count / _line_w など）
//! // 保存: ビルダーで組み立て -> momo_doc_write でバイト列取得（format: 0=MBR,1=BES,3=BASE,4=BRF）
//! MomoDocBuilder momo_doc_builder_new(int32_t lw, int32_t lpp, bool header, int32_t number_start, const uint16_t* title);
//! void           momo_doc_builder_add_line(MomoDocBuilder, const uint16_t* content, bool logical_end, const uint16_t* page_break);
//! MomoDoc        momo_doc_builder_build(MomoDocBuilder); // ビルダーは解放される
//! MomoBytes      momo_doc_write(MomoDoc, int32_t format);
//! int32_t        momo_bytes_len(MomoBytes);  void momo_bytes_copy(MomoBytes, uint8_t* out);  void momo_bytes_free(MomoBytes);
//! // 表示: 印刷イメージ（ページ/物理行/ヘッダ）
//! MomoFormatted  momo_doc_render(MomoDoc);
//! // 逐次: 1論理行の折返し
//! MomoWrapLines  momo_wrap_line_w(const uint16_t* text, int32_t line_width);
//! MomoWrapLines  momo_wrap_suffix_w(const uint16_t* text, int32_t line_width, int32_t first_line_remaining);
//! ```
//!
//! ## 逆点訳（点字 → かな表層、ガイド表示用）
//! 逆変換器を一度作って使い回し、行ごとに逆変換する。セルインデックスは
//! 入力点字の UTF-16 位置に対応する（点字は全て BMP のため UTF-16 と等価）。
//! ```c
//! MomoBackTranslator momo_back_translator_new();              // 組み込みテーブル
//! MomoBackTranslator momo_back_translator_new_from_file_w(const uint16_t* toml_path);
//! void               momo_back_translator_free(MomoBackTranslator);
//! // 逆変換: 点字 -> 結果（全文＋セグメント）
//! MomoBackTrans momo_back_translate_w(MomoBackTranslator, const uint16_t* braille_utf16);
//! void          momo_back_trans_free(MomoBackTrans);
//! int32_t momo_back_trans_text_w(MomoBackTrans, uint16_t* buf, int32_t buf_len);
//! int32_t momo_back_trans_segment_count(MomoBackTrans);
//! void    momo_back_trans_cell_bounds(MomoBackTrans, int32_t* out_start, int32_t* out_end);
//! int32_t momo_back_trans_segment_text_w(MomoBackTrans, int32_t idx, uint16_t* buf, int32_t buf_len);
//! ```

// cbindgen は #[no_mangle] (Rust 2024) 未対応のため edition 2021 を使用する。
// 1.82+ で発生する "unsafe attribute used without unsafe" lint を抑制する。
#![allow(unsafe_op_in_unsafe_fn)]

use std::os::raw::c_int;

use momors_braille::document::{BrailleDocument, DocumentConfig, PageBreak, PhysicalLine};
use momors_braille::formatter::{render, wrap_line, wrap_suffix, FormattedDocument, RenderedLine};
use momors_braille::writer::OutputFormat;
use momors_braille::NabccCase;
use momors_braille::{
    detect_language, embedded_table, embedded_tables, BackTransResult, BrailleBackTranslator,
    BrailleResult, BrailleTranslator, EnglishTranslator, JapaneseTranslator, Language, Table,
    UnknownCharPolicy,
};
use momors_core::{PredictionResult, Predictor, PredictorConfig};

// ============================================================
// 内部ハンドル型
// ============================================================
//
// momors-core / momors-braille の公開型を C へ忠実に写す薄いラッパ。
// それぞれ対応する momo_*_free で解放する。

/// [`Predictor`]（momors-core。漢字かな交じり文 → かな＋境界）。
pub struct PredictorHandle {
    inner: Predictor,
}

/// [`Table`]（変換テーブル。日本語・英語(UEB) 両スキーマを同じ型で保持）。
pub struct TableHandle {
    inner: Table,
}

/// [`JapaneseTranslator`]（かな → 日本語点字）。
pub struct JapaneseTranslatorHandle {
    inner: JapaneseTranslator,
}

/// [`EnglishTranslator`]（英語 → UEB 点字）。
pub struct EnglishTranslatorHandle {
    inner: EnglishTranslator,
}

/// [`BrailleTranslator`]（点訳の入口。行単位で日本語／英語を振り分ける）。
pub struct BrailleTranslatorHandle {
    inner: BrailleTranslator,
}

/// 1行の点訳結果（3層 source/reading/braille とインデックス）。
///
/// momors-braille は漢字列を扱わない（点訳器は読み＝かなを受け取る 2層）。原文
/// （漢字かな交じり）との対応づけは**この FFI が上位として合成する**: 予測器の
/// 原文↔かな と、点訳器の かな↔点字 を突き合わせて 3層に組み立てる。
/// 英語行は予測を通さないので、読み層は原文と同一（恒等写像）になる。
pub struct BrailleResultHandle {
    inner: BrailleResult,
    /// 原文（漢字かな交じり。点訳器へ渡す前のテキスト）。
    source_text: String,
    reading_to_source: Vec<usize>,
    source_to_reading: Vec<Vec<usize>>,
    braille_to_source: Vec<usize>,
    source_to_braille: Vec<Vec<usize>>,
    has_prediction: bool,
}

impl BrailleResultHandle {
    /// 点訳結果と（日本語行なら）予測結果から 3層を組み立てる。
    fn new(source: &str, inner: BrailleResult, pred: Option<PredictionResult>) -> Self {
        let source_count = source.chars().count();
        let braille_count = inner.braille_char_count();
        let (reading_to_source, source_to_reading, braille_to_source) = match &pred {
            // 日本語行: 原文↔かな は予測器が持つ。点字→原文は かな経由で合成する。
            Some(pred) => (
                pred.kana_to_source_char(),
                pred.source_to_kana_char(),
                pred.braille_char_to_source(inner.text_to_braille(), braille_count),
            ),
            // 英語行: 読み層＝原文（恒等）。点字→原文は点訳器の 点字→テキスト そのもの。
            None => (
                (0..source_count).collect(),
                (0..source_count).map(|i| vec![i]).collect(),
                inner.braille_to_text().to_vec(),
            ),
        };
        let source_to_braille = invert(&braille_to_source, source_count);
        Self {
            inner,
            source_text: source.to_owned(),
            reading_to_source,
            source_to_reading,
            braille_to_source,
            source_to_braille,
            has_prediction: pred.is_some(),
        }
    }
}

/// `x → 原文文字` の写像を反転し、`原文文字 → x の集合（昇順）` にする。
/// 範囲外の原文インデックスは無視する。
fn invert(to_source: &[usize], source_count: usize) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); source_count];
    for (idx, &s) in to_source.iter().enumerate() {
        if s < source_count {
            out[s].push(idx);
        }
    }
    out
}

/// 1行を点訳する。日本語行は予測器で かな にしてから点訳器へ渡す。
///
/// `routing` が true なら日本語文字を含まない行を **予測を通さず** 英語（UEB）で点訳する
/// （英語エンジンを持たない点訳器では日本語経路へ落ちる）。
fn translate_line(
    lt: &BrailleTranslator,
    predictor: &Predictor,
    line: &str,
    routing: bool,
) -> Option<BrailleResultHandle> {
    if routing && detect_language(line) == Language::English {
        if let Some(result) = lt.translate_english(line) {
            return Some(BrailleResultHandle::new(line, result, None));
        }
    }
    let pred = predictor.predict(line).ok()?;
    let result = lt.translate_japanese(pred.kana_text()).ok()?;
    Some(BrailleResultHandle::new(line, result, Some(pred)))
}

// ============================================================
// 文字列変換ヘルパ
// ============================================================

unsafe fn lpwstr_to_string(ptr: *const u16) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let len = (0usize..)
        .take_while(|&i| unsafe { *ptr.add(i) } != 0)
        .count();
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    Some(String::from_utf16_lossy(slice).to_owned())
}

// ============================================================
// インデックス書き出しヘルパ
// ============================================================

/// `usize` スライスを i32 配列として out に書く。out は src.len() 要素以上確保すること。
/// out が NULL なら何もしない。
fn write_usize_array(src: &[usize], out: *mut i32) {
    if out.is_null() {
        return;
    }
    for (i, &v) in src.iter().enumerate() {
        unsafe { *out.add(i) = v as i32 };
    }
}

/// 一対多マップ（`&[Vec<usize>]`）を CSR 形式で書く。
/// row_ptr は rows.len()+1 要素、col_idx は総要素数（各 Vec の長さの合計）以上を確保すること。
/// 各ポインタは NULL 可（片方のみ書ける）。
fn write_csr(rows: &[Vec<usize>], row_ptr: *mut i32, col_idx: *mut i32) {
    let mut col_pos = 0usize;
    let mut offset = 0i32;
    for (i, cols) in rows.iter().enumerate() {
        if !row_ptr.is_null() {
            unsafe { *row_ptr.add(i) = offset };
        }
        for &c in cols {
            if !col_idx.is_null() {
                unsafe { *col_idx.add(col_pos) = c as i32 };
            }
            col_pos += 1;
        }
        offset += cols.len() as i32;
    }
    if !row_ptr.is_null() {
        unsafe { *row_ptr.add(rows.len()) = offset };
    }
}

// ============================================================
// 予測器（momors-core::Predictor）
// ============================================================

/// UTF-16 モデルパス（`.mbm`）から予測器を作る。単一漢字辞書はモデル同梱のものを使う。
/// 失敗（ファイル不正・NULL）時は NULL を返す。
#[no_mangle]
pub extern "C" fn momo_predictor_new_w(model_path: *const u16) -> *mut PredictorHandle {
    let model = match unsafe { lpwstr_to_string(model_path) } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    let config = PredictorConfig::new(std::path::Path::new(&model));
    match Predictor::load(config) {
        Ok(inner) => Box::into_raw(Box::new(PredictorHandle { inner })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// 予測器を解放する。NULL は無視する。
#[no_mangle]
pub extern "C" fn momo_predictor_free(handle: *mut PredictorHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

// ============================================================
// 変換テーブル（momors-braille::Table）
// ============================================================

/// 組み込みテーブルの数。
#[no_mangle]
pub extern "C" fn momo_table_embedded_count() -> c_int {
    embedded_tables().len() as c_int
}

/// 組み込みテーブル idx の name（UTF-16, null 終端）を buf に書く。
/// 戻り値は必要な u16 要素数。name が None なら 0、idx が範囲外なら -1。
#[no_mangle]
pub extern "C" fn momo_table_embedded_name_w(idx: c_int, buf: *mut u16, buf_len: c_int) -> c_int {
    match embedded_tables().get(idx as usize) {
        Some(t) => match t.name.as_deref() {
            Some(s) => write_utf16(s, buf, buf_len),
            None => 0,
        },
        None => -1,
    }
}

/// 組み込みテーブル idx の displayname（UTF-16, null 終端）を buf に書く。
/// 戻り値は必要な u16 要素数。displayname が None なら 0、idx が範囲外なら -1。
#[no_mangle]
pub extern "C" fn momo_table_embedded_displayname_w(
    idx: c_int,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    match embedded_tables().get(idx as usize) {
        Some(t) => match t.displayname.as_deref() {
            Some(s) => write_utf16(s, buf, buf_len),
            None => 0,
        },
        None => -1,
    }
}

/// 名前（`[metadata].name`）で組み込みテーブルを引く。見つからない／NULL なら NULL。
#[no_mangle]
pub extern "C" fn momo_table_from_embedded_name_w(name: *const u16) -> *mut TableHandle {
    let name = match unsafe { lpwstr_to_string(name) } {
        Some(n) => n,
        None => return std::ptr::null_mut(),
    };
    match embedded_table(&name) {
        Some(inner) => Box::into_raw(Box::new(TableHandle { inner })),
        None => std::ptr::null_mut(),
    }
}

/// 日本語スキーマの TOML ファイルからテーブルを読む。失敗時 NULL。
#[no_mangle]
pub extern "C" fn momo_table_from_file_w(path: *const u16) -> *mut TableHandle {
    let path = match unsafe { lpwstr_to_string(path) } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    match Table::from_file(&path) {
        Ok(inner) => Box::into_raw(Box::new(TableHandle { inner })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// UEB（英語）スキーマの TOML ファイルからテーブルを読む。失敗時 NULL。
#[no_mangle]
pub extern "C" fn momo_table_from_ueb_file_w(path: *const u16) -> *mut TableHandle {
    let path = match unsafe { lpwstr_to_string(path) } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    match Table::from_ueb_file(&path) {
        Ok(inner) => Box::into_raw(Box::new(TableHandle { inner })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// テーブルの name（UTF-16）。None なら 0、handle NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_table_name_w(
    handle: *const TableHandle,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    let h = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    match h.inner.name.as_deref() {
        Some(s) => write_utf16(s, buf, buf_len),
        None => 0,
    }
}

/// テーブルの displayname（UTF-16）。None なら 0、handle NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_table_displayname_w(
    handle: *const TableHandle,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    let h = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    match h.inner.displayname.as_deref() {
        Some(s) => write_utf16(s, buf, buf_len),
        None => 0,
    }
}

/// テーブルを解放する。NULL は無視する。
#[no_mangle]
pub extern "C" fn momo_table_free(handle: *mut TableHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

// ============================================================
// 日本語変換器（momors-braille::JapaneseTranslator）
// ============================================================

/// テーブルから日本語変換器を作る。**`table` を消費する**（呼び出し後は無効・
/// `momo_table_free` に渡してはならない）。table が NULL なら NULL。
#[no_mangle]
pub extern "C" fn momo_japanese_translator_new(
    table: *mut TableHandle,
) -> *mut JapaneseTranslatorHandle {
    if table.is_null() {
        return std::ptr::null_mut();
    }
    let table = unsafe { *Box::from_raw(table) };
    Box::into_raw(Box::new(JapaneseTranslatorHandle {
        inner: JapaneseTranslator::new(table.inner),
    }))
}

/// 名前で組み込みテーブルを指定して日本語変換器を作る。失敗時 NULL。
#[no_mangle]
pub extern "C" fn momo_japanese_translator_from_embedded_name_w(
    name: *const u16,
) -> *mut JapaneseTranslatorHandle {
    let name = match unsafe { lpwstr_to_string(name) } {
        Some(n) => n,
        None => return std::ptr::null_mut(),
    };
    match JapaneseTranslator::from_embedded_name(&name) {
        Ok(inner) => Box::into_raw(Box::new(JapaneseTranslatorHandle { inner })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// 日本語スキーマの TOML ファイルから日本語変換器を作る。失敗時 NULL。
#[no_mangle]
pub extern "C" fn momo_japanese_translator_from_file_w(
    path: *const u16,
) -> *mut JapaneseTranslatorHandle {
    let path = match unsafe { lpwstr_to_string(path) } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    match JapaneseTranslator::from_file(&path) {
        Ok(inner) => Box::into_raw(Box::new(JapaneseTranslatorHandle { inner })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// テーブル未定義文字のポリシー（0=Space, 1=PassThrough）。handle NULL なら何もしない。
#[no_mangle]
pub extern "C" fn momo_japanese_translator_set_unknown_char_policy(
    handle: *mut JapaneseTranslatorHandle,
    policy: c_int,
) {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.inner.set_unknown_char_policy(policy_from_i32(policy));
    }
}

/// 日本語変換器を解放する。NULL は無視する。
#[no_mangle]
pub extern "C" fn momo_japanese_translator_free(handle: *mut JapaneseTranslatorHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

// ============================================================
// 英語変換器（momors-braille::EnglishTranslator）
// ============================================================

/// テーブルから英語変換器を作る。**`table` を消費する**（呼び出し後は無効）。
/// table が NULL なら NULL。
#[no_mangle]
pub extern "C" fn momo_english_translator_new(
    table: *mut TableHandle,
) -> *mut EnglishTranslatorHandle {
    if table.is_null() {
        return std::ptr::null_mut();
    }
    let table = unsafe { *Box::from_raw(table) };
    Box::into_raw(Box::new(EnglishTranslatorHandle {
        inner: EnglishTranslator::new(table.inner),
    }))
}

/// 名前で組み込みテーブルを指定して英語変換器を作る（例: `"english_ueb_grade2"`）。失敗時 NULL。
#[no_mangle]
pub extern "C" fn momo_english_translator_from_embedded_name_w(
    name: *const u16,
) -> *mut EnglishTranslatorHandle {
    let name = match unsafe { lpwstr_to_string(name) } {
        Some(n) => n,
        None => return std::ptr::null_mut(),
    };
    match EnglishTranslator::from_embedded_name(&name) {
        Ok(inner) => Box::into_raw(Box::new(EnglishTranslatorHandle { inner })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// UEB スキーマの TOML ファイルから英語変換器を作る。失敗時 NULL。
#[no_mangle]
pub extern "C" fn momo_english_translator_from_file_w(
    path: *const u16,
) -> *mut EnglishTranslatorHandle {
    let path = match unsafe { lpwstr_to_string(path) } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    match EnglishTranslator::from_file(&path) {
        Ok(inner) => Box::into_raw(Box::new(EnglishTranslatorHandle { inner })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// テーブル未定義文字のポリシー（0=Space, 1=PassThrough）。handle NULL なら何もしない。
#[no_mangle]
pub extern "C" fn momo_english_translator_set_unknown_char_policy(
    handle: *mut EnglishTranslatorHandle,
    policy: c_int,
) {
    if let Some(h) = unsafe { handle.as_mut() } {
        h.inner.set_unknown_char_policy(policy_from_i32(policy));
    }
}

/// 英語変換器を解放する。NULL は無視する。
#[no_mangle]
pub extern "C" fn momo_english_translator_free(handle: *mut EnglishTranslatorHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

fn policy_from_i32(policy: c_int) -> UnknownCharPolicy {
    match policy {
        1 => UnknownCharPolicy::PassThrough,
        _ => UnknownCharPolicy::Space,
    }
}

// ============================================================
// 点訳器（momors-braille::BrailleTranslator）= 入口
// ============================================================

/// 組み込みテーブル（日本語１級 + UEB grade 2）で点訳器を作る。失敗時 NULL。
#[no_mangle]
pub extern "C" fn momo_braille_translator_from_embedded() -> *mut BrailleTranslatorHandle {
    match BrailleTranslator::from_embedded() {
        Ok(inner) => Box::into_raw(Box::new(BrailleTranslatorHandle { inner })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// 日本語変換器と英語変換器（NULL 可）から点訳器を作る。
/// **両ハンドルを消費する**（呼び出し後は無効）。japanese が NULL なら NULL。
/// english が NULL なら英語行も日本語テーブルで点訳する（no conversion 用途）。
#[no_mangle]
pub extern "C" fn momo_braille_translator_new(
    japanese: *mut JapaneseTranslatorHandle,
    english: *mut EnglishTranslatorHandle,
) -> *mut BrailleTranslatorHandle {
    if japanese.is_null() {
        return std::ptr::null_mut();
    }
    let japanese = unsafe { *Box::from_raw(japanese) };
    let english = if english.is_null() {
        None
    } else {
        Some(unsafe { *Box::from_raw(english) }.inner)
    };
    Box::into_raw(Box::new(BrailleTranslatorHandle {
        inner: BrailleTranslator::new(japanese.inner, english),
    }))
}

/// 英語エンジンを持たない点訳器（全行を日本語テーブルで点訳）。
/// **`japanese` を消費する**。japanese が NULL なら NULL。
#[no_mangle]
pub extern "C" fn momo_braille_translator_japanese_only(
    japanese: *mut JapaneseTranslatorHandle,
) -> *mut BrailleTranslatorHandle {
    if japanese.is_null() {
        return std::ptr::null_mut();
    }
    let japanese = unsafe { *Box::from_raw(japanese) };
    Box::into_raw(Box::new(BrailleTranslatorHandle {
        inner: BrailleTranslator::japanese_only(japanese.inner),
    }))
}

/// 組み込みテーブルを名前で指定して点訳器を作る便宜関数。
/// `english_name` が NULL なら英語エンジンなし。いずれかの名前が無効なら NULL。
#[no_mangle]
pub extern "C" fn momo_braille_translator_from_names_w(
    japanese_name: *const u16,
    english_name: *const u16,
) -> *mut BrailleTranslatorHandle {
    let jp_name = match unsafe { lpwstr_to_string(japanese_name) } {
        Some(n) => n,
        None => return std::ptr::null_mut(),
    };
    let japanese = match JapaneseTranslator::from_embedded_name(&jp_name) {
        Ok(j) => j,
        Err(_) => return std::ptr::null_mut(),
    };
    let english = match unsafe { lpwstr_to_string(english_name) } {
        Some(n) => match EnglishTranslator::from_embedded_name(&n) {
            Ok(e) => Some(e),
            Err(_) => return std::ptr::null_mut(),
        },
        None => None,
    };
    Box::into_raw(Box::new(BrailleTranslatorHandle {
        inner: BrailleTranslator::new(japanese, english),
    }))
}

/// 1行を**言語判定して**点訳する。日本語行のみ predictor を使う。
/// handle / predictor / line が NULL、または点訳失敗時は NULL。
#[no_mangle]
pub extern "C" fn momo_braille_translator_translate_w(
    handle: *const BrailleTranslatorHandle,
    line: *const u16,
    predictor: *const PredictorHandle,
) -> *mut BrailleResultHandle {
    let h = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return std::ptr::null_mut(),
    };
    let p = match unsafe { predictor.as_ref() } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    let line = match unsafe { lpwstr_to_string(line) } {
        Some(l) => l,
        None => return std::ptr::null_mut(),
    };
    match translate_line(&h.inner, &p.inner, &line, true) {
        Some(handle) => Box::into_raw(Box::new(handle)),
        None => std::ptr::null_mut(),
    }
}

/// 1行を**必ず日本語として**点訳する（英字は外字符＋無縮約）。失敗時 NULL。
#[no_mangle]
pub extern "C" fn momo_braille_translator_translate_japanese_w(
    handle: *const BrailleTranslatorHandle,
    line: *const u16,
    predictor: *const PredictorHandle,
) -> *mut BrailleResultHandle {
    let h = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return std::ptr::null_mut(),
    };
    let p = match unsafe { predictor.as_ref() } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    let line = match unsafe { lpwstr_to_string(line) } {
        Some(l) => l,
        None => return std::ptr::null_mut(),
    };
    match translate_line(&h.inner, &p.inner, &line, false) {
        Some(handle) => Box::into_raw(Box::new(handle)),
        None => std::ptr::null_mut(),
    }
}

/// 1行を**必ず英語（UEB）として**点訳する（予測不要）。
/// 英語エンジンを持たない点訳器・NULL 引数の場合は NULL。
#[no_mangle]
pub extern "C" fn momo_braille_translator_translate_english_w(
    handle: *const BrailleTranslatorHandle,
    line: *const u16,
) -> *mut BrailleResultHandle {
    let h = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return std::ptr::null_mut(),
    };
    let line = match unsafe { lpwstr_to_string(line) } {
        Some(l) => l,
        None => return std::ptr::null_mut(),
    };
    match h.inner.translate_english(&line) {
        Some(inner) => Box::into_raw(Box::new(BrailleResultHandle::new(&line, inner, None))),
        None => std::ptr::null_mut(),
    }
}

/// 点訳器を解放する。NULL は無視する。
#[no_mangle]
pub extern "C" fn momo_braille_translator_free(handle: *mut BrailleTranslatorHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

// ============================================================
// 点訳結果（momors-braille::BrailleResult）
// ============================================================

/// 点訳経路。0=Japanese, 1=English。handle NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_braille_result_language(handle: *const BrailleResultHandle) -> c_int {
    match unsafe { handle.as_ref() } {
        Some(h) => match h.inner.language() {
            Language::Japanese => 0,
            Language::English => 1,
        },
        None => -1,
    }
}

/// 原文テキスト（UTF-16, null 終端）。handle NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_braille_result_source_text_w(
    handle: *const BrailleResultHandle,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    match unsafe { handle.as_ref() } {
        Some(h) => write_utf16(&h.source_text, buf, buf_len),
        None => -1,
    }
}

/// 読みテキスト（日本語=かな / 英語=原文。UTF-16, null 終端）。handle NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_braille_result_reading_text_w(
    handle: *const BrailleResultHandle,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    match unsafe { handle.as_ref() } {
        Some(h) => write_utf16(h.inner.text(), buf, buf_len),
        None => -1,
    }
}

/// 点字テキスト（UTF-16, null 終端）。handle NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_braille_result_braille_text_w(
    handle: *const BrailleResultHandle,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    match unsafe { handle.as_ref() } {
        Some(h) => write_utf16(h.inner.braille_text(), buf, buf_len),
        None => -1,
    }
}

/// 原文の文字数（コードポイント）。handle NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_braille_result_source_char_count(
    handle: *const BrailleResultHandle,
) -> c_int {
    match unsafe { handle.as_ref() } {
        Some(h) => h.source_to_reading.len() as c_int,
        None => -1,
    }
}

/// 読みの文字数（コードポイント）。handle NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_braille_result_reading_char_count(
    handle: *const BrailleResultHandle,
) -> c_int {
    match unsafe { handle.as_ref() } {
        Some(h) => h.inner.text_char_count() as c_int,
        None => -1,
    }
}

/// 点字の文字数（コードポイント）。handle NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_braille_result_braille_char_count(
    handle: *const BrailleResultHandle,
) -> c_int {
    match unsafe { handle.as_ref() } {
        Some(h) => h.inner.braille_char_count() as c_int,
        None => -1,
    }
}

/// 読み→原文 インデックス配列（reading_char_count 要素）を out に書く。
/// handle / out が NULL なら何もしない。
#[no_mangle]
pub extern "C" fn momo_braille_result_reading_to_source(
    handle: *const BrailleResultHandle,
    out: *mut i32,
) {
    if let Some(h) = unsafe { handle.as_ref() } {
        write_usize_array(&h.reading_to_source, out);
    }
}

/// 点字→原文 インデックス配列（braille_char_count 要素）を out に書く。
/// handle / out が NULL なら何もしない。
#[no_mangle]
pub extern "C" fn momo_braille_result_braille_to_source(
    handle: *const BrailleResultHandle,
    out: *mut i32,
) {
    if let Some(h) = unsafe { handle.as_ref() } {
        write_usize_array(&h.braille_to_source, out);
    }
}

/// 原文→読み インデックスを CSR で書く。
/// row_ptr: source_char_count+1 要素、col_idx: reading_char_count 要素。各 NULL 可。
#[no_mangle]
pub extern "C" fn momo_braille_result_source_to_reading(
    handle: *const BrailleResultHandle,
    row_ptr: *mut i32,
    col_idx: *mut i32,
) {
    if let Some(h) = unsafe { handle.as_ref() } {
        write_csr(&h.source_to_reading, row_ptr, col_idx);
    }
}

/// 原文→点字 インデックスを CSR で書く。
/// row_ptr: source_char_count+1 要素、col_idx: braille_char_count 要素。各 NULL 可。
#[no_mangle]
pub extern "C" fn momo_braille_result_source_to_braille(
    handle: *const BrailleResultHandle,
    row_ptr: *mut i32,
    col_idx: *mut i32,
) {
    if let Some(h) = unsafe { handle.as_ref() } {
        write_csr(&h.source_to_braille, row_ptr, col_idx);
    }
}

/// 日本語予測（かな・確信度）が付随するか。英語行では false。handle NULL でも false。
#[no_mangle]
pub extern "C" fn momo_braille_result_has_prediction(handle: *const BrailleResultHandle) -> bool {
    unsafe { handle.as_ref() }.is_some_and(|h| h.has_prediction)
}

/// 点訳結果を解放する。NULL は無視する。
#[no_mangle]
pub extern "C" fn momo_braille_result_free(handle: *mut BrailleResultHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

// ============================================================
// 点字ドキュメント（正本 BrailleDocument）
//
// すべての形式の読み書きは Rust 側に集約されている。C# など呼び出し側は
//   読込: momo_doc_read -> getter で自前モデルへ展開
//   保存: momo_doc_builder_* で組み立て -> momo_doc_write でバイト列取得
//   表示: momo_doc_render -> 印刷イメージ(ページ/物理行/ヘッダ)を取得
//   逐次: momo_wrap_line_w / momo_wrap_suffix_w（1論理行の折返し）
// 改ページ情報は MBR の ==== マーカー文字列として不透明に授受する。
// ============================================================

/// 文字列を null 終端 UTF-16 として buf に書く。戻り値は必要な u16 要素数（null 含む）。
/// buf が NULL か buf_len 不足なら書かずにサイズだけ返す。
fn write_utf16(s: &str, buf: *mut u16, buf_len: c_int) -> c_int {
    let mut utf16: Vec<u16> = s.encode_utf16().collect();
    utf16.push(0);
    let needed = utf16.len() as c_int;
    if !buf.is_null() && buf_len >= needed {
        unsafe { std::ptr::copy_nonoverlapping(utf16.as_ptr(), buf, utf16.len()) };
    }
    needed
}

pub struct BrailleDocHandle {
    doc: BrailleDocument,
}

/// バイト列を点字ドキュメントへ読み込む。format: 0=MBR, 1=BES, 2=BET。
/// 失敗時（破損・不正 UTF-8・NULL）は NULL を返す。
#[no_mangle]
pub extern "C" fn momo_doc_read(
    bytes: *const u8,
    len: c_int,
    format: c_int,
) -> *mut BrailleDocHandle {
    if bytes.is_null() || len < 0 {
        return std::ptr::null_mut();
    }
    let slice = unsafe { std::slice::from_raw_parts(bytes, len as usize) };
    let doc = match format {
        1 => match momors_braille::read_bes(slice) {
            Some(d) => d,
            None => return std::ptr::null_mut(),
        },
        2 => match momors_braille::read_bet(slice) {
            Some(d) => d,
            None => return std::ptr::null_mut(),
        },
        _ => match std::str::from_utf8(slice) {
            Ok(t) => BrailleDocument::parse_mbr(t),
            Err(_) => return std::ptr::null_mut(),
        },
    };
    Box::into_raw(Box::new(BrailleDocHandle { doc }))
}

/// ドキュメントハンドルを解放する。NULL は無視する。
#[no_mangle]
pub extern "C" fn momo_doc_free(handle: *mut BrailleDocHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

// ---- 設定 getter ----

#[no_mangle]
pub extern "C" fn momo_doc_line_width(h: *const BrailleDocHandle) -> c_int {
    match unsafe { h.as_ref() } {
        Some(h) => h.doc.config.line_width as c_int,
        None => -1,
    }
}

#[no_mangle]
pub extern "C" fn momo_doc_lines_per_page(h: *const BrailleDocHandle) -> c_int {
    match unsafe { h.as_ref() } {
        Some(h) => h.doc.config.lines_per_page as c_int,
        None => -1,
    }
}

#[no_mangle]
pub extern "C" fn momo_doc_page_header(h: *const BrailleDocHandle) -> bool {
    match unsafe { h.as_ref() } {
        Some(h) => h.doc.config.page_header,
        None => false,
    }
}

#[no_mangle]
pub extern "C" fn momo_doc_number_start(h: *const BrailleDocHandle) -> c_int {
    match unsafe { h.as_ref() } {
        Some(h) => h.doc.config.number_start as c_int,
        None => -1,
    }
}

/// タイトル（UTF-16, null 終端）を buf に書く。タイトル無しなら 0、handle NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_doc_title_w(
    h: *const BrailleDocHandle,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    let h = match unsafe { h.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    match &h.doc.config.title {
        Some(t) => write_utf16(t, buf, buf_len),
        None => 0,
    }
}

// ---- 段落・物理行 getter ----

#[no_mangle]
pub extern "C" fn momo_doc_paragraph_count(h: *const BrailleDocHandle) -> c_int {
    match unsafe { h.as_ref() } {
        Some(h) => h.doc.paragraphs.len() as c_int,
        None => -1,
    }
}

#[no_mangle]
pub extern "C" fn momo_doc_line_count(h: *const BrailleDocHandle, para: c_int) -> c_int {
    let h = match unsafe { h.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    match h.doc.paragraphs.get(para as usize) {
        Some(p) => p.len() as c_int,
        None => -1,
    }
}

fn doc_line(h: &BrailleDocHandle, para: c_int, line: c_int) -> Option<&PhysicalLine> {
    h.doc
        .paragraphs
        .get(para as usize)
        .and_then(|p| p.get(line as usize))
}

/// 物理行のテキスト（UTF-16, null 終端）。無効な引数なら -1。
#[no_mangle]
pub extern "C" fn momo_doc_line_w(
    h: *const BrailleDocHandle,
    para: c_int,
    line: c_int,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    let h = match unsafe { h.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    match doc_line(h, para, line) {
        Some(l) => write_utf16(&l.content, buf, buf_len),
        None => -1,
    }
}

/// 物理行が論理行末尾か。無効な引数なら false。
#[no_mangle]
pub extern "C" fn momo_doc_line_logical_end(
    h: *const BrailleDocHandle,
    para: c_int,
    line: c_int,
) -> bool {
    match unsafe { h.as_ref() } {
        Some(h) => doc_line(h, para, line)
            .map(|l| l.logical_end)
            .unwrap_or(false),
        None => false,
    }
}

/// 物理行の改ページマーカー（==== で始まる行）を UTF-16 で buf に書く。
/// 改ページ無しなら 0、無効な引数なら -1。
#[no_mangle]
pub extern "C" fn momo_doc_line_page_break_w(
    h: *const BrailleDocHandle,
    para: c_int,
    line: c_int,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    let h = match unsafe { h.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    match doc_line(h, para, line) {
        Some(l) => match &l.page_break {
            Some(pb) => write_utf16(&pb.to_marker(), buf, buf_len),
            None => 0,
        },
        None => -1,
    }
}

// ============================================================
// ドキュメントビルダー（保存用）
// ============================================================

pub struct BrailleDocBuilder {
    config: DocumentConfig,
    paragraphs: Vec<Vec<PhysicalLine>>,
    current: Vec<PhysicalLine>,
}

/// ビルダーを作る。number_start: 開始ページ番号。title は NULL/空で無し。
#[no_mangle]
pub extern "C" fn momo_doc_builder_new(
    line_width: c_int,
    lines_per_page: c_int,
    page_header: bool,
    number_start: c_int,
    title: *const u16,
) -> *mut BrailleDocBuilder {
    let title = unsafe { lpwstr_to_string(title) }.filter(|s| !s.is_empty());
    let config = DocumentConfig {
        line_width: line_width.max(1) as usize,
        lines_per_page: lines_per_page.max(1) as usize,
        page_header,
        title,
        number_start: number_start.max(0) as u32,
    };
    Box::into_raw(Box::new(BrailleDocBuilder {
        config,
        paragraphs: Vec::new(),
        current: Vec::new(),
    }))
}

/// 物理行を1行追加する。logical_end が true ならその行で段落を確定する。
/// page_break は ==== マーカー文字列（NULL/空で改ページ無し）。
#[no_mangle]
pub extern "C" fn momo_doc_builder_add_line(
    b: *mut BrailleDocBuilder,
    content: *const u16,
    logical_end: bool,
    page_break: *const u16,
) {
    let b = match unsafe { b.as_mut() } {
        Some(b) => b,
        None => return,
    };
    let content = unsafe { lpwstr_to_string(content) }.unwrap_or_default();
    let pb = unsafe { lpwstr_to_string(page_break) }
        .filter(|s| !s.is_empty())
        .map(|m| PageBreak::from_marker(&m));
    b.current.push(PhysicalLine {
        content,
        logical_end,
        page_break: pb,
    });
    if logical_end {
        b.paragraphs.push(std::mem::take(&mut b.current));
    }
}

/// ビルダーからドキュメントを確定して返す。
///
/// b が NULL なら NULL。
///
/// # 注意（他の `_free` 関数との非対称性）
/// 本関数はビルダーを消費し、内部で解放する。他のハンドル型は「対応する
/// `_free` 関数を呼ぶまで有効」という単純な1対1の対応だが、
/// `BrailleDocBuilder` だけは `momo_doc_builder_build` と
/// `momo_doc_builder_free` の**どちらでも**無効化されうる。呼び出し側は
/// 本関数を呼んだ後、同じポインタを `momo_doc_builder_add_line` や
/// `momo_doc_builder_free` に**絶対に**渡してはならない（use-after-free /
/// 二重解放になる。呼び出し後にポインタを NULL にすることを推奨する。
/// `momo_raii.hpp` の `DocBuilder::build()` はこれを内部で行っている）。
#[no_mangle]
pub extern "C" fn momo_doc_builder_build(b: *mut BrailleDocBuilder) -> *mut BrailleDocHandle {
    if b.is_null() {
        return std::ptr::null_mut();
    }
    let builder = unsafe { *Box::from_raw(b) };
    let BrailleDocBuilder {
        config,
        mut paragraphs,
        current,
    } = builder;
    if !current.is_empty() {
        paragraphs.push(current);
    }
    Box::into_raw(Box::new(BrailleDocHandle {
        doc: BrailleDocument { paragraphs, config },
    }))
}

/// ビルダーを解放する（build を呼ばずに破棄する場合）。NULL は無視する。
#[no_mangle]
pub extern "C" fn momo_doc_builder_free(b: *mut BrailleDocBuilder) {
    if !b.is_null() {
        unsafe { drop(Box::from_raw(b)) };
    }
}

// ============================================================
// 書き出し（バイト列）
// ============================================================

pub struct ByteBuffer {
    bytes: Vec<u8>,
}

/// 出力形式コード → [`OutputFormat`]。不正なコードは `None`。
///
/// 0=MBR, 1=BES, 3=BASE(.bse), 4=BrailleText 大文字 NABCC(.brf),
/// 5=BrailleText 小文字 NABCC(.brf)。
fn output_format_from_code(format: c_int) -> Option<OutputFormat> {
    match format {
        0 => Some(OutputFormat::Mbr),
        1 => Some(OutputFormat::Bes),
        3 => Some(OutputFormat::Base),
        4 => Some(OutputFormat::BrailleText {
            case: NabccCase::Upper,
        }),
        5 => Some(OutputFormat::BrailleText {
            case: NabccCase::Lower,
        }),
        _ => None,
    }
}

/// ドキュメントを指定形式のバイト列へ書き出す。
/// format: [`output_format_from_code`] 参照。無効/NULL なら NULL。
#[no_mangle]
pub extern "C" fn momo_doc_write(h: *const BrailleDocHandle, format: c_int) -> *mut ByteBuffer {
    let h = match unsafe { h.as_ref() } {
        Some(h) => h,
        None => return std::ptr::null_mut(),
    };
    let fmt = match output_format_from_code(format) {
        Some(f) => f,
        None => return std::ptr::null_mut(),
    };
    let bytes = fmt.write(&h.doc);
    Box::into_raw(Box::new(ByteBuffer { bytes }))
}

/// バイト列の長さ。NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_bytes_len(b: *const ByteBuffer) -> c_int {
    match unsafe { b.as_ref() } {
        Some(b) => b.bytes.len() as c_int,
        None => -1,
    }
}

/// バイト列を out へコピーする。out は momo_bytes_len バイト以上確保すること。
/// b または out が NULL なら何もしない。
#[no_mangle]
pub extern "C" fn momo_bytes_copy(b: *const ByteBuffer, out: *mut u8) {
    if let Some(b) = unsafe { b.as_ref() } {
        if !out.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(b.bytes.as_ptr(), out, b.bytes.len()) };
        }
    }
}

/// バイト列を解放する。NULL は無視する。
#[no_mangle]
pub extern "C" fn momo_bytes_free(b: *mut ByteBuffer) {
    if !b.is_null() {
        unsafe { drop(Box::from_raw(b)) };
    }
}

// ============================================================
// 描画（印刷イメージ）
// ============================================================

pub struct FormattedDocHandle {
    doc: FormattedDocument,
}

/// 正本ドキュメントを印刷イメージへ描画してハンドルを返す。NULL なら NULL。
#[no_mangle]
pub extern "C" fn momo_doc_render(h: *const BrailleDocHandle) -> *mut FormattedDocHandle {
    let h = match unsafe { h.as_ref() } {
        Some(h) => h,
        None => return std::ptr::null_mut(),
    };
    Box::into_raw(Box::new(FormattedDocHandle {
        doc: render(&h.doc),
    }))
}

/// 印刷イメージハンドルを解放する。NULL は無視する。
#[no_mangle]
pub extern "C" fn momo_formatted_free(h: *mut FormattedDocHandle) {
    if !h.is_null() {
        unsafe { drop(Box::from_raw(h)) };
    }
}

#[no_mangle]
pub extern "C" fn momo_formatted_page_count(h: *const FormattedDocHandle) -> c_int {
    match unsafe { h.as_ref() } {
        Some(h) => h.doc.page_count() as c_int,
        None => -1,
    }
}

#[no_mangle]
pub extern "C" fn momo_formatted_line_count(h: *const FormattedDocHandle, page: c_int) -> c_int {
    let h = match unsafe { h.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    match h.doc.pages().get(page as usize) {
        Some(p) => p.len() as c_int,
        None => -1,
    }
}

fn fmt_line(h: &FormattedDocHandle, page: c_int, line: c_int) -> Option<&RenderedLine> {
    h.doc
        .pages()
        .get(page as usize)
        .and_then(|p| p.get(line as usize))
}

/// 印刷イメージの物理行テキスト（UTF-16, null 終端）。無効な引数なら -1。
#[no_mangle]
pub extern "C" fn momo_formatted_line_w(
    h: *const FormattedDocHandle,
    page: c_int,
    line: c_int,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    let h = match unsafe { h.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    match fmt_line(h, page, line) {
        Some(l) => write_utf16(&l.content, buf, buf_len),
        None => -1,
    }
}

/// 物理行がページヘッダ行か。無効な引数なら false。
#[no_mangle]
pub extern "C" fn momo_formatted_line_is_header(
    h: *const FormattedDocHandle,
    page: c_int,
    line: c_int,
) -> bool {
    match unsafe { h.as_ref() } {
        Some(h) => fmt_line(h, page, line)
            .map(|l| l.is_header)
            .unwrap_or(false),
        None => false,
    }
}

/// 物理行が論理行末尾か。無効な引数なら false。
#[no_mangle]
pub extern "C" fn momo_formatted_line_logical_end(
    h: *const FormattedDocHandle,
    page: c_int,
    line: c_int,
) -> bool {
    match unsafe { h.as_ref() } {
        Some(h) => fmt_line(h, page, line)
            .map(|l| l.logical_end)
            .unwrap_or(false),
        None => false,
    }
}

/// 物理行の元セグメント通し番号（ヘッダ行や無効な引数なら -1）。
/// エディタが表示行 ↔ 論理位置（セグメント+オフセット）を対応づけるのに使う。
#[no_mangle]
pub extern "C" fn momo_formatted_line_segment_index(
    h: *const FormattedDocHandle,
    page: c_int,
    line: c_int,
) -> c_int {
    match unsafe { h.as_ref() } {
        Some(h) => fmt_line(h, page, line)
            .map(|l| l.segment_index)
            .unwrap_or(-1),
        None => -1,
    }
}

// ============================================================
// 1論理行の折返し（逐次編集の表示用）
// ============================================================

pub struct WrapLinesHandle {
    lines: Vec<PhysicalLine>,
}

/// 1論理行（UTF-16 点字文字列）を line_width マスで折返して物理行リストを返す。
/// 空文字列は count=0 のハンドル（NULL ではない）。text が NULL なら NULL。
#[no_mangle]
pub extern "C" fn momo_wrap_line_w(text: *const u16, line_width: c_int) -> *mut WrapLinesHandle {
    let s = match unsafe { lpwstr_to_string(text) } {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };
    let lines = wrap_line(&s, line_width.max(1) as usize);
    Box::into_raw(Box::new(WrapLinesHandle { lines }))
}

/// 論理行のサフィックスを折返す。first_line_remaining は現在行の残りマス数
/// （0 で先頭に空行を出し以降通常幅）。空文字列は count=0。text が NULL なら NULL。
#[no_mangle]
pub extern "C" fn momo_wrap_suffix_w(
    text: *const u16,
    line_width: c_int,
    first_line_remaining: c_int,
) -> *mut WrapLinesHandle {
    let s = match unsafe { lpwstr_to_string(text) } {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };
    let lines = wrap_suffix(
        &s,
        line_width.max(1) as usize,
        first_line_remaining.max(0) as usize,
    );
    Box::into_raw(Box::new(WrapLinesHandle { lines }))
}

/// 折返しハンドルを解放する。NULL は無視する。
#[no_mangle]
pub extern "C" fn momo_wrap_lines_free(h: *mut WrapLinesHandle) {
    if !h.is_null() {
        unsafe { drop(Box::from_raw(h)) };
    }
}

#[no_mangle]
pub extern "C" fn momo_wrap_lines_count(h: *const WrapLinesHandle) -> c_int {
    match unsafe { h.as_ref() } {
        Some(h) => h.lines.len() as c_int,
        None => -1,
    }
}

/// 物理行テキスト（UTF-16, null 終端）。無効な引数なら -1。
#[no_mangle]
pub extern "C" fn momo_wrap_lines_get_w(
    h: *const WrapLinesHandle,
    index: c_int,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    let h = match unsafe { h.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    match h.lines.get(index as usize) {
        Some(l) => write_utf16(&l.content, buf, buf_len),
        None => -1,
    }
}

/// 物理行が論理行末尾か。無効な引数なら false。
#[no_mangle]
pub extern "C" fn momo_wrap_lines_logical_end(h: *const WrapLinesHandle, index: c_int) -> bool {
    match unsafe { h.as_ref() } {
        Some(h) => h
            .lines
            .get(index as usize)
            .map(|l| l.logical_end)
            .unwrap_or(false),
        None => false,
    }
}
// ============================================================
// 段落ベースの便宜関数（エディタ向け）
//
// エディタは論理段落（\n 区切りテキスト）＋設定だけを保持し、折返し・ページ分割・
// ヘッダ生成・各形式の符号化はすべて Rust に委ねる。
// ============================================================

fn config_from_params(
    line_width: c_int,
    lines_per_page: c_int,
    page_header: bool,
    number_start: c_int,
    title: *const u16,
) -> DocumentConfig {
    let title = unsafe { lpwstr_to_string(title) }.filter(|s| !s.is_empty());
    DocumentConfig {
        line_width: line_width.max(1) as usize,
        lines_per_page: lines_per_page.max(1) as usize,
        page_header,
        title,
        number_start: number_start.max(0) as u32,
    }
}

fn paragraphs_from_text(text: &str) -> Vec<String> {
    text.split('\n').map(str::to_owned).collect()
}

/// 段落 `para` の論理テキスト（物理行を連結したもの）を UTF-16 で buf に書く。
/// 無効な引数なら -1。
#[no_mangle]
pub extern "C" fn momo_doc_logical_text_w(
    h: *const BrailleDocHandle,
    para: c_int,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    let h = match unsafe { h.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    if para < 0 || para as usize >= h.doc.paragraphs.len() {
        return -1;
    }
    write_utf16(&h.doc.logical_text(para as usize), buf, buf_len)
}

/// 論理段落（UTF-16, `\n` 区切り）を折返し・ページ分割して印刷イメージを返す。
/// paragraphs が NULL なら NULL。
#[no_mangle]
pub extern "C" fn momo_doc_render_from_paragraphs(
    paragraphs: *const u16,
    line_width: c_int,
    lines_per_page: c_int,
    page_header: bool,
    number_start: c_int,
    title: *const u16,
) -> *mut FormattedDocHandle {
    let text = match unsafe { lpwstr_to_string(paragraphs) } {
        Some(t) => t,
        None => return std::ptr::null_mut(),
    };
    let config = config_from_params(line_width, lines_per_page, page_header, number_start, title);
    let doc = BrailleDocument::from_paragraphs(&paragraphs_from_text(&text), config);
    Box::into_raw(Box::new(FormattedDocHandle { doc: render(&doc) }))
}

/// 論理段落（UTF-16, `\n` 区切り）を折返した上で指定形式のバイト列へ書き出す。
/// `format`: [`output_format_from_code`] 参照。paragraphs が NULL／format 不正なら NULL。
#[no_mangle]
pub extern "C" fn momo_doc_write_from_paragraphs(
    paragraphs: *const u16,
    line_width: c_int,
    lines_per_page: c_int,
    page_header: bool,
    number_start: c_int,
    title: *const u16,
    format: c_int,
) -> *mut ByteBuffer {
    let text = match unsafe { lpwstr_to_string(paragraphs) } {
        Some(t) => t,
        None => return std::ptr::null_mut(),
    };
    let fmt = match output_format_from_code(format) {
        Some(f) => f,
        None => return std::ptr::null_mut(),
    };
    let config = config_from_params(line_width, lines_per_page, page_header, number_start, title);
    let doc = BrailleDocument::from_paragraphs(&paragraphs_from_text(&text), config);
    Box::into_raw(Box::new(ByteBuffer {
        bytes: fmt.write(&doc),
    }))
}

// ============================================================
// 逆点訳（点字 → かな表層）
//
// エディタの編集画面で各点字セルの下に読みを並べるガイド表示に使う。
// 逆変換器ハンドルを一度作って使い回し、行ごとに momo_back_translate_w を呼ぶ。
//
// 点字セルはすべて BMP（U+2800〜）なので、セルインデックスは UTF-16 の
// コード単位インデックスとして C# の String にそのまま対応する。
// ============================================================

pub struct BackTranslatorHandle {
    inner: BrailleBackTranslator,
}

pub struct BackTransHandle {
    /// null 終端 UTF-16 全文
    text_utf16: Vec<u16>,
    /// 各セグメントの点字セル開始位置
    seg_cell_start: Vec<i32>,
    /// 各セグメントの点字セル終端位置（排他的）
    seg_cell_end: Vec<i32>,
    /// 各セグメントの null 終端 UTF-16 テキスト
    seg_text_utf16: Vec<Vec<u16>>,
}

impl BackTransHandle {
    fn new(result: BackTransResult) -> Self {
        let mut text_utf16: Vec<u16> = result.text.encode_utf16().collect();
        text_utf16.push(0);

        let mut seg_cell_start = Vec::with_capacity(result.segments.len());
        let mut seg_cell_end = Vec::with_capacity(result.segments.len());
        let mut seg_text_utf16 = Vec::with_capacity(result.segments.len());
        for seg in &result.segments {
            seg_cell_start.push(seg.cell_start as i32);
            seg_cell_end.push(seg.cell_end as i32);
            let mut t: Vec<u16> = seg.text.encode_utf16().collect();
            t.push(0);
            seg_text_utf16.push(t);
        }

        Self {
            text_utf16,
            seg_cell_start,
            seg_cell_end,
            seg_text_utf16,
        }
    }
}

/// 組み込みテーブルで逆変換器を作る。失敗時は NULL。
#[no_mangle]
pub extern "C" fn momo_back_translator_new() -> *mut BackTranslatorHandle {
    match BrailleBackTranslator::from_embedded() {
        Ok(t) => Box::into_raw(Box::new(BackTranslatorHandle { inner: t })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// `japanese_grade1_braille.toml`（UTF-16 パス）から逆変換器を作る。失敗時は NULL。
#[no_mangle]
pub extern "C" fn momo_back_translator_new_from_file_w(
    toml_path: *const u16,
) -> *mut BackTranslatorHandle {
    let path = match unsafe { lpwstr_to_string(toml_path) } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    match BrailleBackTranslator::from_file(&path) {
        Ok(t) => Box::into_raw(Box::new(BackTranslatorHandle { inner: t })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// 逆変換器を解放する。NULL は無視する。
#[no_mangle]
pub extern "C" fn momo_back_translator_free(handle: *mut BackTranslatorHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// 点字文字列（UTF-16）を逆変換し、全文とセグメントを持つハンドルを返す。
/// handle または braille が NULL なら NULL。
#[no_mangle]
pub extern "C" fn momo_back_translate_w(
    handle: *const BackTranslatorHandle,
    braille: *const u16,
) -> *mut BackTransHandle {
    let translator = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return std::ptr::null_mut(),
    };
    let text = match unsafe { lpwstr_to_string(braille) } {
        Some(t) => t,
        None => return std::ptr::null_mut(),
    };
    let result = translator.inner.back_translate_aligned(&text);
    Box::into_raw(Box::new(BackTransHandle::new(result)))
}

/// 逆変換結果を解放する。NULL は無視する。
#[no_mangle]
pub extern "C" fn momo_back_trans_free(handle: *mut BackTransHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// 復元された全文（UTF-16, null 終端）を buf に書く。
/// 戻り値は必要な u16 要素数（null 含む）。buf 不足／NULL なら書かずにサイズだけ返す。
/// handle が NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_back_trans_text_w(
    handle: *const BackTransHandle,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    let h = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    let needed = h.text_utf16.len() as c_int;
    if !buf.is_null() && buf_len >= needed {
        unsafe {
            std::ptr::copy_nonoverlapping(h.text_utf16.as_ptr(), buf, h.text_utf16.len());
        }
    }
    needed
}

/// セグメント数を返す。handle が NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_back_trans_segment_count(handle: *const BackTransHandle) -> c_int {
    match unsafe { handle.as_ref() } {
        Some(h) => h.seg_cell_start.len() as c_int,
        None => -1,
    }
}

/// 各セグメントの点字セル範囲を out_start / out_end へまとめて書く。
///
/// 各配列は momo_back_trans_segment_count 要素以上を確保すること。
/// 範囲は半開区間 `[start, end)`、インデックスは入力点字の UTF-16 位置に対応する。
/// handle が NULL なら何もしない。片方のみ NULL なら非 NULL 側だけ書く。
#[no_mangle]
pub extern "C" fn momo_back_trans_cell_bounds(
    handle: *const BackTransHandle,
    out_start: *mut i32,
    out_end: *mut i32,
) {
    if let Some(h) = unsafe { handle.as_ref() } {
        if !out_start.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    h.seg_cell_start.as_ptr(),
                    out_start,
                    h.seg_cell_start.len(),
                );
            }
        }
        if !out_end.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    h.seg_cell_end.as_ptr(),
                    out_end,
                    h.seg_cell_end.len(),
                );
            }
        }
    }
}

/// セグメント idx のテキスト（UTF-16, null 終端）を buf に書く。
/// 戻り値は必要な u16 要素数（null 含む）。buf 不足／NULL なら書かずにサイズだけ返す。
/// handle が NULL、または idx が範囲外なら -1。
#[no_mangle]
pub extern "C" fn momo_back_trans_segment_text_w(
    handle: *const BackTransHandle,
    idx: c_int,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    let h = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    let seg = match h.seg_text_utf16.get(idx as usize) {
        Some(s) => s,
        None => return -1,
    };
    let needed = seg.len() as c_int;
    if !buf.is_null() && buf_len >= needed {
        unsafe {
            std::ptr::copy_nonoverlapping(seg.as_ptr(), buf, seg.len());
        }
    }
    needed
}
