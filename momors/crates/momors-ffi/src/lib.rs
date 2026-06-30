//! Windows DLL 向け C FFI。
//!
//! # 関数一覧
//!
//! ## 予測器ライフサイクル
//! ```c
//! // toml_path が NULL なら組み込みテーブルにフォールバック（モデル横の .toml も自動探索）
//! MomoPredictor momo_predictor_new  (const char*     model_path_utf8,  const char*     toml_path_utf8);
//! MomoPredictor momo_predictor_new_w(const uint16_t* model_path_utf16, const uint16_t* toml_path_utf16);
//! void          momo_predictor_free (MomoPredictor predictor);
//! // 点字テーブルのみ差し替え（モデル再読み込みなし）。失敗時は false。
//! bool          momo_predictor_set_table  (MomoPredictor, const char*     toml_path_utf8);
//! bool          momo_predictor_set_table_w(MomoPredictor, const uint16_t* toml_path_utf16);
//! ```
//!
//! ## 予測実行
//! ```c
//! MomoPrediction momo_predict  (MomoPredictor, const char*     src_utf8);
//! MomoPrediction momo_predict_w(MomoPredictor, const uint16_t* src_utf16);
//! void           momo_prediction_free(MomoPrediction);
//! ```
//!
//! ## かな・点字テキスト取得
//! 戻り値は必要なバッファサイズ（null 終端含む）。
//! `buf_len` が足りない場合は buf に書かず、サイズだけ返す。
//! 点字は予測結果のかなをさらに日本語点字へ変換したもの。
//! ```c
//! int32_t momo_prediction_kana     (MomoPrediction, char*     buf, int32_t buf_len);
//! int32_t momo_prediction_kana_w   (MomoPrediction, uint16_t* buf, int32_t buf_len);
//! int32_t momo_prediction_braille  (MomoPrediction, char*     buf, int32_t buf_len);
//! int32_t momo_prediction_braille_w(MomoPrediction, uint16_t* buf, int32_t buf_len);
//! ```
//!
//! ## サイズ照会
//! ```c
//! int32_t momo_prediction_kana_char_count   (MomoPrediction);
//! int32_t momo_prediction_src_char_count    (MomoPrediction);
//! int32_t momo_prediction_braille_char_count(MomoPrediction);
//! ```
//!
//! ## インデックス配列（コードポイント単位）
//! ```c
//! // kana→src: kana_char_count 要素の int32_t 配列
//! void momo_prediction_kana_to_src(MomoPrediction, int32_t* out);
//!
//! // src→kana: CSR 形式
//! //   row_ptr: src_char_count+1 要素  (row_ptr[i]..row_ptr[i+1] が src[i] に対応するかなインデックス範囲)
//! //   col_idx: kana_char_count 要素
//! void momo_prediction_src_to_kana(MomoPrediction, int32_t* row_ptr, int32_t* col_idx);
//!
//! // src→点字: CSR 形式
//! //   row_ptr: src_char_count+1 要素  (row_ptr[i]..row_ptr[i+1] が src[i] に対応する点字インデックス範囲)
//! //   col_idx: 最大 braille_char_count 要素（複合音の重複を除いた実際の要素数は row_ptr 末尾値で確認）
//! void momo_prediction_src_to_braille(MomoPrediction, int32_t* row_ptr, int32_t* col_idx);
//!
//! // 点字→src: braille_char_count 要素の int32_t 配列
//! void momo_prediction_braille_to_src(MomoPrediction, int32_t* out);
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

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

use momors_braille::document::{BrailleDocument, DocumentConfig, PageBreak, PhysicalLine};
use momors_braille::formatter::{render, wrap_line, wrap_suffix, FormattedDocument, RenderedLine};
use momors_braille::writer::OutputFormat;
use momors_braille::{BackTransResult, BrailleBackTranslator, BrailleConverter};
use momors_core::{PredictionResult, Predictor, PredictorConfig};

// ============================================================
// 内部ハンドル型
// ============================================================

pub struct PredictorHandle {
    inner: Predictor,
    /// かな→点字変換器。モデルと同じ場所の `japanese_grade1_braille.toml`、
    /// 無ければ組み込みテーブルから構築する。
    converter: BrailleConverter,
}

pub struct PredictionHandle {
    /// null 終端 UTF-8 かなテキスト
    kana_utf8: Vec<u8>,
    /// null 終端 UTF-16 かなテキスト
    kana_utf16: Vec<u16>,
    /// null 終端 UTF-8 点字テキスト（かなを日本語点字に変換したもの）
    braille_utf8: Vec<u8>,
    /// null 終端 UTF-16 点字テキスト
    braille_utf16: Vec<u16>,
    /// 点字のコードポイント数
    braille_char_count: i32,
    /// かな char_idx → 原文 char_idx (kana_char_count 要素)
    kana_to_src: Vec<i32>,
    /// CSR 行ポインタ (src_char_count+1 要素)
    src_to_kana_row: Vec<i32>,
    /// CSR 列インデックス (kana_char_count 要素)
    src_to_kana_col: Vec<i32>,
    /// CSR 行ポインタ (src_char_count+1 要素)
    src_to_braille_row: Vec<i32>,
    /// CSR 列インデックス (≤ braille_char_count 要素、複合音の重複を除く)
    src_to_braille_col: Vec<i32>,
    /// 点字 char_idx → 原文 char_idx (braille_char_count 要素)
    braille_to_src: Vec<i32>,
}

impl PredictionHandle {
    fn new(result: PredictionResult, converter: &BrailleConverter) -> Self {
        let k2s = result.kana_to_source_char();
        let s2k = result.source_to_kana_char();

        let mut kana_utf8 = result.kana_text().as_bytes().to_vec();
        kana_utf8.push(0);

        let mut kana_utf16: Vec<u16> = result.kana_text().encode_utf16().collect();
        kana_utf16.push(0);

        // かな→点字。変換に失敗した場合は空文字列にフォールバックする。
        let (braille_str, kana_to_braille) = match converter.convert(result.kana_text()) {
            Ok(b) => {
                let k2b = b.kana_to_braille().to_vec();
                (b.braille_text().to_owned(), k2b)
            }
            Err(_) => (String::new(), Vec::new()),
        };
        let braille_char_count = braille_str.chars().count() as i32;
        let mut braille_utf8 = braille_str.as_bytes().to_vec();
        braille_utf8.push(0);
        let mut braille_utf16: Vec<u16> = braille_str.encode_utf16().collect();
        braille_utf16.push(0);

        let kana_to_src: Vec<i32> = k2s.iter().map(|&i| i as i32).collect();

        let mut src_to_kana_row = Vec::with_capacity(s2k.len() + 1);
        let mut src_to_kana_col = Vec::new();
        let mut offset = 0i32;
        for kanas in &s2k {
            src_to_kana_row.push(offset);
            for &ki in kanas {
                src_to_kana_col.push(ki as i32);
            }
            offset += kanas.len() as i32;
        }
        src_to_kana_row.push(offset);

        let s2b = result.source_to_braille_char(&kana_to_braille);
        let mut src_to_braille_row = Vec::with_capacity(s2b.len() + 1);
        let mut src_to_braille_col = Vec::new();
        let mut offset = 0i32;
        for brailles in &s2b {
            src_to_braille_row.push(offset);
            for &bi in brailles {
                src_to_braille_col.push(bi as i32);
            }
            offset += brailles.len() as i32;
        }
        src_to_braille_row.push(offset);

        let b2s_raw = result.braille_char_to_source(&kana_to_braille, braille_char_count as usize);
        let braille_to_src: Vec<i32> = b2s_raw.iter().map(|&i| i as i32).collect();

        Self {
            kana_utf8,
            kana_utf16,
            braille_utf8,
            braille_utf16,
            braille_char_count,
            kana_to_src,
            src_to_kana_row,
            src_to_kana_col,
            src_to_braille_row,
            src_to_braille_col,
            braille_to_src,
        }
    }
}

// ============================================================
// 文字列変換ヘルパ
// ============================================================

unsafe fn lpstr_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .ok()
        .map(str::to_owned)
}

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
// 予測器: 作成 / 解放
// ============================================================

/// モデルパスから予測器とかな→点字変換器を構築する。失敗時は NULL。
///
/// モデルと同じディレクトリにある `single_character_dic.tsv`（漢字辞書）があれば利用し、
/// 無ければ辞書なしにフォールバックする。
///
/// `braille_table` が `Some(path)` のとき指定ファイルからテーブルを読む（失敗すれば NULL）。
/// `None` のときはモデル横の `japanese_grade1_braille.toml` を優先し、無ければ組み込みテーブルを使う。
fn build_predictor(model_path: &str, braille_table: Option<&str>) -> *mut PredictorHandle {
    let path = std::path::Path::new(model_path);
    let dir = path.parent();

    let mut config = PredictorConfig::new(path);
    if let Some(dir) = dir {
        let dict = dir.join("single_character_dic.tsv");
        if dict.exists() {
            config = config.with_kanji_dict_path(&dict);
        }
    }

    let predictor = match Predictor::load(config) {
        Ok(p) => p,
        Err(_) => return std::ptr::null_mut(),
    };

    let converter = match braille_table {
        Some(table_path) => BrailleConverter::from_file(table_path).ok(),
        None => dir
            .map(|d| d.join("japanese_grade1_braille.toml"))
            .filter(|p| p.exists())
            .and_then(|p| BrailleConverter::from_file(&p).ok())
            .or_else(|| BrailleConverter::from_embedded().ok()),
    };
    let converter = match converter {
        Some(c) => c,
        None => return std::ptr::null_mut(),
    };

    Box::into_raw(Box::new(PredictorHandle {
        inner: predictor,
        converter,
    }))
}

/// 点字テーブルのみを差し替える（UTF-8 パス）。モデルの再読み込みは行わない。
/// 成功時 true、失敗時 false（ハンドルの変換器は変更されない）。
/// handle が NULL または toml_path が NULL なら false を返す。
#[no_mangle]
pub extern "C" fn momo_predictor_set_table(
    handle: *mut PredictorHandle,
    toml_path: *const c_char,
) -> bool {
    let h = match unsafe { handle.as_mut() } {
        Some(h) => h,
        None => return false,
    };
    let path = match unsafe { lpstr_to_string(toml_path) } {
        Some(p) => p,
        None => return false,
    };
    match BrailleConverter::from_file(&path) {
        Ok(c) => {
            h.converter = c;
            true
        }
        Err(_) => false,
    }
}

/// 点字テーブルのみを差し替える（UTF-16 パス）。モデルの再読み込みは行わない。
/// 成功時 true、失敗時 false（ハンドルの変換器は変更されない）。
/// handle が NULL または toml_path が NULL なら false を返す。
#[no_mangle]
pub extern "C" fn momo_predictor_set_table_w(
    handle: *mut PredictorHandle,
    toml_path: *const u16,
) -> bool {
    let h = match unsafe { handle.as_mut() } {
        Some(h) => h,
        None => return false,
    };
    let path = match unsafe { lpwstr_to_string(toml_path) } {
        Some(p) => p,
        None => return false,
    };
    match BrailleConverter::from_file(&path) {
        Ok(c) => {
            h.converter = c;
            true
        }
        Err(_) => false,
    }
}

/// UTF-8 パスからモデルを読み込み予測器を作成する。
/// `toml_path` が NULL なら自動選択（モデル横の .toml → 組み込みテーブル）。
/// 失敗時は NULL を返す。
#[no_mangle]
pub extern "C" fn momo_predictor_new(
    model_path: *const c_char,
    toml_path: *const c_char,
) -> *mut PredictorHandle {
    let model = match unsafe { lpstr_to_string(model_path) } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    let table = unsafe { lpstr_to_string(toml_path) };
    build_predictor(&model, table.as_deref())
}

/// UTF-16 パスからモデルを読み込み予測器を作成する。
/// `toml_path` が NULL なら自動選択（モデル横の .toml → 組み込みテーブル）。
/// 失敗時は NULL を返す。
#[no_mangle]
pub extern "C" fn momo_predictor_new_w(
    model_path: *const u16,
    toml_path: *const u16,
) -> *mut PredictorHandle {
    let model = match unsafe { lpwstr_to_string(model_path) } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    let table = unsafe { lpwstr_to_string(toml_path) };
    build_predictor(&model, table.as_deref())
}

/// 予測器を解放する。NULL は無視する。
#[no_mangle]
pub extern "C" fn momo_predictor_free(handle: *mut PredictorHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

// ============================================================
// 予測: 実行 / 解放
// ============================================================

/// UTF-8 テキストを予測する。失敗時は NULL を返す。
#[no_mangle]
pub extern "C" fn momo_predict(
    handle: *const PredictorHandle,
    src_text: *const c_char,
) -> *mut PredictionHandle {
    let predictor = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return std::ptr::null_mut(),
    };
    let text = match unsafe { lpstr_to_string(src_text) } {
        Some(t) => t,
        None => return std::ptr::null_mut(),
    };
    match predictor.inner.predict(&text) {
        Ok(r) => Box::into_raw(Box::new(PredictionHandle::new(r, &predictor.converter))),
        Err(_) => std::ptr::null_mut(),
    }
}

/// UTF-16 テキストを予測する。失敗時は NULL を返す。
#[no_mangle]
pub extern "C" fn momo_predict_w(
    handle: *const PredictorHandle,
    src_text: *const u16,
) -> *mut PredictionHandle {
    let predictor = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return std::ptr::null_mut(),
    };
    let text = match unsafe { lpwstr_to_string(src_text) } {
        Some(t) => t,
        None => return std::ptr::null_mut(),
    };
    match predictor.inner.predict(&text) {
        Ok(r) => Box::into_raw(Box::new(PredictionHandle::new(r, &predictor.converter))),
        Err(_) => std::ptr::null_mut(),
    }
}

/// 予測結果を解放する。NULL は無視する。
#[no_mangle]
pub extern "C" fn momo_prediction_free(handle: *mut PredictionHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

// ============================================================
// かなテキスト取得
// ============================================================

/// かなテキスト (UTF-8, null 終端) を buf に書き込む。
///
/// 戻り値: 必要なバイト数 (null 終端含む)。
/// buf_len が不足または buf が NULL の場合は書き込まず、必要サイズだけ返す。
/// handle が NULL なら -1 を返す。
#[no_mangle]
pub extern "C" fn momo_prediction_kana(
    handle: *const PredictionHandle,
    buf: *mut c_char,
    buf_len: c_int,
) -> c_int {
    let h = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    let needed = h.kana_utf8.len() as c_int;
    if !buf.is_null() && buf_len >= needed {
        unsafe {
            std::ptr::copy_nonoverlapping(h.kana_utf8.as_ptr(), buf as *mut u8, h.kana_utf8.len());
        }
    }
    needed
}

/// かなテキスト (UTF-16, null 終端) を buf に書き込む。
///
/// 戻り値: 必要な u16 要素数 (null 終端含む)。
/// buf_len が不足または buf が NULL の場合は書き込まず、必要サイズだけ返す。
/// handle が NULL なら -1 を返す。
#[no_mangle]
pub extern "C" fn momo_prediction_kana_w(
    handle: *const PredictionHandle,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    let h = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    let needed = h.kana_utf16.len() as c_int;
    if !buf.is_null() && buf_len >= needed {
        unsafe {
            std::ptr::copy_nonoverlapping(h.kana_utf16.as_ptr(), buf, h.kana_utf16.len());
        }
    }
    needed
}

// ============================================================
// 点字テキスト取得
// ============================================================

/// 点字テキスト (UTF-8, null 終端) を buf に書き込む。
///
/// 戻り値: 必要なバイト数 (null 終端含む)。
/// buf_len が不足または buf が NULL の場合は書き込まず、必要サイズだけ返す。
/// handle が NULL なら -1 を返す。
#[no_mangle]
pub extern "C" fn momo_prediction_braille(
    handle: *const PredictionHandle,
    buf: *mut c_char,
    buf_len: c_int,
) -> c_int {
    let h = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    let needed = h.braille_utf8.len() as c_int;
    if !buf.is_null() && buf_len >= needed {
        unsafe {
            std::ptr::copy_nonoverlapping(
                h.braille_utf8.as_ptr(),
                buf as *mut u8,
                h.braille_utf8.len(),
            );
        }
    }
    needed
}

/// 点字テキスト (UTF-16, null 終端) を buf に書き込む。
///
/// 戻り値: 必要な u16 要素数 (null 終端含む)。
/// buf_len が不足または buf が NULL の場合は書き込まず、必要サイズだけ返す。
/// handle が NULL なら -1 を返す。
#[no_mangle]
pub extern "C" fn momo_prediction_braille_w(
    handle: *const PredictionHandle,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    let h = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    let needed = h.braille_utf16.len() as c_int;
    if !buf.is_null() && buf_len >= needed {
        unsafe {
            std::ptr::copy_nonoverlapping(h.braille_utf16.as_ptr(), buf, h.braille_utf16.len());
        }
    }
    needed
}

// ============================================================
// サイズ照会
// ============================================================

/// かなテキストのコードポイント数を返す。handle が NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_prediction_kana_char_count(handle: *const PredictionHandle) -> c_int {
    match unsafe { handle.as_ref() } {
        Some(h) => h.kana_to_src.len() as c_int,
        None => -1,
    }
}

/// 原文のコードポイント数を返す。handle が NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_prediction_src_char_count(handle: *const PredictionHandle) -> c_int {
    match unsafe { handle.as_ref() } {
        // src_to_kana_row は src_char_count+1 要素
        Some(h) => (h.src_to_kana_row.len() as c_int) - 1,
        None => -1,
    }
}

// ============================================================
// インデックス配列
// ============================================================

/// かな→原文 コードポイントインデックス配列を out に書き込む。
///
/// out は kana_char_count 要素以上の領域を確保しておくこと。
/// handle または out が NULL なら何もしない。
#[no_mangle]
pub extern "C" fn momo_prediction_kana_to_src(handle: *const PredictionHandle, out: *mut i32) {
    if let Some(h) = unsafe { handle.as_ref() } {
        if !out.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(h.kana_to_src.as_ptr(), out, h.kana_to_src.len());
            }
        }
    }
}

/// src→かな インデックスを CSR 形式で書き込む。
///
/// - row_ptr: src_char_count+1 要素以上の領域を確保すること。
///   row_ptr[i]..row_ptr[i+1] が原文文字 i に対応するかな文字インデックスの範囲。
/// - col_idx: kana_char_count 要素以上の領域を確保すること。
///
/// handle が NULL、または両ポインタが NULL なら何もしない。
/// 片方のみ NULL の場合は非 NULL 側だけ書く。
#[no_mangle]
pub extern "C" fn momo_prediction_src_to_kana(
    handle: *const PredictionHandle,
    row_ptr: *mut i32,
    col_idx: *mut i32,
) {
    if let Some(h) = unsafe { handle.as_ref() } {
        if !row_ptr.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    h.src_to_kana_row.as_ptr(),
                    row_ptr,
                    h.src_to_kana_row.len(),
                );
            }
        }
        if !col_idx.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    h.src_to_kana_col.as_ptr(),
                    col_idx,
                    h.src_to_kana_col.len(),
                );
            }
        }
    }
}

/// 点字テキストのコードポイント数を返す。handle が NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_prediction_braille_char_count(handle: *const PredictionHandle) -> c_int {
    match unsafe { handle.as_ref() } {
        Some(h) => h.braille_char_count,
        None => -1,
    }
}

/// src→点字 インデックスを CSR 形式で書き込む。
///
/// - row_ptr: src_char_count+1 要素以上の領域を確保すること。
///   row_ptr[i]..row_ptr[i+1] が原文文字 i に対応する点字文字インデックスの範囲。
/// - col_idx: braille_char_count 要素以上の領域を確保すること。
///   複合音（キャ など）は重複が除去されるため、実際の要素数は
///   row_ptr[src_char_count]（末尾値）で確認する。
///
/// handle が NULL、または両ポインタが NULL なら何もしない。
/// 片方のみ NULL の場合は非 NULL 側だけ書く。
#[no_mangle]
pub extern "C" fn momo_prediction_src_to_braille(
    handle: *const PredictionHandle,
    row_ptr: *mut i32,
    col_idx: *mut i32,
) {
    if let Some(h) = unsafe { handle.as_ref() } {
        if !row_ptr.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    h.src_to_braille_row.as_ptr(),
                    row_ptr,
                    h.src_to_braille_row.len(),
                );
            }
        }
        if !col_idx.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    h.src_to_braille_col.as_ptr(),
                    col_idx,
                    h.src_to_braille_col.len(),
                );
            }
        }
    }
}

/// 点字→原文 コードポイントインデックス配列を out に書き込む。
///
/// out は braille_char_count 要素以上の領域を確保しておくこと。
/// handle または out が NULL なら何もしない。
#[no_mangle]
pub extern "C" fn momo_prediction_braille_to_src(handle: *const PredictionHandle, out: *mut i32) {
    if let Some(h) = unsafe { handle.as_ref() } {
        if !out.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    h.braille_to_src.as_ptr(),
                    out,
                    h.braille_to_src.len(),
                );
            }
        }
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

/// ビルダーからドキュメントを確定して返す。ビルダーは解放される（再利用・再解放不可）。
/// b が NULL なら NULL。
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

/// ドキュメントを指定形式のバイト列へ書き出す。
/// format: 0=MBR, 1=BES, 3=BASE(.bse), 4=BrailleText(.brf)。無効/NULL なら NULL。
#[no_mangle]
pub extern "C" fn momo_doc_write(h: *const BrailleDocHandle, format: c_int) -> *mut ByteBuffer {
    let h = match unsafe { h.as_ref() } {
        Some(h) => h,
        None => return std::ptr::null_mut(),
    };
    let fmt = match format {
        0 => OutputFormat::Mbr,
        1 => OutputFormat::Bes,
        3 => OutputFormat::Base,
        4 => OutputFormat::BrailleText,
        _ => return std::ptr::null_mut(),
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
/// `format`: 0=MBR, 1=BES, 3=BASE, 4=BrailleText。paragraphs が NULL／format 不正なら NULL。
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
    let fmt = match format {
        0 => OutputFormat::Mbr,
        1 => OutputFormat::Bes,
        3 => OutputFormat::Base,
        4 => OutputFormat::BrailleText,
        _ => return std::ptr::null_mut(),
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
