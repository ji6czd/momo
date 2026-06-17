//! Windows DLL 向け C FFI。
//!
//! # 関数一覧
//!
//! ## 予測器ライフサイクル
//! ```c
//! MomoPredictor momo_predictor_new  (const char*     model_path_utf8);
//! MomoPredictor momo_predictor_new_w(const uint16_t* model_path_utf16);
//! void          momo_predictor_free (MomoPredictor predictor);
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
//! int32_t momo_prediction_kana_char_count(MomoPrediction);
//! int32_t momo_prediction_src_char_count (MomoPrediction);
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
//! ```
//!
//! ## 1論理行フォーマット
//! 1論理行（点字文字列）を複数の物理行に折り返す。
//! 空文字列を渡すと count=0 のハンドルが返る（NULL ではない）。
//! `momo_format_suffix_w` は現在の物理行の残りマス数 `first_line_remaining` を指定して
//! 論理行のサフィックスをフォーマットする。
//! ```c
//! MomoPhysicalLines momo_format_logical_line_w(const uint16_t* text, int32_t line_width);
//! MomoPhysicalLines momo_format_suffix_w(const uint16_t* text, int32_t line_width, int32_t first_line_remaining);
//! void              momo_physical_lines_free(MomoPhysicalLines);
//! int32_t           momo_physical_lines_count(MomoPhysicalLines);
//! // 戻り値: 必要な u16 要素数 (null 終端含む)。buf_len 不足または buf NULL なら書かずサイズだけ返す。無効引数なら -1。
//! int32_t           momo_physical_lines_get_w(MomoPhysicalLines, int32_t index, uint16_t* buf, int32_t buf_len);
//! bool              momo_physical_lines_logical_end(MomoPhysicalLines, int32_t index);
//! ```
//!
//! ## BES ファイル読み込み
//! BES バイナリを物理行リストへ復元する。改ページの強制/暗黙判定とヘッダ除去を含む。
//! ```c
//! MomoBesDocument momo_bes_read(const uint8_t* bytes, int32_t len); // 失敗時 NULL
//! void            momo_bes_free(MomoBesDocument);
//! int32_t         momo_bes_line_width(MomoBesDocument);     // xl
//! int32_t         momo_bes_lines_per_page(MomoBesDocument); // yl
//! int32_t         momo_bes_line_count(MomoBesDocument);
//! int32_t         momo_bes_line_get_w(MomoBesDocument, int32_t index, uint16_t* buf, int32_t buf_len);
//! bool            momo_bes_line_logical_end(MomoBesDocument, int32_t index);
//! bool            momo_bes_line_page_break(MomoBesDocument, int32_t index);
//! ```

// cbindgen は #[no_mangle] (Rust 2024) 未対応のため edition 2021 を使用する。
// 1.82+ で発生する "unsafe attribute used without unsafe" lint を抑制する。
#![allow(unsafe_op_in_unsafe_fn)]

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

use momors_braille::formatter::{BrailleFormatter, FormattedDocument, FormatterConfig};
use momors_braille::BrailleConverter;
use momors_core::{PredictionResult, Predictor, PredictorConfig};

// ============================================================
// 内部ハンドル型
// ============================================================

pub struct PredictorHandle {
    inner: Predictor,
    /// かな→点字変換器。モデルと同じ場所の `japanese_braille.toml`、
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
    /// かな char_idx → 原文 char_idx (kana_char_count 要素)
    kana_to_src: Vec<i32>,
    /// CSR 行ポインタ (src_char_count+1 要素)
    src_to_kana_row: Vec<i32>,
    /// CSR 列インデックス (kana_char_count 要素)
    src_to_kana_col: Vec<i32>,
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
        let braille = converter
            .convert(result.kana_text())
            .map(|b| b.braille_text().to_owned())
            .unwrap_or_default();
        let mut braille_utf8 = braille.as_bytes().to_vec();
        braille_utf8.push(0);
        let mut braille_utf16: Vec<u16> = braille.encode_utf16().collect();
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

        Self {
            kana_utf8,
            kana_utf16,
            braille_utf8,
            braille_utf16,
            kana_to_src,
            src_to_kana_row,
            src_to_kana_col,
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
/// モデルと同じディレクトリにある `single_character_dic.tsv`（漢字辞書）と
/// `japanese_braille.toml`（点字テーブル）があれば利用し、無ければ
/// 辞書なし・組み込みテーブルにフォールバックする。
fn build_predictor(model_path: &str) -> *mut PredictorHandle {
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

    let converter = dir
        .map(|d| d.join("japanese_braille.toml"))
        .filter(|p| p.exists())
        .and_then(|p| BrailleConverter::from_file(&p).ok())
        .or_else(|| BrailleConverter::from_embedded().ok());
    let converter = match converter {
        Some(c) => c,
        None => return std::ptr::null_mut(),
    };

    Box::into_raw(Box::new(PredictorHandle {
        inner: predictor,
        converter,
    }))
}

/// UTF-8 パスからモデルを読み込み予測器を作成する。
/// 失敗時は NULL を返す。
#[no_mangle]
pub extern "C" fn momo_predictor_new(model_path: *const c_char) -> *mut PredictorHandle {
    match unsafe { lpstr_to_string(model_path) } {
        Some(p) => build_predictor(&p),
        None => std::ptr::null_mut(),
    }
}

/// UTF-16 パスからモデルを読み込み予測器を作成する。
/// 失敗時は NULL を返す。
#[no_mangle]
pub extern "C" fn momo_predictor_new_w(model_path: *const u16) -> *mut PredictorHandle {
    match unsafe { lpwstr_to_string(model_path) } {
        Some(p) => build_predictor(&p),
        None => std::ptr::null_mut(),
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

// ============================================================
// フォーマット済みドキュメント
// ============================================================

pub struct DocumentHandle {
    doc: FormattedDocument,
    page_header: bool,
}

/// 段落リスト (UTF-16, `\n` 区切り) をフォーマットしてドキュメントハンドルを返す。
/// 失敗時は NULL を返す。
#[no_mangle]
pub extern "C" fn momo_document_format_w(
    paragraphs_text: *const u16,
    line_width: c_int,
    lines_per_page: c_int,
    page_header: bool,
    title: *const u16,
) -> *mut DocumentHandle {
    let text = match unsafe { lpwstr_to_string(paragraphs_text) } {
        Some(t) => t,
        None => return std::ptr::null_mut(),
    };
    let title_str = unsafe { lpwstr_to_string(title) };
    let paragraphs: Vec<String> = if text.is_empty() {
        vec![]
    } else {
        text.split('\n').map(str::to_owned).collect()
    };
    let config = FormatterConfig {
        line_width: line_width as usize,
        lines_per_page: lines_per_page as usize,
        page_header,
        title: title_str,
    };
    let doc = BrailleFormatter::new(config).format(&paragraphs);
    Box::into_raw(Box::new(DocumentHandle { doc, page_header }))
}

/// ドキュメントハンドルを解放する。NULL は無視する。
#[no_mangle]
pub extern "C" fn momo_document_free(handle: *mut DocumentHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// 総ページ数を返す。handle が NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_document_page_count(handle: *const DocumentHandle) -> c_int {
    match unsafe { handle.as_ref() } {
        Some(h) => h.doc.page_count() as c_int,
        None => -1,
    }
}

/// 指定ページの物理行数を返す。無効な引数なら -1。
#[no_mangle]
pub extern "C" fn momo_document_line_count(handle: *const DocumentHandle, page: c_int) -> c_int {
    let h = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    match h.doc.pages().get(page as usize) {
        Some(p) => p.len() as c_int,
        None => -1,
    }
}

/// 物理行テキスト (UTF-16, null 終端) を buf に書き込む。
/// 戻り値: 必要な u16 要素数 (null 終端含む)。buf_len 不足または buf NULL なら書かずサイズだけ返す。
/// 無効な引数なら -1。
#[no_mangle]
pub extern "C" fn momo_document_line_w(
    handle: *const DocumentHandle,
    page: c_int,
    line: c_int,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    let h = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    let content = match h
        .doc
        .pages()
        .get(page as usize)
        .and_then(|p| p.get(line as usize))
    {
        Some(l) => &l.content,
        None => return -1,
    };
    let mut utf16: Vec<u16> = content.encode_utf16().collect();
    utf16.push(0);
    let needed = utf16.len() as c_int;
    if !buf.is_null() && buf_len >= needed {
        unsafe { std::ptr::copy_nonoverlapping(utf16.as_ptr(), buf, utf16.len()) };
    }
    needed
}

/// 指定行がページヘッダ行かどうかを返す。無効な引数なら false。
#[no_mangle]
pub extern "C" fn momo_document_line_is_header(
    handle: *const DocumentHandle,
    page: c_int,
    line: c_int,
) -> bool {
    match unsafe { handle.as_ref() } {
        Some(h) => h.page_header && page >= 0 && line == 0,
        None => false,
    }
}

/// 指定行が論理行の末尾かどうかを返す。無効な引数なら false。
#[no_mangle]
pub extern "C" fn momo_document_line_logical_end(
    handle: *const DocumentHandle,
    page: c_int,
    line: c_int,
) -> bool {
    let h = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return false,
    };
    h.doc
        .pages()
        .get(page as usize)
        .and_then(|p| p.get(line as usize))
        .map(|l| l.logical_end)
        .unwrap_or(false)
}

// ============================================================
// 1論理行フォーマット
// ============================================================

pub struct PhysicalLinesHandle {
    lines: Vec<momors_braille::formatter::PhysicalLine>,
}

/// 1論理行 (UTF-16 点字文字列) を物理行のリストに変換して返す。
///
/// 空文字列を渡すと行数 0 のハンドルが返る（NULL ではない）。
/// `text` が NULL なら NULL を返す。
#[no_mangle]
pub extern "C" fn momo_format_logical_line_w(
    text: *const u16,
    line_width: c_int,
) -> *mut PhysicalLinesHandle {
    let s = match unsafe { lpwstr_to_string(text) } {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };
    let config = FormatterConfig {
        line_width: line_width as usize,
        ..FormatterConfig::default()
    };
    let lines = BrailleFormatter::new(config).format_logical_line(&s);
    Box::into_raw(Box::new(PhysicalLinesHandle { lines }))
}

/// 論理行のサフィックス (UTF-16 点字文字列) を物理行のリストに変換して返す。
///
/// `first_line_remaining`: 現在の物理行の残りマス数。0 のとき先頭に空行を追加して通常幅で折り返す。
/// 空文字列を渡すと行数 0 のハンドルが返る（NULL ではない）。
/// `text` が NULL なら NULL を返す。
#[no_mangle]
pub extern "C" fn momo_format_suffix_w(
    text: *const u16,
    line_width: c_int,
    first_line_remaining: c_int,
) -> *mut PhysicalLinesHandle {
    let s = match unsafe { lpwstr_to_string(text) } {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };
    let config = FormatterConfig {
        line_width: line_width as usize,
        ..FormatterConfig::default()
    };
    let lines = BrailleFormatter::new(config).format_suffix(&s, first_line_remaining as usize);
    Box::into_raw(Box::new(PhysicalLinesHandle { lines }))
}

/// 物理行ハンドルを解放する。NULL は無視する。
#[no_mangle]
pub extern "C" fn momo_physical_lines_free(handle: *mut PhysicalLinesHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// 物理行数を返す。handle が NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_physical_lines_count(handle: *const PhysicalLinesHandle) -> c_int {
    match unsafe { handle.as_ref() } {
        Some(h) => h.lines.len() as c_int,
        None => -1,
    }
}

/// 指定インデックスの物理行テキスト (UTF-16, null 終端) を buf に書き込む。
///
/// 戻り値: 必要な u16 要素数 (null 終端含む)。buf_len 不足または buf が NULL なら書かずサイズだけ返す。
/// 無効な引数なら -1。
#[no_mangle]
pub extern "C" fn momo_physical_lines_get_w(
    handle: *const PhysicalLinesHandle,
    index: c_int,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    let h = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    let content = match h.lines.get(index as usize) {
        Some(l) => &l.content,
        None => return -1,
    };
    let mut utf16: Vec<u16> = content.encode_utf16().collect();
    utf16.push(0);
    let needed = utf16.len() as c_int;
    if !buf.is_null() && buf_len >= needed {
        unsafe { std::ptr::copy_nonoverlapping(utf16.as_ptr(), buf, utf16.len()) };
    }
    needed
}

/// 指定インデックスの物理行が論理行の末尾かどうかを返す。無効な引数なら false。
#[no_mangle]
pub extern "C" fn momo_physical_lines_logical_end(
    handle: *const PhysicalLinesHandle,
    index: c_int,
) -> bool {
    let h = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return false,
    };
    h.lines
        .get(index as usize)
        .map(|l| l.logical_end)
        .unwrap_or(false)
}

// ============================================================
// BES ファイル読み込み
// ============================================================

pub struct BesDocumentHandle {
    doc: momors_braille::reader::BesDocument,
}

/// BES バイナリ列を読み込んでハンドルを返す。
/// マジック不一致や破損では NULL を返す。`bytes` が NULL でも NULL。
#[no_mangle]
pub extern "C" fn momo_bes_read(bytes: *const u8, len: c_int) -> *mut BesDocumentHandle {
    if bytes.is_null() || len < 0 {
        return std::ptr::null_mut();
    }
    let slice = unsafe { std::slice::from_raw_parts(bytes, len as usize) };
    match momors_braille::reader::read_bes_file(slice) {
        Some(doc) => Box::into_raw(Box::new(BesDocumentHandle { doc })),
        None => std::ptr::null_mut(),
    }
}

/// BES ハンドルを解放する。NULL は無視する。
#[no_mangle]
pub extern "C" fn momo_bes_free(handle: *mut BesDocumentHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// 1行あたりのマス数 (xl)。handle が NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_bes_line_width(handle: *const BesDocumentHandle) -> c_int {
    match unsafe { handle.as_ref() } {
        Some(h) => h.doc.line_width as c_int,
        None => -1,
    }
}

/// 1ページあたりの行数 (yl)。handle が NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_bes_lines_per_page(handle: *const BesDocumentHandle) -> c_int {
    match unsafe { handle.as_ref() } {
        Some(h) => h.doc.lines_per_page as c_int,
        None => -1,
    }
}

/// 物理行数を返す。handle が NULL なら -1。
#[no_mangle]
pub extern "C" fn momo_bes_line_count(handle: *const BesDocumentHandle) -> c_int {
    match unsafe { handle.as_ref() } {
        Some(h) => h.doc.lines.len() as c_int,
        None => -1,
    }
}

/// 指定行のテキスト (UTF-16, null 終端) を buf に書き込む。
/// 戻り値: 必要な u16 要素数 (null 終端含む)。buf_len 不足または buf NULL なら書かずサイズだけ返す。
/// 無効な引数なら -1。
#[no_mangle]
pub extern "C" fn momo_bes_line_get_w(
    handle: *const BesDocumentHandle,
    index: c_int,
    buf: *mut u16,
    buf_len: c_int,
) -> c_int {
    let h = match unsafe { handle.as_ref() } {
        Some(h) => h,
        None => return -1,
    };
    let content = match h.doc.lines.get(index as usize) {
        Some(l) => &l.content,
        None => return -1,
    };
    let mut utf16: Vec<u16> = content.encode_utf16().collect();
    utf16.push(0);
    let needed = utf16.len() as c_int;
    if !buf.is_null() && buf_len >= needed {
        unsafe { std::ptr::copy_nonoverlapping(utf16.as_ptr(), buf, utf16.len()) };
    }
    needed
}

/// 指定行が論理行の末尾かどうか。無効な引数なら false。
#[no_mangle]
pub extern "C" fn momo_bes_line_logical_end(
    handle: *const BesDocumentHandle,
    index: c_int,
) -> bool {
    match unsafe { handle.as_ref() } {
        Some(h) => h
            .doc
            .lines
            .get(index as usize)
            .map(|l| l.logical_end)
            .unwrap_or(false),
        None => false,
    }
}

/// 指定行の直後で強制改ページするか。無効な引数なら false。
#[no_mangle]
pub extern "C" fn momo_bes_line_page_break(handle: *const BesDocumentHandle, index: c_int) -> bool {
    match unsafe { handle.as_ref() } {
        Some(h) => h
            .doc
            .lines
            .get(index as usize)
            .map(|l| l.page_break)
            .unwrap_or(false),
        None => false,
    }
}
