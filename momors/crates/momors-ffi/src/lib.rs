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
//! ## かなテキスト取得
//! 戻り値は必要なバッファサイズ（null 終端含む）。
//! `buf_len` が足りない場合は buf に書かず、サイズだけ返す。
//! ```c
//! int32_t momo_prediction_kana  (MomoPrediction, char*     buf, int32_t buf_len);
//! int32_t momo_prediction_kana_w(MomoPrediction, uint16_t* buf, int32_t buf_len);
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

// cbindgen は #[no_mangle] (Rust 2024) 未対応のため edition 2021 を使用する。
// 1.82+ で発生する "unsafe attribute used without unsafe" lint を抑制する。
#![allow(unsafe_op_in_unsafe_fn)]

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

use momors_core::{PredictionResult, Predictor, PredictorConfig};

// ============================================================
// 内部ハンドル型
// ============================================================

pub struct PredictorHandle {
    inner: Predictor,
}

pub struct PredictionHandle {
    /// null 終端 UTF-8 かなテキスト
    kana_utf8: Vec<u8>,
    /// null 終端 UTF-16 かなテキスト
    kana_utf16: Vec<u16>,
    /// かな char_idx → 原文 char_idx (kana_char_count 要素)
    kana_to_src: Vec<i32>,
    /// CSR 行ポインタ (src_char_count+1 要素)
    src_to_kana_row: Vec<i32>,
    /// CSR 列インデックス (kana_char_count 要素)
    src_to_kana_col: Vec<i32>,
}

impl PredictionHandle {
    fn new(result: PredictionResult) -> Self {
        let k2s = result.kana_to_source_char();
        let s2k = result.source_to_kana_char();

        let mut kana_utf8 = result.kana_text().as_bytes().to_vec();
        kana_utf8.push(0);

        let mut kana_utf16: Vec<u16> = result.kana_text().encode_utf16().collect();
        kana_utf16.push(0);

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

/// UTF-8 パスからモデルを読み込み予測器を作成する。
/// 失敗時は NULL を返す。
#[no_mangle]
pub extern "C" fn momo_predictor_new(model_path: *const c_char) -> *mut PredictorHandle {
    let path = match unsafe { lpstr_to_string(model_path) } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    match Predictor::load(PredictorConfig::new(&path)) {
        Ok(p) => Box::into_raw(Box::new(PredictorHandle { inner: p })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// UTF-16 パスからモデルを読み込み予測器を作成する。
/// 失敗時は NULL を返す。
#[no_mangle]
pub extern "C" fn momo_predictor_new_w(model_path: *const u16) -> *mut PredictorHandle {
    let path = match unsafe { lpwstr_to_string(model_path) } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    match Predictor::load(PredictorConfig::new(&path)) {
        Ok(p) => Box::into_raw(Box::new(PredictorHandle { inner: p })),
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
        Ok(r) => Box::into_raw(Box::new(PredictionHandle::new(r))),
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
        Ok(r) => Box::into_raw(Box::new(PredictionHandle::new(r))),
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
