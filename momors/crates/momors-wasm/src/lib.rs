//! momors の WebAssembly バインディング。
//!
//! momors-pyo3（Python）と同じクラス名・メソッド名で揃えている。JS の ES module は
//! Python のモジュールと同様に名前空間を持つため、C FFI が必要とする `Momo` プレフィックス
//! は付けない（プレフィックスは momors-ffi 側のみの事情）。
//!
//! - [`Predictor`]: 漢字かな交じり文 → カナ。`predict()` → [`PredictionResult`]。
//! - [`BrailleTranslator`]: 読み（かな／英文）→ 点字。`translate()` 等 → [`BrailleResult`]。
//! - [`BrailleBackTranslator`]: 点字 → 読み。一括変換とセル逐次入力の両方に対応。
//!
//! 漢字かな交じり文を点字に変換するには [`Predictor::predict`] → [`BrailleTranslator::translate`]
//! の 2 段で呼ぶ（momors-pyo3 と同じ流儀）。

use momors_braille::BackTransState as RustBackTransState;
use momors_braille::BrailleBackTranslator as RustBrailleBackTranslator;
use momors_braille::BrailleResult as RustBrailleResult;
use momors_braille::BrailleTranslator as RustBrailleTranslator;
use momors_braille::Language;
use momors_core::PredictionResult as RustPredictionResult;
use momors_core::Predictor as RustPredictor;
use wasm_bindgen::prelude::*;

// ============================================================
// Predictor / PredictionResult
// ============================================================

/// 漢字かな交じり文 → カナ変換器。
///
/// ```js
/// const p = new Predictor(mbmBytes);
/// const r = p.predict("吾輩は猫である");
/// console.log(r.kana);           // ワガハイワネコデアル
/// console.log(r.sourceSegmented); // 吾輩/は/猫/で/ある
/// ```
#[wasm_bindgen]
pub struct Predictor {
    inner: RustPredictor,
}

#[wasm_bindgen]
impl Predictor {
    /// モデルバイト列 (.mbm) からインスタンスを構築する。
    #[wasm_bindgen(constructor)]
    pub fn new(mbm_data: &[u8]) -> Result<Predictor, JsValue> {
        let inner = RustPredictor::from_model_bytes(mbm_data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Predictor { inner })
    }

    /// テキストをカナに変換する。
    pub fn predict(&self, text: &str) -> Result<PredictionResult, JsValue> {
        let result = self
            .inner
            .predict(text)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(PredictionResult::from_rust(result))
    }
}

/// 予測結果。[`Predictor::predict`] の戻り値。
#[wasm_bindgen]
pub struct PredictionResult {
    source: String,
    kana: String,
    confidences: Vec<f32>,
    segmented: String,
    source_segmented: String,
    kana_to_source: Vec<usize>,
    source_to_kana: Vec<Vec<usize>>,
}

impl PredictionResult {
    fn from_rust(r: RustPredictionResult) -> Self {
        let kana_to_source = r.kana_to_source_char();
        let source_to_kana = r.source_to_kana_char();
        let segmented = r.format_segmented();
        let source_segmented = r.format_source_segmented();
        let confidences: Vec<f32> = r
            .kana_text()
            .char_indices()
            .map(|(b, _)| r.confidences()[b])
            .collect();
        Self {
            source: r.source_text().to_string(),
            kana: r.kana_text().to_string(),
            confidences,
            segmented,
            source_segmented,
            kana_to_source,
            source_to_kana,
        }
    }
}

#[wasm_bindgen]
impl PredictionResult {
    /// 入力文字列（原文）。
    #[wasm_bindgen(getter)]
    pub fn source(&self) -> String {
        self.source.clone()
    }

    /// 変換後のカナ文字列。
    #[wasm_bindgen(getter)]
    pub fn kana(&self) -> String {
        self.kana.clone()
    }

    /// 各カナ文字の自信度（0.0〜1.0）。
    #[wasm_bindgen(getter)]
    pub fn confidences(&self) -> Vec<f32> {
        self.confidences.clone()
    }

    /// 分かち書きされたカナ（`ワガハイ/ワ/ネコ/デ/アル` 形式）。
    #[wasm_bindgen(getter)]
    pub fn segmented(&self) -> String {
        self.segmented.clone()
    }

    /// 分かち書きされた原文（`吾輩/は/猫/で/ある` 形式）。
    #[wasm_bindgen(getter, js_name = sourceSegmented)]
    pub fn source_segmented(&self) -> String {
        self.source_segmented.clone()
    }

    /// かな文字インデックス → 原文文字インデックス。
    #[wasm_bindgen(getter, js_name = kanaToSource)]
    pub fn kana_to_source(&self) -> Vec<u32> {
        self.kana_to_source.iter().map(|&x| x as u32).collect()
    }

    /// 原文文字インデックス → かな文字インデックスの配列（配列の配列）。
    #[wasm_bindgen(getter, js_name = sourceToKana)]
    pub fn source_to_kana(&self) -> js_sys::Array {
        nested_usize_to_js(&self.source_to_kana)
    }
}

// ============================================================
// BrailleTranslator / BrailleResult
// ============================================================

/// 点訳器。読み（かな／英文）を点字に変換する。行の言語を判定して日本語／英語へ振り分ける。
///
/// 入力は**読みに変換済みのテキスト**（日本語ならカナ）。漢字かな交じり文はまず
/// [`Predictor::predict`] でカナにしてから渡す。
///
/// ```js
/// const t = new BrailleTranslator();
/// console.log(t.translate("ワガハイワネコデアル").braille);
/// console.log(t.translate("you should be here today").braille); // UEB grade 2
/// ```
#[wasm_bindgen]
pub struct BrailleTranslator {
    inner: RustBrailleTranslator,
}

#[wasm_bindgen]
impl BrailleTranslator {
    /// 組み込みテーブル（日本語１級 + UEB grade 2）で変換器を作る。
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<BrailleTranslator, JsValue> {
        let inner = RustBrailleTranslator::from_embedded()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(BrailleTranslator { inner })
    }

    /// 1行を言語判定して点字に変換する（日本語=かな / 英語=UEB）。
    pub fn translate(&self, text: &str) -> Result<BrailleResult, JsValue> {
        let result = self
            .inner
            .translate(text)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(BrailleResult::from_rust(result))
    }

    /// 言語判定せず、必ず日本語として点訳する（英字は外字符 ⠰ ＋無縮約）。
    #[wasm_bindgen(js_name = translateJapanese)]
    pub fn translate_japanese(&self, kana: &str) -> Result<BrailleResult, JsValue> {
        let result = self
            .inner
            .translate_japanese(kana)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(BrailleResult::from_rust(result))
    }

    /// 言語判定せず、必ず英語（UEB）として点訳する。
    #[wasm_bindgen(js_name = translateEnglish)]
    pub fn translate_english(&self, text: &str) -> Result<BrailleResult, JsValue> {
        self.inner
            .translate_english(text)
            .map(BrailleResult::from_rust)
            .ok_or_else(|| JsValue::from_str("English translator is not available"))
    }
}

/// 点字変換結果。[`BrailleTranslator::translate`] 等の戻り値。
#[wasm_bindgen]
pub struct BrailleResult {
    language: String,
    text: String,
    braille: String,
    text_to_braille: Vec<usize>,
    braille_to_text: Vec<usize>,
}

impl BrailleResult {
    fn from_rust(r: RustBrailleResult) -> Self {
        Self {
            language: match r.language() {
                Language::Japanese => "japanese".to_string(),
                Language::English => "english".to_string(),
            },
            text: r.text().to_string(),
            braille: r.braille_text().to_string(),
            text_to_braille: r.text_to_braille().to_vec(),
            braille_to_text: r.braille_to_text().to_vec(),
        }
    }
}

#[wasm_bindgen]
impl BrailleResult {
    /// 点訳した経路（`"japanese"` / `"english"`）。
    #[wasm_bindgen(getter)]
    pub fn language(&self) -> String {
        self.language.clone()
    }

    /// 点訳したテキスト（日本語=かな / 英語=英文）。
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
    }

    /// 変換後の点字文字列。
    #[wasm_bindgen(getter)]
    pub fn braille(&self) -> String {
        self.braille.clone()
    }

    /// テキスト文字インデックス → 点字先頭セルインデックス。
    #[wasm_bindgen(getter, js_name = textToBraille)]
    pub fn text_to_braille(&self) -> Vec<u32> {
        self.text_to_braille.iter().map(|&x| x as u32).collect()
    }

    /// 点字セルインデックス → テキスト文字インデックス。
    #[wasm_bindgen(getter, js_name = brailleToText)]
    pub fn braille_to_text(&self) -> Vec<u32> {
        self.braille_to_text.iter().map(|&x| x as u32).collect()
    }
}

// ============================================================
// BrailleBackTranslator / BackTransState / StepResult / BackTransResult
// ============================================================

/// 点字をかな・記号・数字・ラテン文字に逆変換する。
///
/// 順変換 [`BrailleTranslator`] と異なりモデル（.mbm）不要で、組み込みテーブルのみで動作する。
///
/// ```js
/// const bt = new BrailleBackTranslator();
/// console.log(bt.backTranslate("⠁⠃⠉⠋⠊")); // アイウエオ
/// ```
#[wasm_bindgen]
pub struct BrailleBackTranslator {
    inner: RustBrailleBackTranslator,
}

#[wasm_bindgen]
impl BrailleBackTranslator {
    /// 組み込みテーブルからインスタンスを構築する。
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<BrailleBackTranslator, JsValue> {
        let inner = RustBrailleBackTranslator::from_embedded()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(BrailleBackTranslator { inner })
    }

    /// 点字セル列を文字列に逆変換して返す。
    #[wasm_bindgen(js_name = backTranslate)]
    pub fn back_translate(&self, braille: &str) -> String {
        self.inner.back_translate(braille)
    }

    /// 点字セル列を逆変換し、復元文字列と点字セル範囲⇔文字列の対応を返す。
    ///
    /// エディタでの読みガイド表示など、点字セル位置と復元文字を対応付けたい用途向け。
    #[wasm_bindgen(js_name = backTranslateAligned)]
    pub fn back_translate_aligned(&self, braille: &str) -> BackTransResult {
        let result = self.inner.back_translate_aligned(braille);
        BackTransResult {
            text: result.text,
            segments: result.segments,
        }
    }

    /// 点字 1 セルを逐次逆変換する（エディタでの逐次入力向け）。
    ///
    /// `cell` は点字 1 セル分の文字列（先頭の 1 文字のみ使用）。
    /// 行頭では [`BackTransState::new`] の初期状態を渡し、以降は返り値の
    /// [`StepResult::state`] を次回呼び出しに渡して、数符・外字符・濁音前置
    /// などの行内状態を引き継ぐ。前置セル（濁音など）の直後は `text` が空になる
    /// ことがある（次のセルと結合して確定するまで保留されるため）。
    pub fn step(&self, cell: &str, state: &BackTransState) -> Result<StepResult, JsValue> {
        let ch = cell
            .chars()
            .next()
            .ok_or_else(|| JsValue::from_str("cell must be a single character"))?;
        let result = self.inner.step(ch, state.inner);
        Ok(StepResult {
            text: result.text,
            state: result.state,
        })
    }

    /// 行末で呼び、保留中のセル（結合待ちの前置セルなど）を確定させる。
    ///
    /// [`BackTransState::isSettled`] が `false` のまま行が終わる場合に呼ぶ。
    pub fn flush(&self, state: &BackTransState) -> StepResult {
        let result = self.inner.flush(state.inner);
        StepResult {
            text: result.text,
            state: result.state,
        }
    }
}

/// [`BrailleBackTranslator::step`] / [`BrailleBackTranslator::flush`] が引き継ぐ行内状態。
///
/// 不透明なハンドル。JS 側は中身を参照せず、次回の `step`/`flush` にそのまま渡す。
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct BackTransState {
    inner: RustBackTransState,
}

#[wasm_bindgen]
impl BackTransState {
    /// 初期状態（行頭）を生成する。
    #[wasm_bindgen(constructor)]
    pub fn new() -> BackTransState {
        BackTransState {
            inner: RustBackTransState::new(),
        }
    }

    /// 保留セルが残っていない（行末で `flush` が不要）か。
    #[wasm_bindgen(js_name = isSettled)]
    pub fn is_settled(&self) -> bool {
        self.inner.is_settled()
    }
}

impl Default for BackTransState {
    fn default() -> Self {
        Self::new()
    }
}

/// [`BrailleBackTranslator::step`] / [`BrailleBackTranslator::flush`] の結果。
#[wasm_bindgen]
pub struct StepResult {
    text: String,
    state: RustBackTransState,
}

#[wasm_bindgen]
impl StepResult {
    /// このセルで確定した文字列。前置セルの保留中は空、拗音などは 2 文字になりうる。
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
    }

    /// 更新後の状態。次の `step`/`flush` にそのまま渡す。
    #[wasm_bindgen(getter)]
    pub fn state(&self) -> BackTransState {
        BackTransState { inner: self.state }
    }
}

/// [`BrailleBackTranslator::back_translate_aligned`] の結果。
#[wasm_bindgen]
pub struct BackTransResult {
    text: String,
    segments: Vec<momors_braille::BackTransSegment>,
}

#[wasm_bindgen]
impl BackTransResult {
    /// 復元された全文（`segments` のテキストを連結したものと一致する）。
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
    }

    /// 点字セル範囲 → 文字列のセグメント列。各要素は `{ cellStart, cellEnd, text }`。
    #[wasm_bindgen(getter)]
    pub fn segments(&self) -> js_sys::Array {
        let out = js_sys::Array::new();
        for seg in &self.segments {
            let obj = js_sys::Object::new();
            let _ = js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("cellStart"),
                &JsValue::from(seg.cell_start as u32),
            );
            let _ = js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("cellEnd"),
                &JsValue::from(seg.cell_end as u32),
            );
            let _ = js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("text"),
                &JsValue::from_str(&seg.text),
            );
            out.push(&obj);
        }
        out
    }
}

// ============================================================
// ヘルパー
// ============================================================

/// `Vec<Vec<usize>>` を JS の配列の配列（各要素 `Uint32Array`）に変換する。
fn nested_usize_to_js(nested: &[Vec<usize>]) -> js_sys::Array {
    let outer = js_sys::Array::new();
    for inner in nested {
        let arr = js_sys::Uint32Array::new_with_length(inner.len() as u32);
        for (i, &v) in inner.iter().enumerate() {
            arr.set_index(i as u32, v as u32);
        }
        outer.push(&arr);
    }
    outer
}
