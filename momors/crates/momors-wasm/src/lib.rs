use momors_braille::BackTransState;
use momors_braille::BrailleBackTranslator;
use momors_braille::BrailleTranslator;
use momors_core::Predictor;
use wasm_bindgen::prelude::*;

/// 日本語/英語テキストを点字に変換する WASM バインディング。
///
/// JS 側から `new MomoWasm(mbmBytes)` でインスタンスを生成し、
/// `predictKana` / `predictBraille` / `predictSegmented` を呼ぶ。
#[wasm_bindgen]
pub struct MomoWasm {
    predictor: Predictor,
    braille_translator: BrailleTranslator,
}

#[wasm_bindgen]
impl MomoWasm {
    /// モデルバイト列 (.mbm) からインスタンスを構築する。
    #[wasm_bindgen(constructor)]
    pub fn new(mbm_data: &[u8]) -> Result<MomoWasm, JsValue> {
        let predictor =
            Predictor::from_model_bytes(mbm_data).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let braille_translator =
            BrailleTranslator::from_embedded().map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(MomoWasm {
            predictor,
            braille_translator,
        })
    }

    /// 日本語テキストをカタカナに変換して返す。
    #[wasm_bindgen(js_name = predictKana)]
    pub fn predict_kana(&self, text: &str) -> Result<String, JsValue> {
        let result = self
            .predictor
            .predict(text)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(result.kana_text().to_string())
    }

    /// 日本語テキストを点字に変換して返す。
    #[wasm_bindgen(js_name = predictBraille)]
    pub fn predict_braille(&self, text: &str) -> Result<String, JsValue> {
        let result = self
            .predictor
            .predict(text)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let braille = self
            .braille_translator
            .translate(result.kana_text())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(braille.braille_text().to_string())
    }

    /// 日本語テキストを変換し、点字・かな・インデックスを JSON で返す。
    ///
    /// 返り値 JSON のスキーマ:
    /// ```json
    /// {
    ///   "braille": "⠀⠁...",
    ///   "kana": "カナ...",
    ///   "kana_to_src_index": [0, 0, 1, ...],
    ///   "src_to_kana_index": [[0, 1], [2], ...]
    /// }
    /// ```
    #[wasm_bindgen(js_name = predictSegmented)]
    pub fn predict_segmented(&self, text: &str) -> Result<String, JsValue> {
        let result = self
            .predictor
            .predict(text)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let braille = self
            .braille_translator
            .translate(result.kana_text())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let json = serde_json::json!({
            "braille": braille.braille_text(),
            "kana": result.kana_text(),
            "kana_to_src_index": result.kana_to_source_char(),
            "src_to_kana_index": result.source_to_kana_char(),
        });
        Ok(json.to_string())
    }
}

/// 点字をかな・記号・数字・ラテン文字に逆変換する WASM バインディング。
///
/// 順変換 [`MomoWasm`] と異なりモデル（.mbm）不要で、組み込みテーブルのみで動作する。
/// JS 側から `new MomoBackTranslator()` でインスタンスを生成し、
/// `backTranslate` / `backTranslateAligned` を呼ぶ。
#[wasm_bindgen]
pub struct MomoBackTranslator {
    inner: BrailleBackTranslator,
}

#[wasm_bindgen]
impl MomoBackTranslator {
    /// 組み込みテーブルからインスタンスを構築する。
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<MomoBackTranslator, JsValue> {
        let inner = BrailleBackTranslator::from_embedded()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(MomoBackTranslator { inner })
    }

    /// 点字セル列を文字列に逆変換して返す。
    #[wasm_bindgen(js_name = backTranslate)]
    pub fn back_translate(&self, braille: &str) -> String {
        self.inner.back_translate(braille)
    }

    /// 点字セル列を逆変換し、復元文字列と点字セル範囲⇔文字列の対応を JSON で返す。
    ///
    /// エディタでの読みガイド表示など、点字セル位置と復元文字を対応付けたい用途向け。
    ///
    /// 返り値 JSON のスキーマ:
    /// ```json
    /// {
    ///   "text": "アイウ",
    ///   "segments": [
    ///     { "cellStart": 0, "cellEnd": 1, "text": "ア" },
    ///     { "cellStart": 1, "cellEnd": 2, "text": "イ" },
    ///     { "cellStart": 2, "cellEnd": 3, "text": "ウ" }
    ///   ]
    /// }
    /// ```
    #[wasm_bindgen(js_name = backTranslateAligned)]
    pub fn back_translate_aligned(&self, braille: &str) -> String {
        let result = self.inner.back_translate_aligned(braille);
        let segments: Vec<_> = result
            .segments
            .iter()
            .map(|s| {
                serde_json::json!({
                    "cellStart": s.cell_start,
                    "cellEnd": s.cell_end,
                    "text": s.text,
                })
            })
            .collect();
        let json = serde_json::json!({
            "text": result.text,
            "segments": segments,
        });
        json.to_string()
    }

    /// 点字 1 セルを逐次逆変換する（エディタでの逐次入力向け）。
    ///
    /// `cell` は点字 1 セル分の文字列（先頭の 1 文字のみ使用）。
    /// 行頭では [`MomoBackTransState::new`] の初期状態を渡し、以降は返り値の
    /// [`MomoStepResult::state`] を次回呼び出しに渡して、数符・外字符・濁音前置
    /// などの行内状態を引き継ぐ。前置セル（濁音など）の直後は `text` が空になる
    /// ことがある（次のセルと結合して確定するまで保留されるため）。
    #[wasm_bindgen]
    pub fn step(&self, cell: &str, state: &MomoBackTransState) -> Result<MomoStepResult, JsValue> {
        let ch = cell
            .chars()
            .next()
            .ok_or_else(|| JsValue::from_str("cell must be a single character"))?;
        let result = self.inner.step(ch, state.inner);
        Ok(MomoStepResult {
            text: result.text,
            state: result.state,
        })
    }

    /// 行末で呼び、保留中のセル（結合待ちの前置セルなど）を確定させる。
    ///
    /// [`MomoBackTransState::isSettled`] が `false` のまま行が終わる場合に呼ぶ。
    #[wasm_bindgen]
    pub fn flush(&self, state: &MomoBackTransState) -> MomoStepResult {
        let result = self.inner.flush(state.inner);
        MomoStepResult {
            text: result.text,
            state: result.state,
        }
    }
}

/// [`MomoBackTranslator::step`] / [`MomoBackTranslator::flush`] が引き継ぐ行内状態。
///
/// 不透明なハンドル。JS 側は中身を参照せず、次回の `step`/`flush` にそのまま渡す。
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct MomoBackTransState {
    inner: BackTransState,
}

#[wasm_bindgen]
impl MomoBackTransState {
    /// 初期状態（行頭）を生成する。
    #[wasm_bindgen(constructor)]
    pub fn new() -> MomoBackTransState {
        MomoBackTransState {
            inner: BackTransState::new(),
        }
    }

    /// 保留セルが残っていない（行末で `flush` が不要）か。
    #[wasm_bindgen(js_name = isSettled)]
    pub fn is_settled(&self) -> bool {
        self.inner.is_settled()
    }
}

impl Default for MomoBackTransState {
    fn default() -> Self {
        Self::new()
    }
}

/// [`MomoBackTranslator::step`] / [`MomoBackTranslator::flush`] の結果。
#[wasm_bindgen]
pub struct MomoStepResult {
    text: String,
    state: BackTransState,
}

#[wasm_bindgen]
impl MomoStepResult {
    /// このセルで確定した文字列。前置セルの保留中は空、拗音などは 2 文字になりうる。
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
    }

    /// 更新後の状態。次の `step`/`flush` にそのまま渡す。
    #[wasm_bindgen(getter)]
    pub fn state(&self) -> MomoBackTransState {
        MomoBackTransState { inner: self.state }
    }
}
