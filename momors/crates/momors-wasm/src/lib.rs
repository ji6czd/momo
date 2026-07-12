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
