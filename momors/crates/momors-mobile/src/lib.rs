//! momors の Android / iOS バインディング（UniFFI）。
//!
//! momors-wasm / momors-pyo3 と同じ型・メソッド名で揃えている。Kotlin / Swift のコードは
//! `uniffi-bindgen` がこのファイルの定義から生成する。
//!
//! - [`Predictor`]: 漢字かな交じり文 → カナ。`predict()` → [`PredictionResult`]。
//! - [`BrailleTranslator`]: 読み（かな／英文）→ 点字。`translate()` 等 → [`BrailleResult`]。
//! - [`BrailleBackTranslator`]: 点字 → 読み。一括変換とセル逐次入力の両方に対応。
//!
//! 漢字かな交じり文を点字に変換するには [`Predictor::predict`] → [`BrailleTranslator::translate`]
//! の 2 段で呼ぶ。モデル（.mbm）はバイト列で渡せるので、Android の assets や iOS の bundle から
//! 読んだ内容をそのまま渡せる。

use momors_braille::BackTransState as RustBackTransState;
use momors_braille::BrailleBackTranslator as RustBrailleBackTranslator;
use momors_braille::BrailleResult as RustBrailleResult;
use momors_braille::BrailleTranslator as RustBrailleTranslator;
use momors_braille::Language;
use momors_core::PredictionResult as RustPredictionResult;
use momors_core::Predictor as RustPredictor;
use momors_core::PredictorConfig;
use std::path::Path;

uniffi::setup_scaffolding!();

/// すべての API が投げるエラー。メッセージは下位クレートのエラー文字列。
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum MomoError {
    #[error("{message}")]
    Failed { message: String },
}

impl MomoError {
    fn from_display(e: impl std::fmt::Display) -> Self {
        MomoError::Failed {
            message: e.to_string(),
        }
    }
}

// ============================================================
// Predictor / PredictionResult
// ============================================================

/// 漢字かな交じり文 → カナ変換器。
///
/// ```kotlin
/// val p = Predictor(context.assets.open("model.mbm").readBytes())
/// val r = p.predict("吾輩は猫である")
/// println(r.kana)  // ワガハイワネコデアル
/// ```
#[derive(uniffi::Object)]
pub struct Predictor {
    inner: RustPredictor,
}

#[uniffi::export]
impl Predictor {
    /// モデルバイト列 (.mbm) からインスタンスを構築する。
    #[uniffi::constructor]
    pub fn new(mbm_data: Vec<u8>) -> Result<Self, MomoError> {
        let inner = RustPredictor::from_model_bytes(&mbm_data).map_err(MomoError::from_display)?;
        Ok(Self { inner })
    }

    /// ファイルパスのモデル (.mbm) から構築する。`custom_dict_path` はカスタム辞書 TSV（省略可）。
    #[uniffi::constructor]
    pub fn from_path(
        model_path: String,
        custom_dict_path: Option<String>,
    ) -> Result<Self, MomoError> {
        let mut config = PredictorConfig::new(Path::new(&model_path));
        if let Some(dict) = custom_dict_path {
            config = config.with_custom_dict_path(Path::new(&dict));
        }
        let inner = RustPredictor::load(config).map_err(MomoError::from_display)?;
        Ok(Self { inner })
    }

    /// テキストをカナに変換する。
    pub fn predict(&self, text: String) -> Result<PredictionResult, MomoError> {
        let result = self.inner.predict(&text).map_err(MomoError::from_display)?;
        Ok(PredictionResult::from_rust(result))
    }
}

/// 予測結果。[`Predictor::predict`] の戻り値。インデックスはすべて文字（コードポイント）単位。
#[derive(uniffi::Record)]
pub struct PredictionResult {
    /// 入力文字列（原文）。
    pub source: String,
    /// 変換後のカナ文字列。
    pub kana: String,
    /// 各カナ文字の自信度（0.0〜1.0）。
    pub confidences: Vec<f32>,
    /// 分かち書きされたカナ（`ワガハイ/ワ/ネコ/デ/アル` 形式）。
    pub segmented: String,
    /// 分かち書きされた原文（`吾輩/は/猫/で/ある` 形式）。
    pub source_segmented: String,
    /// かな文字インデックス → 原文文字インデックス。
    pub kana_to_source: Vec<u32>,
    /// 原文文字インデックス → かな文字インデックスの配列。
    pub source_to_kana: Vec<Vec<u32>>,
}

impl PredictionResult {
    fn from_rust(r: RustPredictionResult) -> Self {
        let confidences = r
            .kana_text()
            .char_indices()
            .map(|(b, _)| r.confidences()[b])
            .collect();
        Self {
            source: r.source_text().to_string(),
            kana: r.kana_text().to_string(),
            confidences,
            segmented: r.format_segmented(),
            source_segmented: r.format_source_segmented(),
            kana_to_source: to_u32(&r.kana_to_source_char()),
            source_to_kana: r.source_to_kana_char().iter().map(|v| to_u32(v)).collect(),
        }
    }
}

fn to_u32(v: &[usize]) -> Vec<u32> {
    v.iter().map(|&x| x as u32).collect()
}

// ============================================================
// BrailleTranslator / BrailleResult
// ============================================================

/// 点訳器。読み（かな／英文）を点字に変換する。行の言語を判定して日本語／英語へ振り分ける。
///
/// 入力は**読みに変換済みのテキスト**（日本語ならカナ）。漢字かな交じり文はまず
/// [`Predictor::predict`] でカナにしてから渡す。
#[derive(uniffi::Object)]
pub struct BrailleTranslator {
    inner: RustBrailleTranslator,
}

#[uniffi::export]
impl BrailleTranslator {
    /// 組み込みテーブル（日本語１級 + UEB grade 2）で変換器を作る。
    #[uniffi::constructor]
    pub fn new() -> Result<Self, MomoError> {
        let inner = RustBrailleTranslator::from_embedded().map_err(MomoError::from_display)?;
        Ok(Self { inner })
    }

    /// 1行を言語判定して点字に変換する（日本語=かな / 英語=UEB）。
    pub fn translate(&self, text: String) -> Result<BrailleResult, MomoError> {
        self.inner
            .translate(&text)
            .map(BrailleResult::from_rust)
            .map_err(MomoError::from_display)
    }

    /// 言語判定せず、必ず日本語として点訳する（英字は外字符 ⠰ ＋無縮約）。
    pub fn translate_japanese(&self, kana: String) -> Result<BrailleResult, MomoError> {
        self.inner
            .translate_japanese(&kana)
            .map(BrailleResult::from_rust)
            .map_err(MomoError::from_display)
    }

    /// 言語判定せず、必ず英語（UEB）として点訳する。
    pub fn translate_english(&self, text: String) -> Result<BrailleResult, MomoError> {
        self.inner
            .translate_english(&text)
            .map(BrailleResult::from_rust)
            .ok_or_else(|| MomoError::from_display("English translator is not available"))
    }
}

/// 点字変換結果。[`BrailleTranslator::translate`] 等の戻り値。
#[derive(uniffi::Record)]
pub struct BrailleResult {
    /// 点訳した経路（`"japanese"` / `"english"`）。
    pub language: String,
    /// 点訳したテキスト（日本語=かな / 英語=英文）。
    pub text: String,
    /// 変換後の点字文字列。
    pub braille: String,
    /// テキスト文字インデックス → 点字先頭セルインデックス。
    pub text_to_braille: Vec<u32>,
    /// 点字セルインデックス → テキスト文字インデックス。
    pub braille_to_text: Vec<u32>,
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
            text_to_braille: to_u32(r.text_to_braille()),
            braille_to_text: to_u32(r.braille_to_text()),
        }
    }
}

// ============================================================
// BrailleBackTranslator / BackTransState / StepResult / BackTransResult
// ============================================================

/// 点字をかな・記号・数字・ラテン文字に逆変換する。モデル不要で、組み込みテーブルのみで動作する。
#[derive(uniffi::Object)]
pub struct BrailleBackTranslator {
    inner: RustBrailleBackTranslator,
}

#[uniffi::export]
impl BrailleBackTranslator {
    /// 組み込みテーブルからインスタンスを構築する。
    #[uniffi::constructor]
    pub fn new() -> Result<Self, MomoError> {
        let inner = RustBrailleBackTranslator::from_embedded().map_err(MomoError::from_display)?;
        Ok(Self { inner })
    }

    /// 点字セル列を文字列に逆変換して返す。
    pub fn back_translate(&self, braille: String) -> String {
        self.inner.back_translate(&braille)
    }

    /// 点字セル列を逆変換し、復元文字列と点字セル範囲⇔文字列の対応を返す。
    pub fn back_translate_aligned(&self, braille: String) -> BackTransResult {
        let result = self.inner.back_translate_aligned(&braille);
        BackTransResult {
            text: result.text,
            segments: result
                .segments
                .into_iter()
                .map(|s| BackTransSegment {
                    cell_start: s.cell_start as u32,
                    cell_end: s.cell_end as u32,
                    text: s.text,
                })
                .collect(),
        }
    }

    /// 点字 1 セルを逐次逆変換する（逐次入力向け）。
    ///
    /// `cell` は点字 1 セル分の文字列（先頭の 1 文字のみ使用）。行頭では
    /// [`BackTransState::new`] の初期状態を渡し、以降は返り値の `state` を次回に渡す。
    /// 前置セル（濁音など）の直後は `text` が空になることがある。
    pub fn step(&self, cell: String, state: &BackTransState) -> Result<StepResult, MomoError> {
        let ch = cell
            .chars()
            .next()
            .ok_or_else(|| MomoError::from_display("cell must be a single character"))?;
        let result = self.inner.step(ch, state.inner);
        Ok(StepResult::from_rust(result.text, result.state))
    }

    /// 行末で呼び、保留中のセル（結合待ちの前置セルなど）を確定させる。
    pub fn flush(&self, state: &BackTransState) -> StepResult {
        let result = self.inner.flush(state.inner);
        StepResult::from_rust(result.text, result.state)
    }
}

/// [`BrailleBackTranslator::step`] / `flush` が引き継ぐ行内状態。不透明なハンドル。
#[derive(uniffi::Object)]
pub struct BackTransState {
    inner: RustBackTransState,
}

#[uniffi::export]
impl BackTransState {
    /// 初期状態（行頭）を生成する。
    #[uniffi::constructor]
    pub fn new() -> Self {
        Self {
            inner: RustBackTransState::new(),
        }
    }

    /// 保留セルが残っていない（行末で `flush` が不要）か。
    pub fn is_settled(&self) -> bool {
        self.inner.is_settled()
    }
}

/// [`BrailleBackTranslator::step`] / `flush` の結果。
#[derive(uniffi::Record)]
pub struct StepResult {
    /// このセルで確定した文字列。前置セルの保留中は空、拗音などは 2 文字になりうる。
    pub text: String,
    /// 更新後の状態。次の `step` / `flush` にそのまま渡す。
    pub state: std::sync::Arc<BackTransState>,
}

impl StepResult {
    fn from_rust(text: String, state: RustBackTransState) -> Self {
        Self {
            text,
            state: std::sync::Arc::new(BackTransState { inner: state }),
        }
    }
}

/// 点字セル範囲に対応する復元文字列。
#[derive(uniffi::Record)]
pub struct BackTransSegment {
    pub cell_start: u32,
    pub cell_end: u32,
    pub text: String,
}

/// [`BrailleBackTranslator::back_translate_aligned`] の結果。
#[derive(uniffi::Record)]
pub struct BackTransResult {
    /// 復元された全文（`segments` のテキストを連結したものと一致する）。
    pub text: String,
    pub segments: Vec<BackTransSegment>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn braille_round_trip() {
        let t = BrailleTranslator::new().unwrap();
        let r = t.translate_japanese("アイウエオ".to_string()).unwrap();
        assert_eq!(r.language, "japanese");
        let b = BrailleBackTranslator::new().unwrap();
        assert_eq!(b.back_translate(r.braille), "アイウエオ");
    }

    #[test]
    fn step_matches_batch() {
        let b = BrailleBackTranslator::new().unwrap();
        let mut state = std::sync::Arc::new(BackTransState::new());
        let mut out = String::new();
        for c in "⠁⠃⠉⠋⠊".chars() {
            let r = b.step(c.to_string(), &state).unwrap();
            out.push_str(&r.text);
            state = r.state;
        }
        out.push_str(&b.flush(&state).text);
        assert_eq!(out, "アイウエオ");
    }

    #[test]
    fn predict_with_model() {
        // MOMO_DATASET_DIR が無ければ（モデル未配置の環境では）何もしない
        let Ok(dir) = std::env::var("MOMO_DATASET_DIR") else {
            return;
        };
        let bytes = std::fs::read(Path::new(&dir).join("basic_data_4.mbm")).unwrap();
        let p = Predictor::new(bytes).unwrap();
        let r = p.predict("吾輩は猫である".to_string()).unwrap();
        // kana にはマスあけ（空白）が入るので、読みだけを比べる
        assert_eq!(r.kana.replace(' ', ""), "ワガハイワネコデアル");
        assert_eq!(r.confidences.len(), r.kana.chars().count());
    }

    #[test]
    fn invalid_model_is_error() {
        assert!(Predictor::new(vec![0, 1, 2]).is_err());
    }
}
