//! # momors-core
//!
//! 日本語の漢字かな交じり文をカタカナに変換するライブラリ。
//!
//! ## 使い方
//!
//! ```no_run
//! use momors_core::{Predictor, PredictorConfig};
//!
//! # fn main() -> momors_core::Result<()> {
//! let config = PredictorConfig::new("basic_data.mbm");
//!
//! let predictor: Predictor = Predictor::load(config)?;
//!
//! let result = predictor.predict("漢字混じりの文章")?;
//! println!("{}", result.kana_text());
//! # Ok(())
//! # }
//! ```

// ============================================================
// モジュール構成
// ============================================================
// pub mod: 外部に公開するモジュール
// mod    : クレート内部だけで使うモジュール
//
// 公開する型は `pub use` で lib.rs のトップレベルに引き上げて、
// 使う側が `momors_core::Predictor` のように短く書けるようにする。
// ============================================================

pub mod error;
pub mod prediction;

// 内部モジュール（実装が進んだら追加していく）
mod bracket;
mod char_type;
mod feature;
mod featurize;
mod float_loader;
mod float_model;
mod loader;
mod model;
mod name_dict;
mod normalize;
mod numeric;
mod weight_model;

// ============================================================
// 公開 API の再エクスポート
// ============================================================

pub use error::{Error, Result};
pub use prediction::{
    Contribution, DecisionSource, PredictionResult, Predictor, PredictorConfig, TraceResult,
    TraceRow,
};

/// 量子化前 (`.mbmf`) の重みで推論する `Predictor`。
///
/// `.mbm`（int8 量子化）を読む既定の [`Predictor`] と同じ `predict()` API を
/// 持つため、量子化前後でテキスト推論の結果を直接比較できる
/// （`momo-compare-quant` CLI の用途）。
pub type FloatPredictor = Predictor<float_model::FloatMomoModel>;
