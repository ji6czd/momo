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
//! let predictor = Predictor::load(config)?;
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
mod char_type;
mod feature;
mod featurize;
mod loader;
mod model;
mod name_dict;
mod normalize;
mod numeric;

// ============================================================
// 公開 API の再エクスポート
// ============================================================

pub use error::{Error, Result};
pub use prediction::{PredictionResult, Predictor, PredictorConfig};
