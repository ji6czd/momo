//! # momors-braille
//!
//! かな列を日本語点字に変換し、点字ドキュメントを各種ファイル形式で読み書きするライブラリ。
//!
//! ## 構成
//!
//! - [`BrailleConverter`]: かな列 → 点字（[`momors_core::PredictionResult`] のかなを受け取る）。
//! - [`document`]: 点字ドキュメントの**正本** [`BrailleDocument`]。MBR テキスト codec を含む。
//! - [`reader`]: バイト列 → [`BrailleDocument`]（BES デコード）。
//! - [`formatter`]: 正本 → 印刷イメージ [`FormattedDocument`] の導出（[`render`]）とワードラップ。
//! - [`writer`]: 正本 → 各種ファイル形式のバイト列（[`OutputFormat`]）。
//!
//! ## 使い方（変換）
//!
//! ```no_run
//! use momors_braille::BrailleConverter;
//!
//! # fn main() -> momors_braille::Result<()> {
//! let converter = BrailleConverter::from_embedded()?;
//! let result = converter.convert("アイウエオ")?;
//! println!("{}", result.braille_text());
//! # Ok(())
//! # }
//! ```

pub mod backtranslator;
pub mod converter;
pub mod document;
pub mod error;
pub mod formatter;
mod nabcc;
pub mod reader;
pub mod table;
pub mod writer;

pub use backtranslator::{
    BackTransResult, BackTransSegment, BackTransState, BrailleBackTranslator, StepResult,
};
pub use converter::{BrailleConverter, BrailleResult};
pub use document::{BrailleDocument, DocumentConfig, PageBreak, PageNumberStyle, PhysicalLine};
pub use error::{Error, Result};
pub use formatter::{render, FormattedDocument, RenderedLine};
pub use reader::{read_bes, read_bet};
pub use table::BrailleTable;
pub use writer::OutputFormat;
