//! # momors-braille
//!
//! かな列を日本語点字に変換するライブラリ。
//!
//! [`momors_core::PredictionResult`] のかな列を受け取り、点字文字列と
//! かな↔点字の双方向インデックスを返す。
//!
//! ## 使い方
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

pub mod converter;
pub mod error;
pub mod formatter;
mod nabcc;
pub mod reader;
pub mod table;
pub mod writer;

pub use converter::{BrailleConverter, BrailleResult};
pub use error::{Error, Result};
pub use formatter::{BrailleFormatter, FormattedDocument, FormatterConfig, PhysicalLine};
pub use reader::{read_bes_file, BesDocument, BesPhysicalLine};
pub use table::BrailleTable;
pub use writer::OutputFormat;
