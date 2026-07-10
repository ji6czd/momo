use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("braille table parse error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("braille table io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("unknown embedded table: {0}")]
    UnknownTable(String),

    #[error("braille table validation error: {0}")]
    Validation(String),

    /// 日本語行の点訳で `momors_core::Predictor` が失敗した（[`crate::line::LineTranslator`]）。
    #[error("prediction error: {0}")]
    Prediction(String),
}

pub type Result<T> = std::result::Result<T, Error>;
