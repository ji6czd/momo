use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("braille table parse error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("braille table io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
