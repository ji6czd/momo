//! ライブラリ全体で共有するエラー型。
//!
//! `thiserror` クレートを使って `Display` と `std::error::Error` を自動実装する。
//! 使う側は `momors_core::Result<T>` を返り値型にすれば、`?` 演算子で
//! エラーを伝播できる。

use std::path::PathBuf;
use thiserror::Error;

/// `momors-core` で発生しうるエラー。
///
/// `#[non_exhaustive]` を付けておくと、将来バリアントを追加しても
/// 破壊的変更扱いにならない。マッチするときに `_` が必須になる。
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// モデルファイルが開けなかった、または読めなかった。
    #[error("モデルファイルの読み込みに失敗しました: {path}")]
    ModelIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// モデルファイルのマジックバイトが "MOMO" ではない。
    #[error("不正なマジックバイト: {path}")]
    InvalidMagic { path: PathBuf },

    /// モデルファイルのバージョンがサポート外。
    #[error("未対応のモデルバージョン: {version}")]
    UnsupportedVersion { version: u8 },

    /// モデルファイル中の FeatureType 値が未定義。
    #[error("不正な FeatureType 値: 0x{value:02X}")]
    InvalidFeatureType { value: u8 },

    /// モデルファイル中の CharType 値が未定義。
    #[error("不正な CharType 値: 0x{value:02X}")]
    InvalidCharType { value: u8 },

    /// モデルファイル中の文字列が妥当な UTF-8 ではない。
    #[error("ラベル文字列が UTF-8 として不正です")]
    InvalidLabelUtf8 {
        #[source]
        source: std::string::FromUtf8Error,
    },

    /// モデルファイルの構造が壊れている（途中で終端した等）。
    #[error("モデルファイルが壊れています: {reason}")]
    CorruptModel { reason: String },

    /// 漢字辞書ファイルが開けなかった、または読めなかった。
    #[error("漢字辞書ファイルの読み込みに失敗しました: {path}")]
    DictIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// 予測時の内部エラー。
    #[error("予測に失敗しました: {reason}")]
    Prediction { reason: String },

    /// カスタム辞書ファイルが開けなかった、または読めなかった。
    #[error("カスタム辞書ファイルの読み込みに失敗しました: {path}")]
    CustomDictIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// カスタム辞書の記法が不正（境界マーカーの位置・ユニット数不一致等）。
    #[error("カスタム辞書が不正です: {reason}")]
    InvalidCustomDict { reason: String },
}

/// 本クレート用の `Result` 型エイリアス。
///
/// 使う側は `momors_core::Result<T>` と書くだけで済む。
pub type Result<T> = std::result::Result<T, Error>;
