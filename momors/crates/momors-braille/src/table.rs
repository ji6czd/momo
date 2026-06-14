use crate::Result;
use serde::Deserialize;
use std::collections::HashMap;

// ============================================================
// TOML デシリアライズ用の中間型
// ============================================================

/// 句読点エントリ: 文字列だけか、trailing 付きのインラインテーブルか。
///
/// ```toml
/// "ー" = "⠒"
/// "。" = { braille = "⠲", trailing = 2 }
/// ```
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PunctEntry {
    Simple(String),
    WithTrailing { braille: String, trailing: usize },
}

impl PunctEntry {
    fn braille(&self) -> &str {
        match self {
            PunctEntry::Simple(s) => s,
            PunctEntry::WithTrailing { braille, .. } => braille,
        }
    }
    fn trailing(&self) -> usize {
        match self {
            PunctEntry::Simple(_) => 0,
            PunctEntry::WithTrailing { trailing, .. } => *trailing,
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawTable {
    kana: RawKana,
    punct: RawPunct,
    digit: HashMap<String, String>,
    latin: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawPunct {
    /// 日本語（全角）記号。foreign_word フラグが OFF のとき参照する。
    jp: HashMap<String, PunctEntry>,
    /// ASCII 記号。foreign_word フラグが ON のとき参照する。
    latin: HashMap<String, PunctEntry>,
}

#[derive(Debug, Deserialize)]
struct RawKana {
    compound: HashMap<String, String>,
    single: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawFlags {
    digit: FlagDef,
    foreign_word: FlagDef,
    capital: CapitalFlagDef,
}

#[derive(Debug, Deserialize)]
pub(crate) struct FlagDef {
    pub trigger_class: Vec<String>,
    pub entry_prefix: String,
    #[serde(default)]
    pub exit_suffix: String,
    #[serde(default)]
    pub explicit_exit: String,
    #[serde(default)]
    pub exempt_chars: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CapitalFlagDef {
    pub trigger_class: Vec<String>,
    pub entry_prefix: String,
    pub double_entry_prefix: String,
}

#[derive(Debug, Deserialize)]
struct RawBrailleFile {
    flags: RawFlags,
    table: RawTable,
}

// ============================================================
// BrailleTable
// ============================================================

/// 変換テーブルとフラグ定義を保持する。
///
/// [`BrailleTable::from_toml`] でビルドするか、
/// [`BrailleTable::embedded`] でデフォルトの埋め込みテーブルを使う。
#[derive(Debug)]
pub struct BrailleTable {
    /// 複合音テーブル (2文字キー)。先に試みる。
    pub(crate) kana_compound: HashMap<String, String>,
    /// 単音テーブル (1文字キー)。
    pub(crate) kana_single: HashMap<String, String>,
    /// 日本語（全角）記号テーブル。値は (点字文字列, 後続スペース数)。
    /// foreign_word フラグが OFF のとき参照する。
    pub(crate) punct_jp: HashMap<String, (String, usize)>,
    /// ASCII 記号テーブル。値は (点字文字列, 後続スペース数)。
    /// foreign_word フラグが ON のとき参照する。
    pub(crate) punct_latin: HashMap<String, (String, usize)>,
    /// 数字テーブル。
    pub(crate) digit: HashMap<String, String>,
    /// ラテン文字テーブル。
    pub(crate) latin: HashMap<String, String>,
    /// フラグ定義。
    pub(crate) flag_digit: FlagDef,
    pub(crate) flag_foreign_word: FlagDef,
    pub(crate) flag_capital: CapitalFlagDef,
}

impl BrailleTable {
    /// TOML 文字列からテーブルを構築する。
    pub fn from_toml(toml_str: &str) -> Result<Self> {
        let raw: RawBrailleFile = toml::from_str(toml_str)?;
        Ok(Self::from_raw(raw))
    }

    /// コンパイル時に埋め込まれたデフォルトテーブルを使う。
    pub fn embedded() -> Result<Self> {
        Self::from_toml(include_str!("../data/japanese_braille.toml"))
    }

    /// ファイルからテーブルを読み込む。
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let toml_str = std::fs::read_to_string(path)?;
        Self::from_toml(&toml_str)
    }

    fn from_raw(raw: RawBrailleFile) -> Self {
        let to_punct = |map: HashMap<String, PunctEntry>| -> HashMap<String, (String, usize)> {
            map.into_iter()
                .map(|(k, v)| (k, (v.braille().to_owned(), v.trailing())))
                .collect()
        };

        Self {
            kana_compound: raw.table.kana.compound,
            kana_single: raw.table.kana.single,
            punct_jp: to_punct(raw.table.punct.jp),
            punct_latin: to_punct(raw.table.punct.latin),
            digit: raw.table.digit,
            latin: raw.table.latin,
            flag_digit: raw.flags.digit,
            flag_foreign_word: raw.flags.foreign_word,
            flag_capital: raw.flags.capital,
        }
    }
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_table_parses() {
        let table = BrailleTable::embedded().expect("埋め込み TOML がパースできること");
        assert!(!table.kana_single.is_empty(), "単音テーブルが空でない");
        assert!(!table.kana_compound.is_empty(), "複合音テーブルが空でない");
        assert!(!table.digit.is_empty(), "数字テーブルが空でない");
        assert!(!table.latin.is_empty(), "ラテン文字テーブルが空でない");
        assert!(!table.punct_jp.is_empty(), "日本語記号テーブルが空でない");
        assert!(!table.punct_latin.is_empty(), "ASCII記号テーブルが空でない");
    }

    #[test]
    fn spot_check_kana() {
        let table = BrailleTable::embedded().unwrap();
        assert_eq!(table.kana_single.get("ア").map(|s| s.as_str()), Some("⠁"));
        assert_eq!(table.kana_single.get("ン").map(|s| s.as_str()), Some("⠴"));
        assert_eq!(table.kana_single.get("ガ").map(|s| s.as_str()), Some("⠐⠡"));
        assert_eq!(
            table.kana_compound.get("キャ").map(|s| s.as_str()),
            Some("⠈⠡")
        );
        assert_eq!(
            table.kana_compound.get("ジョ").map(|s| s.as_str()),
            Some("⠘⠺")
        );
    }

    #[test]
    fn spot_check_punct_trailing() {
        let table = BrailleTable::embedded().unwrap();

        // 日本語記号
        let (brl, trailing) = table.punct_jp.get("。").unwrap();
        assert_eq!(brl.as_str(), "⠲");
        assert_eq!(*trailing, 2);
        let (brl2, trailing2) = table.punct_jp.get("、").unwrap();
        assert_eq!(brl2.as_str(), "⠰");
        assert_eq!(*trailing2, 1);

        // ASCII 記号
        let (brl3, trailing3) = table.punct_latin.get(".").unwrap();
        assert_eq!(brl3.as_str(), "⠲");
        assert_eq!(*trailing3, 2);
        let (brl4, trailing4) = table.punct_latin.get("-").unwrap();
        assert_eq!(brl4.as_str(), "⠤");
        assert_eq!(*trailing4, 0);
    }

    #[test]
    fn spot_check_digit_and_latin() {
        let table = BrailleTable::embedded().unwrap();
        assert_eq!(table.digit.get("0").map(|s| s.as_str()), Some("⠚"));
        assert_eq!(table.digit.get("５").map(|s| s.as_str()), Some("⠑"));
        assert_eq!(table.latin.get("a").map(|s| s.as_str()), Some("⠁"));
        assert_eq!(table.latin.get("z").map(|s| s.as_str()), Some("⠵"));
    }

    #[test]
    fn flag_digit_defined() {
        let table = BrailleTable::embedded().unwrap();
        assert_eq!(table.flag_digit.entry_prefix, "⠼");
        assert_eq!(table.flag_digit.explicit_exit, "⠤");
        assert!(table.flag_digit.exempt_chars.contains(&".".to_string()));
    }
}
