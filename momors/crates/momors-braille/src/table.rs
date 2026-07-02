use crate::Result;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::LazyLock;

// ============================================================
// TOML デシリアライズ用の中間型
// ============================================================

/// 記号クラスに対する一致条件。
///
/// ```toml
/// suppress_before = ["stop"]              # 次の class がこの中にあれば一致（include）
/// suppress_before = { exclude = ["open"] } # 次の class がこの中になければ一致（exclude）
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum ClassMatcher {
    Include(Vec<String>),
    Exclude { exclude: Vec<String> },
}

impl ClassMatcher {
    /// `class` が条件に一致するか。
    pub(crate) fn matches(&self, class: &str) -> bool {
        match self {
            ClassMatcher::Include(list) => list.iter().any(|c| c == class),
            ClassMatcher::Exclude { exclude } => !exclude.iter().any(|c| c == class),
        }
    }
}

/// 句読点エントリ: 文字列だけか、trailing・class 付きのインラインテーブルか。
///
/// ```toml
/// "ー" = "⠒"
/// "。" = { braille = "⠲", trailing = 2, class = "stop", suppress_before = ["stop"] }
/// ```
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PunctEntry {
    Simple(String),
    WithTrailing {
        braille: String,
        #[serde(default)]
        trailing: usize,
        /// この記号自身の分類（例: "stop", "pause"）。省略時は無分類。
        #[serde(default)]
        class: Option<String>,
        /// 次の文字の class がこの条件に一致するとき、自分の trailing を出さない。
        #[serde(default)]
        suppress_before: Option<ClassMatcher>,
    },
}

impl PunctEntry {
    fn into_cell(self) -> PunctCell {
        match self {
            PunctEntry::Simple(braille) => PunctCell {
                braille,
                trailing: 0,
                class: None,
                suppress_before: None,
            },
            PunctEntry::WithTrailing {
                braille,
                trailing,
                class,
                suppress_before,
            } => PunctCell {
                braille,
                trailing,
                class,
                suppress_before,
            },
        }
    }
}

/// 句読点テーブルの 1 エントリ。
#[derive(Debug, Clone)]
pub(crate) struct PunctCell {
    pub braille: String,
    pub trailing: usize,
    /// この記号自身の分類（例: "stop", "pause"）。未分類なら `None`。
    pub class: Option<String>,
    /// 次の文字の class がこの条件に一致するとき、`trailing` を出さない。
    pub suppress_before: Option<ClassMatcher>,
}

impl PunctCell {
    /// 次の文字の class（分類がなければ `"none"`）を踏まえた実効 trailing 数。
    pub(crate) fn effective_trailing(&self, next_class: &str) -> usize {
        match &self.suppress_before {
            Some(matcher) if matcher.matches(next_class) => 0,
            _ => self.trailing,
        }
    }
}

#[derive(Debug, Deserialize, Default)]
struct RawMetadata {
    name: Option<String>,
    displayname: Option<String>,
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

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct FlagDef {
    #[serde(default)]
    pub entry_prefix: String,
    #[serde(default)]
    pub exit_suffix: String,
    #[serde(default)]
    pub explicit_exit: String,
    /// このモード中に現れても数字モードを終了させない文字（例: 小数点・桁区切り）。
    #[serde(default)]
    pub exempt_chars: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct CapitalFlagDef {
    #[serde(default)]
    pub entry_prefix: String,
    #[serde(default)]
    pub double_entry_prefix: String,
}

#[derive(Debug, Deserialize)]
struct RawBrailleFile {
    #[serde(default)]
    metadata: RawMetadata,
    flags: RawFlags,
    table: RawTable,
}

// ============================================================
// BrailleTable
// ============================================================

/// 変換テーブルとフラグ定義を保持する。
///
/// [`BrailleTable::embedded`] でデフォルトの組み込みテーブルを使うか、
/// [`embedded_tables`] で全テーブルを列挙するか、
/// [`BrailleTable::from_toml`] でビルドする。
#[derive(Debug, Clone)]
pub struct BrailleTable {
    /// テーブルの識別名（TOML `[metadata].name`）。
    pub name: Option<String>,
    /// テーブルの表示名（TOML `[metadata].displayname`）。
    pub displayname: Option<String>,
    /// 複合音テーブル (2文字キー)。先に試みる。
    pub(crate) kana_compound: HashMap<String, String>,
    /// 単音テーブル (1文字キー)。
    pub(crate) kana_single: HashMap<String, String>,
    /// 日本語（全角）記号テーブル。
    /// foreign_word フラグが OFF のとき参照する。
    pub(crate) punct_jp: HashMap<String, PunctCell>,
    /// ASCII 記号テーブル。
    /// foreign_word フラグが ON のとき参照する。
    pub(crate) punct_latin: HashMap<String, PunctCell>,
    /// 数字テーブル。
    pub(crate) digit: HashMap<String, String>,
    /// ラテン文字テーブル。
    pub(crate) latin: HashMap<String, String>,
    /// フラグ定義。
    pub(crate) flag_digit: FlagDef,
    pub(crate) flag_foreign_word: FlagDef,
    pub(crate) flag_capital: CapitalFlagDef,
}

// ============================================================
// 埋め込みテーブルカタログ
// ============================================================

const TOML_GRADE1: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../dataset/japanese_grade1_braille.toml"
));

const TOML_NOCONVERSION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../dataset/japanese_no_conversion_braille.toml"
));

static EMBEDDED: LazyLock<Vec<BrailleTable>> = LazyLock::new(|| {
    vec![
        BrailleTable::from_toml(TOML_GRADE1).expect("grade1 TOML は有効"),
        BrailleTable::from_toml(TOML_NOCONVERSION).expect("noconversion TOML は有効"),
    ]
});

/// 組み込みテーブルの一覧を返す。最初のエントリがデフォルト（[`BrailleTable::embedded`]）。
pub fn embedded_tables() -> &'static [BrailleTable] {
    &EMBEDDED
}

/// 名前で組み込みテーブルを引く。TOML の `[metadata].name` と照合する。
pub fn embedded_table(name: &str) -> Option<BrailleTable> {
    EMBEDDED
        .iter()
        .find(|t| t.name.as_deref() == Some(name))
        .cloned()
}

impl BrailleTable {
    /// TOML 文字列からテーブルを構築する。
    pub fn from_toml(toml_str: &str) -> Result<Self> {
        let raw: RawBrailleFile = toml::from_str(toml_str)?;
        Ok(Self::from_raw(raw))
    }

    /// デフォルトの組み込みテーブル（日本語１級）を返す。
    pub fn embedded() -> Result<Self> {
        Ok(EMBEDDED[0].clone())
    }

    /// ファイルからテーブルを読み込む。
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let toml_str = std::fs::read_to_string(path)?;
        Self::from_toml(&toml_str)
    }

    fn from_raw(raw: RawBrailleFile) -> Self {
        let to_punct = |map: HashMap<String, PunctEntry>| -> HashMap<String, PunctCell> {
            map.into_iter().map(|(k, v)| (k, v.into_cell())).collect()
        };

        Self {
            name: raw.metadata.name,
            displayname: raw.metadata.displayname,
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
    fn embedded_table_metadata() {
        let table = BrailleTable::embedded().unwrap();
        assert_eq!(table.name.as_deref(), Some("japanese_grade1"));
        assert_eq!(table.displayname.as_deref(), Some("日本語１級"));
    }

    #[test]
    fn embedded_tables_list() {
        let tables = embedded_tables();
        assert!(tables.len() >= 2, "テーブルが2つ以上ある");
        assert!(tables
            .iter()
            .any(|t| t.name.as_deref() == Some("japanese_grade1")));
        assert!(tables
            .iter()
            .any(|t| t.name.as_deref() == Some("japanese_noconversion")));
    }

    #[test]
    fn embedded_table_by_name() {
        let t = embedded_table("japanese_noconversion").expect("noconversion テーブルが引ける");
        assert_eq!(t.displayname.as_deref(), Some("日本語無変換"));
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
        let cell = table.punct_jp.get("。").unwrap();
        assert_eq!(cell.braille.as_str(), "⠲");
        assert_eq!(cell.trailing, 2);
        let cell2 = table.punct_jp.get("、").unwrap();
        assert_eq!(cell2.braille.as_str(), "⠰");
        assert_eq!(cell2.trailing, 1);

        // ASCII 記号（"." "," "!" "?" ":" ";" は trailing なし。全角の日本語記号と違い、
        // ASCII 文脈では後続スペースを付けない仕様）
        let cell3 = table.punct_latin.get(".").unwrap();
        assert_eq!(cell3.braille.as_str(), "⠲");
        assert_eq!(cell3.trailing, 0);
        let cell4 = table.punct_latin.get("-").unwrap();
        assert_eq!(cell4.braille.as_str(), "⠤");
        assert_eq!(cell4.trailing, 0);
    }

    #[test]
    fn punct_class_and_suppress_before() {
        let table = BrailleTable::embedded().unwrap();

        // 「！」「？」「。」は stop クラスで、次が stop なら trailing を抑制する
        for key in ["。", "！", "？"] {
            let cell = table.punct_jp.get(key).unwrap();
            assert_eq!(cell.class.as_deref(), Some("stop"), "class of {key}");
            assert_eq!(cell.effective_trailing("stop"), 0, "suppressed before stop");
            assert_eq!(cell.effective_trailing("none"), 2, "normal trailing");
        }

        // 「、」「・」は pause クラス、抑制条件は今のところ設定していない
        for key in ["、", "・"] {
            let cell = table.punct_jp.get(key).unwrap();
            assert_eq!(cell.class.as_deref(), Some("pause"), "class of {key}");
        }
    }

    #[test]
    fn class_matcher_include_and_exclude() {
        let include = ClassMatcher::Include(vec!["stop".to_string()]);
        assert!(include.matches("stop"));
        assert!(!include.matches("pause"));

        let exclude = ClassMatcher::Exclude {
            exclude: vec!["open".to_string()],
        };
        assert!(exclude.matches("stop"));
        assert!(!exclude.matches("open"));
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
