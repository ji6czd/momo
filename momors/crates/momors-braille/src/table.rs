use crate::{Error, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::LazyLock;

/// 予約クラス名。分類を持たない文字（class 未指定・非句読点）を表すため
/// converter が暗黙に使う。
pub(crate) const CLASS_NONE: &str = "none";
/// テーブル所属から暗黙に決まる予約クラス名（`[table.kana]` の文字）。
pub(crate) const CLASS_KANA: &str = "kana";
/// テーブル所属から暗黙に決まる予約クラス名（`[table.digit]` の文字）。
pub(crate) const CLASS_DIGIT: &str = "digit";
/// テーブル所属から暗黙に決まる予約クラス名（`[table.latin]` の文字）。
pub(crate) const CLASS_LATIN: &str = "latin";

/// すべての予約クラス名。`classes` での宣言は不可だが、
/// `suppress_before` や `[transitions]` の条件には使える。
pub(crate) const RESERVED_CLASSES: [&str; 4] = [CLASS_NONE, CLASS_KANA, CLASS_DIGIT, CLASS_LATIN];

// ============================================================
// TOML デシリアライズ用の中間型
// ============================================================

/// 句読点エントリ: 文字列だけか、class 付きのインラインテーブルか。
/// 前後のスペースはセルには書かず、すべて `[transitions]` でクラス単位に宣言する。
///
/// ```toml
/// "ー" = "⠒"
/// "。" = { braille = "⠲", class = "stop" }
/// ```
#[derive(Debug, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
enum PunctEntry {
    Simple(String),
    WithClass {
        braille: String,
        /// この記号自身の分類（例: "stop", "pause"）。省略時は無分類。
        #[serde(default)]
        class: Option<String>,
    },
}

impl PunctEntry {
    fn into_cell(self) -> PunctCell {
        match self {
            PunctEntry::Simple(braille) => PunctCell {
                braille,
                class: None,
            },
            PunctEntry::WithClass { braille, class } => PunctCell { braille, class },
        }
    }
}

/// 句読点テーブルの 1 エントリ。
#[derive(Debug, Clone)]
pub(crate) struct PunctCell {
    pub braille: String,
    /// この記号自身の分類（例: "stop", "pause"）。未分類なら `None`。
    pub class: Option<String>,
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
    /// このテーブルで使えるクラス名の宣言。セルの `class` と `[transitions]` に
    /// 現れる名前はここに宣言されていなければロードエラーになる（typo 検出）。
    #[serde(default)]
    classes: Vec<String>,
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
    /// クラス遷移の境界に挿入する点字スペース数。キーは `"A -> B"` 形式。
    #[serde(default)]
    transitions: HashMap<String, usize>,
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
    /// クラス遷移 (from, to) → 挿入する点字スペース数。エントリ数は少ないため線形探索。
    pub(crate) transitions: Vec<(String, String, usize)>,
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
        Self::from_raw(raw)
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

    fn from_raw(raw: RawBrailleFile) -> Result<Self> {
        let to_punct = |map: HashMap<String, PunctEntry>| -> HashMap<String, PunctCell> {
            map.into_iter().map(|(k, v)| (k, v.into_cell())).collect()
        };

        let declared = raw.table.punct.classes;
        let punct_jp = to_punct(raw.table.punct.jp);
        let punct_latin = to_punct(raw.table.punct.latin);
        validate_punct_classes(&declared, "jp", &punct_jp)?;
        validate_punct_classes(&declared, "latin", &punct_latin)?;
        let transitions = parse_transitions(raw.transitions, &declared)?;

        Ok(Self {
            name: raw.metadata.name,
            displayname: raw.metadata.displayname,
            kana_compound: raw.table.kana.compound,
            kana_single: raw.table.kana.single,
            punct_jp,
            punct_latin,
            digit: raw.table.digit,
            latin: raw.table.latin,
            flag_digit: raw.flags.digit,
            flag_foreign_word: raw.flags.foreign_word,
            flag_capital: raw.flags.capital,
            transitions,
        })
    }

    /// クラス `from` から `to` への遷移で挿入する点字スペース数。
    ///
    /// - 完全一致のペアが最優先（明示した `= 0` による抑制もここに含まれる）
    /// - なければワイルドカード `"from -> *"` と `"* -> to"` の**最大値**
    ///   （例: `"stop -> *" = 2` と `"* -> inline" = 1` が同時に当たる「。→」は 2。
    ///   「あとを二マス」「まえを一マス」の両方を満たすのは大きい方）
    /// - どちらもなければ 0
    pub(crate) fn transition_spaces(&self, from: &str, to: &str) -> usize {
        if let Some(&(_, _, n)) = self
            .transitions
            .iter()
            .find(|(f, t, _)| f == from && t == to)
        {
            return n;
        }
        let lookup = |f: &str, t: &str| {
            self.transitions
                .iter()
                .find(|(tf, tt, _)| tf == f && tt == t)
                .map_or(0, |&(_, _, n)| n)
        };
        lookup(from, "*").max(lookup("*", to))
    }

    /// クラス `from` を起点とする遷移で挿入されうる最大スペース数。
    /// 逆変換（backtranslator）が記号の直後のスペースを吸収する上限として使う。
    pub(crate) fn max_transition_spaces_from(&self, from: &str) -> usize {
        self.transitions
            .iter()
            .filter(|(f, _, _)| f == from)
            .map(|&(_, _, n)| n)
            .max()
            .unwrap_or(0)
    }
}

/// `[transitions]` のキー `"A -> B"` をパースして検証する。
/// 両側のクラス名は予約クラス（kana/digit/latin/none）か
/// `[table.punct]` classes の宣言名、またはワイルドカード `*`（片側のみ）。
fn parse_transitions(
    raw: HashMap<String, usize>,
    declared: &[String],
) -> Result<Vec<(String, String, usize)>> {
    let known = |name: &str| {
        name == "*" || RESERVED_CLASSES.contains(&name) || declared.iter().any(|c| c == name)
    };
    let mut out = Vec::with_capacity(raw.len());
    for (key, spaces) in raw {
        let Some((from, to)) = key.split_once("->") else {
            return Err(Error::Validation(format!(
                "[transitions] \"{key}\": キーは \"クラス名 -> クラス名\" の形式で書いてください"
            )));
        };
        let (from, to) = (from.trim(), to.trim());
        if from == "*" && to == "*" {
            return Err(Error::Validation(format!(
                "[transitions] \"{key}\": 両側をワイルドカード \"*\" にはできません"
            )));
        }
        for name in [from, to] {
            if name.is_empty() || !known(name) {
                return Err(Error::Validation(format!(
                    "[transitions] \"{key}\": クラス \"{name}\" は予約クラス\
                     （kana/digit/latin/none）でも [table.punct] classes の宣言にもありません"
                )));
            }
        }
        out.push((from.to_owned(), to.to_owned(), spaces));
    }
    Ok(out)
}

/// セルの `class` に現れるクラス名がすべて `classes` に宣言されているかを検証する。
/// 予約名（none/kana/digit/latin）は宣言不可・`class` に指定不可
/// （kana/digit/latin はテーブル所属から暗黙に決まるため）。
fn validate_punct_classes(
    declared: &[String],
    section: &str,
    cells: &HashMap<String, PunctCell>,
) -> Result<()> {
    if let Some(reserved) = declared
        .iter()
        .find(|c| RESERVED_CLASSES.contains(&c.as_str()))
    {
        return Err(Error::Validation(format!(
            "[table.punct] classes: \"{reserved}\" は予約名のため宣言できません"
        )));
    }
    let is_declared = |name: &str| declared.iter().any(|c| c == name);
    for (key, cell) in cells {
        if let Some(class) = &cell.class {
            if !is_declared(class) {
                return Err(Error::Validation(format!(
                    "[table.punct.{section}] \"{key}\": class \"{class}\" は \
                     [table.punct] classes に宣言されていません"
                )));
            }
        }
    }
    Ok(())
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
    fn spot_check_punct() {
        let table = BrailleTable::embedded().unwrap();

        // 日本語記号（スペースはセルに書かず [transitions] で宣言する）
        let cell = table.punct_jp.get("。").unwrap();
        assert_eq!(cell.braille.as_str(), "⠲");
        assert_eq!(cell.class.as_deref(), Some("stop"));
        let cell2 = table.punct_jp.get("、").unwrap();
        assert_eq!(cell2.braille.as_str(), "⠰");
        assert_eq!(cell2.class.as_deref(), Some("pause"));

        // ASCII 記号（"." "-" は無分類 = スペースなし）
        let cell3 = table.punct_latin.get(".").unwrap();
        assert_eq!(cell3.braille.as_str(), "⠲");
        assert_eq!(cell3.class, None);
        let cell4 = table.punct_latin.get("-").unwrap();
        assert_eq!(cell4.braille.as_str(), "⠤");
        assert_eq!(cell4.class, None);
    }

    #[test]
    fn punct_classes() {
        let table = BrailleTable::embedded().unwrap();
        for key in ["。", "！", "？"] {
            let cell = table.punct_jp.get(key).unwrap();
            assert_eq!(cell.class.as_deref(), Some("stop"), "class of {key}");
        }
        for key in ["、", "・"] {
            let cell = table.punct_jp.get(key).unwrap();
            assert_eq!(cell.class.as_deref(), Some("pause"), "class of {key}");
        }
        for key in ["→", "←", "…"] {
            let cell = table.punct_jp.get(key).unwrap();
            assert_eq!(cell.class.as_deref(), Some("inline"), "class of {key}");
        }
    }

    /// 検証テスト用の最小 TOML。`punct` に `[table.punct]` 以下のセクションを渡す。
    fn minimal_toml(punct: &str) -> String {
        format!(
            r#"
[flags.digit]
[flags.foreign_word]
[flags.capital]

[table.kana.compound]
[table.kana.single]

{punct}

[table.digit]
[table.latin]
"#
        )
    }

    #[test]
    fn class_validation_accepts_declared_names() {
        let toml = minimal_toml(
            r#"
[table.punct]
classes = ["stop", "pause"]
[table.punct.jp]
"。" = { braille = "⠲", class = "stop" }
"、" = { braille = "⠰", class = "pause" }
[table.punct.latin]
"#,
        );
        BrailleTable::from_toml(&toml).expect("宣言済みクラスは通る");
    }

    #[test]
    fn class_validation_catches_typo_in_class() {
        let toml = minimal_toml(
            r#"
[table.punct]
classes = ["stop"]
[table.punct.jp]
"。" = { braille = "⠲", class = "stpo" }
[table.punct.latin]
"#,
        );
        let err = BrailleTable::from_toml(&toml).unwrap_err();
        assert!(matches!(err, Error::Validation(_)), "{err}");
        assert!(err.to_string().contains("stpo"), "{err}");
    }

    #[test]
    fn punct_entry_rejects_unknown_field() {
        // 旧形式の trailing などの不明フィールドはロードエラーになる（typo・旧テーブル検出）
        let toml = minimal_toml(
            r#"
[table.punct]
classes = ["stop"]
[table.punct.jp]
"。" = { braille = "⠲", trailing = 2, class = "stop" }
[table.punct.latin]
"#,
        );
        assert!(BrailleTable::from_toml(&toml).is_err());
    }

    #[test]
    fn class_validation_requires_declaration() {
        // classes 宣言なしで class を使うとエラー（暗黙の語彙は認めない）
        let toml = minimal_toml(
            r#"
[table.punct.jp]
"。" = { braille = "⠲", class = "stop" }
[table.punct.latin]
"#,
        );
        let err = BrailleTable::from_toml(&toml).unwrap_err();
        assert!(matches!(err, Error::Validation(_)), "{err}");
    }

    #[test]
    fn class_validation_rejects_reserved_none() {
        let toml = minimal_toml(
            r#"
[table.punct]
classes = ["stop", "none"]
[table.punct.jp]
[table.punct.latin]
"#,
        );
        let err = BrailleTable::from_toml(&toml).unwrap_err();
        assert!(matches!(err, Error::Validation(_)), "{err}");
        assert!(err.to_string().contains("予約名"), "{err}");
    }

    #[test]
    fn class_validation_rejects_reserved_table_class() {
        // "kana" / "digit" / "latin" はテーブル所属から暗黙に決まる予約名
        let toml = minimal_toml(
            r#"
[table.punct]
classes = ["stop", "kana"]
[table.punct.jp]
[table.punct.latin]
"#,
        );
        let err = BrailleTable::from_toml(&toml).unwrap_err();
        assert!(matches!(err, Error::Validation(_)), "{err}");
        assert!(err.to_string().contains("予約名"), "{err}");
    }

    // --- [transitions] ---

    #[test]
    fn embedded_transitions() {
        let table = BrailleTable::embedded().unwrap();
        // 文字種境界
        assert_eq!(table.transition_spaces("kana", "latin"), 1);
        assert_eq!(table.transition_spaces("latin", "kana"), 1);
        assert_eq!(
            table.transition_spaces("kana", "digit"),
            0,
            "未宣言ペアは 0"
        );
        // 記号のあと（ワイルドカード）
        assert_eq!(table.transition_spaces("stop", "kana"), 2);
        assert_eq!(table.transition_spaces("stop", "stop"), 0, "明示 0 で抑制");
        assert_eq!(table.transition_spaces("stop", "close"), 0);
        assert_eq!(table.transition_spaces("pause", "kana"), 1);
        // 前後一マスの記号（両ワイルドカードの max）
        assert_eq!(table.transition_spaces("kana", "inline"), 1);
        assert_eq!(table.transition_spaces("inline", "kana"), 1);
        assert_eq!(table.transition_spaces("stop", "inline"), 2, "max(2, 1)");
        // 逆変換用の上限
        assert_eq!(table.max_transition_spaces_from("stop"), 2);
        assert_eq!(table.max_transition_spaces_from("pause"), 1);
        assert_eq!(table.max_transition_spaces_from("open"), 0);
    }

    #[test]
    fn transitions_wildcard_resolution() {
        let toml = minimal_toml(
            r#"
[table.punct]
classes = ["stop", "inline"]
[table.punct.jp]
[table.punct.latin]
"#,
        ) + r#"
[transitions]
"stop -> *" = 2
"stop -> stop" = 0
"* -> inline" = 1
"stop -> inline" = 5
"#;
        let table = BrailleTable::from_toml(&toml).unwrap();
        assert_eq!(table.transition_spaces("stop", "kana"), 2, "from 側 *");
        assert_eq!(table.transition_spaces("kana", "inline"), 1, "to 側 *");
        assert_eq!(
            table.transition_spaces("stop", "stop"),
            0,
            "完全一致の明示 0"
        );
        assert_eq!(
            table.transition_spaces("stop", "inline"),
            5,
            "完全一致はワイルドカードより優先"
        );
        assert_eq!(table.transition_spaces("kana", "latin"), 0);
    }

    #[test]
    fn transitions_reject_double_wildcard() {
        let toml = minimal_toml(
            r#"
[table.punct]
[table.punct.jp]
[table.punct.latin]
"#,
        ) + r#"
[transitions]
"* -> *" = 1
"#;
        let err = BrailleTable::from_toml(&toml).unwrap_err();
        assert!(matches!(err, Error::Validation(_)), "{err}");
    }

    #[test]
    fn transitions_parse_and_validate() {
        let toml = minimal_toml(
            r#"
[table.punct]
classes = ["stop"]
[table.punct.jp]
[table.punct.latin]
"#,
        ) + r#"
[transitions]
"kana -> latin" = 1
"latin -> stop" = 2
"#;
        let table = BrailleTable::from_toml(&toml).unwrap();
        assert_eq!(table.transition_spaces("kana", "latin"), 1);
        assert_eq!(table.transition_spaces("latin", "stop"), 2);
        assert_eq!(table.transition_spaces("stop", "latin"), 0);
    }

    #[test]
    fn transitions_reject_unknown_class() {
        let toml = minimal_toml(
            r#"
[table.punct]
[table.punct.jp]
[table.punct.latin]
"#,
        ) + r#"
[transitions]
"kana -> ltain" = 1
"#;
        let err = BrailleTable::from_toml(&toml).unwrap_err();
        assert!(matches!(err, Error::Validation(_)), "{err}");
        assert!(err.to_string().contains("ltain"), "{err}");
    }

    #[test]
    fn transitions_reject_malformed_key() {
        let toml = minimal_toml(
            r#"
[table.punct]
[table.punct.jp]
[table.punct.latin]
"#,
        ) + r#"
[transitions]
"kana latin" = 1
"#;
        let err = BrailleTable::from_toml(&toml).unwrap_err();
        assert!(matches!(err, Error::Validation(_)), "{err}");
        assert!(err.to_string().contains("形式"), "{err}");
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
