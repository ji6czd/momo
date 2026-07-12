use crate::{Error, Result};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

// ============================================================
// クラスパス（ドット区切りの階層クラス名）
// ============================================================
//
// 各文字は「フルパス」を1本持つ。テーブル所属（構造）＋ 句読点の意味クラス
// （あれば末尾に1セグメント）を `.` で連結したもの。
//
//   ア           → "kana"
//   1            → "digit"
//   a            → "latin"
//   .（ASCII）   → "punct.latin"          （class なし）
//   。           → "punct.jp.stop"        （[table.punct.jp] + class=stop）
//   未定義文字   → "none"
//
// `[transitions]` のキーはこのパスを**セグメント単位のプレフィックス**で参照する：
//   "punct"        → punct.jp / punct.jp.stop / punct.latin すべてに一致
//   "punct.jp"     → punct.jp と punct.jp.stop
//   "punct.jp.stop"→ それだけ
//   "*"            → すべて（match-all。内部表現は空文字列）
//   "punct.*"      → "punct" と同義（末尾 `.*` は読みやすさのための別表記）
// 特異度＝パターンのセグメント数（深いほど優先）。両側の合計が大きいルールが勝ち、
// 同特異度の競合は値の max（現行の「完全一致 > ワイルドカード、両ワイルドカードは max」
// の厳密な一般化）。

/// 分類を持たない文字（テーブル未定義文字）を表す予約クラスパス。
pub(crate) const CLASS_NONE: &str = "none";
/// `[table.kana.*]` の文字の構造クラスパス（single/compound は区別しない）。
pub(crate) const CLASS_KANA: &str = "kana";
/// `[table.digit]` の文字の構造クラスパス。
pub(crate) const CLASS_DIGIT: &str = "digit";
/// `[table.latin]` の文字の構造クラスパス。
pub(crate) const CLASS_LATIN: &str = "latin";
/// `[table.punct.jp]` の構造クラスパス接頭辞。
pub(crate) const CLASS_PUNCT_JP: &str = "punct.jp";
/// `[table.punct.latin]` の構造クラスパス接頭辞。
pub(crate) const CLASS_PUNCT_LATIN: &str = "punct.latin";

/// 予約構造クラス名。`[table.punct] classes`（意味クラス）での宣言は不可。
/// `punct` / `punct.jp` / `punct.latin` は構造から決まるパスなので同様に宣言不可。
pub(crate) const RESERVED_CLASSES: [&str; 5] =
    [CLASS_NONE, CLASS_KANA, CLASS_DIGIT, CLASS_LATIN, "punct"];

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
    /// braille 文字列と（あれば）意味クラス名に分解する。
    fn split(self) -> (String, Option<String>) {
        match self {
            PunctEntry::Simple(braille) => (braille, None),
            PunctEntry::WithClass { braille, class } => (braille, class),
        }
    }
}

/// 句読点テーブルの 1 エントリ。
#[derive(Debug, Clone)]
pub(crate) struct PunctCell {
    pub braille: String,
    /// この記号のフルクラスパス（例: "punct.jp.stop" / "punct.latin"）。
    /// 構造接頭辞（所属テーブル）＋ 意味クラス（あれば末尾）を `.` で連結したもの。
    pub path: String,
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

#[derive(Debug, Clone, Default, Deserialize)]
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
// UEB（英語）縮約型と TOML デシリアライズ
// ============================================================

/// 縮約が適用できる語中の位置。1つの縮約は複数の position を持てる（いずれか一致で適用）。
/// 語 `[start, end)`（連続英字）の中の位置 `i`・長さ `len` に対して判定する。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Position {
    /// 語中どこでも（strong 系 / en / in の groupsign 役）。
    Always,
    /// 語頭でない（`i > start`）。ing 専用。
    NonInitial,
    /// 語頭＋後続が文字（`i == start && i+len < end`）。be / con / dis。
    Initial,
    /// 両側が文字＝語中（`start < i && i+len < end`）。ea / bb / cc / ff / gg。
    Medial,
    /// 単独で1語（`i == start && i+len == end`）。全 wordsign。
    Wordsign,
}

/// UEB 縮約の1エントリ。[`Table::contractions`] が保持し、
/// [`EnglishTranslator`](crate::EnglishTranslator) が適用する。
#[derive(Debug, Clone)]
pub(crate) struct Contraction {
    pub cell: String,
    pub positions: Vec<Position>,
    /// 下方記号（dot 1/4 を含まない）か。lower wordsign の約物接触制限に使う。
    pub lower: bool,
    /// `initial` を解禁する語幹（空なら `initial` は使わない）。be / con / dis 用。
    pub initial_stems: Vec<String>,
    /// shortform（略字）か。
    pub shortform: bool,
}

#[derive(Debug, Deserialize)]
struct RawUeb {
    #[serde(default)]
    metadata: RawMetadata,
    indicators: RawIndicators,
    letters: HashMap<String, String>,
    digits: HashMap<String, String>,
    punctuation: HashMap<String, String>,
    /// grade 1 テーブルは縮約を持たない（セクションを省略可）。
    #[serde(default)]
    contractions: HashMap<String, RawContraction>,
    /// shortform（略字）の whitelist と例外。grade 1 は持たない（省略可）。
    #[serde(default)]
    shortforms: RawShortforms,
}

#[derive(Debug, Deserialize, Default)]
struct RawShortforms {
    /// この語の中では shortform を使ってよい（Appendix 1）。
    #[serde(default)]
    words: Vec<String>,
    /// s / 's を付けても shortform を使わない例外語。
    #[serde(default)]
    plural_exceptions: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawIndicators {
    capital: String,
    capital_word: String,
    number: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawContraction {
    cell: String,
    positions: Vec<String>,
    #[serde(default)]
    lower: bool,
    #[serde(default)]
    initial_stems: Vec<String>,
    #[serde(default)]
    shortform: bool,
}

// ============================================================
// Table
// ============================================================

/// 変換テーブルとフラグ定義を保持する。
///
/// [`Table::embedded`] でデフォルトの組み込みテーブルを使うか、
/// [`embedded_tables`] で全テーブルを列挙するか、
/// [`Table::from_toml`] でビルドする。
#[derive(Debug, Clone)]
pub struct Table {
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
    /// 英語（UEB）の縮約テーブル。綴り（小文字 a–z）→ 縮約定義。
    /// 日本語テーブルでは空。[`EnglishTranslator`](crate::EnglishTranslator) が参照する。
    pub(crate) contractions: HashMap<String, Contraction>,
    /// shortform（略字）を語中でも使ってよい語の whitelist（UEB Appendix 1）。
    /// 縮約を持たないテーブル（grade 1）・日本語テーブルでは空。
    pub(crate) shortform_words: HashSet<String>,
    /// s / 's を付けても shortform を使わない例外語。
    pub(crate) shortform_plural_exceptions: HashSet<String>,
}

// ============================================================
// 埋め込みテーブルカタログ
// ============================================================

const TOML_GRADE1: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../dataset/japanese_grade1.toml"
));

const TOML_NOCONVERSION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../dataset/japanese_no_conversion.toml"
));

const TOML_UEB_GRADE2: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../dataset/english_ueb_grade2.toml"
));

const TOML_UEB_GRADE1: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../dataset/english_ueb_grade1.toml"
));

static EMBEDDED: LazyLock<Vec<Table>> = LazyLock::new(|| {
    vec![
        // [0] がデフォルト（Table::embedded）。日本語テーブルと英語テーブルを同じ
        // カタログに載せ、Japanese/English どちらの from_embedded_name も名前で引く。
        Table::from_toml(TOML_GRADE1).expect("grade1 TOML は有効"),
        Table::from_toml(TOML_NOCONVERSION).expect("noconversion TOML は有効"),
        Table::from_ueb_toml(TOML_UEB_GRADE2).expect("UEB grade2 TOML は有効"),
        Table::from_ueb_toml(TOML_UEB_GRADE1).expect("UEB grade1 TOML は有効"),
    ]
});

/// 組み込みテーブルの一覧を返す。最初のエントリがデフォルト（[`Table::embedded`]）。
pub fn embedded_tables() -> &'static [Table] {
    &EMBEDDED
}

/// 名前で組み込みテーブルを引く。TOML の `[metadata].name` と照合する。
pub fn embedded_table(name: &str) -> Option<Table> {
    EMBEDDED
        .iter()
        .find(|t| t.name.as_deref() == Some(name))
        .cloned()
}

impl Table {
    /// TOML 文字列からテーブルを構築する。
    pub fn from_toml(toml_str: &str) -> Result<Self> {
        let raw: RawBrailleFile = toml::from_str(toml_str)?;
        Self::from_raw(raw)
    }

    /// デフォルトの組み込みテーブル（日本語１級）を返す。
    pub fn embedded() -> Result<Self> {
        Ok(EMBEDDED[0].clone())
    }

    /// ファイルから（日本語スキーマの）テーブルを読み込む。
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let toml_str = std::fs::read_to_string(path)?;
        Self::from_toml(&toml_str)
    }

    /// UEB（英語）スキーマの TOML 文字列からテーブルを構築する。
    ///
    /// 英語の要素は共通フィールドへ写す（letters→`latin`, digits→`digit`,
    /// punctuation→`punct_latin`, 大文字符→`flag_capital`, 数符→`flag_digit`, contractions）。
    /// 日本語側のフィールド（kana / punct_jp / foreign_word / transitions）は空になる。
    pub fn from_ueb_toml(toml_str: &str) -> Result<Self> {
        let raw: RawUeb = toml::from_str(toml_str)?;
        let punct_latin = raw
            .punctuation
            .into_iter()
            .map(|(k, braille)| {
                (
                    k,
                    PunctCell {
                        braille,
                        path: CLASS_PUNCT_LATIN.to_owned(),
                    },
                )
            })
            .collect();
        Ok(Self {
            name: raw.metadata.name,
            displayname: raw.metadata.displayname,
            kana_compound: HashMap::new(),
            kana_single: HashMap::new(),
            punct_jp: HashMap::new(),
            punct_latin,
            digit: raw.digits,
            latin: raw.letters,
            flag_digit: FlagDef {
                entry_prefix: raw.indicators.number,
                ..FlagDef::default()
            },
            flag_foreign_word: FlagDef::default(),
            flag_capital: CapitalFlagDef {
                entry_prefix: raw.indicators.capital,
                double_entry_prefix: raw.indicators.capital_word,
            },
            transitions: Vec::new(),
            contractions: parse_contractions(raw.contractions)?,
            shortform_words: raw.shortforms.words.into_iter().collect(),
            shortform_plural_exceptions: raw.shortforms.plural_exceptions.into_iter().collect(),
        })
    }

    /// UEB（英語）スキーマのファイルからテーブルを読み込む。
    pub fn from_ueb_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let toml_str = std::fs::read_to_string(path)?;
        Self::from_ueb_toml(&toml_str)
    }

    fn from_raw(raw: RawBrailleFile) -> Result<Self> {
        let declared = raw.table.punct.classes;
        // 予約名（none/kana/digit/latin/punct）を意味クラスとして宣言していないか検証
        if let Some(reserved) = declared
            .iter()
            .find(|c| RESERVED_CLASSES.contains(&c.as_str()))
        {
            return Err(Error::Validation(format!(
                "[table.punct] classes: \"{reserved}\" は予約名のため宣言できません"
            )));
        }

        let punct_jp = build_punct_cells(CLASS_PUNCT_JP, "jp", raw.table.punct.jp, &declared)?;
        let punct_latin =
            build_punct_cells(CLASS_PUNCT_LATIN, "latin", raw.table.punct.latin, &declared)?;
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
            // 日本語テーブルは縮約・shortform を持たない
            contractions: HashMap::new(),
            shortform_words: HashSet::new(),
            shortform_plural_exceptions: HashSet::new(),
        })
    }

    /// クラスパス `from` から `to` への遷移で挿入する点字スペース数。
    ///
    /// - 各遷移ルールの両側パターンが `from` / `to` に**プレフィックス一致**するものを集める
    /// - 特異度（両側パターンのセグメント数の合計）が最大のルールが勝つ
    ///   （完全一致は深いパスなので必ずワイルドカードより優先される）
    /// - 同特異度で複数当たったら値の **max**
    ///   （例: `"punct.jp.stop -> *" = 2` と `"* -> punct.jp.inline" = 1` が同時に当たる
    ///   「。→」は max の 2）
    /// - どれも当たらなければ 0
    pub(crate) fn transition_spaces(&self, from: &str, to: &str) -> usize {
        let mut best_spec: Option<usize> = None;
        let mut best_val: usize = 0;
        for (pf, pt, n) in &self.transitions {
            if !pattern_matches(pf, from) || !pattern_matches(pt, to) {
                continue;
            }
            let spec = pattern_spec(pf) + pattern_spec(pt);
            match best_spec {
                Some(b) if spec < b => {}
                Some(b) if spec == b => best_val = best_val.max(*n),
                _ => {
                    best_spec = Some(spec);
                    best_val = *n;
                }
            }
        }
        best_val
    }

    /// クラスパス `from` を起点とする遷移で挿入されうる最大スペース数。
    /// 逆変換（backtranslator）が記号の直後のスペースを吸収する上限として使う。
    ///
    /// この記号自身が原因で挿入されるスペースだけを数えるため、from 側が
    /// match-all（`*`）のルールは除外する（それらは「後続の to が原因」であって
    /// この記号のトレイリングではない。例: `"* -> punct.jp.inline"` は … の前の
    /// スペースであり、直前がどの記号でも起きるので開き括弧のトレイリングにしない）。
    pub(crate) fn max_transition_spaces_from(&self, from: &str) -> usize {
        self.transitions
            .iter()
            .filter(|(pf, _, _)| !pf.is_empty() && pattern_matches(pf, from))
            .map(|&(_, _, n)| n)
            .max()
            .unwrap_or(0)
    }
}

/// 遷移パターン `pat`（正規化済み: `*` は空文字列、`X.*` は `X` に短縮済み）が
/// クラスパス `class` にセグメント単位でプレフィックス一致するか。
fn pattern_matches(pat: &str, class: &str) -> bool {
    if pat.is_empty() {
        return true; // match-all（元 "*"）
    }
    class == pat || class.starts_with(pat) && class[pat.len()..].starts_with('.')
}

/// 遷移パターンの特異度＝セグメント数（match-all は 0）。
fn pattern_spec(pat: &str) -> usize {
    if pat.is_empty() {
        0
    } else {
        pat.split('.').count()
    }
}

/// UEB `[contractions]` をパース・検証して綴り → [`Contraction`] のマップにする。
/// 綴りは小文字 a–z、position は既知名、initial_stems も小文字 a–z のみ。
fn parse_contractions(
    raw: HashMap<String, RawContraction>,
) -> Result<HashMap<String, Contraction>> {
    let mut out = HashMap::with_capacity(raw.len());
    for (spelling, rc) in raw {
        if !spelling.chars().all(|c| c.is_ascii_lowercase()) {
            return Err(Error::Validation(format!(
                "[contractions] \"{spelling}\": 綴りは小文字 a-z のみ"
            )));
        }
        if rc.positions.is_empty() {
            return Err(Error::Validation(format!(
                "[contractions] \"{spelling}\": positions が空です"
            )));
        }
        let mut positions = Vec::with_capacity(rc.positions.len());
        for p in &rc.positions {
            let pos = match p.as_str() {
                "always" => Position::Always,
                "noninitial" => Position::NonInitial,
                "initial" => Position::Initial,
                "medial" => Position::Medial,
                "wordsign" => Position::Wordsign,
                other => {
                    return Err(Error::Validation(format!(
                        "[contractions] \"{spelling}\": position \"{other}\" は \
                         always/noninitial/initial/medial/wordsign のいずれか"
                    )));
                }
            };
            positions.push(pos);
        }
        for stem in &rc.initial_stems {
            if !stem.chars().all(|c| c.is_ascii_lowercase()) {
                return Err(Error::Validation(format!(
                    "[contractions] \"{spelling}\": initial_stems \"{stem}\" は小文字 a-z のみ"
                )));
            }
        }
        out.insert(
            spelling,
            Contraction {
                cell: rc.cell,
                positions,
                lower: rc.lower,
                initial_stems: rc.initial_stems,
                shortform: rc.shortform,
            },
        );
    }
    Ok(out)
}

/// `[table.punct.{section}]` のエントリからフルパス付き [`PunctCell`] を構築する。
/// セルの `class` は `declared`（`[table.punct] classes`）に宣言されていなければエラー。
/// パスは `prefix`（"punct.jp" / "punct.latin"）＋ class（あれば `.class`）。
fn build_punct_cells(
    prefix: &str,
    section: &str,
    map: HashMap<String, PunctEntry>,
    declared: &[String],
) -> Result<HashMap<String, PunctCell>> {
    let mut out = HashMap::with_capacity(map.len());
    for (key, entry) in map {
        let (braille, class) = entry.split();
        let path = match class {
            None => prefix.to_owned(),
            Some(c) => {
                if !declared.iter().any(|d| d == &c) {
                    return Err(Error::Validation(format!(
                        "[table.punct.{section}] \"{key}\": class \"{c}\" は \
                         [table.punct] classes に宣言されていません"
                    )));
                }
                format!("{prefix}.{c}")
            }
        };
        out.insert(key, PunctCell { braille, path });
    }
    Ok(out)
}

/// `[transitions]` のキー `"A -> B"` をパースして検証・正規化する。
/// 両側はクラスパスのプレフィックス（`kana` / `punct.jp` / `punct.jp.stop` …）か
/// ワイルドカード `*`（片側のみ）。`X.*` は `X` に短縮、`*` は空文字列に正規化する。
fn parse_transitions(
    raw: HashMap<String, usize>,
    declared: &[String],
) -> Result<Vec<(String, String, usize)>> {
    // 有効なパスプレフィックスの集合を作る（typo 検出用）。
    let mut allowed: Vec<String> = vec![
        CLASS_NONE.into(),
        CLASS_KANA.into(),
        CLASS_DIGIT.into(),
        CLASS_LATIN.into(),
        "punct".into(),
        CLASS_PUNCT_JP.into(),
        CLASS_PUNCT_LATIN.into(),
    ];
    for c in declared {
        allowed.push(format!("{CLASS_PUNCT_JP}.{c}"));
        allowed.push(format!("{CLASS_PUNCT_LATIN}.{c}"));
    }

    let mut out = Vec::with_capacity(raw.len());
    for (key, spaces) in raw {
        let Some((from, to)) = key.split_once("->") else {
            return Err(Error::Validation(format!(
                "[transitions] \"{key}\": キーは \"クラス名 -> クラス名\" の形式で書いてください"
            )));
        };
        let from = normalize_pattern(from.trim());
        let to = normalize_pattern(to.trim());
        if from.is_empty() && to.is_empty() {
            return Err(Error::Validation(format!(
                "[transitions] \"{key}\": 両側をワイルドカード \"*\" にはできません"
            )));
        }
        for pat in [&from, &to] {
            if !pat.is_empty() && !allowed.iter().any(|a| a == pat) {
                return Err(Error::Validation(format!(
                    "[transitions] \"{key}\": クラス \"{pat}\" は予約構造クラス\
                     （kana/digit/latin/none/punct/punct.jp/punct.latin）でも \
                     [table.punct] classes 由来のパス（punct.jp.<class> 等）にもありません"
                )));
            }
        }
        out.push((from, to, spaces));
    }
    Ok(out)
}

/// 遷移パターンを内部表現に正規化する：`*` → 空文字列（match-all）、
/// 末尾 `.*` → 除去（`punct.*` は `punct` と同義）。
fn normalize_pattern(s: &str) -> String {
    if s == "*" {
        String::new()
    } else if let Some(prefix) = s.strip_suffix(".*") {
        prefix.to_owned()
    } else {
        s.to_owned()
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
        let table = Table::embedded().expect("埋め込み TOML がパースできること");
        assert!(!table.kana_single.is_empty(), "単音テーブルが空でない");
        assert!(!table.kana_compound.is_empty(), "複合音テーブルが空でない");
        assert!(!table.digit.is_empty(), "数字テーブルが空でない");
        assert!(!table.latin.is_empty(), "ラテン文字テーブルが空でない");
        assert!(!table.punct_jp.is_empty(), "日本語記号テーブルが空でない");
        assert!(!table.punct_latin.is_empty(), "ASCII記号テーブルが空でない");
    }

    #[test]
    fn embedded_table_metadata() {
        let table = Table::embedded().unwrap();
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
        let table = Table::embedded().unwrap();
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
        let table = Table::embedded().unwrap();

        // 日本語記号（フルパス = 構造接頭辞 + 意味クラス）
        let cell = table.punct_jp.get("。").unwrap();
        assert_eq!(cell.braille.as_str(), "⠲");
        assert_eq!(cell.path.as_str(), "punct.jp.stop");
        let cell2 = table.punct_jp.get("、").unwrap();
        assert_eq!(cell2.braille.as_str(), "⠰");
        assert_eq!(cell2.path.as_str(), "punct.jp.pause");

        // ASCII 記号（"." "-" は class なし = 構造パスのみ）
        let cell3 = table.punct_latin.get(".").unwrap();
        assert_eq!(cell3.braille.as_str(), "⠲");
        assert_eq!(cell3.path.as_str(), "punct.latin");
        let cell4 = table.punct_latin.get("-").unwrap();
        assert_eq!(cell4.braille.as_str(), "⠤");
        assert_eq!(cell4.path.as_str(), "punct.latin");
    }

    #[test]
    fn punct_classes() {
        let table = Table::embedded().unwrap();
        for key in ["。", "！", "？"] {
            let cell = table.punct_jp.get(key).unwrap();
            assert_eq!(cell.path.as_str(), "punct.jp.stop", "path of {key}");
        }
        for key in ["、", "・"] {
            let cell = table.punct_jp.get(key).unwrap();
            assert_eq!(cell.path.as_str(), "punct.jp.pause", "path of {key}");
        }
        for key in ["→", "←", "…"] {
            let cell = table.punct_jp.get(key).unwrap();
            assert_eq!(cell.path.as_str(), "punct.jp.inline", "path of {key}");
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
        Table::from_toml(&toml).expect("宣言済みクラスは通る");
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
        let err = Table::from_toml(&toml).unwrap_err();
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
        assert!(Table::from_toml(&toml).is_err());
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
        let err = Table::from_toml(&toml).unwrap_err();
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
        let err = Table::from_toml(&toml).unwrap_err();
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
        let err = Table::from_toml(&toml).unwrap_err();
        assert!(matches!(err, Error::Validation(_)), "{err}");
        assert!(err.to_string().contains("予約名"), "{err}");
    }

    // --- [transitions] ---

    #[test]
    fn embedded_transitions() {
        let table = Table::embedded().unwrap();
        // 文字種境界
        assert_eq!(table.transition_spaces("kana", "latin"), 1);
        assert_eq!(table.transition_spaces("latin", "kana"), 1);
        assert_eq!(
            table.transition_spaces("kana", "digit"),
            0,
            "未宣言ペアは 0"
        );
        // 英文句読点→かな（新規: Hello.こんにちは の分割）
        assert_eq!(table.transition_spaces("punct.latin", "kana"), 1);
        assert_eq!(
            table.transition_spaces("punct.latin", "digit"),
            0,
            "3.14 は分割しない"
        );
        // 記号のあと（プレフィックスのワイルドカード）
        assert_eq!(table.transition_spaces("punct.jp.stop", "kana"), 2);
        assert_eq!(
            table.transition_spaces("punct.jp.stop", "punct.jp.stop"),
            0,
            "明示 0 で抑制"
        );
        assert_eq!(
            table.transition_spaces("punct.jp.stop", "punct.jp.close"),
            0
        );
        assert_eq!(table.transition_spaces("punct.jp.pause", "kana"), 1);
        // 前後一マスの記号（両ワイルドカードの max）
        assert_eq!(table.transition_spaces("kana", "punct.jp.inline"), 1);
        assert_eq!(table.transition_spaces("punct.jp.inline", "kana"), 1);
        assert_eq!(
            table.transition_spaces("punct.jp.stop", "punct.jp.inline"),
            2,
            "max(2, 1)"
        );
        // 逆変換用の上限
        assert_eq!(table.max_transition_spaces_from("punct.jp.stop"), 2);
        assert_eq!(table.max_transition_spaces_from("punct.jp.pause"), 1);
        assert_eq!(table.max_transition_spaces_from("punct.jp.open"), 0);
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
"punct.jp.stop -> *" = 2
"punct.jp.stop -> punct.jp.stop" = 0
"* -> punct.jp.inline" = 1
"punct.jp.stop -> punct.jp.inline" = 5
"#;
        let table = Table::from_toml(&toml).unwrap();
        assert_eq!(
            table.transition_spaces("punct.jp.stop", "kana"),
            2,
            "from 側 *"
        );
        assert_eq!(
            table.transition_spaces("kana", "punct.jp.inline"),
            1,
            "to 側 *"
        );
        assert_eq!(
            table.transition_spaces("punct.jp.stop", "punct.jp.stop"),
            0,
            "完全一致の明示 0"
        );
        assert_eq!(
            table.transition_spaces("punct.jp.stop", "punct.jp.inline"),
            5,
            "深いパス（完全一致）はワイルドカードより優先"
        );
        assert_eq!(table.transition_spaces("kana", "latin"), 0);
    }

    #[test]
    fn transitions_prefix_and_specificity_override() {
        // 一般則 punct.jp.* -> kana = 1 に、特定の punct.jp.stop -> kana = 2 を重ねる。
        // 階層クラスの主目的（デフォルト＋例外）が第一級で書けることの確認。
        let toml = minimal_toml(
            r#"
[table.punct]
classes = ["stop", "pause"]
[table.punct.jp]
[table.punct.latin]
"#,
        ) + r#"
[transitions]
"punct.jp.* -> kana" = 1
"punct.jp.stop -> kana" = 2
"#;
        let table = Table::from_toml(&toml).unwrap();
        // 一般則（プレフィックス punct.jp が pause に一致）
        assert_eq!(table.transition_spaces("punct.jp.pause", "kana"), 1);
        // 深い方が勝つ（特異度 3+1 > 2+1）
        assert_eq!(table.transition_spaces("punct.jp.stop", "kana"), 2);
        // punct.* も punct.jp.* と同様にプレフィックス一致する
        assert_eq!(table.transition_spaces("punct.latin", "latin"), 0, "未宣言");
    }

    #[test]
    fn transitions_accept_dotstar_sugar() {
        // "punct.*" は "punct" と同義（末尾 .* は読みやすさのための別表記）
        let toml = minimal_toml(
            r#"
[table.punct]
classes = ["stop"]
[table.punct.jp]
[table.punct.latin]
"#,
        ) + r#"
[transitions]
"punct.* -> kana" = 1
"#;
        let table = Table::from_toml(&toml).unwrap();
        assert_eq!(table.transition_spaces("punct.jp.stop", "kana"), 1);
        assert_eq!(table.transition_spaces("punct.latin", "kana"), 1);
        assert_eq!(table.transition_spaces("kana", "kana"), 0);
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
        let err = Table::from_toml(&toml).unwrap_err();
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
"latin -> punct.jp.stop" = 2
"#;
        let table = Table::from_toml(&toml).unwrap();
        assert_eq!(table.transition_spaces("kana", "latin"), 1);
        assert_eq!(table.transition_spaces("latin", "punct.jp.stop"), 2);
        assert_eq!(table.transition_spaces("punct.jp.stop", "latin"), 0);
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
        let err = Table::from_toml(&toml).unwrap_err();
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
        let err = Table::from_toml(&toml).unwrap_err();
        assert!(matches!(err, Error::Validation(_)), "{err}");
        assert!(err.to_string().contains("形式"), "{err}");
    }

    #[test]
    fn spot_check_digit_and_latin() {
        let table = Table::embedded().unwrap();
        assert_eq!(table.digit.get("0").map(|s| s.as_str()), Some("⠚"));
        assert_eq!(table.digit.get("５").map(|s| s.as_str()), Some("⠑"));
        assert_eq!(table.latin.get("a").map(|s| s.as_str()), Some("⠁"));
        assert_eq!(table.latin.get("z").map(|s| s.as_str()), Some("⠵"));
    }

    #[test]
    fn flag_digit_defined() {
        let table = Table::embedded().unwrap();
        assert_eq!(table.flag_digit.entry_prefix, "⠼");
        assert_eq!(table.flag_digit.explicit_exit, "⠤");
        assert!(table.flag_digit.exempt_chars.contains(&".".to_string()));
    }
}
