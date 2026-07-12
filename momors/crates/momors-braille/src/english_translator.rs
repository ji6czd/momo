//! UEB (Unified English Braille) の英語点字変換。
//!
//! 変換規則は [`Table`] が持ち、縮約（`Table::contractions`）を含む grade 2 テーブル
//! （`english_ueb_grade2.toml`）なら縮約あり、縮約を持たない grade 1 テーブル
//! （`english_ueb_grade1.toml`）なら無縮約になる。適用ロジック（最長一致・語境界・
//! ing 語頭例外・位置制約）はこのモジュールが持つ。shortform の allowlist は
//! テーブルの `[shortforms]` セクション（`Table::shortform_words`）。
//!
//! **Phase 1「文脈非依存コア」＋ Phase 2a「lower signs」＋ Phase 3/4（頭字符・尾字符・shortforms）** を実装。
//! 仕様: `docs/ueb-grade2-spec.md` ほか `docs/ueb-grade2-*.md`。
//!
//! この変換器は**スタンドアロン**（入出力は英語→UEB点字に閉じる）。日本語の状態
//! （外字符・数符・prev_class）を一切持ち込まない。日本語中の英語は外国語引用符
//! `⠦…⠴` で囲まれた「日本語が入らない島」なので、その中身をこの変換器に渡せば同じ
//! エンジンで賄える（島の検出・引用符の出力は日本語側の別レイヤーの仕事）。
//!
//! ## 適用ロジック（コードが持つ usage）
//!
//! - **最長一致**: 各位置で最長の適用可能な縮約を選ぶ。
//! - **位置プリミティブ [`Position`]**: 縮約は複数の位置（always / noninitial / initial /
//!   medial / wordsign）を持ち、いずれかが語中の現在位置で満たされれば適用。
//!   `wordsign` は単独で1語のとき、`initial` は語頭＋後続文字、`medial` は両側が文字、など。
//! - **lower wordsign の約物接触制限（Phase 2a 近似）**: 下方 wordsign（be/his/was/were/in/enough）が
//!   下方約物に隣接するときは縮約しない（`be,` → `b-e`）。
//!
//! ## 既知の割り切り（後続フェーズで対応）
//!
//! - 形態素境界を跨ぐ強縮約を抑制しない（`mishap` の `sh` を縮約してしまう）。
//! - lower sign rule（dot 1/4 アンカー）は位置カテゴリで近似（厳密実装は Phase 2b）。
//! - 大文字は語単位（先頭大文字→`⠠`、全大文字語→`⠠⠠`）のみ。語中の散在大文字は近似。
//! - 数符後の grade-1 曖昧性回避は未対応。

use crate::table::{Contraction, Position, Table};
use crate::{Error, Result, UnknownCharPolicy};
use std::collections::HashMap;
use std::sync::LazyLock;

const BRAILLE_SPACE: char = '⠀'; // U+2800

// shortform の allowlist と例外は [`Table`]（`[shortforms]` セクション）が持つ。
// grade1 テーブルには無いので grade1 は shortform を使わない。

// ============================================================
// EnglishResult
// ============================================================

/// 英語点字変換の結果。
#[derive(Debug, Clone)]
pub struct EnglishResult {
    braille_text: String,
    /// 入力文字インデックス → 点字文字インデックス（その文字の出力先頭セル位置）。
    src_to_braille: Vec<usize>,
}

impl EnglishResult {
    /// 変換後の点字文字列。
    pub fn braille_text(&self) -> &str {
        &self.braille_text
    }

    /// 入力文字インデックス → 点字先頭セルのインデックス。
    ///
    /// 縮約でまとめられた複数文字は、先頭文字がその縮約セル（および語頭の大文字符）を、
    /// 残りの文字は同じセル位置を指す。
    pub fn src_to_braille(&self) -> &[usize] {
        &self.src_to_braille
    }
}

// ============================================================
// EnglishTranslator
// ============================================================

/// UEB 英語点字変換器。[`Table`] を保持し、その `latin`/`digit`/`punct_latin`/
/// `flag_capital`/`flag_digit`/`contractions` を参照して点訳する。縮約を持たない
/// テーブル（grade 1）なら無縮約になる。char キーの参照キャッシュは `table` から派生する。
///
/// [`JapaneseTranslator`](crate::JapaneseTranslator) と同じインターフェース
/// （`new`/`from_embedded`/`from_embedded_name`/`from_file`/`with_unknown_char_policy`/
/// `set_unknown_char_policy`/`translate`）を持つ。
pub struct EnglishTranslator {
    /// 変換テーブル（正本）。
    table: Table,
    /// テーブル未定義文字の扱い。
    unknown_char: UnknownCharPolicy,
    // --- 以下は table から派生したキャッシュ（char キーで高速参照する） ---
    letters: HashMap<char, String>,
    digits: HashMap<char, String>,
    punctuation: HashMap<char, String>,
    cap: String,
    cap_word: String,
    number: String,
    max_contraction_len: usize,
}

impl EnglishTranslator {
    /// テーブルを指定して変換器を作る。char キーのキャッシュと最長縮約長を派生する。
    pub fn new(table: Table) -> Self {
        let letters = char_keyed(&table.latin);
        let digits = char_keyed(&table.digit);
        let punctuation = table
            .punct_latin
            .iter()
            .filter_map(|(k, pc)| single_char(k).map(|c| (c, pc.braille.clone())))
            .collect();
        let cap = table.flag_capital.entry_prefix.clone();
        let cap_word = table.flag_capital.double_entry_prefix.clone();
        let number = table.flag_digit.entry_prefix.clone();
        let max_contraction_len = table
            .contractions
            .keys()
            .map(|s| s.chars().count())
            .max()
            .unwrap_or(0);
        Self {
            table,
            unknown_char: UnknownCharPolicy::default(),
            letters,
            digits,
            punctuation,
            cap,
            cap_word,
            number,
            max_contraction_len,
        }
    }

    /// 組み込みの UEB grade 2（縮約あり）テーブルで変換器を作る。
    pub fn from_embedded() -> Result<Self> {
        Self::from_embedded_name("ueb_english_grade2")
    }

    /// 名前で組み込みテーブルを指定して変換器を作る（`[metadata].name` と照合）。
    /// 例: `"ueb_english_grade1"` / `"ueb_english_grade2"`。見つからなければエラー。
    pub fn from_embedded_name(name: &str) -> Result<Self> {
        let table = crate::table::embedded_table(name)
            .ok_or_else(|| Error::UnknownTable(name.to_owned()))?;
        Ok(Self::new(table))
    }

    /// UEB スキーマの TOML ファイルからテーブルを読み込んで変換器を作る。
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self::new(Table::from_ueb_file(path)?))
    }

    /// テーブル未定義文字の扱いを設定する（ビルダーメソッド）。
    pub fn with_unknown_char_policy(mut self, policy: UnknownCharPolicy) -> Self {
        self.unknown_char = policy;
        self
    }

    /// テーブル未定義文字の扱いをその場で変更する。
    pub fn set_unknown_char_policy(&mut self, policy: UnknownCharPolicy) {
        self.unknown_char = policy;
    }

    /// 保持しているテーブルへの参照。
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// テーブルの識別名（`[metadata].name`。例: `"ueb_english_grade2"`）。
    pub fn name(&self) -> Option<&str> {
        self.table.name.as_deref()
    }

    /// テーブルの表示名（`[metadata].displayname`。例: `"UEB English (Grade 2)"`）。
    pub fn displayname(&self) -> Option<&str> {
        self.table.displayname.as_deref()
    }

    /// 英語テキストを UEB 点字に変換する。
    pub fn translate(&self, text: &str) -> EnglishResult {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut braille = String::new();
        let mut src_to_braille = vec![0usize; n];

        let mut i = 0;
        while i < n {
            let c = chars[i];
            if c.is_ascii_alphabetic() {
                // 語（連続する英字）をまとめて処理する。
                let mut j = i;
                while j < n && chars[j].is_ascii_alphabetic() {
                    j += 1;
                }
                self.translate_word(&chars, i, j, &mut braille, &mut src_to_braille);
                i = j;
            } else if c.is_ascii_digit() {
                i = self.translate_number(&chars, i, &mut braille, &mut src_to_braille);
            } else if c == ' ' || c == '\t' {
                src_to_braille[i] = braille.chars().count();
                braille.push(BRAILLE_SPACE);
                i += 1;
            } else if let Some(cell) = self.punctuation.get(&c) {
                src_to_braille[i] = braille.chars().count();
                braille.push_str(cell);
                i += 1;
            } else {
                // テーブル未定義文字。ポリシーに従う（既定はスペース、
                // PassThrough は原文の文字をそのまま出す＝スクリーンリーダー用途）。
                src_to_braille[i] = braille.chars().count();
                match self.unknown_char {
                    UnknownCharPolicy::PassThrough => braille.push(c),
                    UnknownCharPolicy::Space => braille.push(BRAILLE_SPACE),
                }
                i += 1;
            }
        }

        EnglishResult {
            braille_text: braille,
            src_to_braille,
        }
    }

    /// 語 `chars[start..end]`（すべて英字）を変換して `braille` に追記する。
    fn translate_word(
        &self,
        chars: &[char],
        start: usize,
        end: usize,
        braille: &mut String,
        src_to_braille: &mut [usize],
    ) {
        // 大文字符（語単位）。語頭の文字にこの位置を帰属させる。
        let word_head_pos = braille.chars().count();
        let all_upper =
            end - start >= 2 && chars[start..end].iter().all(|c| c.is_ascii_uppercase());
        if all_upper {
            braille.push_str(&self.cap_word);
        } else if chars[start].is_ascii_uppercase() {
            braille.push_str(&self.cap);
        }

        let mut i = start;
        let mut first = true;
        while i < end {
            let (len, cell) = self.longest_match(chars, i, start, end);
            let cell_pos = braille.chars().count();
            braille.push_str(&cell);
            for k in i..i + len {
                // 語頭文字は大文字符ごと（word_head_pos）、それ以外はセル位置を指す。
                src_to_braille[k] = if first && k == start {
                    word_head_pos
                } else {
                    cell_pos
                };
            }
            first = false;
            i += len;
        }
    }

    /// 位置 `i` から始まる最長の適用可能な縮約を返す。無ければ1文字（grade 1）。
    fn longest_match(
        &self,
        chars: &[char],
        i: usize,
        word_start: usize,
        word_end: usize,
    ) -> (usize, String) {
        let maxl = self.max_contraction_len.min(word_end - i);
        // 縮約はすべて2文字以上なので len>=2 だけ試す。
        for len in (2..=maxl).rev() {
            let sub: String = chars[i..i + len]
                .iter()
                .flat_map(|c| c.to_lowercase())
                .collect();
            if let Some(con) = self.table.contractions.get(&sub) {
                if self.applicable(con, chars, i, len, word_start, word_end) {
                    return (len, con.cell.clone());
                }
            }
        }
        // フォールバック: 1文字（小文字化して letters を引く）。
        let lc = chars[i].to_ascii_lowercase();
        let cell = self
            .letters
            .get(&lc)
            .cloned()
            .unwrap_or_else(|| BRAILLE_SPACE.to_string());
        (1, cell)
    }

    /// 位置 `i`・長さ `len` の縮約が適用可能か。
    fn applicable(
        &self,
        con: &Contraction,
        chars: &[char],
        i: usize,
        len: usize,
        word_start: usize,
        word_end: usize,
    ) -> bool {
        let whole_word = i == word_start && i + len == word_end;

        // Phase 2a 近似（4A）: lower wordsign が下方約物に接触するなら縮約しない。
        // 語全体に一致し、かつ wordsign 役を持つ lower 縮約が対象。左右どちらかの隣接文字が
        // 下方約物なら不適用（例: "be," "was." は縮約せず b-e / w-a-s と書く）。
        if con.lower && whole_word && con.positions.contains(&Position::Wordsign) {
            let left_bad = word_start > 0 && self.is_lower_punct(chars[word_start - 1]);
            let right_bad = word_end < chars.len() && self.is_lower_punct(chars[word_end]);
            if left_bad || right_bad {
                return false;
            }
        }

        con.positions.iter().any(|p| match p {
            Position::Always => true,
            Position::NonInitial => i > word_start,
            // be/con/dis は「第一音節であるとき」だけ。音節は綴りから決まらないので、
            // 語幹リスト（initial_stems）に載る語でだけ解禁する。既定は不許可。
            Position::Initial => {
                i == word_start
                    && i + len < word_end
                    && self.stem_allows_initial(con, chars, word_start, word_end)
            }
            Position::Medial => i > word_start && i + len < word_end,
            // 単独で1語のとき。加えて shortform は、語全体が Appendix 1 の
            // Shortforms List に載っていれば語中のどこでも使える（Phase 4b）。
            Position::Wordsign => {
                whole_word
                    || (con.shortform && self.word_allows_shortform(chars, word_start, word_end))
            }
        })
    }

    /// 語 `chars[word_start..word_end]` の中で shortform を使ってよいか。
    ///
    /// Appendix 1 のリストに載っている語、またはその語に `s` を付けた語（`friend`→`friends`）。
    /// ただし `abouts` / `almosts` / `hims` の3語は例外で使わない。
    /// （`'s` は語の切れ目になるので、`children's` の `children` は素の語として一致する。）
    fn word_allows_shortform(&self, chars: &[char], word_start: usize, word_end: usize) -> bool {
        let word: String = chars[word_start..word_end]
            .iter()
            .flat_map(|c| c.to_lowercase())
            .collect();
        if self.table.shortform_plural_exceptions.contains(&word) {
            return false;
        }
        if self.table.shortform_words.contains(&word) {
            return true;
        }
        // 末尾の "s" を落とした語がリストにあれば可
        word.strip_suffix('s')
            .is_some_and(|stem| self.table.shortform_words.contains(stem))
    }

    /// 語 `chars[word_start..word_end]` が `con.initial_stems` のいずれかで始まるか。
    /// 語幹が空なら `initial` は使わない（誤らない側に倒す既定）。
    fn stem_allows_initial(
        &self,
        con: &Contraction,
        chars: &[char],
        word_start: usize,
        word_end: usize,
    ) -> bool {
        con.initial_stems.iter().any(|stem| {
            let stem: Vec<char> = stem.chars().collect();
            word_end - word_start >= stem.len()
                && stem
                    .iter()
                    .zip(&chars[word_start..word_start + stem.len()])
                    .all(|(s, c)| *s == c.to_ascii_lowercase())
        })
    }

    /// 文字 `c` が下方約物（点字が dot 1/4 を含まない約物）か。
    /// 現状の punctuation は全て下方だが、将来の上方約物追加に備えて点で判定する。
    fn is_lower_punct(&self, c: char) -> bool {
        self.punctuation.get(&c).is_some_and(|braille| {
            braille.chars().all(|cell| {
                let bits = (cell as u32).wrapping_sub(0x2800);
                cell as u32 >= 0x2800 && bits & 0x09 == 0 // dot1=0x01, dot4=0x08 を含まない
            })
        })
    }

    /// 数字の並び（間の小数点・桁区切りを含む）を変換する。次の位置を返す。
    fn translate_number(
        &self,
        chars: &[char],
        start: usize,
        braille: &mut String,
        src_to_braille: &mut [usize],
    ) -> usize {
        let head_pos = braille.chars().count();
        braille.push_str(&self.number);
        let n = chars.len();
        let mut i = start;
        while i < n {
            let c = chars[i];
            if c.is_ascii_digit() {
                let pos = braille.chars().count();
                if let Some(cell) = self.digits.get(&c) {
                    braille.push_str(cell);
                }
                src_to_braille[i] = if i == start { head_pos } else { pos };
                i += 1;
            } else if (c == '.' || c == ',') && i + 1 < n && chars[i + 1].is_ascii_digit() {
                // 数字に挟まれた小数点・桁区切りは数モードを続ける。
                let pos = braille.chars().count();
                if let Some(cell) = self.punctuation.get(&c) {
                    braille.push_str(cell);
                }
                src_to_braille[i] = pos;
                i += 1;
            } else {
                break;
            }
        }
        i
    }
}

/// 1文字だけの文字列ならその char。0文字・2文字以上なら None。
fn single_char(s: &str) -> Option<char> {
    let mut it = s.chars();
    match (it.next(), it.next()) {
        (Some(c), None) => Some(c),
        _ => None,
    }
}

/// `String` キーのマップを char キーのキャッシュに変換する（1文字キーのみ採用）。
fn char_keyed(m: &HashMap<String, String>) -> HashMap<char, String> {
    m.iter()
        .filter_map(|(k, v)| single_char(k).map(|c| (c, v.clone())))
        .collect()
}

/// 組み込み UEB 英語テーブル（grade 2）の表示名（例: `"UEB English (Grade 2)"`）。
/// メニュー表示用。grade を選べるようにする段階で grade 引数版へ一般化する予定。
pub fn embedded_english_displayname() -> Option<&'static str> {
    static DISPLAYNAME: LazyLock<Option<String>> = LazyLock::new(|| {
        crate::table::embedded_table("ueb_english_grade2").and_then(|t| t.displayname)
    });
    DISPLAYNAME.as_deref()
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn tr() -> EnglishTranslator {
        EnglishTranslator::from_embedded().expect("UEB 英語テーブルをロードできること")
    }

    fn t(s: &str) -> String {
        tr().translate(s).braille_text().to_string()
    }

    fn t1(s: &str) -> String {
        EnglishTranslator::from_embedded_name("ueb_english_grade1")
            .expect("UEB 英語テーブル(grade1)をロードできること")
            .translate(s)
            .braille_text()
            .to_string()
    }

    // --- grade 1（無縮約） ---

    #[test]
    fn grade1_has_no_contractions() {
        // grade 2 で縮約される語が grade 1 では素の文字列になる。
        assert_eq!(t("and"), "⠯"); // grade 2: strong contraction
        assert_eq!(t1("and"), "⠁⠝⠙"); // grade 1: a + n + d
        assert_eq!(t("the"), "⠮");
        assert_eq!(t1("the"), "⠞⠓⠑");
        assert_eq!(t("you"), "⠽"); // grade 2: wordsign
        assert_eq!(t1("you"), "⠽⠕⠥");
        assert_eq!(t("should"), "⠩⠙"); // grade 2: shortform
        assert_eq!(t1("should"), "⠎⠓⠕⠥⠇⠙");
    }

    #[test]
    fn grade1_keeps_capitals_and_numbers() {
        // 大文字符・数符は grade 非依存。
        assert_eq!(t1("Cat"), "⠠⠉⠁⠞"); // 大文字符 + c a t
        assert_eq!(t1("NHK"), "⠠⠠⠝⠓⠅"); // 大文字語符 + n h k
        assert_eq!(t1("123"), "⠼⠁⠃⠉"); // 数符 + 1 2 3
    }

    #[test]
    fn table_metadata() {
        let g2 = tr();
        assert_eq!(g2.name(), Some("ueb_english_grade2"));
        assert_eq!(g2.displayname(), Some("UEB English (Grade 2)"));

        let g1 = EnglishTranslator::from_embedded_name("ueb_english_grade1").unwrap();
        assert_eq!(g1.name(), Some("ueb_english_grade1"));
        assert_eq!(g1.displayname(), Some("UEB English (Grade 1)"));
        assert!(
            g1.table().contractions.is_empty(),
            "grade1 は縮約を持たない"
        );
    }

    // --- 基本文字 ---

    #[test]
    fn plain_letters() {
        // "is" = i + s（縮約なし）
        assert_eq!(t("is"), "⠊⠎");
    }

    // --- strong groupsign（always） ---

    #[test]
    fn groupsign_th_within_word() {
        // path = p + a + th
        assert_eq!(t("path"), "⠏⠁⠹");
    }

    #[test]
    fn groupsign_st_within_word() {
        // fast = f + a + st
        assert_eq!(t("fast"), "⠋⠁⠌");
    }

    #[test]
    fn strong_contraction_within_word() {
        // profit = p + r + of + i + t（of は境界跨ぎでも使う）
        assert_eq!(t("profit"), "⠏⠗⠷⠊⠞");
    }

    #[test]
    fn strong_contraction_the_within_word() {
        // gather = g + a + the + r
        // （mother/father は Phase 3 で頭字符 ⠐⠍/⠐⠋ になるため、
        //   the の語中 groupsign 検証には頭字符を持たない語を使う）
        assert_eq!(t("gather"), "⠛⠁⠮⠗");
    }

    #[test]
    fn ed_and_and_within_word() {
        // landed = l + and + ed
        assert_eq!(t("landed"), "⠇⠯⠫");
    }

    // --- ing 語頭例外 ---

    #[test]
    fn ing_used_mid_and_end() {
        // sing = s + ing
        assert_eq!(t("sing"), "⠎⠬");
    }

    #[test]
    fn ing_not_at_word_start() {
        // 語頭の ing は縮約しない。ただし Phase 2a で「in」groupsign が入ったので、
        // ingot = in + g + o + t = ⠔⠛⠕⠞（ing は語頭不可・in は語頭可＝後続 g がアンカー）。
        assert_eq!(t("ingot"), "⠔⠛⠕⠞");
    }

    // --- Phase 2a: lower signs（位置制約） ---

    #[test]
    fn lower_be_initial_groupsign() {
        // become = be + c + o + m + e（be は語頭 groupsign）
        assert_eq!(t("become"), "⠆⠉⠕⠍⠑");
    }

    #[test]
    fn lower_con_dis_initial() {
        // concern = con + c + er + n / distrust = dis + t + r + u + st
        assert_eq!(t("concern"), "⠒⠉⠻⠝");
        assert_eq!(t("distrust"), "⠲⠞⠗⠥⠌");
    }

    #[test]
    fn lower_be_con_dis_not_in_middle() {
        // 語中の be/con/dis は使わない: rebel = r+e+b+e+l（be は語頭のみ）
        assert_eq!(t("rebel"), "⠗⠑⠃⠑⠇");
    }

    // --- Phase 2b: be/con/dis の第一音節規則（initial_stems ゲート） ---

    #[test]
    fn initial_stems_block_when_not_first_syllable() {
        // be が第一音節でない語は縮約しない（語幹リストに無い）。
        assert_eq!(t("benzene"), "⠃⠢⠵⠢⠑"); // b + en + z + en + e（出典の "b5z5e"）
        assert_eq!(t("been"), "⠃⠑⠢"); // b + e + en
        assert_eq!(t("bench"), "⠃⠢⠡"); // b + en + ch
        assert_eq!(t("bed"), "⠃⠫"); // b + ed
                                    // con が第一音節でない語も同様
        assert_eq!(t("cone"), "⠉⠐⠕"); // c + one（頭字符）
    }

    #[test]
    fn initial_stems_allow_when_first_syllable() {
        assert_eq!(t("become"), "⠆⠉⠕⠍⠑"); // becom
        assert_eq!(t("believe"), "⠆⠇⠊⠑⠧⠑"); // believ
        assert_eq!(t("congenial"), "⠒⠛⠢⠊⠁⠇"); // 出典の "3g5ial"
        assert_eq!(t("distrust"), "⠲⠞⠗⠥⠌"); // distrust
    }

    #[test]
    fn only_first_of_two_lower_prefixes() {
        // be/con/dis が連続するときは最初のものだけ縮約する（`initial` が語頭限定なので自然に成立）
        assert_eq!(t("disbelief"), "⠲⠃⠑⠇⠊⠑⠋"); // 出典の "4belief"
        assert_eq!(t("disconnect"), "⠲⠉⠕⠝⠝⠑⠉⠞"); // 出典の "4connect"
    }

    #[test]
    fn lower_en_anywhere() {
        // end = en + d（en は位置無制限。語頭でも可）
        assert_eq!(t("end"), "⠢⠙");
    }

    #[test]
    fn lower_in_groupsign_and_wordsign() {
        // into = in + t + o（groupsign）/ in（単独）= ⠔（wordsign）
        assert_eq!(t("into"), "⠔⠞⠕");
        assert_eq!(t("in"), "⠔");
    }

    #[test]
    fn lower_ea_medial_only() {
        // meat = m + ea + t（語中）/ eat = e + a + t（語頭 ea は不可）
        assert_eq!(t("meat"), "⠍⠂⠞");
        assert_eq!(t("eat"), "⠑⠁⠞");
    }

    #[test]
    fn lower_double_letter_medial() {
        // rubber = r+u+bb+er / bigger = b+i+gg+er / egg = e+g+g（語末 gg は不可）
        assert_eq!(t("rubber"), "⠗⠥⠆⠻");
        assert_eq!(t("bigger"), "⠃⠊⠶⠻");
        assert_eq!(t("egg"), "⠑⠛⠛");
    }

    #[test]
    fn lower_wordsigns_standalone() {
        assert_eq!(t("be"), "⠆");
        assert_eq!(t("his"), "⠦");
        assert_eq!(t("was"), "⠴");
        assert_eq!(t("were"), "⠶");
        assert_eq!(t("enough"), "⠢");
    }

    #[test]
    fn lower_wordsign_blocked_by_lower_punctuation() {
        // Phase 2a 近似（4A）: 下方 wordsign が下方約物に接触すると縮約しない。
        assert_eq!(t("was."), "⠺⠁⠎⠲"); // ⠴ ではなく w+a+s
        assert_eq!(t("be,"), "⠃⠑⠂"); // ⠆ ではなく b+e
        assert_eq!(t("in."), "⠊⠝⠲"); // ⠔ ではなく i+n
    }

    #[test]
    fn lower_wordsign_ok_between_spaces() {
        // 空白に挟まれた lower wordsign は縮約する
        assert_eq!(t("he was in"), "⠓⠑⠀⠴⠀⠔");
    }

    // --- Phase 3: 頭字符（always）・尾字符（noninitial） ---

    #[test]
    fn initial_letter_dot5_standalone() {
        assert_eq!(t("mother"), "⠐⠍");
        assert_eq!(t("father"), "⠐⠋");
        assert_eq!(t("one"), "⠐⠕");
        assert_eq!(t("ought"), "⠐⠳");
    }

    #[test]
    fn initial_letter_within_word() {
        // 頭字符は語中でも使う（always）
        // （today は Phase 4a で shortform td=⠞⠙ になるため、頭字符の語中検証には別語を使う）
        assert_eq!(t("sunday"), "⠎⠥⠝⠐⠙"); // s + u + n + day
        assert_eq!(t("money"), "⠍⠐⠕⠽"); // m + one + y
        assert_eq!(t("somewhere"), "⠐⠎⠐⠱"); // some + where
        assert_eq!(t("thought"), "⠹⠐⠳"); // th + ought
    }

    #[test]
    fn initial_letter_indicator_triples() {
        // 指示符ちがいの三つ組（セル検算の要）
        assert_eq!(t("work"), "⠐⠺"); // dot 5
        assert_eq!(t("word"), "⠘⠺"); // dots 45
        assert_eq!(t("world"), "⠸⠺"); // dots 456
        assert_eq!(t("there"), "⠐⠮");
        assert_eq!(t("these"), "⠘⠮");
        assert_eq!(t("their"), "⠸⠮");
    }

    #[test]
    fn final_letter_dots46() {
        assert_eq!(t("sound"), "⠎⠨⠙"); // s + ound
        assert_eq!(t("count"), "⠉⠨⠞"); // c + ount
        assert_eq!(t("vision"), "⠧⠊⠨⠝"); // v + i + sion
        assert_eq!(t("balance"), "⠃⠁⠇⠨⠑"); // b + a + l + ance
        assert_eq!(t("unless"), "⠥⠝⠨⠎"); // u + n + less
    }

    #[test]
    fn final_letter_dots56() {
        assert_eq!(t("nation"), "⠝⠁⠰⠝"); // n + a + tion
        assert_eq!(t("city"), "⠉⠰⠽"); // c + ity
        assert_eq!(t("careful"), "⠉⠜⠑⠰⠇"); // c + ar + e + ful
        assert_eq!(t("hence"), "⠓⠰⠑"); // h + ence
        assert_eq!(t("long"), "⠇⠰⠛"); // l + ong
        assert_eq!(t("kindness"), "⠅⠔⠙⠰⠎"); // k + in + d + ness（Phase 2a の in と併用）
    }

    #[test]
    fn final_letter_not_at_word_start() {
        // mention の ment は語頭なので使わない → m + en + tion
        assert_eq!(t("mention"), "⠍⠢⠰⠝");
    }

    // --- Phase 4a: shortforms（単独で1語のときだけ） ---

    #[test]
    fn shortform_basic() {
        assert_eq!(t("about"), "⠁⠃");
        assert_eq!(t("your"), "⠽⠗"); // Phase 1 からの宿題を回収
        assert_eq!(t("today"), "⠞⠙");
        assert_eq!(t("said"), "⠎⠙");
        assert_eq!(t("would"), "⠺⠙");
        assert_eq!(t("braille"), "⠃⠗⠇");
    }

    #[test]
    fn shortform_with_embedded_contractions() {
        // 略記に縮約が埋まっているもの
        assert_eq!(t("should"), "⠩⠙"); // shd: sh + d
        assert_eq!(t("much"), "⠍⠡"); // mch: m + ch
        assert_eq!(t("such"), "⠎⠡"); // sch: s + ch
        assert_eq!(t("must"), "⠍⠌"); // mst: m + st
        assert_eq!(t("first"), "⠋⠌"); // fst: f + st
        assert_eq!(t("against"), "⠁⠛⠌"); // agst: a + g + st
        assert_eq!(t("although"), "⠁⠇⠹"); // alth: a + l + th
        assert_eq!(t("children"), "⠡⠝"); // chn: ch + n
    }

    #[test]
    fn shortform_with_lower_groupsigns() {
        assert_eq!(t("because"), "⠆⠉"); // bec: be + c
        assert_eq!(t("before"), "⠆⠋"); // bef: be + f
        assert_eq!(t("beyond"), "⠆⠽"); // bey: be + y
        assert_eq!(t("conceive"), "⠒⠉⠧"); // concv: con + c + v
    }

    #[test]
    fn shortform_with_er_and_initial_letter() {
        assert_eq!(t("herself"), "⠓⠻⠋"); // herf: h + er + f
        assert_eq!(t("perhaps"), "⠏⠻⠓"); // perh: p + er + h
        assert_eq!(t("perceive"), "⠏⠻⠉⠧"); // percv
        assert_eq!(t("oneself"), "⠐⠕⠋"); // onef: one(頭字符) + f
        assert_eq!(t("themselves"), "⠮⠍⠧⠎"); // themvs: the + m + v + s
        assert_eq!(t("ourselves"), "⠳⠗⠧⠎"); // ourvs: ou + r + v + s
        assert_eq!(t("thyself"), "⠹⠽⠋"); // thyf: th + y + f
    }

    // --- Phase 4b: Appendix 1「Shortforms List」に載る長い語の中でも使う ---

    #[test]
    fn shortform_inside_listed_word() {
        assert_eq!(t("aboutface"), "⠁⠃⠋⠁⠉⠑"); // about + f + a + c + e
        assert_eq!(t("gadabout"), "⠛⠁⠙⠁⠃"); // g + a + d + about（語頭でなくてよい）
        assert_eq!(t("friendly"), "⠋⠗⠇⠽"); // friend + l + y
        assert_eq!(t("goodness"), "⠛⠙⠰⠎"); // good + ness（尾字符と併用）
    }

    #[test]
    fn shortform_plural_uses_shortform() {
        // リストの語に "s" を足した語も shortform を使う
        assert_eq!(t("friends"), "⠋⠗⠎"); // fr + s
    }

    #[test]
    fn shortform_plural_three_exceptions() {
        // Appendix 1 の3例外。出典の ASCII 点字と一致すること。
        assert_eq!(t("abouts"), "⠁⠃⠳⠞⠎"); // "ab\ts" = a+b+ou+t+s
        assert_eq!(t("almosts"), "⠁⠇⠍⠕⠌⠎"); // "almo/s" = a+l+m+o+st+s
        assert_eq!(t("hims"), "⠓⠊⠍⠎"); // "hims" = h+i+m+s
    }

    #[test]
    fn shortform_not_used_in_unlisted_word() {
        // リストに無い語では shortform を使わない。期待値は出典の例そのもの。
        assert_eq!(t("blinded"), "⠃⠇⠔⠙⠫"); // "bl9d$" = b+l+in+d+ed
        assert_eq!(t("blinding"), "⠃⠇⠔⠙⠬"); // "bl9d+" = b+l+in+d+ing
        assert_eq!(t("aftereffect"), "⠁⠋⠞⠻⠑⠖⠑⠉⠞"); // "aft]e6ect" = a+f+t+er+e+ff+e+c+t
        assert_eq!(t("misconceived"), "⠍⠊⠎⠉⠕⠝⠉⠑⠊⠧⠫"); // "misconceiv$"
        assert_eq!(t("inbetween"), "⠔⠃⠑⠞⠺⠑⠢"); // "9betwe5" = in+b+e+t+w+e+en
        assert_eq!(t("hereinbefore"), "⠐⠓⠔⠃⠑⠿⠑"); // "\"h9be=e" = here+in+b+e+for+e
    }

    #[test]
    fn shortform_applies_next_to_punctuation() {
        // shortform は lower wordsign ではないので、約物に接触しても縮約する
        // （`was.` が非縮約になるのと対照的）
        assert_eq!(t("said."), "⠎⠙⠲");
    }

    #[test]
    fn sentence_across_all_phases() {
        // you(wordsign) / should(shortform) / be(lower wordsign) /
        // here(頭字符) / today(shortform)
        assert_eq!(t("you should be here today"), "⠽⠀⠩⠙⠀⠆⠀⠐⠓⠀⠞⠙");
    }

    // --- wordsign vs groupsign（語境界で解決） ---

    #[test]
    fn strong_wordsign_standalone() {
        // this（単独）= ⠹（th の wordsign）
        assert_eq!(t("this"), "⠹");
    }

    #[test]
    fn strong_wordsign_not_standalone_uses_groupsign() {
        // thistle（単独でない）= th + i + st + l + e
        assert_eq!(t("thistle"), "⠹⠊⠌⠇⠑");
    }

    #[test]
    fn alphabetic_wordsign_standalone() {
        // "you"（単独）= ⠽
        assert_eq!(t("you"), "⠽");
        // "us"（単独）= ⠥
        assert_eq!(t("us"), "⠥");
        // "knowledge"（単独）= ⠅
        assert_eq!(t("knowledge"), "⠅");
    }

    #[test]
    fn alphabetic_wordsign_not_standalone() {
        // youth（単独でない you）= y + ou + th
        // （your は Phase 4a で shortform yr=⠽⠗ になったので例から外した）
        assert_eq!(t("youth"), "⠽⠳⠹");
        // butter（単独でない but）= b + u + t + t + er
        assert_eq!(t("butter"), "⠃⠥⠞⠞⠻");
    }

    // --- 文（語境界・スペース） ---

    #[test]
    fn sentence_with_wordsigns_and_groupsigns() {
        // "this is a test"
        //   this → ⠹ / is → ⠊⠎ / a → ⠁ / test → t+e+st = ⠞⠑⠌
        assert_eq!(t("this is a test"), "⠹⠀⠊⠎⠀⠁⠀⠞⠑⠌");
    }

    // --- 大文字 ---

    #[test]
    fn capital_first_letter() {
        // "This" = ⠠ + this(⠹)
        assert_eq!(t("This"), "⠠⠹");
    }

    #[test]
    fn capital_word_all_upper() {
        // "GO"（全大文字, 2文字以上）= ⠠⠠ + g + o （単独 wordsign go=⠛ が効く: 全体が "go"）
        assert_eq!(t("GO"), "⠠⠠⠛");
    }

    // --- 数字 ---

    #[test]
    fn number_basic() {
        // 123 = ⠼ + a b c
        assert_eq!(t("123"), "⠼⠁⠃⠉");
    }

    #[test]
    fn number_decimal() {
        // 3.14 = ⠼ + c + . + a + d（小数点で数モード継続）
        assert_eq!(t("3.14"), "⠼⠉⠲⠁⠙");
    }

    // --- 約物・インデックス ---

    #[test]
    fn punctuation_period() {
        // "go." = go(⠛) + .(⠲)
        assert_eq!(t("go."), "⠛⠲");
    }

    #[test]
    fn index_contraction_maps_to_cell() {
        // "this" → ⠹。4文字とも点字位置0を指す。
        let r = tr().translate("this");
        assert_eq!(r.braille_text(), "⠹");
        assert_eq!(r.src_to_braille(), &[0, 0, 0, 0]);
    }

    #[test]
    fn index_capital_belongs_to_first_char() {
        // "This" → ⠠⠹。T は大文字符ごと位置0、his は ⠹ の位置1。
        let r = tr().translate("This");
        assert_eq!(r.braille_text(), "⠠⠹");
        assert_eq!(r.src_to_braille(), &[0, 1, 1, 1]);
    }

    #[test]
    fn index_sentence() {
        // "go" = ⠛（wordsign）, " " = ⠀, "on" = ⠕⠝
        let r = tr().translate("go on");
        assert_eq!(r.braille_text(), "⠛⠀⠕⠝");
        // g,o→0（wordsign セル）, space→1, o→2, n→3
        assert_eq!(r.src_to_braille(), &[0, 0, 1, 2, 3]);
    }
}
