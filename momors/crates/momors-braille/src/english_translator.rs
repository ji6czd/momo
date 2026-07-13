//! UEB (Unified English Braille) の英語点字変換。
//!
//! 変換規則は [`Table`] が持ち、縮約（`Table::contractions`）を含む grade 2 テーブル
//! （`english_ueb_grade2.toml`）なら縮約あり、縮約を持たない grade 1 テーブル
//! （`english_ueb_grade1.toml`）なら無縮約になる。適用ロジック（セル数最小化・位置制約・
//! standing alone・grade 1 モード）はこのモジュールが持つ。shortform の allowlist は
//! テーブルの `[shortforms]` セクション（`Table::shortform_words`）。
//!
//! Phase 1〜9 を実装（縮約体系・約物・記号・語境界・grade 1 モード・大文字モード・優先順位）。
//! 仕様: `docs/ueb-grade2-*.md`。規範は The Rules of Unified English Braille 2024 (ICEB)。
//! 規範の Appendix 2「Word List」1,028語を回帰コーパスにしている（`tests/ueb_appendix2.rs`。
//! 98.1% 一致。未対応語は `KNOWN_FAIL`）。
//!
//! この変換器は**スタンドアロン**（入出力は英語→UEB点字に閉じる）。日本語の状態
//! （外字符・数符・prev_class）を一切持ち込まない。日本語中の英語は外国語引用符
//! `⠦…⠴` で囲まれた「日本語が入らない島」なので、その中身をこの変換器に渡せば同じ
//! エンジンで賄える（島の検出・引用符の出力は日本語側の別レイヤーの仕事）。
//!
//! ## 適用ロジック（コードが持つ usage）
//!
//! - **セル数最小化と優先順位（§10.10）**: 語を DP で分割し、総セル数が最小になる縮約の組を
//!   選ぶ。同数なら種類の優先順位（be/con/dis > strong > lower > 頭字符・尾字符。
//!   [`Self::preference`]）。`bear` は `⠃⠂⠗`（ea）ではなく `⠃⠑⠜`（ar）。
//! - **位置プリミティブ [`Position`]**: 縮約は複数の位置（always / noninitial / initial /
//!   medial / wordsign）を持ち、いずれかが語中の現在位置で満たされれば適用。
//!   `wordsign` は単独で1語のとき、`initial` は語頭＋後続文字、`medial` は両側が文字、など。
//! - **standing alone（§2.6）**: 空白・ハイフン・ダッシュに挟まれること。決められた約物
//!   （開き括弧・引用符・アポストロフィ、コンマ・終止符など）の介在だけが許される。
//!   スラッシュ・数字・`@` は語境界にならない（`could/should` は縮約されない）。
//!   語中アポストロフィの語は並び全体で1語（`it's` = `⠭⠄⠎`）。
//! - **wordsign / shortform は standing alone が要る**（§10.9.1/§10.9.2）。10語だけは
//!   リストに無い語の中でも使う（`free` / `not_before_vowel`。§10.9.3。`Braillette` = `⠠⠃⠗⠇⠞⠞⠑`）。
//! - **誤読を防ぐ grade 1（§10.9.8/§10.9.9）**: セル列を shortform として読み戻した語を
//!   読み手が適用するなら、grade 1 語符 `⠰⠰` で語全体を無縮約に（`Dobrljin` `UNSD`）、
//!   語頭なら記号符 `⠰`（`Blvd`）。
//! - **区切り（§10.4/§10.6/§10.7/§10.11）**: テーブルの `[divisions]`（`mis|hap`）に載る語では、
//!   形態素・音節の切れ目を跨ぐ縮約を使わない。
//! - **下方 wordsign の制限（§10.5）**: `be/were/his/was` は下方約物（ハイフン・ダッシュを含む）に
//!   接すると使わない（`avoid_lower_punct`）。`in` は並びに上方の点があるときだけ（`needs_upper_dot`）。
//!   `enough` には制限がない。
//! - **§10.4.2**: groupsign が語全体に一致して wordsign の語と読まれるときは文字で書く（`sh` = `⠎⠓`）。
//! - **grade 1 記号符（§5.7）**: 単独語が縮約として読めるなら前置（`b` = `⠰⠃`、`ab` = `⠰⠁⠃`）。
//!   数値モード中の a–j にも前置（`1a` = `⠼⠁⠰⠁`。数値モードは `.` `,` を跨いで続く）。
//! - **大文字の3モード（§8）**: 大文字符 `⠠`（1文字）／大文字語符 `⠠⠠`（大文字の連なり）／
//!   大文字句符 `⠠⠠⠠`（3つ以上の並び）。語符・句符は**大文字終止符 `⠠⠄`** で閉じる
//!   （`ABCs` = `⠠⠠⠁⠃⠉⠠⠄⠎`）。指示符が入る位置を縮約は跨げない（`case_boundaries`）。
//! - **引用符**: `[quotes]` の開き／閉じを文脈で選ぶ（直前が空白・行頭・開き括弧なら開き）。
//!
//! ## 既知の割り切り（grade 2 の外）
//!
//! - 区切りはデータ（`[divisions]`）に載っている語だけ。載っていない語は素朴に縮約する。
//! - セル数同数時の tie-break は綴りの長さ（規範の「発音をよく表す方」は綴りから決まらない）。
//! - 大文字語符と単独の大文字符の選択は機械的に決める（§8.8 が「指針」と呼ぶ裁量。
//!   `TVOntario` は `⠠⠠⠞⠧⠕⠠⠄⠝⠞⠜⠊⠕`。出典は `O` を Ontario の頭文字と見て `⠠⠠⠞⠧⠠⠕…`）。
//! - typeform 符（イタリック・太字・下線）、grade 1 語符・句符（`⠰⠰` / `⠰⠰⠰`）は未対応。

use crate::table::{Contraction, FreeScope, Position, Table};
use crate::{Error, Result, UnknownCharPolicy};
use std::collections::{HashMap, HashSet};
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

/// 語を訳すときに効いているモード（§5 数値モード / §8 大文字句）。
#[derive(Debug, Clone, Copy)]
struct Modes {
    /// 数値モードの直後か（続く a–j に grade 1 記号符が要る）。
    numeric: bool,
    /// 大文字句の内側か（語ごとの大文字符は打たない）。
    passage: bool,
    /// この語で大文字句が始まるか（大文字句符を打つ）。
    passage_start: bool,
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
    /// 大文字句符（3つ以上の並びが大文字）と大文字終止符。
    cap_passage: String,
    cap_terminator: String,
    number: String,
    /// grade 1 記号符（数符直後の a–j、および縮約と読めてしまう単独語に前置する）。
    grade1: String,
    /// grade 1 語符（その語を丸ごと無縮約にする。§10.9.9）。
    grade1_word: String,
    /// 引用符。文字 → (開き, 閉じ)。開き／閉じは文脈で決める。
    quotes: HashMap<char, (String, String)>,
    /// 「語全体として使えるセル列」→ その綴り。単独語がこのセル列になるとき、綴りが
    /// 違えば縮約として読まれてしまうので grade 1 記号符を前置する（`b` → `⠰⠃`、
    /// `ab` → `⠰⠁⠃`。Rules of UEB §5.7 / §10.9.7）。
    whole_word_cells: HashMap<String, HashSet<String>>,
    /// wordsign のセル列 → その綴り。groupsign が語全体に一致すると「別の語」として
    /// 読まれてしまうので使わない（§10.4.2。`sh` = `⠎⠓`。`⠩` は「shall」と読まれる）。
    wordsign_cells: HashMap<String, HashSet<String>>,
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
        let cap_passage = table.capital_passage.clone();
        let cap_terminator = table.capital_terminator.clone();
        let number = table.flag_digit.entry_prefix.clone();
        let grade1 = table.grade1_indicator.clone();
        let grade1_word = table.grade1_word.clone();
        let quotes = table
            .quotes
            .iter()
            .filter_map(|(k, (o, c))| single_char(k).map(|ch| (ch, (o.clone(), c.clone()))))
            .collect();
        // 語全体として使える縮約（wordsign 役、および always で語全体にも一致するもの）を
        // セル列で引けるようにする。単独語の grade 1 記号符の判定に使う。
        let mut whole_word_cells: HashMap<String, HashSet<String>> = HashMap::new();
        for (spelling, con) in &table.contractions {
            if con
                .positions
                .iter()
                .any(|p| matches!(p, Position::Wordsign | Position::Always))
            {
                whole_word_cells
                    .entry(con.cell.clone())
                    .or_default()
                    .insert(spelling.clone());
            }
        }
        let mut wordsign_cells: HashMap<String, HashSet<String>> = HashMap::new();
        for (spelling, con) in &table.contractions {
            if con.positions.contains(&Position::Wordsign) {
                wordsign_cells
                    .entry(con.cell.clone())
                    .or_default()
                    .insert(spelling.clone());
            }
        }
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
            cap_passage,
            cap_terminator,
            number,
            grade1,
            grade1_word,
            quotes,
            whole_word_cells,
            wordsign_cells,
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

        // 数値モード（§6）。数符で始まり、数字・小数点・桁区切り（`.` `,`）の間は続く。
        // それ以外の記号・空白で終わる。数値モード中の a–j は数字と同じセルなので、
        // 続く語には grade 1 記号符が要る（`take2.com` = `⠞⠁⠅⠑⠼⠃⠲⠰⠉⠕⠍`）。
        let mut numeric = false;

        // 大文字句（§8.5）。3つ以上の並びが大文字なら、句符 ⠠⠠⠠ で開いて終止符 ⠠⠄ で閉じる。
        let passages = self.capital_passages(&chars);
        let mut passage: Option<&(usize, usize)> = None;

        let mut i = 0;
        while i < n {
            // 句の終わり（終止符を入れる位置）に来たら大文字モードを閉じる。
            if let Some((_, term)) = passage
                && i == *term
            {
                braille.push_str(&self.cap_terminator);
                passage = None;
            }
            if let Some(p) = passages.iter().find(|(open, _)| *open == i) {
                passage = Some(p);
            }
            let c = chars[i];
            if c.is_ascii_alphabetic() {
                // 語（連続する英字）をまとめて処理する。
                let mut j = i;
                while j < n && chars[j].is_ascii_alphabetic() {
                    j += 1;
                }
                let mode = Modes {
                    numeric,
                    passage: passage.is_some(),
                    passage_start: passage.is_some_and(|(open, _)| *open == i),
                };
                self.translate_word(&chars, i, j, mode, &mut braille, &mut src_to_braille);
                numeric = false;
                i = j;
            } else if c.is_ascii_digit() {
                i = self.translate_number(&chars, i, &mut braille, &mut src_to_braille);
                numeric = true;
            } else if numeric && (c == '.' || c == ',') {
                // 数値モードは小数点・桁区切りを跨いで続く（§6.2.1）。
                src_to_braille[i] = braille.chars().count();
                if let Some(cell) = self.punctuation.get(&c) {
                    braille.push_str(cell);
                }
                i += 1;
            } else if c == ' ' || c == '\t' {
                numeric = false;
                src_to_braille[i] = braille.chars().count();
                braille.push(BRAILLE_SPACE);
                i += 1;
            } else if let Some((open, close)) = self.quotes.get(&c) {
                numeric = false;
                // 引用符は開き／閉じで形が違う。直前が空白・行頭・開き括弧なら開き、でなければ閉じ。
                let opening = i == 0
                    || matches!(
                        chars[i - 1],
                        ' ' | '\t' | '(' | '[' | '{' | '“' | '‘' | '-' | '–' | '—' | '/'
                    );
                src_to_braille[i] = braille.chars().count();
                braille.push_str(if opening { open } else { close });
                i += 1;
            } else if let Some(cell) = self.punctuation.get(&c) {
                numeric = false;
                src_to_braille[i] = braille.chars().count();
                braille.push_str(cell);
                i += 1;
            } else {
                numeric = false;
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
        // 句がテキスト末尾で終わるとき（終止符の位置が n）。
        if passage.is_some_and(|(_, term)| *term == n) {
            braille.push_str(&self.cap_terminator);
        }

        EnglishResult {
            braille_text: braille,
            src_to_braille,
        }
    }

    /// 大文字句（§8.5）を見つける。返すのは (句符を置く位置, 終止符を置く位置) の並び。
    ///
    /// 「句」は**3つ以上の symbols-sequence**（空白で区切られた並び）で、すべてが
    /// 「大文字だけ（小文字を含まない）」か「英字を含まない」もの。先頭と末尾は大文字の並び。
    /// 終止符は最後の並びの直後に置くが（`PAINT!` → `⠏⠁⠔⠞⠖⠠⠄`）、句より前に開いた
    /// 閉じ括弧・閉じ引用符の**内側**に置く（§8.6.2 の入れ子。`"I WILL NOT!"` → `⠝⠖⠠⠄⠴`）。
    fn capital_passages(&self, chars: &[char]) -> Vec<(usize, usize)> {
        if self.cap_passage.is_empty() || self.cap_terminator.is_empty() {
            return Vec::new();
        }
        // 空白で区切られた並びに分ける。
        let mut tokens: Vec<(usize, usize)> = Vec::new();
        let mut k = 0;
        while k < chars.len() {
            if chars[k].is_whitespace() {
                k += 1;
                continue;
            }
            let s = k;
            while k < chars.len() && !chars[k].is_whitespace() {
                k += 1;
            }
            tokens.push((s, k));
        }

        // 大文字だけの並び / 英字を含まない並び
        let all_caps = |&(s, e): &(usize, usize)| {
            chars[s..e].iter().any(|c| c.is_ascii_uppercase())
                && !chars[s..e].iter().any(|c| c.is_ascii_lowercase())
        };
        let no_letter = |&(s, e): &(usize, usize)| !chars[s..e].iter().any(|c| c.is_ascii_alphabetic());

        let mut out = Vec::new();
        let mut t = 0;
        while t < tokens.len() {
            if !all_caps(&tokens[t]) {
                t += 1;
                continue;
            }
            // 大文字（または英字なし）の並びが続く限り伸ばし、最後の「大文字の並び」で切る。
            let mut last_caps = t;
            let mut u = t;
            while u < tokens.len() && (all_caps(&tokens[u]) || no_letter(&tokens[u])) {
                if all_caps(&tokens[u]) {
                    last_caps = u;
                }
                u += 1;
            }
            let len = last_caps - t + 1;
            if len >= 3 {
                let (first_s, first_e) = tokens[t];
                let (_, last_e) = tokens[last_caps];
                // 句符は最初の大文字の直前（開き括弧・引用符の後）。
                let open = (first_s..first_e)
                    .find(|&x| chars[x].is_ascii_uppercase())
                    .expect("大文字の並びには大文字がある");
                out.push((open, self.passage_terminator_pos(chars, open, last_e)));
            }
            t = u.max(t + 1);
        }
        out
    }

    /// 大文字句の終止符を置く位置。並びの末尾から、**句より前に開かれた**閉じ括弧・
    /// 閉じ引用符の分だけ手前に戻す（入れ子で閉じる。§8.6.2）。
    fn passage_terminator_pos(&self, chars: &[char], open: usize, end: usize) -> usize {
        let matching_open = |c: char| match c {
            ')' => Some('('),
            ']' => Some('['),
            '}' => Some('{'),
            '”' => Some('“'),
            '’' => Some('‘'),
            '"' => Some('"'),
            _ => None,
        };
        let mut pos = end;
        while pos > open {
            let Some(opener) = matching_open(chars[pos - 1]) else {
                break;
            };
            // その開き括弧・引用符が句の内側にあるなら、閉じてから終止符（入れ子）。
            if chars[open..pos - 1].contains(&opener) {
                break;
            }
            pos -= 1;
        }
        pos
    }

    /// 語の中で、大文字の指示符が入るために縮約が跨げなくなる位置（絶対位置）。
    ///
    /// - **小文字→大文字**: 大文字符・語符がその大文字の直前に入る（`McDonald` の `cD`）。
    /// - **大文字→小文字**: その大文字の連なりが2文字以上なら大文字終止符が入る
    ///   （`XXIInd` の `In`）。1文字だけなら大文字符が縮約の頭に付くので跨いでよい
    ///   （`StepChildren` の `St` = `⠠⠌`）。
    fn case_boundaries(chars: &[char], start: usize, end: usize) -> Vec<usize> {
        let mut out = Vec::new();
        for i in start + 1..end {
            let prev = chars[i - 1];
            let cur = chars[i];
            if !prev.is_ascii_uppercase() && cur.is_ascii_uppercase() {
                out.push(i);
            } else if prev.is_ascii_uppercase() && !cur.is_ascii_uppercase() {
                let run = chars[start..i]
                    .iter()
                    .rev()
                    .take_while(|c| c.is_ascii_uppercase())
                    .count();
                if run >= 2 {
                    out.push(i);
                }
            }
        }
        out
    }

    /// 語 `chars[start..end]`（すべて英字）を変換して `braille` に追記する。
    ///
    /// 大文字は3つのモードを使い分ける（§8）:
    /// **大文字符 `⠠`**（次の1文字）／**大文字語符 `⠠⠠`**（大文字の連なり。空白・非英字・
    /// 終止符まで）／**大文字句符 `⠠⠠⠠`**（3つ以上の並び。`passage` が真なら既に効いている）。
    /// 語符の途中で小文字に戻るときは**大文字終止符 `⠠⠄`** で切る（`ABCs` = `⠠⠠⠁⠃⠉⠠⠄⠎`）。
    fn translate_word(
        &self,
        chars: &[char],
        start: usize,
        end: usize,
        mode: Modes,
        braille: &mut String,
        src_to_braille: &mut [usize],
    ) {
        let Modes {
            numeric,
            passage,
            passage_start,
        } = mode;
        let standing_alone = Self::standing_alone(chars, start, end);
        let mut boundaries = self.division_boundaries(chars, start, end);
        // 大文字符・終止符が入る位置は縮約が跨げない（`XXIInd` の `In` は縮約しない）。
        // 区切りと同じ仕組みに載せる。句符が効いている間は指示符が入らないので不要。
        if !passage {
            boundaries.extend(Self::case_boundaries(chars, start, end));
            boundaries.sort_unstable();
        }
        let mut segments = self.segment(chars, start, end, standing_alone, &boundaries);

        // §10.9.8/§10.9.9: 語の中に shortform と読めてしまう並びがあるとき。
        // 語頭なら grade 1 記号符（`Blvd` = `⠰⠠⠃⠇⠧⠙`）、語中なら grade 1 語符で語全体を
        // 無縮約にする（`Dobrljin` = `⠰⠰⠠⠙⠕⠃⠗⠇⠚⠊⠝`。他の縮約も一切使わない）。
        let hazard = if standing_alone {
            self.misread_as_shortform(chars, start, end, &segments)
        } else {
            None
        };
        let grade1_word_needed = hazard.is_some_and(|li| li > start);
        if grade1_word_needed {
            segments = self.uncontracted(chars, start, end);
        }

        // 語頭の指示符。語頭の文字にこの位置を帰属させる。
        let word_head_pos = braille.chars().count();
        if grade1_word_needed {
            braille.push_str(&self.grade1_word);
        } else if hazard.is_some()
            || (standing_alone && self.reads_as_contraction(chars, start, end, &segments))
        {
            // 単独語のセル列が縮約として読めてしまうなら grade 1 記号符を前置する
            // （`b` は「but」、`ab` は「about」と読まれる）。大文字符より前に置く（`⠰⠠⠭`）。
            braille.push_str(&self.grade1);
        }
        if passage_start {
            braille.push_str(&self.cap_passage);
        }

        // 大文字語モードに入っているか（句符が効いている間は語符を使わない）。
        let mut in_word_mode = false;
        let mut i = start;
        for (idx, (len, cell)) in segments.iter().enumerate() {
            let upper = chars[i].is_ascii_uppercase();
            let mut caps_cells = String::new();
            if !passage {
                if in_word_mode && !upper {
                    // 大文字の連なりが途切れた。ここで大文字モードを切る（WALKing）。
                    caps_cells.push_str(&self.cap_terminator);
                    in_word_mode = false;
                } else if !in_word_mode && upper {
                    // 大文字が2文字以上続くなら語符、1文字だけなら大文字符（TVOntario）。
                    let run = chars[i..end]
                        .iter()
                        .take_while(|c| c.is_ascii_uppercase())
                        .count();
                    if run >= 2 {
                        caps_cells.push_str(&self.cap_word);
                        in_word_mode = true;
                    } else {
                        caps_cells.push_str(&self.cap);
                    }
                }
            }
            // grade 1 記号符: 数値モード中の a–j は数字と同じセルなので前置して弁別する
            // （"1a" = ⠼⠁⠰⠁）。大文字符が挟まるなら曖昧性は消えるので不要（"4A" = ⠼⠙⠠⠁）。
            if idx == 0
                && numeric
                && caps_cells.is_empty()
                && Self::digit_ambiguous(*len, chars[i])
            {
                braille.push_str(&self.grade1);
            }
            braille.push_str(&caps_cells);
            let cell_pos = braille.chars().count();
            braille.push_str(cell);
            for k in i..i + len {
                // 語頭の文字だけは語頭の指示符ごと（word_head_pos）、それ以外はセル位置を指す。
                src_to_braille[k] = if k == start { word_head_pos } else { cell_pos };
            }
            i += len;
        }
    }

    /// 語 `chars[start..end]` を「総セル数が最小」になるように分割する（§10.10）。
    ///
    /// **第1基準はセル数**（§10.10.2「最も字数を節約する縮約を優先」）。綴りの長さとセル数は
    /// 比例しない（頭字符は2セルで4〜8文字、下方 groupsign は1セルで2〜3文字）ので、綴りの
    /// 最長一致（貪欲）では最適にならない。適用可能性は位置と語だけで決まり他の分割に依存
    /// しないので、後ろ向きの DP が最適解を返す。
    ///
    /// **同数のときは種類の優先順位**（[`Self::preference`]。§10.10.4/5/7）。`bear` は
    /// `⠃⠂⠗`（lower の `ea`）ではなく `⠃⠑⠜`（strong の `ar`）。それも同じなら綴りの長い方。
    fn segment(
        &self,
        chars: &[char],
        start: usize,
        end: usize,
        standing_alone: bool,
        boundaries: &[usize],
    ) -> Vec<(usize, String)> {
        let wlen = end - start;
        // best[k] = (最小セル数, 優先順位の合計, その位置で選ぶ長さ, そのセル)
        let mut best: Vec<(usize, u32, usize, String)> =
            vec![(0, 0, 0, String::new()); wlen + 1];
        for k in (0..wlen).rev() {
            let i = start + k;
            let maxl = self.max_contraction_len.min(wlen - k);
            let mut chosen: Option<(usize, u32, usize, String)> = None;
            for len in 1..=maxl.max(1) {
                let mut rank = 0; // 素の文字は優先順位の対象外
                let cell = if len == 1 {
                    // 1文字（grade 1）。テーブルに無い文字はスペースへ倒す。
                    self.letters
                        .get(&chars[i].to_ascii_lowercase())
                        .cloned()
                        .unwrap_or_else(|| BRAILLE_SPACE.to_string())
                } else {
                    let sub: String = chars[i..i + len]
                        .iter()
                        .flat_map(|c| c.to_lowercase())
                        .collect();
                    match self.table.contractions.get(&sub) {
                        Some(con)
                            if self.applicable(
                                &sub,
                                con,
                                chars,
                                i,
                                len,
                                start,
                                end,
                                standing_alone,
                                boundaries,
                            ) =>
                        {
                            rank = Self::preference(con);
                            con.cell.clone()
                        }
                        _ => continue,
                    }
                };
                let total = cell.chars().count() + best[k + len].0;
                let total_rank = rank + best[k + len].1;
                let better = match &chosen {
                    None => true,
                    // セル数 → 種類の優先順位 → 綴りの長さ、の順で決める。
                    Some((best_total, best_rank, best_len, _)) => {
                        (total, total_rank) < (*best_total, *best_rank)
                            || ((total, total_rank) == (*best_total, *best_rank) && len > *best_len)
                    }
                };
                if better {
                    chosen = Some((total, total_rank, len, cell));
                }
            }
            // len==1 は必ず候補になるので unwrap して問題ない。
            best[k] = chosen.expect("1文字の候補が必ずある");
        }

        let mut out = Vec::new();
        let mut k = 0;
        while k < wlen {
            let (_, _, len, cell) = &best[k];
            out.push((*len, cell.clone()));
            k += len;
        }
        out
    }

    /// 縮約の優先順位（§10.10.4/5/7）。小さいほど優先。セル数が同じときの決着に使う。
    ///
    /// - `1` 語頭の下方 groupsign（be / con / dis）。第一音節なら他に優先する（§10.10.4）。
    /// - `2` strong groupsign / strong contraction。下方 groupsign に優先する（§10.10.5）。
    /// - `3` それ以外の下方 groupsign（en / in / ea / bb / cc / ff / gg）。
    /// - `8` 2セルのもの（頭字符・尾字符・shortform）。strong/lower に劣後する（§10.10.7）。
    ///   strong 2つ（2+2）より重くしてある。`adhered` は `⠁⠙⠐⠓⠙`（here）ではなく
    ///   `⠁⠙⠓⠻⠫`（er + ed）。
    fn preference(con: &Contraction) -> u32 {
        if con.cell.chars().count() > 1 {
            8
        } else if con.lower && con.positions.contains(&Position::Initial) {
            1
        } else if !con.lower {
            2
        } else {
            3
        }
    }

    /// 位置 `i`・長さ `len` の縮約が適用可能か。
    #[allow(clippy::too_many_arguments)]
    fn applicable(
        &self,
        spelling: &str,
        con: &Contraction,
        chars: &[char],
        i: usize,
        len: usize,
        word_start: usize,
        word_end: usize,
        standing_alone: bool,
        boundaries: &[usize],
    ) -> bool {
        // Phase 5: 語の区切り（形態素・音節）を跨ぐ縮約は使わない（mishap の sh）。境界にちょうど接する
        // 縮約は使える（with|hold の with は可）。
        if boundaries.iter().any(|&b| i < b && b < i + len) {
            return false;
        }

        // §10.7.4: 直前の文字で使えなくなる縮約（`ever` は `e` / `i` の後では使わない）。
        if i > word_start && con.not_after.contains(chars[i - 1].to_ascii_lowercase()) {
            return false;
        }

        let whole_word = i == word_start && i + len == word_end;

        // §10.4.2: groupsign が語全体に一致すると、その語ではなく wordsign の語として
        // 読まれてしまう（`sh` は `⠩`＝「shall」に見える）。そういうときは縮約しない。
        if whole_word
            && standing_alone
            && self
                .wordsign_cells
                .get(&con.cell)
                .is_some_and(|words| !words.contains(spelling))
        {
            return false;
        }

        // §10.5.3: `in` は「その並びに上方の点を持つ記号がある」ときだけ使う。
        // `in-depth`（⠙ に上方の点）は可、`in.`（⠔⠲ はどちらも下方）は不可。
        if con.needs_upper_dot && whole_word && !self.sequence_has_upper_dot(chars, word_start, word_end)
        {
            return false;
        }

        // §10.5.1: be / were / his / was は下方約物（ハイフン・ダッシュ・引用符を含む）に
        // 接するときは使わない（`would-be` = `⠺⠙⠤⠃⠑`、`be,` = `⠃⠑⠂`）。
        // enough（§10.5.2）と in（§10.5.3）にはこの制限がない（`had-enough` = `⠸⠓⠤⠢`）。
        if con.avoid_lower_punct && whole_word && con.positions.contains(&Position::Wordsign) {
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
            // 単独で1語（standing alone）のとき。加えて shortform は、語全体が Appendix 1 の
            // Shortforms List に載っていれば語中のどこでも使える（§10.9.2）。どちらの道でも
            // **語が standing alone であること**が要る（§10.9.1/§10.9.2）。だから
            // `could/should` はスラッシュで単独でなくなり、縮約されない。
            Position::Wordsign => {
                standing_alone
                    && (whole_word
                        || (con.shortform
                            && (self.word_allows_shortform(chars, word_start, word_end)
                                || Self::free_shortform_ok(
                                    con, chars, i, len, word_start, word_end,
                                ))))
            }
        })
    }

    /// §10.9.3: Appendix 1 に無い長い語の中でも使える10の shortform か。
    ///
    /// `braille` / `great` はどこに現れても、`children` は母音・`y` が続かなければどこでも、
    /// 残る7つ（blind/first/friend/good/letter/little/quick）は**語頭**で母音・`y` が
    /// 続かないときだけ。`Braillette` = `⠠⠃⠗⠇⠞⠞⠑`、`Blindoc` = `⠠⠃⠇⠔⠙⠕⠉`（母音で止まる）。
    fn free_shortform_ok(
        con: &Contraction,
        chars: &[char],
        i: usize,
        len: usize,
        word_start: usize,
        word_end: usize,
    ) -> bool {
        let Some(free) = con.free else {
            return false;
        };
        if free == FreeScope::Initial && i != word_start {
            return false;
        }
        if con.not_before_vowel
            && i + len < word_end
            && matches!(
                chars[i + len].to_ascii_lowercase(),
                'a' | 'e' | 'i' | 'o' | 'u' | 'y'
            )
        {
            return false;
        }
        true
    }

    /// 語 `chars[word_start..word_end]` の区切り（絶対位置・昇順）。リストに無ければ空。
    ///
    /// 語そのもので引き、無ければ屈折語尾（s / es / ed / ing / ly）を落とした語幹でも引く
    /// （`mishaps` → `mishap`、`mishandling` → `mishandle`）。境界は語幹側にあるので
    /// オフセットはそのまま使える。
    fn division_boundaries(
        &self,
        chars: &[char],
        word_start: usize,
        word_end: usize,
    ) -> Vec<usize> {
        if self.table.division_boundaries.is_empty() {
            return Vec::new();
        }
        let word: String = chars[word_start..word_end]
            .iter()
            .flat_map(|c| c.to_lowercase())
            .collect();
        let mut candidates = vec![word.clone()];
        for suffix in ["s", "es", "ed", "ing", "ly"] {
            if let Some(stem) = word.strip_suffix(suffix)
                && !stem.is_empty()
            {
                candidates.push(stem.to_string());
                // 語尾の e が落ちる屈折（mishandle → mishandling）
                candidates.push(format!("{stem}e"));
            }
        }
        candidates
            .iter()
            .find_map(|w| self.table.division_boundaries.get(w))
            .map(|cuts| cuts.iter().map(|b| word_start + b).collect())
            .unwrap_or_default()
    }

    /// 語 `chars[start..end]` が「単独で1語（standing alone）」か（Rules of UEB §2.6）。
    ///
    /// 規範の定義は「空白・ハイフン・ダッシュに前後を挟まれること」で、約物一般ではない。
    /// ただし決められた約物の介在は許される（§2.6.2/§2.6.3）:
    /// 前は開き括弧・開き引用符・アポストロフィ、後ろはコンマ・終止符・省略記号・
    /// 閉じ括弧・閉じ引用符・アポストロフィなど。**スラッシュや数字は許されない**ので
    /// `ab/cd` や `4th` は単独ではない。
    ///
    /// 内部アポストロフィを持つ語（`it's` / `p's`）は語全体で1つの並びとして扱い（§2.6.4）、
    /// **先頭の文字列だけ**が単独になる（`it's` = `⠭⠄⠎`。末尾の `s` は縮約されない）。
    fn standing_alone(chars: &[char], start: usize, end: usize) -> bool {
        /// 語境界そのもの（空白・ハイフン・ダッシュ・行頭行末）。
        const BOUNDARY: &[char] = &[' ', '\t', '\n', '\r', '-', '–', '—'];
        /// 語の前に介在してよい約物（§2.6.2）。
        const BEFORE: &[char] = &['(', '[', '{', '"', '“', '‘', '\''];
        /// 語の後に介在してよい約物（§2.6.3）。
        const AFTER: &[char] = &[
            ',', ';', ':', '.', '…', '!', '?', ')', ']', '}', '"', '”', '’', '\'',
        ];

        let n = chars.len();
        let mut i = start;
        while i > 0 && BEFORE.contains(&chars[i - 1]) {
            i -= 1;
        }
        let left_ok = i == 0 || BOUNDARY.contains(&chars[i - 1]);

        let mut j = end;
        // 内部アポストロフィは、続くのが決まった語尾（'s 't 'd 'll 're 've）のときだけ
        // 語の一部として読み飛ばす（§10.1.2 / §10.2.2）。`it's` `wouldn't've` は語の一部だが、
        // `t'do` `more'n` `you'm` はそうではない（＝先頭部分は単独でなく wordsign を使わない）。
        while j + 1 < n && chars[j] == '\'' && chars[j + 1].is_ascii_alphabetic() {
            let mut k = j + 1;
            while k < n && chars[k].is_ascii_alphabetic() {
                k += 1;
            }
            let tail: String = chars[j + 1..k]
                .iter()
                .flat_map(|c| c.to_lowercase())
                .collect();
            if !matches!(tail.as_str(), "s" | "t" | "d" | "ll" | "re" | "ve") {
                break;
            }
            j = k;
        }
        while j < n && AFTER.contains(&chars[j]) {
            j += 1;
        }
        let right_ok = j == n || BOUNDARY.contains(&chars[j]);

        left_ok && right_ok
    }

    /// 語を1文字ずつ（無縮約で）分割する。grade 1 語符を打った語に使う。
    fn uncontracted(&self, chars: &[char], start: usize, end: usize) -> Vec<(usize, String)> {
        chars[start..end]
            .iter()
            .map(|c| {
                let cell = self
                    .letters
                    .get(&c.to_ascii_lowercase())
                    .cloned()
                    .unwrap_or_else(|| BRAILLE_SPACE.to_string());
                (1, cell)
            })
            .collect()
    }

    /// 出力セル列の中に「読み手が shortform として読んでしまう並び」があるか（§10.9.8/§10.9.9）。
    /// 返すのはその並びが始まる文字位置。
    ///
    /// **読み手の規則をそのまま逆に回す**。あるセル列を shortform として読み戻し、できあがる語が
    ///
    /// - §10.9.3 の10語（`free`）… リストに無い語の中でも読まれる（`Dobrljin` の `brl` → braille）
    /// - それ以外の shortform … 読み戻した語が Appendix 1 のリストにあるときだけ読まれる
    ///   （`UNSD` の `sd` は「unsaid」になり、それはリストにある → 誤読される）
    ///
    /// のいずれかなら危険。読み手が適用しない位置（`table` の `bl` は語頭でないので blind に
    /// ならない。`wisdom` の `sd` は「wisaidom」になりリストに無い）は危険ではない。
    fn misread_as_shortform(
        &self,
        chars: &[char],
        start: usize,
        end: usize,
        segments: &[(usize, String)],
    ) -> Option<usize> {
        if self.grade1_word.is_empty() {
            return None;
        }
        // セグメントの開始文字位置
        let mut letter_at = Vec::with_capacity(segments.len());
        let mut li = start;
        for (len, _) in segments {
            letter_at.push(li);
            li += len;
        }

        let word: String = chars[start..end]
            .iter()
            .flat_map(|c| c.to_lowercase())
            .collect();

        for (sidx, (seg_len, seg_cell)) in segments.iter().enumerate() {
            let li = letter_at[sidx];
            for (spelling, con) in &self.table.contractions {
                if !con.shortform {
                    continue;
                }
                // 我々がその shortform を実際に使ったセグメントなら、それは正しい読み。
                if seg_cell == &con.cell && *seg_len == spelling.chars().count() {
                    continue;
                }
                // セグメント境界にきっちり収まる形で縮約のセル列が現れるか
                let want = con.cell.chars().count();
                let (mut cells, mut letters) = (0, 0);
                for (len, cell) in &segments[sidx..] {
                    cells += cell.chars().count();
                    letters += len;
                    if cells >= want {
                        break;
                    }
                }
                if cells != want {
                    continue;
                }
                let got: String = segments[sidx..]
                    .iter()
                    .map(|(_, c)| c.as_str())
                    .collect::<String>()
                    .chars()
                    .take(want)
                    .collect();
                if got != con.cell {
                    continue;
                }

                // 読み手がそこで shortform を適用するか。
                let applies = match con.free {
                    // §10.9.3 の10語: 位置と母音の制限だけ（リストに無い語でも読まれる）
                    Some(free) => {
                        let initial_ok = free != FreeScope::Initial || li == start;
                        let vowel_ok = !con.not_before_vowel || {
                            let next = li + letters;
                            next >= end
                                || !matches!(
                                    chars[next].to_ascii_lowercase(),
                                    'a' | 'e' | 'i' | 'o' | 'u' | 'y'
                                )
                        };
                        initial_ok && vowel_ok
                    }
                    // それ以外: セル列を綴りへ読み戻した語がリストにあるときだけ読まれる
                    None => {
                        let (a, b) = (li - start, li - start + letters);
                        let read_back = format!("{}{}{}", &word[..a], spelling, &word[b..]);
                        read_back != word
                            && (self.table.shortform_words.contains(&read_back)
                                || read_back
                                    .strip_suffix('s')
                                    .is_some_and(|w| self.table.shortform_words.contains(w)))
                    }
                };
                if applies {
                    return Some(li);
                }
            }
        }
        None
    }

    /// 単独語 `chars[word_start..word_end]` の出力セル列が、**別の語の縮約**として
    /// 読めてしまうか（Rules of UEB §5.7 / §10.9.7）。
    ///
    /// `b` は「but」、`ab` は「about」、`cd` は「could」と読まれるので grade 1 記号符が要る。
    /// `a` / `i` / `o` は縮約セルを持たないので自然に除外される。その語自身の縮約
    /// （`do` → `⠙`、`the` → `⠮`）は当然そのままでよい。
    fn reads_as_contraction(
        &self,
        chars: &[char],
        word_start: usize,
        word_end: usize,
        segments: &[(usize, String)],
    ) -> bool {
        if self.grade1.is_empty() {
            return false;
        }
        let cells: String = segments.iter().map(|(_, c)| c.as_str()).collect();
        let Some(spellings) = self.whole_word_cells.get(&cells) else {
            return false;
        };
        let word: String = chars[word_start..word_end]
            .iter()
            .flat_map(|c| c.to_lowercase())
            .collect();
        !spellings.contains(&word)
    }

    /// 数符の直後に置くと数字と読めてしまうセルか（1文字の a–j。数字 1–0 と同じセル）。
    fn digit_ambiguous(len: usize, c: char) -> bool {
        len == 1 && matches!(c.to_ascii_lowercase(), 'a'..='j')
    }

    /// 語 `chars[word_start..word_end]` を含む並びの中で shortform を使ってよいか（§10.9.2）。
    ///
    /// 判定は**アポストロフィで繋がった並び全体**で行う（§2.6.4）。Appendix 1 は
    /// `wouldn't` のようなアポストロフィ付きの語もリストに載せているので、`wouldn't` の
    /// `would`（並びは `wouldn't`、英字の連なりは `wouldn`）に shortform が使える。
    ///
    /// リストに載っている語、またはその語に `s` / `'s` を付けた語（`friend` → `friends`）。
    /// ただし `abouts` / `almosts` / `hims` の3語は例外で使わない（§10.9.5）。
    fn word_allows_shortform(&self, chars: &[char], word_start: usize, word_end: usize) -> bool {
        let word = Self::apostrophe_sequence(chars, word_start, word_end);
        if self.table.shortform_plural_exceptions.contains(&word) {
            return false;
        }
        if self.table.shortform_words.contains(&word) {
            return true;
        }
        // 末尾の "s" / "'s" を落とした語がリストにあれば可
        ["'s", "s"].iter().any(|suffix| {
            word.strip_suffix(suffix)
                .is_some_and(|stem| self.table.shortform_words.contains(stem))
        })
    }

    /// `chars[word_start..word_end]`（英字の連なり）を含む、アポストロフィで繋がった並び
    /// （小文字化・末尾のアポストロフィは落とす）。`it's` → `it's`、`friends'` → `friends`、
    /// `'twould` → `'twould`。
    fn apostrophe_sequence(chars: &[char], word_start: usize, word_end: usize) -> String {
        let is_part = |c: char| c.is_ascii_alphabetic() || c == '\'';
        let mut start = word_start;
        while start > 0 && is_part(chars[start - 1]) {
            start -= 1;
        }
        let mut end = word_end;
        while end < chars.len() && is_part(chars[end]) {
            end += 1;
        }
        let seq: String = chars[start..end]
            .iter()
            .flat_map(|c| c.to_lowercase())
            .collect();
        seq.trim_end_matches('\'').to_owned()
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

    /// 語を含む並び（空白で区切られた symbols-sequence）に、上方の点（dot 1/4）を持つ記号が
    /// あるか（§10.5.3）。語そのもの以外に記号が無ければ、比べる相手がいないので真とする
    /// （`not in here` の `in` は単独で使える）。
    fn sequence_has_upper_dot(&self, chars: &[char], word_start: usize, word_end: usize) -> bool {
        let n = chars.len();
        let mut s = word_start;
        while s > 0 && !chars[s - 1].is_whitespace() {
            s -= 1;
        }
        let mut e = word_end;
        while e < n && !chars[e].is_whitespace() {
            e += 1;
        }
        let mut others = (s..e).filter(|&k| k < word_start || k >= word_end).peekable();
        if others.peek().is_none() {
            return true;
        }
        others.any(|k| {
            let c = chars[k];
            c.is_ascii_alphanumeric() || (self.punctuation.contains_key(&c) && !self.is_lower_punct(c))
        })
    }

    /// 文字 `c` が下方約物（点字が dot 1/4 を含まない約物）か。§10.5.1 の接触判定に使う。
    /// ハイフン・ダッシュ（⠤ = 36）も下方約物として数える（規範が明示している）。
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
    // --- Phase 5: セル数最小化（規範「最も字数を節約する縮約を優先」）---

    #[test]
    fn cell_minimizing_beats_longest_spelling() {
        // 綴りの最長一致（貪欲）は語中の in(⠔) を先に取って f+r+l+in+e+s+s = 7セルにするが、
        // 尾字符 ness(⠰⠎) を使う分割の方が 6セルで少ない。DP はこちらを選ぶ。
        assert_eq!(t("friendliness"), "⠋⠗⠇⠊⠰⠎"); // fr(shortform) + l + i + ness
        assert_eq!(t("mustiness"), "⠍⠌⠊⠰⠎"); // m + st + i + ness
    }

    #[test]
    fn cell_minimizing_prefers_fewer_cells_over_more_contractions() {
        // shortform yr(2セル) が y+ou+r(3セル) に勝つ。縮約の個数ではなくセル数で選ぶ。
        assert_eq!(t("your"), "⠽⠗");
    }

    // --- Phase 5: 形態素境界を跨ぐ縮約の抑制（[morphemes]）---

    #[test]
    fn morpheme_boundary_blocks_contraction() {
        // mis|hap の sh は語構成の切れ目を跨ぐので使わない
        assert_eq!(t("mishap"), "⠍⠊⠎⠓⠁⠏");
        assert_eq!(t("hothouse"), "⠓⠕⠞⠓⠳⠎⠑"); // th を使わず ou は使う
        assert_eq!(t("coworker"), "⠉⠕⠐⠺⠻"); // ow を使わず work(頭字符) + er
    }

    #[test]
    fn morpheme_boundary_only_for_listed_words() {
        // リストに無い語は素朴に縮約する（mishmash は mis+hap ではない）
        assert_eq!(t("mishmash"), "⠍⠊⠩⠍⠁⠩");
    }

    #[test]
    fn morpheme_boundary_absorbs_inflections() {
        // 屈折語尾（s / ed / ing …）を落とした語幹でも境界を引く
        assert_eq!(t("mishaps"), "⠍⠊⠎⠓⠁⠏⠎");
        assert_eq!(t("mishandling"), "⠍⠊⠎⠓⠯⠇⠬"); // mishandle → m+i+s+h+and+l+ing
    }

    #[test]
    fn contraction_touching_boundary_is_allowed() {
        // 境界にちょうど接する縮約は使える（with|hold の with）
        assert_eq!(t("withhold"), "⠾⠓⠕⠇⠙");
    }

    // --- Phase 5: standing alone（数字は語境界ではない）---

    #[test]
    fn wordsign_not_used_next_to_digit() {
        // "4th" の th は単独で1語ではない（数字に接する）ので wordsign "this" にならない。
        // groupsign としての th は使う。
        assert_eq!(t("4th"), "⠼⠙⠹");
        assert_eq!(t("14th"), "⠼⠁⠙⠹");
        // 単独なら wordsign（同じセルだが意味は this）
        assert_eq!(t("this"), "⠹");
    }

    #[test]
    fn wordsign_used_next_to_punctuation() {
        // 約物は語境界。アポストロフィ・ハイフンに接していても単独扱い。
        assert_eq!(t("it's"), "⠭⠄⠎"); // it(wordsign) + ' + s
        assert_eq!(t("can-do"), "⠉⠤⠙"); // can(wordsign) + - + do(wordsign)
    }

    // --- Phase 5: grade 1 記号符（数符直後の a–j）---

    #[test]
    fn grade1_indicator_after_digit() {
        // a–j は数字 1–0 と同じセルなので、数字の直後では grade 1 記号符で弁別する
        assert_eq!(t("1a"), "⠼⠁⠰⠁");
        // a–j 以外は曖昧でないので不要
        assert_eq!(t("3rd"), "⠼⠉⠗⠙");
        // 大文字符が挟まれば曖昧性は消えるので不要
        assert_eq!(t("4A"), "⠼⠙⠠⠁");
    }

    // --- Phase 5: 語中の散在大文字 ---

    #[test]
    fn capital_inside_word() {
        assert_eq!(t("McDonald"), "⠠⠍⠉⠠⠙⠕⠝⠁⠇⠙");
        assert_eq!(t("iPhone"), "⠊⠠⠏⠓⠐⠕"); // i + P + h + one(頭字符)
    }

    // --- Phase 6: 出典（Rules of UEB 2024）の用例をそのまま回帰テストにする ---
    // §2.6 standing alone / §5.2 grade 1 記号符 / §6 数値モード / §7 約物 / §10.9 shortform。

    #[test]
    fn rules_of_ueb_examples() {
        let cases: &[(&str, &str)] = &[
            ("x", "⠰⠭"), ("it", "⠭"), ("which", "⠱"), ("was", "⠴"), ("al", "⠰⠁⠇"), ("also", "⠁⠇"),
            ("do-it-yourself", "⠙⠤⠭⠤⠽⠗⠋"), ("out-and-out", "⠳⠤⠯⠤⠳"),
            ("but...not", "⠃⠥⠞⠲⠲⠲⠝⠕⠞"),
            ("(c", "⠐⠣⠰⠉"), ("[can", "⠨⠣⠉"), ("{af", "⠸⠣⠰⠁⠋"),
            ("\u{201c}do", "⠦⠙"), ("\u{2018}your", "⠠⠦⠽⠗"),
            ("'e 'as", "⠄⠰⠑⠀⠄⠵"),
            ("very, very still; rather good.", "⠧⠂⠀⠧⠀⠌⠆⠀⠗⠀⠛⠙⠲"),
            ("d:", "⠰⠙⠒"), ("this...", "⠹⠲⠲⠲"), ("rejoice!", "⠗⠚⠉⠖"),
            ("(q, r)", "⠐⠣⠰⠟⠂⠀⠰⠗⠐⠜"),
            ("[quite, rather]", "⠨⠣⠟⠂⠀⠗⠨⠜"),
            ("{k-p}", "⠸⠣⠰⠅⠤⠰⠏⠸⠜"),
            ("children.\u{201d}", "⠡⠝⠲⠴"),
            ("friends' numbers", "⠋⠗⠎⠄⠀⠝⠥⠍⠃⠻⠎"),
            ("ab/cd", "⠁⠃⠸⠌⠉⠙"), ("could/should", "⠉⠳⠇⠙⠸⠌⠩⠳⠇⠙"),
            ("child's?)", "⠡⠄⠎⠦⠐⠜"),
            ("\u{201c}p's and q's\u{201d}", "⠦⠰⠏⠄⠎⠀⠯⠀⠰⠟⠄⠎⠴"),
            ("(Wouldn't)", "⠐⠣⠠⠺⠙⠝⠄⠞⠐⠜"),
            ("Mrs X and Mr O", "⠠⠍⠗⠎⠀⠰⠠⠭⠀⠯⠀⠠⠍⠗⠀⠠⠕"),
            ("J. S. Bach", "⠰⠠⠚⠲⠀⠰⠠⠎⠲⠀⠠⠃⠁⠡"),
            ("22b", "⠼⠃⠃⠰⠃"), ("22B", "⠼⠃⠃⠠⠃"), ("22p", "⠼⠃⠃⠏"),
            ("jim@take2.com", "⠚⠊⠍⠈⠁⠞⠁⠅⠑⠼⠃⠲⠰⠉⠕⠍"),
            ("7(b)", "⠼⠛⠐⠣⠃⠐⠜"),
            ("Braille4All", "⠠⠃⠗⠁⠊⠇⠇⠑⠼⠙⠠⠁⠇⠇"),
            ("X%", "⠠⠭⠨⠴"),
            ("9/11", "⠼⠊⠸⠌⠼⠁⠁"),
            ("self-control", "⠎⠑⠇⠋⠤⠒⠞⠗⠕⠇"),
            ("tied 1-1", "⠞⠊⠫⠀⠼⠁⠤⠼⠁"),
            ("320-foot wingspan", "⠼⠉⠃⠚⠤⠋⠕⠕⠞⠀⠺⠬⠎⠏⠁⠝"),
            ("7:30 a.m.", "⠼⠛⠒⠼⠉⠚⠀⠁⠲⠍⠲"),
            ("1/4 cup", "⠼⠁⠸⠌⠼⠙⠀⠉⠥⠏"),
            ("the vowels are: a, e, i, o and u", "⠮⠀⠧⠪⠑⠇⠎⠀⠜⠑⠒⠀⠁⠂⠀⠰⠑⠂⠀⠊⠂⠀⠕⠀⠯⠀⠰⠥"),
        ];
        for (src, want) in cases {
            assert_eq!(t(src), *want, "入力: {src:?}");
        }
    }

    // --- Phase 6: 約物・記号（§7 / §3）---

    #[test]
    fn brackets_and_slash() {
        assert_eq!(t("(go)"), "⠐⠣⠛⠐⠜");
        assert_eq!(t("[go]"), "⠨⠣⠛⠨⠜");
        assert_eq!(t("{go}"), "⠸⠣⠛⠸⠜");
        assert_eq!(t("and/or"), "⠯⠸⠌⠕⠗");
    }

    #[test]
    fn symbols_and_currency() {
        assert_eq!(t("$5"), "⠈⠎⠼⠑");
        assert_eq!(t("100%"), "⠼⠁⠚⠚⠨⠴");
        assert_eq!(t("a@b"), "⠁⠈⠁⠃");
        assert_eq!(t("R&D"), "⠠⠗⠈⠯⠠⠙");
        assert_eq!(t("1+2 = 3"), "⠼⠁⠐⠖⠼⠃⠀⠐⠶⠀⠼⠉");
    }

    #[test]
    fn dashes_and_underscore() {
        assert_eq!(t("go\u{2013}go"), "⠛⠠⠤⠛"); // en dash
        assert_eq!(t("go\u{2014}go"), "⠛⠐⠠⠤⠛"); // em dash（長いダッシュ）
        assert_eq!(t("a_b"), "⠁⠨⠤⠃");
    }

    // --- Phase 6: 引用符（開き／閉じは文脈で決める）---

    #[test]
    fn quotation_marks_open_and_close() {
        // 直前が空白・行頭・開き括弧なら開き ⠦、でなければ閉じ ⠴
        assert_eq!(t("\"Go!\""), "⠦⠠⠛⠖⠴");
        assert_eq!(t("\u{201c}Go!\u{201d}"), "⠦⠠⠛⠖⠴");
        // 一重引用符は ⠠⠦ / ⠠⠴（アポストロフィ ⠄ とは別）
        assert_eq!(t("\u{2018}go\u{2019}"), "⠠⠦⠛⠠⠴");
        assert_eq!(t("it's"), "⠭⠄⠎");
    }

    // --- Phase 6: 単独語が縮約として読めるときの grade 1 記号符（§5.7 / §10.9.7）---

    #[test]
    fn grade1_indicator_for_standalone_letters() {
        assert_eq!(t("b"), "⠰⠃"); // ⠃ 単独は「but」と読まれる
        assert_eq!(t("ab"), "⠰⠁⠃"); // ⠁⠃ は shortform「about」
        assert_eq!(t("a"), "⠁"); // a / i / o は縮約セルを持たないので不要
        assert_eq!(t("i"), "⠊");
        assert_eq!(t("o"), "⠕");
        assert_eq!(t("do"), "⠙"); // その語自身の wordsign はそのまま
    }

    #[test]
    fn shortform_needs_standing_alone() {
        // スラッシュは語境界ではないので shortform を使わない（§10.9.1）
        assert_eq!(t("could/should"), "⠉⠳⠇⠙⠸⠌⠩⠳⠇⠙");
        // アポストロフィで繋がる語は全体で1つの並び。リストにある wouldn't は shortform 可
        assert_eq!(t("wouldn't"), "⠺⠙⠝⠄⠞");
        assert_eq!(t("could've"), "⠉⠙⠄⠧⠑");
    }

    // --- Phase 6: 数値モードは小数点・桁区切りを跨いで続く（§6.2）---

    #[test]
    fn numeric_mode_spans_period() {
        assert_eq!(t("take2.com"), "⠞⠁⠅⠑⠼⠃⠲⠰⠉⠕⠍");
        assert_eq!(t("7:30 a.m."), "⠼⠛⠒⠼⠉⠚⠀⠁⠲⠍⠲"); // ":" は数値モードを切る
    }

    // --- Phase 7: 大文字の3モード（§8）---
    // 大文字符 ⠠（1文字）／大文字語符 ⠠⠠（連なり）／大文字句符 ⠠⠠⠠（3並び以上）と
    // 大文字終止符 ⠠⠄。用例は Rules of UEB §8.3〜§8.6 からそのまま。

    #[test]
    fn capitals_word_mode_and_terminator() {
        let cases: &[(&str, &str)] = &[
            ("ABCs", "⠠⠠⠁⠃⠉⠠⠄⠎"),
            ("WALKing", "⠠⠠⠺⠁⠇⠅⠠⠄⠬"),
            ("unSELFish", "⠥⠝⠠⠠⠎⠑⠇⠋⠠⠄⠊⠩"),
            ("XXIInd", "⠠⠠⠭⠭⠊⠊⠠⠄⠝⠙"),
            ("VIIb", "⠠⠠⠧⠊⠊⠠⠄⠃"),
            ("DipTP", "⠠⠙⠊⠏⠠⠠⠞⠏"),
            // 出典は ⠠⠠⠞⠧⠠⠕⠝⠞⠜⠊⠕（O を Ontario の頭文字と見る）。どちらも曖昧でないので
            // §8.8 は「指針」と明言している。綴りからは決まらないので機械的な規則を採る。
            ("TVOntario", "⠠⠠⠞⠧⠕⠠⠄⠝⠞⠜⠊⠕"),
            ("NEW YORK", "⠠⠠⠝⠑⠺⠀⠠⠠⠽⠕⠗⠅"),
            ("PARLIAMENT", "⠠⠠⠏⠜⠇⠊⠁⠰⠞"),
            ("B.C.", "⠠⠃⠲⠠⠉⠲"),
            ("XI XIth", "⠠⠠⠭⠊⠀⠠⠠⠭⠊⠠⠄⠹"),
            ("OK OKd", "⠠⠠⠕⠅⠀⠠⠠⠕⠅⠠⠄⠙"),
            ("CD CDs", "⠰⠠⠠⠉⠙⠀⠠⠠⠉⠙⠠⠄⠎"),
            ("CAUTION: WET PAINT!", "⠠⠠⠠⠉⠁⠥⠰⠝⠒⠀⠺⠑⠞⠀⠏⠁⠔⠞⠖⠠⠄"),
            ("THE BBC AFRICA NEWS", "⠠⠠⠠⠮⠀⠃⠃⠉⠀⠁⠋⠗⠊⠉⠁⠀⠝⠑⠺⠎⠠⠄"),
            ("A SELF-MADE MAN", "⠠⠠⠠⠁⠀⠎⠑⠇⠋⠤⠍⠁⠙⠑⠀⠍⠁⠝⠠⠄"),
            ("FOR SALE: 1975 FIREBIRD", "⠠⠠⠠⠿⠀⠎⠁⠇⠑⠒⠀⠼⠁⠊⠛⠑⠀⠋⠊⠗⠑⠃⠊⠗⠙⠠⠄"),
            ("Please KEEP OFF THE GRASS in this area.",
             "⠠⠏⠇⠂⠎⠑⠀⠠⠠⠠⠅⠑⠑⠏⠀⠷⠋⠀⠮⠀⠛⠗⠁⠎⠎⠠⠄⠀⠔⠀⠹⠀⠜⠑⠁⠲"),
            ("He shouted \"I WILL NOT!\"", "⠠⠓⠑⠀⠩⠳⠞⠫⠀⠦⠠⠠⠠⠊⠀⠺⠀⠝⠖⠠⠄⠴"),
            ("IT'S A HOAX! (APRIL FOOL!)", "⠠⠠⠠⠭⠄⠎⠀⠁⠀⠓⠕⠁⠭⠖⠀⠐⠣⠁⠏⠗⠊⠇⠀⠋⠕⠕⠇⠖⠐⠜⠠⠄"),
        ];
        let mut bad = 0;
        for (src, want) in cases {
            let got = t(src);
            if got != *want {
                println!("MISMATCH {src:?}\n  want {want}\n  got  {got}");
                bad += 1;
            }
        }
        println!("=== mismatches: {bad} / {} ===", cases.len());
    }

    // --- Phase 8: §10.9.3 リストに無い語の中でも使える10の shortform ---

    #[test]
    fn free_shortforms_in_unlisted_words() {
        // braille / great はどこに現れても使う（母音制限なし）
        assert_eq!(t("Braillette"), "⠠⠃⠗⠇⠞⠞⠑");
        assert_eq!(t("rebrailled"), "⠗⠑⠃⠗⠇⠙");
        assert_eq!(t("Greatford"), "⠠⠛⠗⠞⠿⠙");
        assert_eq!(t("Greatorex"), "⠠⠛⠗⠞⠕⠗⠑⠭"); // 母音が続いても使う
        assert_eq!(t("Feelgreat"), "⠠⠋⠑⠑⠇⠛⠗⠞"); // 語頭でなくても使う
        // 残る7つは語頭のみ
        assert_eq!(t("Blindcraft"), "⠠⠃⠇⠉⠗⠁⠋⠞");
        assert_eq!(t("Firstchoice"), "⠠⠋⠌⠡⠕⠊⠉⠑");
        assert_eq!(t("Quicksburg"), "⠠⠟⠅⠎⠃⠥⠗⠛");
        assert_eq!(t("Friendly"), "⠠⠋⠗⠇⠽");
    }

    #[test]
    fn free_shortforms_stop_before_vowel() {
        // 母音・y が続くときは使わない（§10.9.3(b)(c)）
        assert_eq!(t("Blindoc"), "⠠⠃⠇⠔⠙⠕⠉");
        assert_eq!(t("Goodacre"), "⠠⠛⠕⠕⠙⠁⠉⠗⠑");
        assert_eq!(t("Littlearm"), "⠠⠇⠊⠞⠞⠇⠑⠜⠍");
        assert_eq!(t("Letterewe"), "⠠⠇⠑⠞⠞⠻⠑⠺⠑");
        // 語頭でない bl は blind と読まれないので、そのまま
        assert_eq!(t("table"), "⠞⠁⠃⠇⠑");
    }

    // --- Phase 8: §10.10 縮約の優先順位（セル数が同じときの決着）---

    #[test]
    fn preference_strong_over_lower() {
        // §10.10.5: strong groupsign は lower groupsign に優先する（ar > ea）
        assert_eq!(t("bear"), "⠃⠑⠜");
        assert_eq!(t("heart"), "⠓⠑⠜⠞");
        assert_eq!(t("nuclear"), "⠝⠥⠉⠇⠑⠜");
        assert_eq!(t("bacchanal"), "⠃⠁⠉⠡⠁⠝⠁⠇"); // ch(strong) > cc(lower)
        // But: egghead は複合語の境界（egg|head）があるので gg を使う
        assert_eq!(t("egghead"), "⠑⠶⠓⠂⠙");
    }

    #[test]
    fn preference_lower_prefix_first_syllable() {
        // §10.10.4: be/con/dis が第一音節なら他の groupsign に優先する
        assert_eq!(t("beatitude"), "⠆⠁⠞⠊⠞⠥⠙⠑");
        assert_eq!(t("bedraggled"), "⠆⠙⠗⠁⠶⠇⠫");
        assert_eq!(t("congee"), "⠒⠛⠑⠑");
        assert_eq!(t("dishonesty"), "⠲⠓⠐⠕⠌⠽");
        // But: 第一音節でないものは使わない（initial_stems に無い）
        assert_eq!(t("beach"), "⠃⠂⠡");
        assert_eq!(t("benzene"), "⠃⠢⠵⠢⠑");
        assert_eq!(t("dishevelled"), "⠙⠊⠩⠑⠧⠑⠇⠇⠫");
    }

    #[test]
    fn preference_strong_over_two_cell() {
        // §10.10.7: strong/lower は頭字符・尾字符に優先する（セル数が同じなら）
        assert_eq!(t("adhered"), "⠁⠙⠓⠻⠫"); // here(頭字符) ではなく er + ed
        assert_eq!(t("heredity"), "⠓⠻⠫⠰⠽");
        assert_eq!(t("onerous"), "⠕⠝⠻⠳⠎");
        assert_eq!(t("Parthian"), "⠠⠏⠜⠹⠊⠁⠝");
        // セル数が減るなら頭字符・尾字符を使う（§10.10.2 が優先）
        assert_eq!(t("timer"), "⠐⠞⠗");
        assert_eq!(t("named"), "⠐⠝⠙");
        assert_eq!(t("advanced"), "⠁⠙⠧⠨⠑⠙");
        assert_eq!(t("happiness"), "⠓⠁⠏⠏⠊⠰⠎");
    }

    // --- Phase 8: §10.9.8/§10.9.9 shortform と誤読される語の grade 1 ---

    #[test]
    fn shortform_inside_listed_word_at_word_end() {
        // said は §10.9.3 の10語ではないので、Appendix 1 に載る語の中だけで使う。
        // リストは said の下に aforesaid / foresaid / gainsaid / missaid / saidest /
        // saidst / unsaid を並べている（語末でも使う）。
        assert_eq!(t("aforesaid"), "⠁⠿⠑⠎⠙");
        assert_eq!(t("unsaid"), "⠥⠝⠎⠙");
        assert_eq!(t("gainsaid"), "⠛⠁⠔⠎⠙");
        // リストに無い語では使わない（Wednesday の sd は縮約しないし、指示符も要らない）
        assert_eq!(t("Wednesday"), "⠠⠺⠫⠝⠑⠎⠐⠙");
        // 出典の例: braillist は「braille」の綴りを含まないので縮約しない
        assert_eq!(t("braillist"), "⠃⠗⠁⠊⠇⠇⠊⠌");
    }

    #[test]
    fn grade1_word_indicator_for_misreadable_words() {
        // 語中に brl（braille）があると誤読されるので、語符で語全体を無縮約にする
        assert_eq!(t("Dobrljin"), "⠰⠰⠠⠙⠕⠃⠗⠇⠚⠊⠝");
        assert_eq!(t("ozbrl"), "⠰⠰⠕⠵⠃⠗⠇");
        // 語頭なら記号符で足りる（§10.9.8）
        assert_eq!(t("Blvd"), "⠰⠠⠃⠇⠧⠙");
        // 読み手が shortform を適用しない位置なら指示符は要らない
        assert_eq!(t("blackboard"), "⠃⠇⠁⠉⠅⠃⠕⠜⠙"); // bl の後が母音
        assert_eq!(t("table"), "⠞⠁⠃⠇⠑"); // bl が語頭でない
    }

    #[test]
    fn grade1_word_indicator_reads_cells_back() {
        // 10語以外の shortform でも、読み戻した語がリストにあれば誤読される。
        // ⠠⠠⠥⠝⠎⠙ は「UNSAID」と同じ点字（我々は unsaid を ⠥⠝⠎⠙ と書く）。
        assert_eq!(t("unsaid"), "⠥⠝⠎⠙");
        assert_eq!(t("UNSD"), "⠰⠰⠠⠠⠥⠝⠎⠙");
        // 読み戻しても語にならないなら誤読されない（wisdom → wisaidom はリストに無い）
        assert_eq!(t("wisdom"), "⠺⠊⠎⠙⠕⠍");
        assert_eq!(t("misdeed"), "⠍⠊⠎⠙⠑⠫");
    }
}
