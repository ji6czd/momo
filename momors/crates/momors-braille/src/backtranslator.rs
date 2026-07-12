use crate::{Result, Table};
use std::collections::{HashMap, HashSet};

const SPACE: char = '⠀'; // U+2800 点字スペース
const DECIMAL_CELL: char = '⠲'; // 数字モード中の小数点
const COMMA_CELL: char = '⠄'; // 数字モード中の桁区切り

// ============================================================
// 状態
// ============================================================

/// 文脈モード。点字のフラグは前置なので、状態は左→右に一方向にしか流れない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// 通常（かな・日本語記号）。
    Normal,
    /// 数字モード（⠼ で開始）。
    Digit,
    /// 外来語モード（⠰ で開始）。
    Foreign,
}

/// 外来語モード中の大文字状態。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Capital {
    /// 小文字。
    None,
    /// 次の 1 文字だけ大文字（⠠）。
    Single,
    /// 以降すべて大文字（⠠⠠、モード終了まで継続）。
    Double,
}

/// 1 セル分の保留状態。前置セルを見た直後、次のセルで確定する。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pending {
    /// 保留なし。
    None,
    /// かなの前置セル（濁音・拗音・半濁音など）を保持中。
    /// 次のセルと結合できれば 1〜2 文字を確定、できなければ単独記号として吐く。
    Kana(char),
    /// ⠰ を保持中。次が英字なら外来語モード、それ以外なら読点「、」。
    ForeignOrComma,
}

/// 逆点訳の途中状態。点字フラグ（数符・外字符・大文字符）と保留セルを引き継ぐ。
///
/// `Copy` なので呼び出し側（エディタ・キー入力）が各セル境界の状態を
/// そのまま保持・キャッシュできる。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackTransState {
    mode: Mode,
    capital: Capital,
    pending: Pending,
    /// 直前の記号の trailing スペースとして読み飛ばすセル数。
    skip_spaces: u8,
}

impl Default for BackTransState {
    fn default() -> Self {
        Self {
            mode: Mode::Normal,
            capital: Capital::None,
            pending: Pending::None,
            skip_spaces: 0,
        }
    }
}

impl BackTransState {
    /// 初期状態（行頭）。
    pub fn new() -> Self {
        Self::default()
    }

    /// 保留セルが残っていない（行末で [`BrailleBackTranslator::flush`] が不要）か。
    pub fn is_settled(&self) -> bool {
        matches!(self.pending, Pending::None)
    }
}

/// [`BrailleBackTranslator::step`] / [`BrailleBackTranslator::flush`] の結果。
#[derive(Debug, Clone)]
pub struct StepResult {
    /// このセルで確定した文字列。前置セルなら空、拗音なら 2 文字になりうる。
    pub text: String,
    /// 更新後の状態。次の [`BrailleBackTranslator::step`] にそのまま渡す。
    pub state: BackTransState,
}

/// 点字セル範囲 → 復元文字列の対応（ガイド表示のアラインメント用）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackTransSegment {
    /// 対応する点字セルの範囲。半開区間 `[cell_start, cell_end)`、
    /// インデックスは `braille.chars()`（点字セル単位）に対応する。
    pub cell_start: usize,
    /// 範囲の終端（排他的）。
    pub cell_end: usize,
    /// このセル範囲が復元する文字列（空にはならない）。
    pub text: String,
}

/// [`BrailleBackTranslator::back_translate_aligned`] の結果。
#[derive(Debug, Clone)]
pub struct BackTransResult {
    /// 復元された全文（`segments` のテキストを連結したものと一致する）。
    pub text: String,
    /// 点字セル範囲 → 文字列のセグメント列。
    ///
    /// - 前置 + 基底（ガ＝⠐⠡）は両セルにまたがる 1 セグメント。
    /// - モードフラグ（数符・外字符）は後続文字のセグメントに吸収される。
    /// - 記号の trailing スペースは直前のセグメントに吸収される。
    pub segments: Vec<BackTransSegment>,
}

// ============================================================
// BrailleBackTranslator
// ============================================================

/// 点字逆変換器（逆点訳）。
///
/// 点字 1 セルと状態を受け取り、復元した文字と更新後の状態を返す。
/// フラグはすべて前置なので、確定済みの文字が後続セルで書き換わることはない。
///
/// # 対象と制約
///
/// - **対象**: かな（清音・濁音・半濁音・拗音）、数字、grade 1 ラテン、基本的な記号、スペース。
/// - **不可逆**: 順変換は漢字→かなで情報が落ちるため、復元できるのはかな表層まで。
/// - **スコープ外**: 丸数字・元号などの特殊記号（順変換専用の便宜エントリ）、Grade 2／UEB。
pub struct BrailleBackTranslator {
    /// 1 セル → かな。
    kana1: HashMap<char, String>,
    /// 2 セル（前置 + 基底）→ かな。
    kana2: HashMap<(char, char), String>,
    /// 2 セルかなの先頭（前置）セル集合。これを見たら保留する。
    prefix_cells: HashSet<char>,
    /// 1 セル日本語記号 → (文字, trailing スペース数)。
    punct_jp_rev: HashMap<char, (String, u8)>,
    /// 1 セル ASCII 記号 → (文字, trailing スペース数)。外来語モードで参照。
    punct_latin_rev: HashMap<char, (String, u8)>,
    /// 数字セル → ASCII 数字。
    digit_rev: HashMap<char, char>,
    /// ラテン文字セル → 小文字。
    latin_rev: HashMap<char, char>,
    // フラグセル
    digit_flag: char,
    foreign_flag: char,
    capital_flag: char,
    explicit_exit: char,
}

impl BrailleBackTranslator {
    /// テーブルを指定して逆変換器を作る（順方向 [`Table`] を反転構築）。
    pub fn new(table: Table) -> Self {
        let mut kana1: HashMap<char, String> = HashMap::new();
        let mut kana2: HashMap<(char, char), String> = HashMap::new();
        let mut prefix_cells: HashSet<char> = HashSet::new();

        for map in [&table.kana_single, &table.kana_compound] {
            for (kana, brl) in map {
                let cells: Vec<char> = brl.chars().collect();
                match cells.len() {
                    1 => {
                        // 語境界スペース（" " → ⠀）は逆引きしない（スペースとして別処理）
                        if cells[0] != SPACE {
                            kana1.insert(cells[0], kana.clone());
                        }
                    }
                    2 => {
                        kana2.insert((cells[0], cells[1]), kana.clone());
                        prefix_cells.insert(cells[0]);
                    }
                    _ => {} // 3 セル以上のかなは現状存在しない
                }
            }
        }

        // 記号の直後に吸収するスペース数は、その記号のクラスを起点とする
        // [transitions] の最大値（順変換が挿入しうる上限）から導出する。
        let to_rev = |m: &HashMap<String, crate::table::PunctCell>| -> HashMap<char, (String, u8)> {
            let mut r = HashMap::new();
            for (ch, cell) in m {
                let cells: Vec<char> = cell.braille.chars().collect();
                if cells.len() == 1 {
                    let trailing = table.max_transition_spaces_from(&cell.path);
                    r.insert(cells[0], (ch.clone(), trailing as u8));
                }
            }
            r
        };

        let mut digit_rev: HashMap<char, char> = HashMap::new();
        for (ch, brl) in &table.digit {
            // ASCII 数字キーのみ採用（全角は同じ点字に重複するため）
            if ch.chars().all(|c| c.is_ascii_digit()) {
                if let (Some(d), Some(cell)) = (ch.chars().next(), brl.chars().next()) {
                    digit_rev.insert(cell, d);
                }
            }
        }

        let mut latin_rev: HashMap<char, char> = HashMap::new();
        for (ch, brl) in &table.latin {
            if let (Some(c), Some(cell)) = (ch.chars().next(), brl.chars().next()) {
                latin_rev.insert(cell, c);
            }
        }

        let first = |s: &str| s.chars().next().unwrap_or(SPACE);

        Self {
            kana1,
            kana2,
            prefix_cells,
            punct_jp_rev: to_rev(&table.punct_jp),
            punct_latin_rev: to_rev(&table.punct_latin),
            digit_rev,
            latin_rev,
            digit_flag: first(&table.flag_digit.entry_prefix),
            foreign_flag: first(&table.flag_foreign_word.entry_prefix),
            capital_flag: first(&table.flag_capital.entry_prefix),
            explicit_exit: first(&table.flag_digit.explicit_exit),
        }
    }

    /// 組み込みテーブルで逆変換器を作る。
    pub fn from_embedded() -> Result<Self> {
        Ok(Self::new(Table::embedded()?))
    }

    /// ファイルからテーブルを読み込んで逆変換器を作る。
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self::new(Table::from_file(path)?))
    }

    /// 点字 1 セルを逆変換する純粋関数。
    ///
    /// 状態は呼び出し側が保持する。キー入力など 1 セルずつ届く状況でも、
    /// 行全体を流す [`Self::back_translate`] でも、この関数を土台にする。
    pub fn step(&self, cell: char, state: BackTransState) -> StepResult {
        let (flushed, current, state) = self.step_parts(cell, state);
        let mut text = flushed;
        text.push_str(&current);
        StepResult { text, state }
    }

    /// [`Self::step`] と同じ処理だが、出力を「直前の保留セルが確定した分（flushed）」と
    /// 「このセル自身の確定分（current）」に分けて返す。
    ///
    /// `flushed` は 1 つ前までのセル範囲、`current` はこのセル（および直前の前置セル）に
    /// 属する。アラインメント構築（[`Self::back_translate_aligned`]）に使う。
    fn step_parts(
        &self,
        cell: char,
        mut state: BackTransState,
    ) -> (String, String, BackTransState) {
        // 0. 直前の記号の trailing スペースを読み飛ばす
        if cell == SPACE && state.skip_spaces > 0 && matches!(state.pending, Pending::None) {
            state.skip_spaces -= 1;
            return (String::new(), String::new(), state);
        }
        if cell != SPACE {
            state.skip_spaces = 0;
        }

        // 1. 保留セルの解決
        match state.pending {
            Pending::Kana(prefix) => {
                state.pending = Pending::None;
                if let Some(kana) = self.kana2.get(&(prefix, cell)) {
                    // 前置 + 基底が結合 → かな確定（前置セルとこのセルで 1 セグメント）
                    return (String::new(), kana.clone(), state);
                }
                // 結合できない → 前置セルを単独記号として吐き（flushed）、cell は新規処理
                let mut flushed = String::new();
                if let Some((sym, trailing)) = self.punct_jp_rev.get(&prefix) {
                    flushed.push_str(sym);
                    state.skip_spaces = *trailing;
                }
                if cell == SPACE && state.skip_spaces > 0 {
                    state.skip_spaces -= 1;
                    return (flushed, String::new(), state);
                }
                let (cur, s) = self.process_fresh(cell, state);
                (flushed, cur, s)
            }
            Pending::ForeignOrComma => {
                state.pending = Pending::None;
                if cell == self.capital_flag || self.latin_rev.contains_key(&cell) {
                    // ⠰ + 英字（または大文字符）→ 外来語モード
                    state.mode = Mode::Foreign;
                    state.capital = Capital::None;
                    let (cur, s) = self.process_fresh(cell, state);
                    return (String::new(), cur, s);
                }
                // それ以外 → 読点「、」（直前の ⠰ セルに属する flushed）
                state.skip_spaces = 1;
                if cell == SPACE {
                    state.skip_spaces -= 1;
                    return (String::from("、"), String::new(), state);
                }
                let (cur, s) = self.process_fresh(cell, state);
                (String::from("、"), cur, s)
            }
            Pending::None => {
                let (cur, s) = self.process_fresh(cell, state);
                (String::new(), cur, s)
            }
        }
    }

    /// 入力終端で呼び、保留中のセルを確定させる。
    pub fn flush(&self, mut state: BackTransState) -> StepResult {
        let mut out = String::new();
        match state.pending {
            Pending::Kana(prefix) => {
                if let Some((sym, _)) = self.punct_jp_rev.get(&prefix) {
                    out.push_str(sym);
                }
            }
            Pending::ForeignOrComma => out.push('、'),
            Pending::None => {}
        }
        state.pending = Pending::None;
        state.skip_spaces = 0;
        StepResult { text: out, state }
    }

    /// 点字文字列全体を逆変換する便利メソッド。
    pub fn back_translate(&self, braille: &str) -> String {
        let mut state = BackTransState::default();
        let mut out = String::new();
        for cell in braille.chars() {
            let r = self.step(cell, state);
            out.push_str(&r.text);
            state = r.state;
        }
        out.push_str(&self.flush(state).text);
        out
    }

    /// 点字文字列全体を逆変換し、点字セル範囲 → 文字列の対応も返す。
    ///
    /// エディタの編集画面で、各点字セルの下（または横）に読みを並べるガイド表示に使う。
    /// セルのインデックスは `braille.chars()` に対応する。
    pub fn back_translate_aligned(&self, braille: &str) -> BackTransResult {
        let cells: Vec<char> = braille.chars().collect();
        let mut state = BackTransState::default();
        let mut segments: Vec<BackTransSegment> = Vec::new();
        let mut text = String::new();
        // まだセグメントに割り当てていないセル run の開始位置
        let mut run_start = 0usize;

        for (i, &cell) in cells.iter().enumerate() {
            // 記号の trailing スペースは直前のセグメントに吸収する
            let is_trailing_space =
                cell == SPACE && state.skip_spaces > 0 && matches!(state.pending, Pending::None);

            let (flushed, current, next_state) = self.step_parts(cell, state);
            state = next_state;

            if is_trailing_space {
                if let Some(last) = segments.last_mut() {
                    last.cell_end = i + 1;
                }
                run_start = i + 1;
                continue;
            }

            // flushed は直前の保留セル run（[run_start, i)）に属する
            if !flushed.is_empty() {
                text.push_str(&flushed);
                segments.push(BackTransSegment {
                    cell_start: run_start,
                    cell_end: i,
                    text: flushed,
                });
                run_start = i;
            }
            // current はこのセル（直前の前置セルを含む run [run_start, i]）に属する
            if !current.is_empty() {
                text.push_str(&current);
                segments.push(BackTransSegment {
                    cell_start: run_start,
                    cell_end: i + 1,
                    text: current,
                });
                run_start = i + 1;
            }
            // 両方空（前置・モードフラグ・消費スペース）→ run_start 据え置きで次に吸収
        }

        // 行末フラッシュ（保留セルを確定）
        let f = self.flush(state);
        if !f.text.is_empty() {
            segments.push(BackTransSegment {
                cell_start: run_start,
                cell_end: cells.len(),
                text: f.text.clone(),
            });
            text.push_str(&f.text);
        }

        BackTransResult { text, segments }
    }

    // ============================================================
    // private
    // ============================================================

    /// 保留なしの状態で 1 セルを処理する。モード遷移時は自身を再帰呼び出し（深さ ≤ 2）。
    fn process_fresh(&self, cell: char, mut state: BackTransState) -> (String, BackTransState) {
        let mut out = String::new();
        match state.mode {
            Mode::Normal => {
                if cell == SPACE {
                    out.push(' ');
                } else if cell == self.digit_flag {
                    state.mode = Mode::Digit;
                } else if cell == self.foreign_flag {
                    state.pending = Pending::ForeignOrComma;
                } else if self.prefix_cells.contains(&cell) {
                    state.pending = Pending::Kana(cell);
                } else if let Some(kana) = self.kana1.get(&cell) {
                    out.push_str(kana);
                } else if let Some((sym, trailing)) = self.punct_jp_rev.get(&cell) {
                    out.push_str(sym);
                    state.skip_spaces = *trailing;
                }
                // それ以外（未知セル）は読み捨て
            }
            Mode::Digit => {
                if let Some(d) = self.digit_rev.get(&cell) {
                    out.push(*d);
                } else if cell == DECIMAL_CELL {
                    out.push('.');
                    state.skip_spaces = 2;
                } else if cell == COMMA_CELL {
                    out.push(',');
                    state.skip_spaces = 1;
                } else if cell == self.explicit_exit {
                    state.mode = Mode::Normal; // ⠤ による明示的終端
                } else {
                    // 数字でないセル → 数字モード終了、通常モードで処理し直す
                    state.mode = Mode::Normal;
                    let (t, s) = self.process_fresh(cell, state);
                    out.push_str(&t);
                    state = s;
                }
            }
            Mode::Foreign => {
                if cell == SPACE {
                    // 外来語モード終了（exit_suffix / 語境界）
                    state.mode = Mode::Normal;
                    state.capital = Capital::None;
                    out.push(' ');
                } else if cell == self.capital_flag {
                    state.capital = match state.capital {
                        Capital::None => Capital::Single,
                        _ => Capital::Double,
                    };
                } else if let Some(l) = self.latin_rev.get(&cell) {
                    let ch = if state.capital == Capital::None {
                        *l
                    } else {
                        l.to_ascii_uppercase()
                    };
                    out.push(ch);
                    if state.capital == Capital::Single {
                        state.capital = Capital::None;
                    }
                } else if let Some((sym, trailing)) = self.punct_latin_rev.get(&cell) {
                    out.push_str(sym);
                    state.skip_spaces = *trailing;
                } else {
                    // 英字でないセル → 外来語モード終了、通常モードで処理し直す
                    state.mode = Mode::Normal;
                    state.capital = Capital::None;
                    let (t, s) = self.process_fresh(cell, state);
                    out.push_str(&t);
                    state = s;
                }
            }
        }
        (out, state)
    }
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn bt() -> BrailleBackTranslator {
        BrailleBackTranslator::from_embedded().expect("テーブルをロードできること")
    }

    // --- 基本変換（converter.rs の出力を逆変換） ---

    #[test]
    fn kana_aiueo() {
        assert_eq!(bt().back_translate("⠁⠃⠉⠋⠊"), "アイウエオ");
    }

    #[test]
    fn kana_voiced() {
        // ⠐（濁音前置）+ ⠡（カ）→ ガ
        assert_eq!(bt().back_translate("⠐⠡"), "ガ");
    }

    #[test]
    fn kana_compound_kya() {
        // ⠈（拗音前置）+ ⠡ → キャ（2 文字）
        assert_eq!(bt().back_translate("⠈⠡"), "キャ");
    }

    #[test]
    fn kana_semivoiced() {
        // ⠠（半濁音前置・通常モード）+ ⠥（ハ）→ パ
        assert_eq!(bt().back_translate("⠠⠥"), "パ");
    }

    #[test]
    fn sentence_with_punct() {
        // コンニチワ、セカイ！
        assert_eq!(bt().back_translate("⠪⠴⠇⠗⠄⠰⠀⠻⠡⠃⠖⠀⠀"), "コンニチワ、セカイ！");
    }

    // --- 数字モード ---

    #[test]
    fn digits_12345() {
        assert_eq!(bt().back_translate("⠼⠁⠃⠉⠙⠑"), "12345");
    }

    #[test]
    fn digit_to_kana_explicit_exit() {
        // ⠤ で数字モード終了 → レツ
        assert_eq!(bt().back_translate("⠼⠁⠤⠛⠝"), "1レツ");
    }

    #[test]
    fn digit_to_kana_no_explicit_exit() {
        // ⠐ は数字でない → 数字モード終了して濁音前置として処理
        assert_eq!(bt().back_translate("⠼⠁⠐⠡"), "1ガ");
    }

    #[test]
    fn decimal_point_preserves_digit_mode() {
        // 小数点の trailing スペースを読み飛ばし、数字モードを維持
        assert_eq!(bt().back_translate("⠼⠉⠲⠀⠀⠁⠙"), "3.14");
    }

    // --- 外来語モード ---

    #[test]
    fn latin_lowercase() {
        assert_eq!(bt().back_translate("⠰⠁⠃⠉"), "abc");
    }

    #[test]
    fn latin_single_capital() {
        assert_eq!(bt().back_translate("⠰⠠⠁"), "A");
    }

    #[test]
    fn latin_all_caps() {
        assert_eq!(bt().back_translate("⠰⠠⠠⠝⠓⠅"), "NHK");
    }

    #[test]
    fn latin_followed_by_kana() {
        // 外来語終了の ⠀ はスペースとして復元される（空白付きの綴りに正規化）
        assert_eq!(bt().back_translate("⠰⠠⠠⠝⠓⠅⠀⠑⠐⠳⠊"), "NHK ラジオ");
    }

    // --- 単独記号（前置セルが結合しなかった場合のフラッシュ） ---

    #[test]
    fn middle_dot_standalone() {
        // ⠐ は濁音前置だが、結合相手がなければ「・」
        assert_eq!(bt().back_translate("⠐⠀"), "・");
    }

    // --- 逐次ステップ / フラッシュ ---

    #[test]
    fn step_defers_then_resolves() {
        let t = bt();
        let r1 = t.step('⠐', BackTransState::new());
        assert_eq!(r1.text, ""); // 前置セルは保留（出力なし）
        assert!(!r1.state.is_settled());
        let r2 = t.step('⠡', r1.state);
        assert_eq!(r2.text, "ガ");
        assert!(r2.state.is_settled());
    }

    #[test]
    fn flush_emits_pending() {
        let t = bt();
        let r1 = t.step('⠐', BackTransState::new()); // 保留
        let r2 = t.flush(r1.state);
        assert_eq!(r2.text, "・");
        assert!(r2.state.is_settled());
    }

    // --- 順変換との往復 ---

    #[test]
    fn round_trip_via_converter() {
        use crate::JapaneseTranslator;
        let conv = JapaneseTranslator::from_embedded().unwrap();
        let back = bt();
        for kana in [
            "アイウエオ",
            "ガギグゲゴ",
            "キャシュチョ",
            "パピプペポ",
            "ッー",
        ] {
            let brl = conv.translate(kana).unwrap();
            assert_eq!(
                back.back_translate(brl.braille_text()),
                kana,
                "入力: {kana}"
            );
        }
    }

    // --- アラインメント（ガイド表示用） ---

    /// セグメントを (cell_start, cell_end, text) のタプル列に落とす。
    fn segs(r: &BackTransResult) -> Vec<(usize, usize, &str)> {
        r.segments
            .iter()
            .map(|s| (s.cell_start, s.cell_end, s.text.as_str()))
            .collect()
    }

    #[test]
    fn aligned_simple() {
        let r = bt().back_translate_aligned("⠁⠃⠉");
        assert_eq!(r.text, "アイウ");
        assert_eq!(segs(&r), vec![(0, 1, "ア"), (1, 2, "イ"), (2, 3, "ウ")]);
    }

    #[test]
    fn aligned_voiced_spans_two_cells() {
        // ガ ＝ ⠐⠡ は両セルにまたがる 1 セグメント
        let r = bt().back_translate_aligned("⠐⠡");
        assert_eq!(segs(&r), vec![(0, 2, "ガ")]);
    }

    #[test]
    fn aligned_compound_spans_two_cells() {
        // キャ ＝ ⠈⠡（2 文字でも 1 セグメント）
        let r = bt().back_translate_aligned("⠈⠡");
        assert_eq!(segs(&r), vec![(0, 2, "キャ")]);
    }

    #[test]
    fn aligned_digit_flag_absorbed() {
        // 数符 ⠼ は後続の "1" に吸収される
        let r = bt().back_translate_aligned("⠼⠁⠃");
        assert_eq!(r.text, "12");
        assert_eq!(segs(&r), vec![(0, 2, "1"), (2, 3, "2")]);
    }

    #[test]
    fn aligned_trailing_space_absorbed() {
        // 。の trailing スペース 2 つは「。」のセグメントに吸収される
        let r = bt().back_translate_aligned("⠲⠀⠀");
        assert_eq!(r.text, "。");
        assert_eq!(segs(&r), vec![(0, 3, "。")]);
    }

    #[test]
    fn aligned_flush_then_current_split() {
        // ⠐ が結合できず「・」、続く ⠁ が「ア」→ セルごとに分かれる
        let r = bt().back_translate_aligned("⠐⠁");
        assert_eq!(r.text, "・ア");
        assert_eq!(segs(&r), vec![(0, 1, "・"), (1, 2, "ア")]);
    }

    #[test]
    fn aligned_text_matches_back_translate() {
        let t = bt();
        for brl in ["⠪⠴⠇⠗⠄⠰⠀⠻⠡⠃⠖⠀⠀", "⠼⠉⠲⠀⠀⠁⠙", "⠰⠠⠠⠝⠓⠅⠀⠑⠐⠳⠊"]
        {
            let r = t.back_translate_aligned(brl);
            assert_eq!(r.text, t.back_translate(brl), "入力: {brl}");
            // セグメントのテキスト連結は全文と一致
            let joined: String = r.segments.iter().map(|s| s.text.as_str()).collect();
            assert_eq!(joined, r.text, "入力: {brl}");
        }
    }
}
