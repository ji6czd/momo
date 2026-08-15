use crate::table::{CLASS_DIGIT, CLASS_KANA, CLASS_LATIN, CLASS_NONE, PunctCell};
use crate::{Result, Table};
use std::collections::HashSet;

const BRAILLE_SPACE: char = '⠀'; // U+2800

/// テーブルに定義されていない文字の扱い。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UnknownCharPolicy {
    /// 点字スペース（U+2800）に置換する（デフォルト）。
    #[default]
    Space,
    /// 元の文字をそのまま出力する（スクリーンリーダー用途）。
    PassThrough,
}

// ============================================================
// JapaneseResult
// ============================================================

/// 点字変換の結果。
#[derive(Debug, Clone)]
pub struct JapaneseResult {
    braille_text: String,
    /// かな文字インデックス → 点字文字インデックス（文字単位、先頭セル位置）
    kana_to_braille: Vec<usize>,
}

impl JapaneseResult {
    /// 変換後の点字文字列。
    pub fn braille_text(&self) -> &str {
        &self.braille_text
    }

    /// かな文字インデックス → 点字先頭セルのインデックス。
    ///
    /// - 複合音（キャ など）の 2 文字はどちらも同じ点字位置を指す。
    /// - 開始フラグ（⠼ ⠰ ⠠）はそのフラグを発動させた文字の点字位置に含まれる。
    /// - 終了フラグ（⠤ など）とクラス遷移スペースは直前の文字の範囲に含まれる。
    pub fn kana_to_braille(&self) -> &[usize] {
        &self.kana_to_braille
    }
}

// ============================================================
// JapaneseTranslator
// ============================================================

/// 点字変換器。
///
/// [`JapaneseTranslator::from_embedded`] で組み込みテーブルを使うか、
/// [`JapaneseTranslator::new`] でテーブルを直接渡す。
/// テーブル未定義文字の扱いは [`JapaneseTranslator::with_unknown_char_policy`] で変更できる。
pub struct JapaneseTranslator {
    table: Table,
    /// 数字モードで使われる点字セルの先頭文字集合。⠤ 挿入判定に使う。
    digit_cells: HashSet<char>,
    /// 数字モード中に現れても数字モードを終了させない文字（`flags.digit.exempt_chars` 由来）。
    digit_exempt_chars: HashSet<char>,
    unknown_char: UnknownCharPolicy,
}

impl JapaneseTranslator {
    /// 保持しているテーブルへの参照を返す。
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// テーブルを指定して変換器を作る。
    pub fn new(table: Table) -> Self {
        // 数字テーブルの各エントリの先頭セルを集める
        let digit_cells = table
            .digit
            .values()
            .filter_map(|s| s.chars().next())
            .collect();
        let digit_exempt_chars = table
            .flag_digit
            .exempt_chars
            .iter()
            .filter_map(|s| s.chars().next())
            .collect();
        Self {
            table,
            digit_cells,
            digit_exempt_chars,
            unknown_char: UnknownCharPolicy::default(),
        }
    }

    /// デフォルトの組み込みテーブル（日本語１級）で変換器を作る。
    pub fn from_embedded() -> Result<Self> {
        Ok(Self::new(Table::embedded()?))
    }

    /// 名前で組み込みテーブルを指定して変換器を作る。
    ///
    /// `name` は TOML の `[metadata].name` と照合する。見つからない場合はエラー。
    pub fn from_embedded_name(name: &str) -> Result<Self> {
        let table = crate::table::embedded_table(name)
            .ok_or_else(|| crate::Error::UnknownTable(name.to_owned()))?;
        Ok(Self::new(table))
    }

    /// ファイルからテーブルを読み込んで変換器を作る。
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self::new(Table::from_file(path)?))
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

    /// かな文字列を点字に変換する。
    ///
    /// `kana_text` は読みに変換済みのテキスト（上位の予測器の出力）。漢字は扱わない。
    /// ASCII 文字・数字・句読点が混在していても処理できる。
    pub fn translate(&self, kana_text: &str) -> Result<JapaneseResult> {
        // 幅正規化: 半角カナ → 全角カタカナ（`ｶﾞ→ガ` の合成含む）、全角英数字 → ASCII。
        // src_len[k] は正規化後の k 文字目が消費した原文文字数（`ｶﾞ→ガ` なら 2）。
        // 末尾で kana_to_braille を原文インデックスへ展開するのに使う。
        let (chars, src_len) = crate::width::normalize(kana_text);
        let n = chars.len();

        let mut braille = String::new();
        let mut kana_to_braille: Vec<usize> = Vec::with_capacity(n);

        // フラグ状態
        let mut in_digit = false; // ⠼ が有効な数字モード
        let mut in_foreign_word = false; // ⠰ が有効な外来語モード
        let mut in_capital_word = false; // 大文字モード（先頭 ⠠ か ⠠⠠ 済み）

        // 直前に出力した文字のクラス（[transitions] の判定に使う）。テキスト先頭は None。
        let mut prev_class: Option<String> = None;

        let mut i = 0;
        while i < n {
            let c = chars[i];

            // この文字の点字列が始まる位置。遷移スペースの直後・開始フラグの前で
            // 記録するため、終了フラグ（⠤ など）と遷移スペースは直前の文字の範囲に、
            // 開始フラグ（⠼ ⠰ ⠠）はこの文字の範囲に入る。
            let brl_pos;

            if (c as u32) < 0x80 {
                // ==================================================
                // ASCII 文字
                // ==================================================
                if c.is_ascii_alphabetic() {
                    // 数字モード終了
                    in_digit = false;

                    // クラス遷移スペース（⠰ より前に挿入する）
                    self.push_transition_spaces(&mut braille, prev_class.as_deref(), CLASS_LATIN);
                    brl_pos = braille.chars().count();

                    // 外来語モード開始（初回のみ ⠰ を挿入）
                    if !in_foreign_word {
                        braille.push_str(&self.table.flag_foreign_word.entry_prefix);
                        in_foreign_word = true;
                    }

                    // 大文字フラグ
                    if c.is_ascii_uppercase() {
                        if !in_capital_word {
                            // 語頭で次も大文字なら全大文字モード（⠠⠠）、そうでなければ単独大文字（⠠）
                            if i + 1 < n && chars[i + 1].is_ascii_uppercase() {
                                braille.push_str(&self.table.flag_capital.double_entry_prefix);
                            } else {
                                braille.push_str(&self.table.flag_capital.entry_prefix);
                            }
                        }
                        in_capital_word = true;
                    } else if in_capital_word {
                        // 小文字に戻ったら大文字モード解除
                        in_capital_word = false;
                    }

                    // ラテン文字テーブルを引く。
                    // テーブルに大文字キーがあればそちらを優先し（no-conversion など）、
                    // なければ小文字キーで引く（grade1 など: 大文字フラグで区別）。
                    let exact_key = c.to_string();
                    let lower_key = c.to_ascii_lowercase().to_string();
                    if let Some(cell) = self
                        .table
                        .latin
                        .get(&exact_key)
                        .or_else(|| self.table.latin.get(&lower_key))
                    {
                        braille.push_str(cell);
                    } else {
                        self.emit_unknown(&mut braille, c);
                    }
                    prev_class = Some(CLASS_LATIN.to_string());
                } else if c.is_ascii_digit() {
                    // 外来語モード終了（exit_suffix なし: ⠼ がモード変更を示す）
                    in_foreign_word = false;
                    in_capital_word = false;

                    // クラス遷移スペース（⠼ より前に挿入する）
                    self.push_transition_spaces(&mut braille, prev_class.as_deref(), CLASS_DIGIT);
                    brl_pos = braille.chars().count();

                    // 数字モード開始（初回のみ ⠼ を挿入）
                    if !in_digit {
                        braille.push_str(&self.table.flag_digit.entry_prefix);
                        in_digit = true;
                    }

                    let key = c.to_string();
                    if let Some(cell) = self.table.digit.get(&key) {
                        braille.push_str(cell);
                    } else {
                        self.emit_unknown(&mut braille, c);
                    }
                    prev_class = Some(CLASS_DIGIT.to_string());
                } else if self.digit_exempt_chars.contains(&c) {
                    // 数字モード中の小数点・桁区切り（flags.digit.exempt_chars）: 数字モードをリセットしない。
                    // 外来語モード中なら punct_latin を、そうでなければ punct_jp を参照。
                    let key = c.to_string();
                    let entry = if in_foreign_word {
                        self.table.punct_latin.get(&key)
                    } else {
                        self.table
                            .punct_jp
                            .get(&key)
                            .or_else(|| self.table.punct_latin.get(&key))
                    };
                    let cur = Self::cell_class(entry);
                    self.push_transition_spaces(&mut braille, prev_class.as_deref(), &cur);
                    brl_pos = braille.chars().count();
                    self.emit_punct(&mut braille, entry, c);
                    prev_class = Some(cur);
                } else {
                    // その他 ASCII（スペース含む）: 全フラグをリセット
                    in_foreign_word = false;
                    in_digit = false;
                    in_capital_word = false;

                    let key = c.to_string();
                    let entry = self.table.punct_latin.get(&key);
                    let cur = Self::cell_class(entry);
                    self.push_transition_spaces(&mut braille, prev_class.as_deref(), &cur);
                    brl_pos = braille.chars().count();
                    self.emit_punct(&mut braille, entry, c);
                    prev_class = Some(cur);
                }

                kana_to_braille.push(brl_pos);
                i += 1;
            } else {
                // ==================================================
                // 非 ASCII（日本語文字・記号）
                // ==================================================

                // 外来語モード終了: 終了マーカー exit_suffix を挿入
                // （grade1 では空。latin→kana の境界スペースは [transitions] が担う）
                if in_foreign_word {
                    braille.push_str(&self.table.flag_foreign_word.exit_suffix);
                }

                // 数字モード終了: 次の点字先頭セルが数字セルと衝突する場合は ⠤ を挿入
                if in_digit {
                    let first = self.peek_first_kana_cell(c, chars.get(i + 1).copied());
                    if first.map_or(false, |ch| self.digit_cells.contains(&ch)) {
                        braille.push_str(&self.table.flag_digit.explicit_exit);
                    }
                }

                in_digit = false;
                in_foreign_word = false;
                in_capital_word = false;

                // 複合音（2文字キー）を先に試みる
                if i + 1 < n {
                    let mut key2 = String::with_capacity(8);
                    key2.push(c);
                    key2.push(chars[i + 1]);
                    if let Some(brl) = self.table.kana_compound.get(&key2) {
                        self.push_transition_spaces(
                            &mut braille,
                            prev_class.as_deref(),
                            CLASS_KANA,
                        );
                        let brl_pos = braille.chars().count();
                        braille.push_str(brl);
                        kana_to_braille.push(brl_pos); // 複合音 1 文字目
                        kana_to_braille.push(brl_pos); // 複合音 2 文字目（同じ点字位置）
                        prev_class = Some(CLASS_KANA.to_string());
                        i += 2;
                        continue;
                    }
                }

                // 単音・日本語記号
                let key = c.to_string();
                if let Some(brl) = self.table.kana_single.get(&key) {
                    self.push_transition_spaces(&mut braille, prev_class.as_deref(), CLASS_KANA);
                    brl_pos = braille.chars().count();
                    braille.push_str(brl);
                    prev_class = Some(CLASS_KANA.to_string());
                } else if let Some(cell) = self.table.punct_jp.get(&key) {
                    let cur = Self::cell_class(Some(cell));
                    self.push_transition_spaces(&mut braille, prev_class.as_deref(), &cur);
                    brl_pos = braille.chars().count();
                    braille.push_str(&cell.braille);
                    prev_class = Some(cur);
                } else {
                    brl_pos = braille.chars().count();
                    self.emit_unknown(&mut braille, c);
                    prev_class = Some(CLASS_NONE.to_string());
                }

                kana_to_braille.push(brl_pos);
                i += 1;
            }
        }

        // kana_to_braille は正規化後の文字ごとに1エントリ。合成（`ｶﾞ→ガ`）で
        // 潰れた原文文字も同じ点字位置を指すよう、src_len で原文インデックスへ展開する。
        // 合成が無ければ src_len は全て 1 で、展開は恒等（従来と同一）。
        let kana_to_braille = if src_len.iter().any(|&l| l != 1) {
            let mut expanded = Vec::with_capacity(src_len.iter().sum());
            for (pos, &len) in kana_to_braille.iter().zip(&src_len) {
                for _ in 0..len {
                    expanded.push(*pos);
                }
            }
            expanded
        } else {
            kana_to_braille
        };

        Ok(JapaneseResult {
            braille_text: braille,
            kana_to_braille,
        })
    }

    // ============================================================
    // private helpers
    // ============================================================

    /// クラス遷移 `prev` → `cur` に `[transitions]` で宣言されたスペースを挿入する。
    /// テキスト先頭（`prev` が `None`）では発火しない。直前の出力がすでに
    /// 点字スペースで終わっていれば挿入しない（二重スペース防止）。
    fn push_transition_spaces(&self, braille: &mut String, prev: Option<&str>, cur: &str) {
        let Some(prev) = prev else { return };
        let spaces = self.table.transition_spaces(prev, cur);
        if spaces > 0 && !braille.ends_with(BRAILLE_SPACE) {
            for _ in 0..spaces {
                braille.push(BRAILLE_SPACE);
            }
        }
    }

    /// 句読点エントリのフルクラスパスを返す（エントリなしは予約パス `"none"`）。
    fn cell_class(entry: Option<&PunctCell>) -> String {
        entry
            .map(|cell| cell.path.clone())
            .unwrap_or_else(|| CLASS_NONE.to_string())
    }

    /// 句読点エントリを braille に追記する。エントリがなければ `fallback` 文字で `emit_unknown` する。
    /// 前後のスペースはここでは扱わない（呼び出し側の `push_transition_spaces` が担う）。
    fn emit_punct(&self, braille: &mut String, entry: Option<&PunctCell>, fallback: char) {
        if let Some(cell) = entry {
            braille.push_str(&cell.braille);
        } else {
            self.emit_unknown(braille, fallback);
        }
    }

    /// テーブル未定義文字を `unknown_char` ポリシーに従って出力する。
    fn emit_unknown(&self, braille: &mut String, c: char) {
        match self.unknown_char {
            UnknownCharPolicy::Space => braille.push(BRAILLE_SPACE),
            UnknownCharPolicy::PassThrough => braille.push(c),
        }
    }

    /// 文字 `c`（次の文字 `next` も考慮）の点字出力の先頭セルを返す。
    /// 数字モード終了時の ⠤ 挿入判定専用。
    fn peek_first_kana_cell(&self, c: char, next: Option<char>) -> Option<char> {
        // 複合音を先に試みる
        if let Some(nc) = next {
            let mut key2 = String::new();
            key2.push(c);
            key2.push(nc);
            if let Some(brl) = self.table.kana_compound.get(&key2) {
                return brl.chars().next();
            }
        }
        // 単音
        let key = c.to_string();
        if let Some(brl) = self.table.kana_single.get(&key) {
            return brl.chars().next();
        }
        if let Some(cell) = self.table.punct_jp.get(&key) {
            return cell.braille.chars().next();
        }
        Some(BRAILLE_SPACE)
    }
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn conv() -> JapaneseTranslator {
        JapaneseTranslator::from_embedded().expect("テーブルをロードできること")
    }

    // --- 基本変換 ---

    #[test]
    fn kana_aiueo() {
        let r = conv().translate("アイウエオ").unwrap();
        assert_eq!(r.braille_text(), "⠁⠃⠉⠋⠊");
    }

    #[test]
    fn kana_voiced() {
        // ガ = ⠐⠡（濁音プレフィックス + カ行）
        let r = conv().translate("ガ").unwrap();
        assert_eq!(r.braille_text(), "⠐⠡");
    }

    #[test]
    fn kana_compound_kya() {
        // キャ → ⠈⠡（2文字 → 1エントリ）
        let r = conv().translate("キャ").unwrap();
        assert_eq!(r.braille_text(), "⠈⠡");
        // 2 文字ともに同じ点字位置
        assert_eq!(r.kana_to_braille(), &[0, 0]);
    }

    #[test]
    fn kana_with_punct() {
        // コンニチワ、セカイ！
        // 、（pause）→ セ（kana）の境界に "pause -> *" = 1 のスペース。
        // 末尾の！は次の文字がないので遷移スペースなし。
        let r = conv().translate("コンニチワ、セカイ！").unwrap();
        assert_eq!(r.braille_text(), "⠪⠴⠇⠗⠄⠰⠀⠻⠡⠃⠖");
    }

    #[test]
    fn stop_to_stop_suppressed() {
        // ！？ はどちらも stop。"stop -> stop" = 0 の明示宣言でスペースが入らない。
        let r = conv().translate("！？").unwrap();
        assert_eq!(r.braille_text(), "⠖⠢");
    }

    #[test]
    fn stop_to_kana_inserts_two_spaces() {
        // ！→ ア は "stop -> *" = 2 で二マスあけ。
        let r = conv().translate("！ア").unwrap();
        assert_eq!(r.braille_text(), "⠖⠀⠀⠁");
    }

    #[test]
    fn three_stop_marks_no_spaces() {
        // ！？！ の連続はすべて "stop -> stop" = 0。
        let r = conv().translate("！？！").unwrap();
        assert_eq!(r.braille_text(), "⠖⠢⠖");
    }

    #[test]
    fn no_trailing_spaces_at_text_end() {
        // 遷移は「次の文字」がないと発火しない。文末の。にスペースは付かない。
        let r = conv().translate("ア。").unwrap();
        assert_eq!(r.braille_text(), "⠁⠲");
    }

    // --- ワイルドカード遷移: '→' のような前後必須スペース記号を想定した最小テーブル ---

    fn conv_with_inline_arrow() -> JapaneseTranslator {
        let toml = r#"
[flags.digit]
[flags.foreign_word]
[flags.capital]

[table.kana.compound]
[table.kana.single]
"ア" = "⠁"

[table.punct]
classes = ["stop", "inline"]
[table.punct.jp]
"。" = { braille = "⠲", class = "stop" }
"→" = { braille = "⠿⠿", class = "inline" }
[table.punct.latin]

[table.digit]
[table.latin]

[transitions]
"punct.jp.stop -> *" = 2
"punct.jp.stop -> punct.jp.stop" = 0
"* -> punct.jp.inline" = 1
"punct.jp.inline -> *" = 1
"#;
        JapaneseTranslator::new(Table::from_toml(toml).expect("テストテーブルは有効"))
    }

    #[test]
    fn inline_no_space_at_text_start() {
        // 行頭には prev がないので "* -> inline" は発火しない
        let r = conv_with_inline_arrow().translate("→").unwrap();
        assert_eq!(r.braille_text(), "⠿⠿");
    }

    #[test]
    fn inline_spaces_between_kana() {
        // ア→ア: "* -> inline" と "inline -> *" で前後一マスずつ
        let r = conv_with_inline_arrow().translate("ア→ア").unwrap();
        assert_eq!(r.braille_text(), "⠁⠀⠿⠿⠀⠁");
    }

    #[test]
    fn wildcard_conflict_takes_max() {
        // 。→ は "stop -> *" = 2 と "* -> inline" = 1 が両方当たる → max の 2
        let r = conv_with_inline_arrow().translate("。→").unwrap();
        assert_eq!(r.braille_text(), "⠲⠀⠀⠿⠿");
    }

    // --- 数字モード ---

    #[test]
    fn digits_12345() {
        let r = conv().translate("12345").unwrap();
        // ⠼ + 1〜5
        assert_eq!(r.braille_text(), "⠼⠁⠃⠉⠙⠑");
        // 最初の '1' → ⠼ 込みで位置 0
        assert_eq!(r.kana_to_braille()[0], 0);
        // '2' → 位置 2（⠼⠁ の次）
        assert_eq!(r.kana_to_braille()[1], 2);
    }

    #[test]
    fn digit_to_kana_explicit_exit() {
        // 1レツ → ⠼⠁⠤⠛⠝
        // レ = ⠛ はデジットセル（7）と衝突するので ⠤ が挿入される
        let r = conv().translate("1レツ").unwrap();
        assert_eq!(r.braille_text(), "⠼⠁⠤⠛⠝");
    }

    #[test]
    fn digit_to_kana_no_explicit_exit() {
        // 1ガ → ⠼⠁⠐⠡
        // ガ の先頭セル ⠐ はデジットセルでないので ⠤ 不要
        let r = conv().translate("1ガ").unwrap();
        assert_eq!(r.braille_text(), "⠼⠁⠐⠡");
    }

    #[test]
    fn decimal_point_preserves_digit_mode() {
        // 3.14 → ⠼⠉⠲⠁⠙
        // '.' は数字モードをリセットしない。ASCII 記号なので trailing なし。
        let r = conv().translate("3.14").unwrap();
        // ⠼⠉ (3) + ⠲ (.) + ⠁ (1) + ⠙ (4)
        assert_eq!(r.braille_text(), "⠼⠉⠲⠁⠙");
    }

    // --- 半角カナ ---

    #[test]
    fn halfwidth_kana_same_as_fullwidth() {
        // ｱｲｳｴｵ は全角と同じ点字
        let r = conv().translate("ｱｲｳｴｵ").unwrap();
        assert_eq!(r.braille_text(), "⠁⠃⠉⠋⠊");
    }

    #[test]
    fn halfwidth_voiced_composes() {
        // ｶﾞ（2文字）→ ガ → ⠐⠡。両文字とも同じ点字位置を指す。
        let r = conv().translate("ｶﾞ").unwrap();
        assert_eq!(r.braille_text(), "⠐⠡");
        assert_eq!(r.kana_to_braille(), &[0, 0]);
    }

    #[test]
    fn halfwidth_semivoiced_composes() {
        // ﾊﾟﾝ → パン。パ=⠠⠡相当（濁点系）、ン=⠴
        let full = conv().translate("パン").unwrap();
        let half = conv().translate("ﾊﾟﾝ").unwrap();
        assert_eq!(half.braille_text(), full.braille_text());
        // ﾊ ﾟ ﾝ の3文字。ﾊ と ﾟ はパの位置(0)、ﾝ はンの位置。
        assert_eq!(half.kana_to_braille(), &[0, 0, full.kana_to_braille()[1]]);
    }

    #[test]
    fn halfwidth_compound_youon() {
        // ｷｬ（半角、拗音）→ キャ → ⠈⠡
        let r = conv().translate("ｷｬ").unwrap();
        assert_eq!(r.braille_text(), "⠈⠡");
        assert_eq!(r.kana_to_braille(), &[0, 0]);
    }

    #[test]
    fn halfwidth_voiced_youon() {
        // ｷﾞｬ（3文字: ｷ ﾞ ｬ）→ ギャ。ｷ と ﾞ はギャの位置(0)、ｬ も同じ(複合音)。
        let full = conv().translate("ギャ").unwrap();
        let half = conv().translate("ｷﾞｬ").unwrap();
        assert_eq!(half.braille_text(), full.braille_text());
        assert_eq!(half.kana_to_braille(), &[0, 0, 0]);
    }

    #[test]
    fn fullwidth_digit_treated_as_digit() {
        // momors-core がバイパスした全角数字 → ASCII に正規化されて数字モードで変換
        // １レツ → ⠼⠁⠤⠛⠝（ASCII "1レツ" と同じ結果）
        let r = conv().translate("１レツ").unwrap();
        assert_eq!(r.braille_text(), "⠼⠁⠤⠛⠝");
    }

    // --- 外来語モード ---

    #[test]
    fn latin_lowercase() {
        // abc → ⠰⠁⠃⠉（外来語フラグ + a + b + c）
        let r = conv().translate("abc").unwrap();
        assert_eq!(r.braille_text(), "⠰⠁⠃⠉");
    }

    #[test]
    fn latin_single_capital() {
        // A → ⠰⠠⠁（外来語 + 大文字 + a）
        let r = conv().translate("A").unwrap();
        assert_eq!(r.braille_text(), "⠰⠠⠁");
    }

    #[test]
    fn latin_all_caps() {
        // NHK → ⠰⠠⠠⠝⠓⠅（外来語 + ⠠⠠ + n + h + k）
        let r = conv().translate("NHK").unwrap();
        assert_eq!(r.braille_text(), "⠰⠠⠠⠝⠓⠅");
    }

    #[test]
    fn latin_followed_by_kana() {
        // NHK ラジオ → ⠰⠠⠠⠝⠓⠅ + スペース(⠀) + ラ(⠑) + ジ(⠐⠳) + オ(⠊)
        let r = conv().translate("NHK ラジオ").unwrap();
        assert_eq!(r.braille_text(), "⠰⠠⠠⠝⠓⠅⠀⠑⠐⠳⠊");
    }

    #[test]
    fn latin_to_kana_no_space_adds_transition_space() {
        // NHKラジオ（スペースなし）→ [transitions] "latin -> kana" でスペースが挿入される
        // スペースあり版と同じ結果になる
        let r = conv().translate("NHKラジオ").unwrap();
        assert_eq!(r.braille_text(), "⠰⠠⠠⠝⠓⠅⠀⠑⠐⠳⠊");
    }

    #[test]
    fn latin_punct_to_kana_splits() {
        // 「a.ア」: 英文ピリオド（punct.latin）→ かな境界で "punct.latin -> kana" = 1
        // が発火し一マスあく。⠰⠁(a) ⠲(.) ⠀(遷移スペース) ⠁(ア)
        let r = conv().translate("a.ア").unwrap();
        assert_eq!(r.braille_text(), "⠰⠁⠲⠀⠁");
    }

    #[test]
    fn decimal_point_not_split_by_latin_punct_rule() {
        // 「3.14」の '.' は punct.latin だが後続は数字（kana ではない）ので
        // "punct.latin -> kana" は発火しない。3.14 は分割されないまま。
        let r = conv().translate("3.14").unwrap();
        assert_eq!(r.braille_text(), "⠼⠉⠲⠁⠙");
    }

    // --- クラス遷移スペース（[transitions]） ---

    #[test]
    fn kana_to_latin_inserts_transition_space() {
        // かな→latin 境界: "kana -> latin" = 1 により外字符（⠰）の前を一マスあける
        let r = conv().translate("アイウエオabc").unwrap();
        assert_eq!(r.braille_text(), "⠁⠃⠉⠋⠊⠀⠰⠁⠃⠉");
    }

    #[test]
    fn transition_space_idempotent_with_source_space() {
        // ソース側にスペースがあってもなくても同じ出力になる
        let with_space = conv().translate("アイウエオ abc").unwrap();
        let without = conv().translate("アイウエオabc").unwrap();
        assert_eq!(with_space.braille_text(), without.braille_text());
    }

    #[test]
    fn kana_to_digit_no_transition_space() {
        // かな→数字は宣言がないのでスペースなし（語中数字: ダイ3カイ など）
        let r = conv().translate("ア1").unwrap();
        assert_eq!(r.braille_text(), "⠁⠼⠁");
    }

    #[test]
    fn latin_to_punct_no_transition_space() {
        // latin→句読点（stop）は宣言がないのでスペースなし。
        // 旧 exit_suffix 方式では latin→非ASCII に一律スペースが入っていた。
        // 文末の。にも遷移スペースは付かない。
        let r = conv().translate("abc。").unwrap();
        assert_eq!(r.braille_text(), "⠰⠁⠃⠉⠲");
    }

    #[test]
    fn no_transition_space_at_text_start() {
        // テキスト先頭では prev がないため発火しない
        let r = conv().translate("abc").unwrap();
        assert_eq!(r.braille_text(), "⠰⠁⠃⠉");
    }

    #[test]
    fn index_with_transition_space() {
        // アa → （⠁ + 遷移スペース⠀）+ （⠰ + ⠁）
        // 遷移スペースは遷移前の文字（ア）の範囲に含まれる
        let r = conv().translate("アa").unwrap();
        assert_eq!(r.braille_text(), "⠁⠀⠰⠁");
        assert_eq!(r.kana_to_braille(), &[0, 2]);
    }

    #[test]
    fn index_pause_space_belongs_to_pause() {
        // ア、カ → ⠁ + （⠰ + スペース⠀）+ ⠡
        // "pause -> *" のスペースは 、の範囲に含まれる
        let r = conv().translate("ア、カ").unwrap();
        assert_eq!(r.braille_text(), "⠁⠰⠀⠡");
        assert_eq!(r.kana_to_braille(), &[0, 1, 3]);
    }

    #[test]
    fn index_stop_spaces_belong_to_stop() {
        // ！ア → （⠖ + 二マスあけ⠀⠀）+ ⠁
        // "stop -> *" の 2 スペースはどちらも ！の範囲に含まれる
        let r = conv().translate("！ア").unwrap();
        assert_eq!(r.braille_text(), "⠖⠀⠀⠁");
        assert_eq!(r.kana_to_braille(), &[0, 3]);
    }

    // --- インデックス ---

    #[test]
    fn index_simple() {
        let r = conv().translate("アイウ").unwrap();
        assert_eq!(r.kana_to_braille(), &[0, 1, 2]);
    }

    #[test]
    fn index_with_digit_prefix() {
        // "1ア" → ⠼⠁⠤⠁
        // '1' → pos 0（⠼ 込み）。終了フラグ ⠤ は数字側に帰属し、'ア' → pos 3
        let r = conv().translate("1ア").unwrap();
        assert_eq!(r.kana_to_braille(), &[0, 3]);
    }

    #[test]
    fn index_compound_and_single() {
        // キャア → ⠈⠡⠁
        // キ→0, ャ→0（複合）, ア→2
        let r = conv().translate("キャア").unwrap();
        assert_eq!(r.kana_to_braille(), &[0, 0, 2]);
    }
}
