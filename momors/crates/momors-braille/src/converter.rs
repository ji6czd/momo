use std::collections::HashSet;
use crate::{BrailleTable, Result};

const BRAILLE_SPACE: char = '⠀'; // U+2800

// ============================================================
// BrailleResult
// ============================================================

/// 点字変換の結果。
#[derive(Debug, Clone)]
pub struct BrailleResult {
    braille_text: String,
    /// かな文字インデックス → 点字文字インデックス（文字単位、先頭セル位置）
    kana_to_braille: Vec<usize>,
}

impl BrailleResult {
    /// 変換後の点字文字列。
    pub fn braille_text(&self) -> &str {
        &self.braille_text
    }

    /// かな文字インデックス → 点字先頭セルのインデックス。
    ///
    /// - 複合音（キャ など）の 2 文字はどちらも同じ点字位置を指す。
    /// - フラグ記号（⠼ ⠰ ⠠ ⠤ など）はそのフラグを発動させた文字の点字位置に含まれる。
    pub fn kana_to_braille(&self) -> &[usize] {
        &self.kana_to_braille
    }
}

// ============================================================
// BrailleConverter
// ============================================================

/// 点字変換器。
///
/// [`BrailleConverter::from_embedded`] で組み込みテーブルを使うか、
/// [`BrailleConverter::new`] でテーブルを直接渡す。
pub struct BrailleConverter {
    table: BrailleTable,
    /// 数字モードで使われる点字セルの先頭文字集合。⠤ 挿入判定に使う。
    digit_cells: HashSet<char>,
}

impl BrailleConverter {
    /// テーブルを指定して変換器を作る。
    pub fn new(table: BrailleTable) -> Self {
        // 数字テーブルの各エントリの先頭セルを集める
        let digit_cells = table
            .digit
            .values()
            .filter_map(|s| s.chars().next())
            .collect();
        Self { table, digit_cells }
    }

    /// 組み込みテーブルで変換器を作る。
    pub fn from_embedded() -> Result<Self> {
        Ok(Self::new(BrailleTable::embedded()?))
    }

    /// かな文字列を点字に変換する。
    ///
    /// `kana_text` には [`momors_core::PredictionResult::kana_text`] の出力を渡す。
    /// ASCII 文字・数字・句読点が混在していても処理できる。
    pub fn convert(&self, kana_text: &str) -> Result<BrailleResult> {
        // momors-core がバイパスした全角数字・英字を ASCII に正規化する
        let normalized: String = kana_text.chars().map(|c| match c as u32 {
            0xFF10..=0xFF19 => char::from_u32(c as u32 - 0xFF10 + 0x30).unwrap_or(c), // ０-９→0-9
            0xFF21..=0xFF3A => char::from_u32(c as u32 - 0xFF21 + 0x41).unwrap_or(c), // Ａ-Ｚ→A-Z
            0xFF41..=0xFF5A => char::from_u32(c as u32 - 0xFF41 + 0x61).unwrap_or(c), // ａ-ｚ→a-z
            _ => c,
        }).collect();
        let chars: Vec<char> = normalized.chars().collect();
        let n = chars.len();

        let mut braille = String::new();
        let mut kana_to_braille: Vec<usize> = Vec::with_capacity(n);

        // フラグ状態
        let mut in_digit = false;       // ⠼ が有効な数字モード
        let mut in_foreign_word = false; // ⠰ が有効な外来語モード
        let mut in_capital_word = false; // 大文字モード（先頭 ⠠ 済み）
        let mut in_double_capital = false; // 全大文字モード（⠠⠠ 済み）

        let mut i = 0;
        while i < n {
            let c = chars[i];

            // この文字の点字列が始まる位置（フラグ記号も含む）を記録
            let brl_pos = braille.chars().count();

            if (c as u32) < 0x80 {
                // ==================================================
                // ASCII 文字
                // ==================================================
                if c.is_ascii_alphabetic() {
                    // 数字モード終了
                    in_digit = false;

                    // 外来語モード開始（初回のみ ⠰ を挿入）
                    if !in_foreign_word {
                        braille.push_str(&self.table.flag_foreign_word.entry_prefix);
                        in_foreign_word = true;
                    }

                    // 大文字フラグ
                    if c.is_ascii_uppercase() {
                        if !in_capital_word {
                            // 1文字目の大文字: ⠠ を 1 つ挿入
                            braille.push_str(&self.table.flag_capital.entry_prefix);
                        }
                        in_capital_word = true;
                        // 次も大文字なら全大文字モード: さらに ⠠ を 1 つ追加（合計 ⠠⠠）
                        if i + 1 < n && chars[i + 1].is_ascii_uppercase() && !in_double_capital {
                            braille.push_str(&self.table.flag_capital.entry_prefix);
                            in_double_capital = true;
                        }
                    } else if in_capital_word {
                        // 小文字に戻ったら大文字モード解除
                        in_capital_word = false;
                        in_double_capital = false;
                    }

                    // ラテン文字テーブルを引く（小文字キーで統一）
                    let key = c.to_ascii_lowercase().to_string();
                    let cell = self.table.latin.get(&key).map_or(BRAILLE_SPACE.to_string(), |s| s.clone());
                    braille.push_str(&cell);
                } else if c.is_ascii_digit() {
                    // 外来語モード終了（exit_suffix なし: ⠼ がモード変更を示す）
                    in_foreign_word = false;
                    in_capital_word = false;
                    in_double_capital = false;

                    // 数字モード開始（初回のみ ⠼ を挿入）
                    if !in_digit {
                        braille.push_str(&self.table.flag_digit.entry_prefix);
                        in_digit = true;
                    }

                    let key = c.to_string();
                    let cell = self.table.digit.get(&key).map_or(BRAILLE_SPACE.to_string(), |s| s.clone());
                    braille.push_str(&cell);
                } else if c == '.' || c == ',' {
                    // 数字の小数点・桁区切りの可能性: 数字モードをリセットしない。
                    // 外来語モード中なら punct_latin を、そうでなければ punct_jp を参照。
                    let key = c.to_string();
                    let entry = if in_foreign_word {
                        self.table.punct_latin.get(&key)
                    } else {
                        self.table.punct_jp.get(&key).or_else(|| self.table.punct_latin.get(&key))
                    };
                    self.emit_punct(&mut braille, entry);
                } else {
                    // その他 ASCII（スペース含む）: 全フラグをリセット
                    in_foreign_word = false;
                    in_digit = false;
                    in_capital_word = false;
                    in_double_capital = false;

                    let key = c.to_string();
                    let entry = self.table.punct_latin.get(&key);
                    self.emit_punct(&mut braille, entry);
                }

                kana_to_braille.push(brl_pos);
                i += 1;
            } else {
                // ==================================================
                // 非 ASCII（日本語文字・記号）
                // ==================================================

                // 外来語モード終了: 日本語文字への切り替えを示す exit_suffix（⠀）を挿入
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
                in_double_capital = false;

                // 複合音（2文字キー）を先に試みる
                if i + 1 < n {
                    let mut key2 = String::with_capacity(8);
                    key2.push(c);
                    key2.push(chars[i + 1]);
                    if let Some(brl) = self.table.kana_compound.get(&key2) {
                        braille.push_str(brl);
                        kana_to_braille.push(brl_pos); // 複合音 1 文字目
                        kana_to_braille.push(brl_pos); // 複合音 2 文字目（同じ点字位置）
                        i += 2;
                        continue;
                    }
                }

                // 単音・日本語記号
                let key = c.to_string();
                if let Some(brl) = self.table.kana_single.get(&key) {
                    braille.push_str(brl);
                } else if let Some((brl, trailing)) = self.table.punct_jp.get(&key) {
                    braille.push_str(brl);
                    for _ in 0..*trailing {
                        braille.push(BRAILLE_SPACE);
                    }
                } else {
                    braille.push(BRAILLE_SPACE);
                }

                kana_to_braille.push(brl_pos);
                i += 1;
            }
        }

        Ok(BrailleResult {
            braille_text: braille,
            kana_to_braille,
        })
    }

    // ============================================================
    // private helpers
    // ============================================================

    /// 句読点エントリを braille に追記する。
    /// エントリがなければ点字スペース（⠀）を追記する。
    fn emit_punct(&self, braille: &mut String, entry: Option<&(String, usize)>) {
        if let Some((brl, trailing)) = entry {
            braille.push_str(brl);
            for _ in 0..*trailing {
                braille.push(BRAILLE_SPACE);
            }
        } else {
            braille.push(BRAILLE_SPACE);
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
        if let Some((brl, _)) = self.table.punct_jp.get(&key) {
            return brl.chars().next();
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

    fn conv() -> BrailleConverter {
        BrailleConverter::from_embedded().expect("テーブルをロードできること")
    }

    // --- 基本変換 ---

    #[test]
    fn kana_aiueo() {
        let r = conv().convert("アイウエオ").unwrap();
        assert_eq!(r.braille_text(), "⠁⠃⠉⠋⠊");
    }

    #[test]
    fn kana_voiced() {
        // ガ = ⠐⠡（濁音プレフィックス + カ行）
        let r = conv().convert("ガ").unwrap();
        assert_eq!(r.braille_text(), "⠐⠡");
    }

    #[test]
    fn kana_compound_kya() {
        // キャ → ⠈⠡（2文字 → 1エントリ）
        let r = conv().convert("キャ").unwrap();
        assert_eq!(r.braille_text(), "⠈⠡");
        // 2 文字ともに同じ点字位置
        assert_eq!(r.kana_to_braille(), &[0, 0]);
    }

    #[test]
    fn kana_with_punct() {
        // コンニチワ、セカイ！
        // 、= ⠰ + 後続スペース1、！→ ！ は ASCII 分岐で punct_latin から引く
        let r = conv().convert("コンニチワ、セカイ！").unwrap();
        // ⠪⠴⠇⠗⠄⠰⠀⠻⠡⠃⠖⠀⠀
        assert_eq!(r.braille_text(), "⠪⠴⠇⠗⠄⠰⠀⠻⠡⠃⠖⠀⠀");
    }

    // --- 数字モード ---

    #[test]
    fn digits_12345() {
        let r = conv().convert("12345").unwrap();
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
        let r = conv().convert("1レツ").unwrap();
        assert_eq!(r.braille_text(), "⠼⠁⠤⠛⠝");
    }

    #[test]
    fn digit_to_kana_no_explicit_exit() {
        // 1ガ → ⠼⠁⠐⠡
        // ガ の先頭セル ⠐ はデジットセルでないので ⠤ 不要
        let r = conv().convert("1ガ").unwrap();
        assert_eq!(r.braille_text(), "⠼⠁⠐⠡");
    }

    #[test]
    fn decimal_point_preserves_digit_mode() {
        // 3.14 → ⠼⠉⠲⠀⠀⠁⠙
        // '.' は数字モードをリセットしない
        let r = conv().convert("3.14").unwrap();
        // ⠼⠉ (3) + ⠲⠀⠀ (.) + ⠁ (1) + ⠙ (4)
        assert_eq!(r.braille_text(), "⠼⠉⠲⠀⠀⠁⠙");
    }

    #[test]
    fn fullwidth_digit_treated_as_digit() {
        // momors-core がバイパスした全角数字 → ASCII に正規化されて数字モードで変換
        // １レツ → ⠼⠁⠤⠛⠝（ASCII "1レツ" と同じ結果）
        let r = conv().convert("１レツ").unwrap();
        assert_eq!(r.braille_text(), "⠼⠁⠤⠛⠝");
    }

    // --- 外来語モード ---

    #[test]
    fn latin_lowercase() {
        // abc → ⠰⠁⠃⠉（外来語フラグ + a + b + c）
        let r = conv().convert("abc").unwrap();
        assert_eq!(r.braille_text(), "⠰⠁⠃⠉");
    }

    #[test]
    fn latin_single_capital() {
        // A → ⠰⠠⠁（外来語 + 大文字 + a）
        let r = conv().convert("A").unwrap();
        assert_eq!(r.braille_text(), "⠰⠠⠁");
    }

    #[test]
    fn latin_all_caps() {
        // NHK → ⠰⠠⠠⠝⠓⠅（外来語 + ⠠⠠ + n + h + k）
        let r = conv().convert("NHK").unwrap();
        assert_eq!(r.braille_text(), "⠰⠠⠠⠝⠓⠅");
    }

    #[test]
    fn latin_followed_by_kana() {
        // NHK ラジオ → ⠰⠠⠠⠝⠓⠅ + スペース(⠀) + ラ(⠑) + ジ(⠐⠳) + オ(⠊)
        let r = conv().convert("NHK ラジオ").unwrap();
        assert_eq!(r.braille_text(), "⠰⠠⠠⠝⠓⠅⠀⠑⠐⠳⠊");
    }

    #[test]
    fn latin_to_kana_no_space_adds_exit_suffix() {
        // NHKラジオ（スペースなし）→ 外来語終了で exit_suffix(⠀) が挿入される
        // スペースあり版と同じ結果になる
        let r = conv().convert("NHKラジオ").unwrap();
        assert_eq!(r.braille_text(), "⠰⠠⠠⠝⠓⠅⠀⠑⠐⠳⠊");
    }

    // --- インデックス ---

    #[test]
    fn index_simple() {
        let r = conv().convert("アイウ").unwrap();
        assert_eq!(r.kana_to_braille(), &[0, 1, 2]);
    }

    #[test]
    fn index_with_digit_prefix() {
        // "1ア" → ⠼⠁⠤⠁
        // '1' → pos 0（⠼ 込み）, 'ア' → pos 2（⠤ 込み）
        let r = conv().convert("1ア").unwrap();
        assert_eq!(r.kana_to_braille(), &[0, 2]);
    }

    #[test]
    fn index_compound_and_single() {
        // キャア → ⠈⠡⠁
        // キ→0, ャ→0（複合）, ア→2
        let r = conv().convert("キャア").unwrap();
        assert_eq!(r.kana_to_braille(), &[0, 0, 2]);
    }
}
