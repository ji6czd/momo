//! 文字種判定モジュール。
//!
//! C++ 版の `char_type.hpp` + `utf8.cpp` の `get_char_type` 部分に相当。
//! Python 版 `momopy/src/momo_py/utils.py` の `get_char_type()` と
//! 挙動を一致させることを目的とする。
//!
//! 記号テーブルは `char_type_table.rs` に分離されており、
//! `tools/gen_char_type_rs.py` で Python から自動生成される。

mod table;

// ============================================================
// CharType
// ============================================================

/// 文字種列挙型。
///
/// 値の割り当てルール（C++ 版 char_type.hpp と一致）:
///
/// - bit7-4 : カテゴリ
///   - `0x0_` : 空白系
///   - `0x1_` : 文字系（ラテン・数字）
///   - `0x2_` : 予約
///   - `0x3_` : 記号系
///   - `0x4_` : 日本語文字系
///   - `0x5_` : スキップ系（NBSP・ZWJ など、仮名に現れず原文位置のみ保持）
///   - `0xF_` : その他・不明（バイパス）
///
/// モデルファイル（.mbm）にこの値が `u8` として書き込まれている。
/// `#[repr(u8)]` でメモリ表現を明示し、`as u8` で値を取得できる。
///
/// `Default` は [`Space`] (0x00) を返す。これは C++ 版で `CharType ct[3] = {}`
/// がゼロ初期化されて `SPACE` になるのと一致する。
///
/// # variant の宣言順について
///
/// `derive(Ord)` は **variant の宣言順** で比較する。本 enum では
/// **variant を u8 値の昇順** に並べているため、`Ord` の挙動は `as u8`
/// での比較と等価になる。新規 variant を追加するときも、u8 値の
/// 昇順位置に挿入すること。
///
/// [`Space`]: CharType::Space
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(u8)]
pub enum CharType {
    #[default]
    Space = 0x00,

    Alpha = 0x10,
    Numeric = 0x11,

    /// 予約（将来拡張用）
    Reserved = 0x20,

    Symbol = 0x30,
    SymbolClose = 0x31,
    SymbolOpen = 0x32,
    SymbolStop = 0x33,
    SymbolPause = 0x34,

    Hiragana = 0x40,
    Katakana = 0x41,
    Kanji = 0x42,
    JapaneseNumeric = 0x43,

    /// スキップ系（ZWJ・ゼロ幅文字など）。
    /// 仮名テキストには現れず、`src_to_kana_index` のエントリだけ保持される。
    Skip = 0x50,

    /// 未定義文字。バイパス扱いで原文のまま仮名に出力し、
    /// momors-braille 側で 1:1 変換する。
    Other = 0xFF,
}

// `category` / `is_japanese` / `is_symbol` / `is_latin` は内部モジュール
// (`mod char_type`) のため crate 外からは見えないが、診断・将来用途のため
// 公開メソッドとして残しておく。クレート内未使用でも dead_code を抑制する。
#[allow(dead_code)]
impl CharType {
    /// 上位ニブル（カテゴリ）を返す。
    #[inline]
    pub fn category(self) -> u8 {
        (self as u8) & 0xF0
    }

    /// 日本語文字系（ひらがな・カタカナ・漢字・漢数字）か。
    #[inline]
    pub fn is_japanese(self) -> bool {
        self.category() == 0x40
    }

    /// 記号系か。
    #[inline]
    pub fn is_symbol(self) -> bool {
        self.category() == 0x30
    }

    /// ラテン・数字系か。
    #[inline]
    pub fn is_latin(self) -> bool {
        self.category() == 0x10
    }

    /// 素通し (bypass) 扱いにすべき文字種か。**読みの**素通し可否のみを表す。
    ///
    /// `Space`・記号系・`Other` は読みを推論せず、原文をそのまま出力する。
    /// 英字 `Alpha` は読みこそ素通しだが、直後のマスあけは境界モデルに委ねるため
    /// bypass には含めない（カタカナと同じ「読み素通し＋境界モデル」扱い。
    /// 推論本体 `predict_normalized` の ALPHA 分岐を参照）。
    ///
    /// 境界（マスあけ）判定を実際にスキップするかどうかは別軸で、
    /// [`skips_boundary_check`](Self::skips_boundary_check) が担う。
    /// `Symbol`/`SymbolOpen`/`SymbolClose`（括弧類を含む）は読みは素通ししつつ
    /// 境界はモデルに問い合わせる（ALPHA・カタカナと同型）。
    #[inline]
    pub fn is_bypass(self) -> bool {
        matches!(
            self,
            CharType::Space
                | CharType::Symbol
                | CharType::SymbolClose
                | CharType::SymbolOpen
                | CharType::SymbolStop
                | CharType::SymbolPause
                | CharType::Other
        )
    }

    /// 境界（マスあけ）判定もスキップすべき文字種か。
    ///
    /// `is_bypass()` が読みの素通し可否を表すのに対し、こちらは境界判定の
    /// 可否を表す独立した軸。`SymbolStop`（文末記号 `。！?.`）・
    /// `SymbolPause`（読点・中点 `、・,`）はそれ自体が既に文の区切りとして
    /// 機能しており、直後にさらに境界モデルでマスあけを重ねて判定する意味が
    /// 薄いためスキップする。`Space`（原文の空白そのもの）・`Other`
    /// （未定義文字）も同様に対象外とする。
    ///
    /// 一方 `Symbol`・`SymbolOpen`・`SymbolClose`（括弧類を含む）は境界としての
    /// 意味合いが薄く、直後に語が続くかどうかは文脈次第（例:「」の直後が
    /// 助詞なら分割しない・体言が続くなら分割する）なので、ここには含めず
    /// 境界モデルに判定させる。学習データ（trainer.py）はこれらの文字種の
    /// 行も boundary ラベル（+S）付きで学習に使っており、モデル自体は既に
    /// 対応済み。単に推論側がこれまで問い合わせていなかっただけ。
    #[inline]
    pub fn skips_boundary_check(self) -> bool {
        matches!(
            self,
            CharType::Space | CharType::SymbolStop | CharType::SymbolPause | CharType::Other
        )
    }

    /// スキップ扱いにすべき文字種か。
    ///
    /// 仮名テキストには現れないが、`src_to_kana_index` のエントリは保持される。
    #[inline]
    pub fn is_skip(self) -> bool {
        self.category() == 0x50
    }

    /// Python 版 `CharType.value` と一致する文字列名を返す。
    ///
    /// 特徴量名の整形・解析（`type_s=KANJI` 等）に使う。exporter の
    /// `CHARTYPE_TO_INT` が扱う名前と同一でなければならない。
    pub fn as_str(self) -> &'static str {
        use CharType::*;
        match self {
            Space => "SPACE",
            Alpha => "ALPHA",
            Numeric => "NUM",
            Reserved => "RESERVED",
            Symbol => "SYMBOL",
            SymbolClose => "SYMBOL_CLOSE",
            SymbolOpen => "SYMBOL_OPEN",
            SymbolStop => "SYMBOL_STOP",
            SymbolPause => "SYMBOL_PAUSE",
            Hiragana => "HIRAGANA",
            Katakana => "KATAKANA",
            Kanji => "KANJI",
            JapaneseNumeric => "JAPANESE_NUMERIC",
            Skip => "SKIP",
            Other => "OTHER",
        }
    }

    /// [`as_str`](CharType::as_str) の逆変換。未知の名前は `None`。
    pub fn from_name(s: &str) -> Option<Self> {
        use CharType::*;
        Some(match s {
            "SPACE" => Space,
            "ALPHA" => Alpha,
            "NUM" => Numeric,
            "RESERVED" => Reserved,
            "SYMBOL" => Symbol,
            "SYMBOL_CLOSE" => SymbolClose,
            "SYMBOL_OPEN" => SymbolOpen,
            "SYMBOL_STOP" => SymbolStop,
            "SYMBOL_PAUSE" => SymbolPause,
            "HIRAGANA" => Hiragana,
            "KATAKANA" => Katakana,
            "KANJI" => Kanji,
            "JAPANESE_NUMERIC" => JapaneseNumeric,
            "SKIP" => Skip,
            "OTHER" => Other,
            _ => return None,
        })
    }

    /// `u8` 値から [`CharType`] を構築する。
    ///
    /// 主にモデルファイル読み込み時に使う。未定義の値は `None` を返す。
    pub fn from_u8(v: u8) -> Option<Self> {
        use CharType::*;
        Some(match v {
            0x00 => Space,
            0x10 => Alpha,
            0x11 => Numeric,
            0x20 => Reserved,
            0x30 => Symbol,
            0x31 => SymbolClose,
            0x32 => SymbolOpen,
            0x33 => SymbolStop,
            0x34 => SymbolPause,
            0x40 => Hiragana,
            0x41 => Katakana,
            0x42 => Kanji,
            0x43 => JapaneseNumeric,
            0x50 => Skip,
            0xFF => Other,
            _ => return None,
        })
    }
}

// ============================================================
// get_char_type
// ============================================================

/// 文字を [`CharType`] に分類する。
///
/// C++ 版の `get_char_type()` 相当。
///
/// 位取り文字（十・百・千・万・億・兆）の [`JapaneseNumeric`] への
/// 昇格は文脈判定が必要なため、ここでは [`Kanji`] を返す。
/// 文脈判定は呼び出し側（`to_source_seq()` 相当の前処理）の責務。
///
/// [`JapaneseNumeric`]: CharType::JapaneseNumeric
/// [`Kanji`]: CharType::Kanji
pub fn get_char_type(c: char) -> CharType {
    // --- 空白（Unicode White_Space プロパティ全体）---
    // Python の str.isspace() と等価。NBSP・全角スペース・各種幅スペースをすべて含む。
    if c.is_whitespace() {
        return CharType::Space;
    }

    let cp = c as u32;

    // --- スキップ文字（ゼロ幅・フォーマット制御）---
    if is_skip_cp(cp) {
        return CharType::Skip;
    }

    // --- 数字（ASCII・全角）---
    if c.is_ascii_digit() || (0xFF10..=0xFF19).contains(&cp) {
        return CharType::Numeric;
    }

    // --- 括弧の圧縮アイデンティティ・トークン（bracket.rs参照）---
    // 実文書には出現しない合成コードポイント（Private Use Area）なので、
    // Pythonの記号テーブル（tools/gen_char_type_rs.py で自動生成される
    // symbol_type_lookup）には載せず、ここで直接分類する。
    if c == crate::bracket::INLINE_OPEN_TOKEN || c == crate::bracket::ASIDE_TOKEN {
        return CharType::SymbolOpen;
    }
    if c == crate::bracket::INLINE_CLOSE_TOKEN {
        return CharType::SymbolClose;
    }

    // --- 記号ルックアップ ---
    // Python は unicodedata.category() の P*/S* を先に判定する。
    // テーブルがその挙動を再現する。
    let st = table::symbol_type_lookup(c);
    if st != CharType::Other {
        return st;
    }

    // --- ひらがな (U+3041–U+309F) ---
    if (0x3041..=0x309F).contains(&cp) {
        return CharType::Hiragana;
    }

    // --- カタカナ (U+30A0–U+30FF) ---
    if (0x30A0..=0x30FF).contains(&cp) {
        return CharType::Katakana;
    }

    // --- 漢数字（〇一二三四五六七八九）---
    if is_japanese_numeric_char(c) {
        return CharType::JapaneseNumeric;
    }

    // --- CJK統合漢字 (U+4E00–U+9FFF) および拡張A (U+3400–U+4DBF) ---
    if (0x4E00..=0x9FFF).contains(&cp) || (0x3400..=0x4DBF).contains(&cp) {
        return CharType::Kanji;
    }

    // --- CJK統合漢字拡張B〜F・I (U+20000–U+2EE5F、サロゲートペア必須) ---
    // 拡張B (𠮟 U+20B9F 等、人名・地名用漢字を含む) が実用上重要。
    if (0x20000..=0x2EE5F).contains(&cp) {
        return CharType::Kanji;
    }

    // --- CJK統合漢字拡張G・H (U+30000–U+323AF、サロゲートペア必須) ---
    if (0x30000..=0x323AF).contains(&cp) {
        return CharType::Kanji;
    }

    // --- 漢字繰り返し符号 (U+3005 々, U+303B 〻) ---
    // CJK Symbols and Punctuation ブロック内のため範囲判定では拾えない。
    if matches!(c, '々' | '〻') {
        return CharType::Kanji;
    }

    // --- ラテン文字（ASCII・全角）---
    if c.is_ascii_alphabetic() {
        return CharType::Alpha;
    }
    if (0xFF21..=0xFF3A).contains(&cp) || (0xFF41..=0xFF5A).contains(&cp) {
        return CharType::Alpha;
    }

    CharType::Other
}

/// スキップすべきコードポイントか。
///
/// ゼロ幅文字・Unicode フォーマット制御文字など、
/// 仮名・点字出力には現れるべきでない文字を列挙する。
/// 可視スペース系（NBSP 等）は `is_whitespace()` で Space に分類されるため含まない。
fn is_skip_cp(cp: u32) -> bool {
    matches!(
        cp,
        0x00AD // SOFT HYPHEN
        | 0x200B // ZERO WIDTH SPACE
        | 0x200C // ZERO WIDTH NON-JOINER
        | 0x200D // ZERO WIDTH JOINER
        | 0x2060 // WORD JOINER
        | 0x2061 // FUNCTION APPLICATION
        | 0x2062 // INVISIBLE TIMES
        | 0x2063 // INVISIBLE SEPARATOR
        | 0x2064 // INVISIBLE PLUS
        | 0xFEFF // ZERO WIDTH NO-BREAK SPACE (BOM)
    )
}

/// 常に [`CharType::JapaneseNumeric`] となる漢数字。
fn is_japanese_numeric_char(c: char) -> bool {
    matches!(
        c,
        '〇' | '一' | '二' | '三' | '四' | '五' | '六' | '七' | '八' | '九'
    )
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_values_match_cpp() {
        // C++ 版 char_type.hpp と同じ値であることを保証する。
        // モデルファイル (.mbm) のバイト互換性に直結するため、絶対に変えない。
        assert_eq!(CharType::Space as u8, 0x00);
        assert_eq!(CharType::Alpha as u8, 0x10);
        assert_eq!(CharType::Numeric as u8, 0x11);
        assert_eq!(CharType::Symbol as u8, 0x30);
        assert_eq!(CharType::SymbolClose as u8, 0x31);
        assert_eq!(CharType::SymbolOpen as u8, 0x32);
        assert_eq!(CharType::SymbolStop as u8, 0x33);
        assert_eq!(CharType::SymbolPause as u8, 0x34);
        assert_eq!(CharType::Hiragana as u8, 0x40);
        assert_eq!(CharType::Katakana as u8, 0x41);
        assert_eq!(CharType::Kanji as u8, 0x42);
        assert_eq!(CharType::JapaneseNumeric as u8, 0x43);
        assert_eq!(CharType::Other as u8, 0xFF);
    }

    #[test]
    fn category_helpers() {
        assert!(CharType::Hiragana.is_japanese());
        assert!(CharType::Kanji.is_japanese());
        assert!(!CharType::Alpha.is_japanese());

        assert!(CharType::SymbolStop.is_symbol());
        assert!(!CharType::Hiragana.is_symbol());

        assert!(CharType::Alpha.is_latin());
        assert!(CharType::Numeric.is_latin());
        assert!(!CharType::Symbol.is_latin());
    }

    #[test]
    fn space_is_bypass() {
        assert!(CharType::Space.is_bypass());
        assert!(!CharType::Numeric.is_bypass());
        assert!(!CharType::Hiragana.is_bypass());
    }

    #[test]
    fn other_is_bypass() {
        assert!(CharType::Other.is_bypass());
    }

    #[test]
    fn stop_and_pause_skip_boundary_check() {
        // それ自体が既に文の区切りとして機能するため、直後のマスあけ判定は
        // 重ねてモデルに問い合わせない。
        assert!(CharType::SymbolStop.skips_boundary_check());
        assert!(CharType::SymbolPause.skips_boundary_check());
    }

    #[test]
    fn space_and_other_skip_boundary_check() {
        assert!(CharType::Space.skips_boundary_check());
        assert!(CharType::Other.skips_boundary_check());
    }

    #[test]
    fn symbol_and_brackets_do_not_skip_boundary_check() {
        // 括弧類（SymbolOpen/SymbolClose）を含む一般記号は境界としての意味合いが
        // 薄く、直後に語が続くかは文脈次第（例:「」の直後が助詞なら分割しない・
        // 体言が続くなら分割する）なので、読みはbypassしつつ境界はモデルに
        // 判定させる（ALPHA・カタカナと同型）。
        assert!(!CharType::Symbol.skips_boundary_check());
        assert!(!CharType::SymbolOpen.skips_boundary_check());
        assert!(!CharType::SymbolClose.skips_boundary_check());
    }

    #[test]
    fn non_bypass_types_do_not_skip_boundary_check() {
        // is_bypass()がfalseの型は別経路(ALPHA/カタカナ専用分岐や通常LR分岐)で
        // 境界判定されるため、この述語の対象外(falseで一貫)。
        assert!(!CharType::Hiragana.skips_boundary_check());
        assert!(!CharType::Kanji.skips_boundary_check());
        assert!(!CharType::Alpha.skips_boundary_check());
    }

    #[test]
    fn skip_chars() {
        assert_eq!(get_char_type('\u{200B}'), CharType::Skip); // ZERO WIDTH SPACE
        assert_eq!(get_char_type('\u{200D}'), CharType::Skip); // ZWJ
        assert_eq!(get_char_type('\u{2060}'), CharType::Skip); // WORD JOINER
        assert_eq!(get_char_type('\u{FEFF}'), CharType::Skip); // BOM
        assert!(CharType::Skip.is_skip());
        assert!(!CharType::Space.is_skip());
        assert!(!CharType::Other.is_skip());
    }

    #[test]
    fn space_chars() {
        assert_eq!(get_char_type(' '), CharType::Space);
        assert_eq!(get_char_type('\t'), CharType::Space);
        assert_eq!(get_char_type('\n'), CharType::Space);
        assert_eq!(get_char_type('\u{00A0}'), CharType::Space); // NO-BREAK SPACE
        assert_eq!(get_char_type('\u{2003}'), CharType::Space); // EM SPACE
        assert_eq!(get_char_type('\u{202F}'), CharType::Space); // NARROW NO-BREAK SPACE
        assert_eq!(get_char_type('\u{205F}'), CharType::Space); // MEDIUM MATHEMATICAL SPACE
        assert_eq!(get_char_type('\u{3000}'), CharType::Space); // 全角スペース
    }

    #[test]
    fn numeric_chars() {
        assert_eq!(get_char_type('0'), CharType::Numeric);
        assert_eq!(get_char_type('5'), CharType::Numeric);
        assert_eq!(get_char_type('9'), CharType::Numeric);
        assert_eq!(get_char_type('０'), CharType::Numeric); // 全角
        assert_eq!(get_char_type('９'), CharType::Numeric);
    }

    #[test]
    fn alpha_chars() {
        assert_eq!(get_char_type('a'), CharType::Alpha);
        assert_eq!(get_char_type('Z'), CharType::Alpha);
        assert_eq!(get_char_type('Ａ'), CharType::Alpha); // 全角
        assert_eq!(get_char_type('ｚ'), CharType::Alpha);
    }

    #[test]
    fn hiragana_katakana() {
        assert_eq!(get_char_type('あ'), CharType::Hiragana);
        assert_eq!(get_char_type('ん'), CharType::Hiragana);
        assert_eq!(get_char_type('ア'), CharType::Katakana);
        assert_eq!(get_char_type('ン'), CharType::Katakana);
        assert_eq!(get_char_type('ー'), CharType::Katakana); // 長音記号
    }

    #[test]
    fn kanji_basic() {
        assert_eq!(get_char_type('漢'), CharType::Kanji);
        assert_eq!(get_char_type('字'), CharType::Kanji);
        assert_eq!(get_char_type('百'), CharType::Kanji); // 位取り文字はここでは Kanji
        assert_eq!(get_char_type('花'), CharType::Kanji);
    }

    #[test]
    fn japanese_numeric() {
        for c in "〇一二三四五六七八九".chars() {
            assert_eq!(get_char_type(c), CharType::JapaneseNumeric, "char: {c}");
        }
    }

    #[test]
    fn symbol_subcategories() {
        // 文末記号
        assert_eq!(get_char_type('。'), CharType::SymbolStop);
        assert_eq!(get_char_type('！'), CharType::SymbolStop);
        assert_eq!(get_char_type('?'), CharType::SymbolStop);
        assert_eq!(get_char_type('.'), CharType::SymbolStop);

        // 読点・中点
        assert_eq!(get_char_type('、'), CharType::SymbolPause);
        assert_eq!(get_char_type('・'), CharType::SymbolPause);
        assert_eq!(get_char_type(','), CharType::SymbolPause);

        // 開き括弧
        assert_eq!(get_char_type('「'), CharType::SymbolOpen);
        assert_eq!(get_char_type('('), CharType::SymbolOpen);
        assert_eq!(get_char_type('（'), CharType::SymbolOpen);

        // 閉じ括弧
        assert_eq!(get_char_type('」'), CharType::SymbolClose);
        assert_eq!(get_char_type(')'), CharType::SymbolClose);
        assert_eq!(get_char_type('）'), CharType::SymbolClose);

        // その他の記号
        assert_eq!(get_char_type('+'), CharType::Symbol);
        assert_eq!(get_char_type('='), CharType::Symbol);
    }

    #[test]
    fn bracket_identity_tokens_classify_as_symbol_open_close() {
        assert_eq!(
            get_char_type(crate::bracket::INLINE_OPEN_TOKEN),
            CharType::SymbolOpen
        );
        assert_eq!(
            get_char_type(crate::bracket::INLINE_CLOSE_TOKEN),
            CharType::SymbolClose
        );
        assert_eq!(
            get_char_type(crate::bracket::ASIDE_TOKEN),
            CharType::SymbolOpen
        );
    }

    #[test]
    fn kanji_iteration_marks() {
        assert_eq!(get_char_type('々'), CharType::Kanji); // U+3005
        assert_eq!(get_char_type('〻'), CharType::Kanji); // U+303B
    }

    #[test]
    fn other() {
        // 絵文字 (BMP外) は OTHER
        assert_eq!(get_char_type('🐕'), CharType::Other);
    }

    #[test]
    fn kanji_extension_b() {
        // 𠮟る の「𠮟」(U+20B9F、人名・地名用漢字) は拡張B
        assert_eq!(get_char_type('\u{20B9F}'), CharType::Kanji);
    }

    #[test]
    fn kanji_extension_g_h() {
        assert_eq!(get_char_type('\u{30000}'), CharType::Kanji); // 拡張G 先頭
        assert_eq!(get_char_type('\u{323AF}'), CharType::Kanji); // 拡張H 末尾
    }

    #[test]
    fn kanji_extension_gap_is_other() {
        // 拡張B〜F・I の直後 (U+2EE60) と 拡張G直前 (U+2FFFF) は非漢字ブロック
        assert_eq!(get_char_type('\u{2EE60}'), CharType::Other);
        assert_eq!(get_char_type('\u{2FFFF}'), CharType::Other);
    }
}
