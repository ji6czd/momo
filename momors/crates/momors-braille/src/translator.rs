//! 行単位の言語ルーティング。
//!
//! **入力はすでに読みに変換されたテキスト**（日本語なら かな）。漢字かな交じり文の
//! 予測（`momors_core::Predictor`）は上位の責務で、momors-braille は漢字列を扱わない。
//!
//! **行に日本語が1文字も含まれなければ英語（UEB）として点訳し、含まれれば従来どおり
//! 日本語として点訳する。** 実務でも使われる分け方で、「行」という自然な境界を使うため
//! 島の検出が要らない。英語の縮約有無は英語テーブル（grade1/grade2）で決まる。
//!
//! 日本語行の中に埋め込まれた英語は従来どおり外字符 `⠰` ＋無縮約で書かれる——これは
//! 日本語点字の慣行に合致する。上位は [`detect_language`] を先に呼べば、英語行に対して
//! 予測を走らせずに [`BrailleTranslator::translate_english`] へ直接渡せる。
//!
//! **英語エンジンは省略可能**（[`BrailleTranslator::japanese_only`]）。無ければ英語行も
//! 日本語テーブルへ流す。no_conversion（生の1:1・大小保持の計算機点字）のように言語別
//! 処理が不要なときに使う。
//!
//! ```text
//! ＮＨＫ                    → 英語   ⠠⠠⠝⠓⠅           （外字符なし・UEB）
//! NHKラジオ                → 日本語 ⠰⠠⠠⠝⠓⠅⠀⠑⠐⠳⠊  （外字符あり・従来）
//! you should be here today → 英語   ⠽⠀⠩⠙⠀⠆⠀⠐⠓⠀⠞⠙ （grade2 縮約）
//! ```

use crate::english_translator::{EnglishResult, EnglishTranslator};
use crate::japanese_translator::{JapaneseResult, JapaneseTranslator};
use crate::Result;

/// 行の言語。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    /// 日本語文字を1つ以上含む行。従来の日本語点訳を行う。
    Japanese,
    /// 日本語文字を含まない行。UEB grade 2 で点訳する。
    English,
}

/// 文字が日本語か。ひらがな・カタカナ（半角含む）・漢字・和文約物・全角形。
pub fn is_japanese_char(c: char) -> bool {
    matches!(c as u32,
        0x3000..=0x303F   // 和文約物（　、。「」）
        | 0x3040..=0x309F // ひらがな
        | 0x30A0..=0x30FF // カタカナ
        | 0x31F0..=0x31FF // カタカナ拡張
        | 0x3400..=0x4DBF // CJK 統合漢字拡張A
        | 0x4E00..=0x9FFF // CJK 統合漢字
        | 0xF900..=0xFAFF // CJK 互換漢字
        | 0xFF01..=0xFF60 // 全角形（Ａ１！）
        | 0xFF66..=0xFF9F // 半角カナ
    )
}

/// 行の言語を判定する。日本語文字が1つでもあれば [`Language::Japanese`]。
pub fn detect_language(line: &str) -> Language {
    if line.chars().any(is_japanese_char) {
        Language::Japanese
    } else {
        Language::English
    }
}

// ============================================================
// BrailleResult
// ============================================================

/// 1行（または1まとまり）の点訳結果。
///
/// 2層 **テキスト(text) / 点字(braille)** を保持し、層間のインデックス対応
/// （コードポイント単位）を日本語・英語で**同じ形**で提供する。text は点訳器へ渡した
/// テキストそのもの（日本語=かな / 英語=英文）。漢字かな交じりの原文との対応は上位が
/// `momors_core::PredictionResult` 側で持ち、必要なら合成する。
pub struct BrailleResult {
    language: Language,
    text: String,
    braille_text: String,
    /// テキスト文字 → 点字文字（先頭セル位置）。
    text_to_braille: Vec<usize>,
    /// 点字文字 → テキスト文字（カーソル対応づけ用）。
    braille_to_text: Vec<usize>,
}

impl BrailleResult {
    /// 日本語（かな）の結果を組み立てる。
    fn from_japanese(kana: &str, jp: JapaneseResult) -> Self {
        Self::new(
            Language::Japanese,
            kana,
            jp.braille_text(),
            jp.kana_to_braille(),
        )
    }

    /// 英語の結果を組み立てる。
    fn from_english(text: &str, en: EnglishResult) -> Self {
        Self::new(
            Language::English,
            text,
            en.braille_text(),
            en.src_to_braille(),
        )
    }

    fn new(language: Language, text: &str, braille_text: &str, text_to_braille: &[usize]) -> Self {
        let braille_count = braille_text.chars().count();
        Self {
            language,
            text: text.to_owned(),
            braille_text: braille_text.to_owned(),
            text_to_braille: text_to_braille.to_vec(),
            braille_to_text: invert_text_to_braille(text_to_braille, braille_count),
        }
    }

    /// この行がどちらの経路で点訳されたか。
    pub fn language(&self) -> Language {
        self.language
    }

    /// 点訳したテキスト（日本語=かな / 英語=英文）。
    pub fn text(&self) -> &str {
        &self.text
    }

    /// 点訳された点字テキスト。
    pub fn braille_text(&self) -> &str {
        &self.braille_text
    }

    /// テキストの文字数（コードポイント）。
    pub fn text_char_count(&self) -> usize {
        self.text_to_braille.len()
    }

    /// 点字の文字数（コードポイント）。
    pub fn braille_char_count(&self) -> usize {
        self.braille_to_text.len()
    }

    /// テキスト文字 → 点字文字（先頭セル位置）。
    pub fn text_to_braille(&self) -> &[usize] {
        &self.text_to_braille
    }

    /// 点字文字 → テキスト文字（カーソル対応づけ用）。
    pub fn braille_to_text(&self) -> &[usize] {
        &self.braille_to_text
    }
}

// ============================================================
// BrailleTranslator
// ============================================================

/// 点訳の**唯一の入口**。行単位で日本語／英語を振り分けて点字に変換する。
///
/// 入力は**読みに変換済みのテキスト**（日本語なら かな）。日本語（かな→点字）は
/// [`JapaneseTranslator`]、英語（UEB）は [`EnglishTranslator`] に委譲する。
/// 逆変換は [`BrailleBackTranslator`](crate::BrailleBackTranslator)。
/// cli / ffi / pyo3 / wasm はこれを使う。
///
/// **英語エンジンは省略可能**（`english: None`）。無い場合は英語行も日本語テーブルへ
/// 流す。「no conversion（生の1:1）」のように言語別処理が不要なときは、日本語側に
/// no_conversion テーブルを置いて英語を `None` にすれば、全行が同じテーブルで点訳される
/// （UEB の小文字化・約物欠落を避けられる。詳細は [`japanese_only`](Self::japanese_only)）。
pub struct BrailleTranslator {
    japanese: JapaneseTranslator,
    english: Option<EnglishTranslator>,
}

impl BrailleTranslator {
    /// 組み込みテーブル（日本語１級 + UEB grade 2）で作る。
    pub fn from_embedded() -> Result<Self> {
        Ok(Self::new(
            JapaneseTranslator::from_embedded()?,
            Some(EnglishTranslator::from_embedded()?),
        ))
    }

    /// 変換器を指定して作る。`english` が `None` なら英語行も日本語テーブルで点訳する。
    pub fn new(japanese: JapaneseTranslator, english: Option<EnglishTranslator>) -> Self {
        Self { japanese, english }
    }

    /// 英語エンジンを持たない変換器（全行を日本語テーブルで点訳）。
    ///
    /// no_conversion のように言語別処理が不要なときに使う。英語行も UEB に回さず、
    /// 日本語テーブル（例: `japanese_no_conversion`＝大小保持の計算機点字）で 1:1 変換する。
    pub fn japanese_only(japanese: JapaneseTranslator) -> Self {
        Self::new(japanese, None)
    }

    /// 日本語点字変換器への参照。
    pub fn japanese(&self) -> &JapaneseTranslator {
        &self.japanese
    }

    /// 英語点字変換器への参照（無ければ `None`）。
    pub fn english(&self) -> Option<&EnglishTranslator> {
        self.english.as_ref()
    }

    /// 1行を**言語判定して**点訳する。
    ///
    /// `text` は読みに変換済み（日本語なら かな）であること。英語行でも英語エンジンが
    /// 無ければ日本語テーブルで点訳する。
    pub fn translate(&self, text: &str) -> Result<BrailleResult> {
        match detect_language(text) {
            Language::English => match self.translate_english(text) {
                Some(r) => Ok(r),
                None => self.translate_japanese(text),
            },
            Language::Japanese => self.translate_japanese(text),
        }
    }

    /// 言語判定せず、**必ず日本語として**点訳する（英字は外字符 `⠰` ＋無縮約）。
    /// ルーティングを使わない従来どおりの経路。
    pub fn translate_japanese(&self, kana: &str) -> Result<BrailleResult> {
        let jp = self.japanese.translate(kana)?;
        Ok(BrailleResult::from_japanese(kana, jp))
    }

    /// 言語判定せず、**必ず英語（UEB）として**点訳する。
    /// 英語エンジンを持たない場合は `None`。
    pub fn translate_english(&self, text: &str) -> Option<BrailleResult> {
        self.english
            .as_ref()
            .map(|en| BrailleResult::from_english(text, en.translate(text)))
    }
}

/// `text_to_braille`（テキスト→点字先頭位置）を反転して 点字→テキスト を作る。
///
/// 縮約や複合音で複数のテキスト文字が同じ点字位置を指す場合、その点字セルは**最初の**
/// テキスト文字に帰属する（`this` → `⠹` なら `t`）。どのテキスト文字にも直接指されない
/// 点字位置（フラグ・遷移スペースなど）は直前の帰属を引き継ぐ。
fn invert_text_to_braille(text_to_braille: &[usize], braille_char_count: usize) -> Vec<usize> {
    let mut out = vec![usize::MAX; braille_char_count];
    for (i, &p) in text_to_braille.iter().enumerate() {
        if p < braille_char_count && out[p] == usize::MAX {
            out[p] = i;
        }
    }
    let mut last = 0;
    for slot in out.iter_mut() {
        if *slot == usize::MAX {
            *slot = last;
        } else {
            last = *slot;
        }
    }
    out
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn lt() -> BrailleTranslator {
        BrailleTranslator::from_embedded().expect("組み込みテーブルをロードできること")
    }

    #[test]
    fn detects_japanese_lines() {
        assert_eq!(detect_language("コンニチワ"), Language::Japanese);
        assert_eq!(detect_language("ワガハイワ ネコデアル"), Language::Japanese);
        assert_eq!(detect_language("NHKラジオ"), Language::Japanese);
        assert_eq!(detect_language("Hello.コンニチワ。"), Language::Japanese);
        assert_eq!(
            detect_language("ＡＢＣ"),
            Language::Japanese,
            "全角形は日本語扱い"
        );
        assert_eq!(detect_language("ｶﾀｶﾅ"), Language::Japanese, "半角カナ");
    }

    #[test]
    fn detects_english_lines() {
        assert_eq!(detect_language("Hello world."), Language::English);
        assert_eq!(detect_language("NHK"), Language::English);
        assert_eq!(detect_language("3.14"), Language::English);
        assert_eq!(detect_language(""), Language::English, "空行");
    }

    #[test]
    fn english_line_uses_ueb_without_foreign_indicator() {
        // 英語行なので外字符 ⠰ は付かない（日本語行の埋め込み英語とは別扱い）
        let r = lt().translate("NHK").expect("点訳できること");
        assert_eq!(r.braille_text(), "⠠⠠⠝⠓⠅");
        assert_eq!(r.language(), Language::English);
    }

    #[test]
    fn japanese_line_keeps_foreign_indicator() {
        // かな混じりの行は日本語経路。埋め込み英語は外字符 ⠰ ＋無縮約。
        let r = lt().translate("NHKラジオ").expect("点訳できること");
        assert_eq!(r.braille_text(), "⠰⠠⠠⠝⠓⠅⠀⠑⠐⠳⠊");
        assert_eq!(r.language(), Language::Japanese);
    }

    #[test]
    fn compose_japanese_grade1_with_english_grade1() {
        // 日本語＝１級、英語＝UEB grade 1（無縮約）を名前で選んで合成する。
        let bt = BrailleTranslator::new(
            JapaneseTranslator::from_embedded_name("japanese_grade1").unwrap(),
            Some(EnglishTranslator::from_embedded_name("english_ueb_grade1").unwrap()),
        );
        // grade 1 なので縮約されない: "and the" = a n d / t h e。
        let r = bt.translate_english("and the").unwrap();
        assert_eq!(r.braille_text(), "⠁⠝⠙⠀⠞⠓⠑");
        // 参考: grade 2 なら縮約されて ⠯⠀⠮。
        let g2 = lt().translate_english("and the").unwrap();
        assert_eq!(g2.braille_text(), "⠯⠀⠮");
    }

    #[test]
    fn english_none_falls_back_to_japanese_table() {
        // english=None なら英語行も日本語テーブルへ回す（no conversion 用途）。
        let with_en = BrailleTranslator::from_embedded().unwrap();
        let a = with_en.translate("the").expect("点訳できること");
        assert_eq!(a.language(), Language::English);
        assert_eq!(a.braille_text(), "⠮"); // UEB 縮約

        // english=None（日本語 no_conversion）: 同じ英語行が日本語経路へ。
        let jp_only = BrailleTranslator::japanese_only(
            JapaneseTranslator::from_embedded_name("japanese_no_conversion").unwrap(),
        );
        let b = jp_only.translate("the").expect("点訳できること");
        assert_eq!(
            b.language(),
            Language::Japanese,
            "english が None なら英語行も日本語経路"
        );
        assert_ne!(
            b.braille_text(),
            a.braille_text(),
            "no_conversion は UEB と別出力（縮約しない）"
        );
    }

    #[test]
    fn english_line_is_contracted() {
        let r = lt().translate("you should be here today").unwrap();
        assert_eq!(r.braille_text(), "⠽⠀⠩⠙⠀⠆⠀⠐⠓⠀⠞⠙");
    }

    #[test]
    fn english_empty_line() {
        let r = lt().translate("").unwrap();
        assert_eq!(r.braille_text(), "");
        assert!(r.braille_to_text().is_empty());
    }

    // --- 2層インデックス（text ↔ braille） ---

    #[test]
    fn english_braille_to_text_maps_contraction_to_first_char() {
        // "this" → ⠹（1セル）。そのセルは先頭の 't'（index 0）に帰属する。
        let r = lt().translate("this").unwrap();
        assert_eq!(r.braille_text(), "⠹");
        assert_eq!(r.text(), "this");
        assert_eq!(r.text_to_braille(), &[0, 0, 0, 0]);
        assert_eq!(r.braille_to_text(), &[0]);
    }

    #[test]
    fn english_braille_to_text_sentence() {
        // "go on" → ⠛⠀⠕⠝
        let r = lt().translate("go on").unwrap();
        assert_eq!(r.braille_text(), "⠛⠀⠕⠝");
        assert_eq!(r.text_char_count(), 5);
        assert_eq!(r.braille_char_count(), 4);
        // ⠛→'g'(0), ⠀→' '(2), ⠕→'o'(3), ⠝→'n'(4)
        assert_eq!(r.braille_to_text(), &[0, 2, 3, 4]);
    }

    #[test]
    fn japanese_two_layers_are_consistent() {
        // かな→点字の対応が、逆写像（点字→かな）と整合すること。
        let r = lt().translate("キャア").expect("点訳できること");
        assert_eq!(r.language(), Language::Japanese);
        assert_eq!(r.braille_text(), "⠈⠡⠁");
        // キ・ャ は同じ点字位置（複合音）、ア は次のセル。
        assert_eq!(r.text_to_braille(), &[0, 0, 2]);
        // 点字セルは先頭のかな文字に帰属する（⠈⠡ は キ、⠁ は ア）。
        assert_eq!(r.braille_to_text(), &[0, 0, 2]);
    }

    #[test]
    fn japanese_flag_cells_belong_to_triggering_char() {
        // "1レツ" → ⠼⠁⠤⠛⠝。開始フラグ ⠼ は '1' に、終了フラグ ⠤ は直前（'1'）に帰属する。
        let r = lt().translate("1レツ").expect("点訳できること");
        assert_eq!(r.braille_text(), "⠼⠁⠤⠛⠝");
        assert_eq!(r.text_to_braille(), &[0, 3, 4]);
        assert_eq!(r.braille_to_text(), &[0, 0, 0, 1, 2]);
    }
}
