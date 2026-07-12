//! 行単位の言語ルーティング。
//!
//! **行に日本語が1文字も含まれなければ英語（UEB grade 2）として点訳し、
//! 含まれれば従来どおり日本語として点訳する。** 実務でも使われる分け方で、
//! 「行」という自然な境界を使うため島の検出が要らない。
//!
//! 英語行は**予測を通さない**（純英語なら `Predictor` は実質パススルーなので、
//! モデルを走らせる意味がない）。日本語行の中に埋め込まれた英語は従来どおり
//! 外字符 `⠰` ＋無縮約で書かれる——これは日本語点字の慣行に合致する。
//!
//! ```text
//! NHK                      → 英語   ⠠⠠⠝⠓⠅           （外字符なし・UEB）
//! NHKラジオ                → 日本語 ⠰⠠⠠⠝⠓⠅⠀⠑⠐⠳⠊  （外字符あり・従来）
//! you should be here today → 英語   ⠽⠀⠩⠙⠀⠆⠀⠐⠓⠀⠞⠙ （縮約）
//! ```

use crate::converter::BrailleConverter;
use crate::english::EnglishTranslator;
use crate::{Error, Result};
use momors_core::{PredictionResult, Predictor};

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
// LineResult
// ============================================================

/// 1行の点訳結果。
pub struct LineResult {
    braille_text: String,
    braille_to_source: Vec<usize>,
    language: Language,
    prediction: Option<PredictionResult>,
}

impl LineResult {
    /// 点訳された点字テキスト。
    pub fn braille_text(&self) -> &str {
        &self.braille_text
    }

    /// 点字文字インデックス → 原文文字インデックス（カーソル対応づけ用）。
    pub fn braille_to_source(&self) -> &[usize] {
        &self.braille_to_source
    }

    /// この行がどちらの経路で点訳されたか。
    pub fn language(&self) -> Language {
        self.language
    }

    /// 日本語行のときの予測結果（かな・確信度など）。英語行では `None`。
    pub fn prediction(&self) -> Option<&PredictionResult> {
        self.prediction.as_ref()
    }
}

// ============================================================
// LineTranslator
// ============================================================

/// 行単位で日本語／英語を振り分けて点訳する。
///
/// 振り分けを1箇所に集約するための型。cli / ffi / pyo3 / wasm はこれを使う。
pub struct LineTranslator {
    japanese: BrailleConverter,
    english: EnglishTranslator,
}

impl LineTranslator {
    /// 組み込みテーブル（日本語１級 + UEB grade 2）で作る。
    pub fn from_embedded() -> Result<Self> {
        Ok(Self::new(
            BrailleConverter::from_embedded()?,
            EnglishTranslator::from_embedded()?,
        ))
    }

    /// 変換器を指定して作る。
    pub fn new(japanese: BrailleConverter, english: EnglishTranslator) -> Self {
        Self { japanese, english }
    }

    /// 日本語点字変換器への参照。
    pub fn japanese(&self) -> &BrailleConverter {
        &self.japanese
    }

    /// 英語点字変換器への参照。
    pub fn english(&self) -> &EnglishTranslator {
        &self.english
    }

    /// 1行を**言語判定して**点訳する。日本語行のみ `predictor` を使う。
    pub fn translate_line(&self, line: &str, predictor: &Predictor) -> Result<LineResult> {
        match detect_language(line) {
            Language::English => Ok(self.translate_english(line)),
            Language::Japanese => self.translate_japanese(line, predictor),
        }
    }

    /// 言語判定せず、**必ず日本語として**点訳する（英字は外字符 `⠰` ＋無縮約）。
    /// ルーティングを使わない従来どおりの経路。
    pub fn translate_japanese(&self, line: &str, predictor: &Predictor) -> Result<LineResult> {
        let pred = predictor
            .predict(line)
            .map_err(|e| Error::Prediction(e.to_string()))?;
        let brl = self.japanese.convert(pred.kana_text())?;
        let count = brl.braille_text().chars().count();
        let braille_to_source = pred.braille_char_to_source(brl.kana_to_braille(), count);
        Ok(LineResult {
            braille_text: brl.braille_text().to_owned(),
            braille_to_source,
            language: Language::Japanese,
            prediction: Some(pred),
        })
    }

    /// 英語行として点訳する（予測を通さない）。判定を自分で済ませている呼び出し用。
    pub fn translate_english(&self, line: &str) -> LineResult {
        let r = self.english.translate(line);
        let count = r.braille_text().chars().count();
        let braille_to_source = invert_src_to_braille(r.src_to_braille(), count);
        LineResult {
            braille_text: r.braille_text().to_owned(),
            braille_to_source,
            language: Language::English,
            prediction: None,
        }
    }
}

/// `src_to_braille`（原文→点字先頭位置）を反転して 点字→原文 を作る。
///
/// 縮約で複数の原文文字が同じ点字位置を指す場合、その点字セルは**最初の**原文文字に帰属する
/// （`this` → `⠹` なら `t`）。どの原文文字にも直接指されない点字位置は直前の帰属を引き継ぐ。
fn invert_src_to_braille(src_to_braille: &[usize], braille_char_count: usize) -> Vec<usize> {
    let mut out = vec![usize::MAX; braille_char_count];
    for (i, &p) in src_to_braille.iter().enumerate() {
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

    fn lt() -> LineTranslator {
        LineTranslator::from_embedded().expect("組み込みテーブルをロードできること")
    }

    #[test]
    fn detects_japanese_lines() {
        assert_eq!(detect_language("こんにちは"), Language::Japanese);
        assert_eq!(detect_language("吾輩は猫である"), Language::Japanese);
        assert_eq!(detect_language("NHKラジオ"), Language::Japanese);
        assert_eq!(detect_language("Hello.こんにちは。"), Language::Japanese);
        assert_eq!(detect_language("これは、テストです。"), Language::Japanese);
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
        let r = lt().translate_english("NHK");
        assert_eq!(r.braille_text(), "⠠⠠⠝⠓⠅");
        assert_eq!(r.language(), Language::English);
        assert!(r.prediction().is_none());
    }

    #[test]
    fn english_line_is_contracted() {
        let r = lt().translate_english("you should be here today");
        assert_eq!(r.braille_text(), "⠽⠀⠩⠙⠀⠆⠀⠐⠓⠀⠞⠙");
    }

    #[test]
    fn english_empty_line() {
        let r = lt().translate_english("");
        assert_eq!(r.braille_text(), "");
        assert!(r.braille_to_source().is_empty());
    }

    #[test]
    fn english_braille_to_source_maps_contraction_to_first_char() {
        // "this" → ⠹（1セル）。そのセルは先頭の 't'（index 0）に帰属する。
        let r = lt().translate_english("this");
        assert_eq!(r.braille_text(), "⠹");
        assert_eq!(r.braille_to_source(), &[0]);
    }

    #[test]
    fn english_braille_to_source_sentence() {
        // "go on" → ⠛⠀⠕⠝
        let r = lt().translate_english("go on");
        assert_eq!(r.braille_text(), "⠛⠀⠕⠝");
        // ⠛→'g'(0), ⠀→' '(2), ⠕→'o'(3), ⠝→'n'(4)
        assert_eq!(r.braille_to_source(), &[0, 2, 3, 4]);
    }
}
