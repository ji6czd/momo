//! JAPANESE_NUMERIC ルールベース変換 (フォールバック)。
//!
//! C++ 版 `predictor.cpp` の `digit_table()` / `kurai_fallback()` /
//! `convert_japanese_numeric()` に対応する。
//!
//! 読みモデルの自信度が `numeric_confidence_threshold` を下回ったときに、
//! ルールベースで漢数字・位取り文字を変換するために使う。
//!
//! ## 変換ルール
//!
//! ### 漢数字 ([`digit_table`])
//!
//! | 文字 | 出力 |
//! |---|---|
//! | 〇 | "0" |
//! | 一 / 壱 | "1" |
//! | 二 / 弐 | "2" |
//! | 三 / 参 | "3" |
//! | 四〜九 | "4"〜"9" |
//!
//! ### 位取り文字 ([`kurai_fallback`])
//!
//! 「左右に JapaneseNumeric が隣接するか」で出力が変わる:
//!
//! - `十`: 単独 → "10"、左のみ数字 → "0"、右のみ数字 → "1"、両側数字 → SKIP
//! - `百`: 単独 → "100"、左のみ数字 → "00"、右のみ数字 → "1"、両側数字 → "0"
//! - `千`: 単独 → "セン" (ただし左に「三」があれば "ゼン")、右に数字 → "0"
//! - `万` → "マン"、`億` → "オク"、`兆` → "チョー"
//!
//! ## 文字定数
//!
//! Unicode コードポイントを直接 `u32` リテラルで書いている (例: `0x4E09` = 「三」)。
//! `char` リテラルで書くこともできるが、ホットパスでの比較速度と、
//! C++ 版との対応が見やすくなることを優先した。

use crate::char_type::CharType;
use crate::featurize::SourceEntry;

// ============================================================
// 公開型
// ============================================================

/// JAPANESE_NUMERIC ルールベース変換の結果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NumericFallback {
    /// この文字は出力せずスキップする (例: 「三百二十」の「十」)。
    Skip,
    /// 指定の文字列を出力する。
    Output(&'static str),
}

// ============================================================
// 公開関数
// ============================================================

/// 数字漢字の訓読み（みっか・ふたり・ここのつ 等の助数詞語の第1要素）。
///
/// これらは数ではなく訓読みの助数詞語であり、自信度が低くても数字化してはならない。
/// ホワイトリストは読み（表層形ではない）に張るので、十三日・九十三日間のような
/// 多桁ケースは LR が音読み（サン等）を出して自然に弾かれ、列挙は不要。
fn is_kun_numeral_reading(label: &str) -> bool {
    matches!(
        label,
        "ヒト"      // 一つ・一人
            | "ツイ"  // 一日(ついたち)
            | "フツ"  // 二日(ふつか)
            | "フタ"  // 二つ・二人
            | "ミッ"  // 三日・三つ
            | "ヨッ"  // 四日・四つ
            | "ヨ"    // 四人(よにん)
            | "イツ"  // 五日・五つ
            | "ムイ"  // 六日(むいか)
            | "ムッ"  // 六つ
            | "ナノ"  // 七日(なのか)
            | "ナナ"  // 七つ
            | "ヨー"  // 八日(ようか)
            | "ヤッ"  // 八つ
            | "ココノ" // 九日・九つ
            | "トオ" // 十日
    )
}

/// `seq[i]` が訓読み助数詞として保護対象か（数字化を抑止すべきか）を判定する。
///
/// LR の読み `label` が数字漢字の訓読みで、かつ多桁数の一部（左右が漢数字）でなければ
/// `true`。Python 版 `_convert_japanese_numeric` の訓読み保護に対応する。
pub(crate) fn is_protected_kun_numeral(label: &str, i: usize, seq: &[SourceEntry]) -> bool {
    if !is_kun_numeral_reading(label) {
        return false;
    }
    let n = seq.len();
    let left_is_numeric = i > 0 && seq[i - 1].ctype == CharType::JapaneseNumeric;
    let right_is_numeric = i + 1 < n && seq[i + 1].ctype == CharType::JapaneseNumeric;
    !left_is_numeric && !right_is_numeric
}

/// JAPANESE_NUMERIC として認識された `seq[i]` をフォールバック変換する。
///
/// 前提: `seq[i].ctype == CharType::JapaneseNumeric` であること。
///
/// 漢数字 (〇一二三四五六七八九 / 壱弐参) は無条件で ASCII 数字に変換、
/// 位取り文字 (十百千万億兆) は左右の文脈に応じて変換する。
pub(crate) fn convert_japanese_numeric(i: usize, seq: &[SourceEntry]) -> NumericFallback {
    let cp = seq[i].cp;
    let n = seq.len();
    let left_cp = if i > 0 { seq[i - 1].cp } else { 0 };
    let left_is_numeric = i > 0 && seq[i - 1].ctype == CharType::JapaneseNumeric;
    let right_is_numeric = i + 1 < n && seq[i + 1].ctype == CharType::JapaneseNumeric;

    // まず漢数字テーブルを引く
    if let Some(d) = digit_table(cp) {
        return NumericFallback::Output(d);
    }

    // 漢数字でなければ位取り文字フォールバックへ
    kurai_fallback(cp, left_cp, left_is_numeric, right_is_numeric)
}

// ============================================================
// digit_table
// ============================================================

/// 漢数字 (〇一二三四五六七八九 / 壱弐参) を ASCII 数字 1 文字に変換する。
///
/// 該当しなければ `None`。位取り文字 (十百千万億兆) はここでは扱わない。
fn digit_table(cp: u32) -> Option<&'static str> {
    Some(match cp {
        0x3007 => "0", // 〇
        0x4E00 => "1", // 一
        0x4E8C => "2", // 二
        0x4E09 => "3", // 三
        0x56DB => "4", // 四
        0x4E94 => "5", // 五
        0x516D => "6", // 六
        0x4E03 => "7", // 七
        0x516B => "8", // 八
        0x4E5D => "9", // 九
        0x58F1 => "1", // 壱 (大字)
        0x5F10 => "2", // 弐 (大字)
        0x53C2 => "3", // 参 (大字)
        _ => return None,
    })
}

// ============================================================
// kurai_fallback
// ============================================================

/// 位取り文字 (十百千万億兆) のフォールバック変換。
///
/// 漢数字でも位取り文字でもない文字 (本来到達しないはず) には
/// [`NumericFallback::Skip`] を返す。
fn kurai_fallback(
    cp: u32,
    left_cp: u32,
    left_is_numeric: bool,
    right_is_numeric: bool,
) -> NumericFallback {
    use NumericFallback::{Output, Skip};

    match cp {
        // 十
        0x5341 => match (left_is_numeric, right_is_numeric) {
            (true, true) => Skip,           // 例: 「三百二十」の「十」
            (true, false) => Output("0"),   // 例: 「二十」
            (false, true) => Output("1"),   // 例: 「十一」 ※ to_source_seq で右→左パスで昇格
            (false, false) => Output("10"), // 単独の「十」
        },

        // 百
        0x767E => match (left_is_numeric, right_is_numeric) {
            (true, true) => Output("0"),
            (true, false) => Output("00"),
            (false, true) => Output("1"),
            (false, false) => Output("100"),
        },

        // 千
        0x5343 => {
            if right_is_numeric {
                Output("0")
            } else if left_cp == 0x4E09 {
                // 「三千」だけは「ゼン」と読む特例
                Output("ゼン")
            } else {
                Output("セン")
            }
        }

        // 単独語化する位取り文字
        0x4E07 => Output("マン"),   // 万
        0x5104 => Output("オク"),   // 億
        0x5146 => Output("チョー"), // 兆

        // 到達しないはず (JapaneseNumeric なのに漢数字でも位取り文字でもない)
        _ => Skip,
    }
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::featurize::to_source_seq;

    /// テキストから SourceEntry 列を作るヘルパ
    fn seq(text: &str) -> Vec<SourceEntry> {
        to_source_seq(text)
    }

    // --- digit_table ---

    #[test]
    fn digit_table_basic_kanji_numerals() {
        // 〇一二三四五六七八九
        let chars = ['〇', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
        let expected = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
        for (c, e) in chars.iter().zip(expected.iter()) {
            assert_eq!(digit_table(*c as u32), Some(*e), "char: {c}");
        }
    }

    #[test]
    fn digit_table_daiji() {
        // 大字
        assert_eq!(digit_table('壱' as u32), Some("1"));
        assert_eq!(digit_table('弐' as u32), Some("2"));
        assert_eq!(digit_table('参' as u32), Some("3"));
    }

    #[test]
    fn digit_table_kurai_returns_none() {
        // 位取り文字は digit_table では拾わない
        for c in "十百千万億兆".chars() {
            assert_eq!(digit_table(c as u32), None, "char: {c}");
        }
    }

    #[test]
    fn digit_table_non_numeric_returns_none() {
        assert_eq!(digit_table('あ' as u32), None);
        assert_eq!(digit_table('漢' as u32), None);
        assert_eq!(digit_table(0), None);
    }

    // --- convert_japanese_numeric: 漢数字 ---

    #[test]
    fn convert_simple_digits() {
        // 「三」だけ
        let s = seq("三");
        assert_eq!(
            convert_japanese_numeric(0, &s),
            NumericFallback::Output("3")
        );
    }

    // --- convert_japanese_numeric: 十 ---

    #[test]
    fn convert_juu_alone() {
        // 単独の「十」は単体としては JapaneseNumeric にならないので、
        // 直接 seq に放り込んでテストする
        let entries = vec![SourceEntry {
            cp: 0x5341,
            cp2: 0,
            cp3: 0,
            orig_idx: 0,
            ctype: CharType::JapaneseNumeric,
            compound_len: 1,
        }];
        assert_eq!(
            convert_japanese_numeric(0, &entries),
            NumericFallback::Output("10")
        );
    }

    #[test]
    fn convert_juu_left_numeric() {
        // 「二十」: 二(JapaneseNumeric) - 十
        let s = seq("二十");
        assert_eq!(
            convert_japanese_numeric(1, &s),
            NumericFallback::Output("0")
        );
    }

    #[test]
    fn convert_juu_right_numeric() {
        // 「十一」: 十 - 一(JapaneseNumeric) ← 十は右→左パスで昇格
        let s = seq("十一");
        assert_eq!(
            convert_japanese_numeric(0, &s),
            NumericFallback::Output("1")
        );
    }

    #[test]
    fn convert_juu_both_numeric() {
        // 「二十一」: 二 - 十 - 一
        let s = seq("二十一");
        assert_eq!(convert_japanese_numeric(1, &s), NumericFallback::Skip);
    }

    // --- convert_japanese_numeric: 百 ---

    #[test]
    fn convert_hyaku_alone() {
        let entries = vec![SourceEntry {
            cp: 0x767E,
            cp2: 0,
            cp3: 0,
            orig_idx: 0,
            ctype: CharType::JapaneseNumeric,
            compound_len: 1,
        }];
        assert_eq!(
            convert_japanese_numeric(0, &entries),
            NumericFallback::Output("100")
        );
    }

    #[test]
    fn convert_hyaku_combinations() {
        // 「三百」→「百」は "00"
        let s = seq("三百");
        assert_eq!(
            convert_japanese_numeric(1, &s),
            NumericFallback::Output("00")
        );

        // 「百一」→「百」は "1"
        let s = seq("百一");
        assert_eq!(
            convert_japanese_numeric(0, &s),
            NumericFallback::Output("1")
        );

        // 「三百二」→「百」は "0"
        let s = seq("三百二");
        assert_eq!(
            convert_japanese_numeric(1, &s),
            NumericFallback::Output("0")
        );
    }

    // --- convert_japanese_numeric: 千 ---

    #[test]
    fn convert_sen_alone() {
        // 単独の「千」: kurai 単体は昇格しないので直接 entry を作る
        let entries = vec![SourceEntry {
            cp: 0x5343,
            cp2: 0,
            cp3: 0,
            orig_idx: 0,
            ctype: CharType::JapaneseNumeric,
            compound_len: 1,
        }];
        assert_eq!(
            convert_japanese_numeric(0, &entries),
            NumericFallback::Output("セン")
        );
    }

    #[test]
    fn convert_sen_after_san() {
        // 「三千」→「千」は「ゼン」(連濁)
        let s = seq("三千");
        assert_eq!(
            convert_japanese_numeric(1, &s),
            NumericFallback::Output("ゼン")
        );
    }

    #[test]
    fn convert_sen_after_other() {
        // 「二千」→「千」は「セン」
        let s = seq("二千");
        assert_eq!(
            convert_japanese_numeric(1, &s),
            NumericFallback::Output("セン")
        );
    }

    #[test]
    fn convert_sen_with_right_numeric() {
        // 「千一」→「千」は "0"
        let s = seq("千一");
        assert_eq!(
            convert_japanese_numeric(0, &s),
            NumericFallback::Output("0")
        );
    }

    // --- convert_japanese_numeric: 万・億・兆 ---

    #[test]
    fn convert_man_oku_chou() {
        // 「三万」→「万」は「マン」
        let s = seq("三万");
        assert_eq!(
            convert_japanese_numeric(1, &s),
            NumericFallback::Output("マン")
        );

        // 「三億」→「億」は「オク」
        let s = seq("三億");
        assert_eq!(
            convert_japanese_numeric(1, &s),
            NumericFallback::Output("オク")
        );

        // 「三兆」→「兆」は「チョー」
        let s = seq("三兆");
        assert_eq!(
            convert_japanese_numeric(1, &s),
            NumericFallback::Output("チョー")
        );
    }

    // --- 連鎖ケース ---

    #[test]
    fn convert_san_sen_ni_hyaku_juu_ichi() {
        // 「三千二百十一」→ 全部 JapaneseNumeric として処理されたとき、
        // フォールバックは「桁としての展開」になる:
        //   三 → "3"     (digit_table)
        //   千 → "0"     (right_is_numeric なので桁の "0"。"ゼン"は単独/末尾のとき)
        //   二 → "2"     (digit_table)
        //   百 → "0"     (left_is_numeric && right_is_numeric)
        //   十 → Skip    (left_is_numeric && right_is_numeric)
        //   一 → "1"     (digit_table)
        // 合成結果: "3" + "0" + "2" + "0" + (skip) + "1" = "30201"
        //
        // これは数字の正しい読みではないが、フォールバックの目的 (モデルが低自信
        // 度のときに数字っぽく出す) としては想定通りの挙動。
        let s = seq("三千二百十一");
        assert_eq!(
            convert_japanese_numeric(0, &s),
            NumericFallback::Output("3")
        );
        assert_eq!(
            convert_japanese_numeric(1, &s),
            NumericFallback::Output("0")
        );
        assert_eq!(
            convert_japanese_numeric(2, &s),
            NumericFallback::Output("2")
        );
        assert_eq!(
            convert_japanese_numeric(3, &s),
            NumericFallback::Output("0")
        );
        assert_eq!(convert_japanese_numeric(4, &s), NumericFallback::Skip);
        assert_eq!(
            convert_japanese_numeric(5, &s),
            NumericFallback::Output("1")
        );
    }

    // --- is_protected_kun_numeral: 訓読み助数詞の保護 ---

    #[test]
    fn protect_mikka_standalone() {
        // 「三日(みっか)」の三: LR が訓読み「ミッ」を出し、左右が漢数字でない → 保護
        let s = seq("三日");
        assert!(is_protected_kun_numeral("ミッ", 0, &s));
    }

    #[test]
    fn protect_futari() {
        // 「二人(ふたり)」の二: 訓読み「フタ」、右は KANJI → 保護
        let s = seq("二人");
        assert!(is_protected_kun_numeral("フタ", 0, &s));
    }

    #[test]
    fn no_protect_when_left_numeric() {
        // 「十三日」の三: 左に十(漢数字) ＝多桁数の一部 → たとえ「ミッ」でも保護しない
        let s = seq("十三日");
        assert!(!is_protected_kun_numeral("ミッ", 1, &s));
    }

    #[test]
    fn no_protect_when_right_numeric() {
        // 「三〇」の三: 右に〇(漢数字) → 保護しない（数字化させる）
        let s = seq("三〇");
        assert!(!is_protected_kun_numeral("ミッ", 0, &s));
    }

    #[test]
    fn no_protect_on_yomi_reading() {
        // 音読み「サン」はホワイトリスト外 → 保護しない
        let s = seq("三日");
        assert!(!is_protected_kun_numeral("サン", 0, &s));
    }
}
