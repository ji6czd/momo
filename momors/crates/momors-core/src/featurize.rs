//! 特徴量計算モジュール。
//!
//! C++ 版の `momo_features.cpp` に対応する。
//! 入力テキストを [`SourceEntry`] 列に変換し、各文字に対する
//! [`FeatureKey`] 列を計算する。
//!
//! ## 主な公開関数
//!
//! - [`to_source_seq`]: テキスト → [`SourceEntry`] 列
//!   - 拗音（ひらがな/カタカナ＋小書き仮名）を複合ユニットとしてまとめる
//!   - 位取り文字 (十百千万億兆) の `JapaneseNumeric` 昇格判定もここで行う
//! - [`compute_source_features`]: [`SourceEntry`] 列 → 各文字の [`FeatureKey`] 列
//!
//! ## 原文位置の表現
//!
//! C++ 版は `orig_idx` を **コードポイント番目** として扱っているが、
//! Rust 版は **UTF-8 バイト位置** で統一する。`str::char_indices()` から
//! 自然に取れること、後段で原文スライスを取るときに変換が不要なこと、
//! Rust の文字列表現と整合的なことが理由。

use std::collections::HashSet;

use crate::char_type::{get_char_type, CharType};
use crate::feature::{FeatureKey, FeatureType};

// ============================================================
// SourceEntry
// ============================================================

/// 1 文字（または複合ユニット）分のソース情報。
///
/// テキスト前処理 ([`to_source_seq`]) の出力で、特徴量計算
/// ([`compute_source_features`]) の入力でもある。
///
/// 拗音（例: "きゃ"）や辞書登録された漢字語（例: "今日"）は、
/// `compound_len == 2` または `3` で cp2/cp3 に追加コードポイントを格納。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SourceEntry {
    /// 先頭コードポイント値 (`u32`)。文脈特徴量 (bigram/trigram) に使用。
    pub cp: u32,
    /// 2文字目のコードポイント（複合ユニットのみ、それ以外は 0）
    pub cp2: u32,
    /// 3文字目のコードポイント（3文字複合ユニットのみ、それ以外は 0）
    pub cp3: u32,
    /// 原文中の UTF-8 バイト位置（先頭文字）
    pub orig_idx: u32,
    /// 文字種 (位取り文字昇格後の最終値)。先頭文字の種別。
    pub ctype: CharType,
    /// ユニットのコードポイント数（通常 1、拗音など複合は 2 か 3）
    pub compound_len: u8,
}

// ============================================================
// to_source_seq
// ============================================================

/// テキストを [`SourceEntry`] 列に変換する。
///
/// 処理の流れ:
/// 1. 各文字の文字種を仮計算
/// 2. 位取り文字 (`十`, `百`, `千`, `万`, `億`, `兆`) の `JapaneseNumeric` 昇格
/// 3. 拗音複合ユニット（ひらがな/カタカナ基底文字 + 小書き仮名）の検出
/// 4. `compound_units` が指定されていれば辞書による複合ユニット検出（最大3文字）
///
/// C++ 版の `to_source_seq()` と `_preprocess_text()` の推論時パスに対応。
pub(crate) fn to_source_seq(
    text: &str,
    compound_units: Option<&HashSet<String>>,
) -> Vec<SourceEntry> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();

    if n == 0 {
        return Vec::new();
    }

    // --- Step 1: 各文字種を仮計算 ---
    let mut ctypes: Vec<CharType> = chars.iter().map(|&(_, c)| get_char_type(c)).collect();

    // --- Step 2: 位取り文字昇格 (左→右パス) ---
    for i in 1..n {
        if is_kurai_char(chars[i].1) && ctypes[i - 1] == CharType::JapaneseNumeric {
            ctypes[i] = CharType::JapaneseNumeric;
        }
    }

    // --- Step 3: 位取り文字昇格 (右→左パス) ---
    // 例: 「十一」の「十」は、右隣の「一」(JapaneseNumeric) を見て昇格する
    for i in (0..n - 1).rev() {
        if is_kurai_char(chars[i].1) && ctypes[i + 1] == CharType::JapaneseNumeric {
            ctypes[i] = CharType::JapaneseNumeric;
        }
    }

    // --- Step 4: SourceEntry を組み立て (拗音・compound unit 対応) ---
    let mut seq = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        // 拗音複合ユニット検出（アルゴリズム）
        // ひらがな/カタカナ基底文字 + 小書き仮名
        if is_base_kana(chars[i].1) && i + 1 < n && is_small_kana(chars[i + 1].1 as u32) {
            seq.push(SourceEntry {
                cp: chars[i].1 as u32,
                cp2: chars[i + 1].1 as u32,
                cp3: 0,
                orig_idx: chars[i].0 as u32,
                ctype: ctypes[i],
                compound_len: 2,
            });
            i += 2;
            continue;
        }

        // 漢字複合ユニット検出（辞書ベース）
        if let Some(cu) = compound_units {
            if !cu.is_empty() {
                let mut found = false;
                // 最長一致（最大3文字まで）
                for len in (2usize..=3).rev() {
                    if i + len > n {
                        continue;
                    }
                    let candidate: String = chars[i..i + len].iter().map(|&(_, c)| c).collect();
                    if cu.contains(&candidate) {
                        seq.push(SourceEntry {
                            cp: chars[i].1 as u32,
                            cp2: chars[i + 1].1 as u32,
                            cp3: if len >= 3 { chars[i + 2].1 as u32 } else { 0 },
                            orig_idx: chars[i].0 as u32,
                            ctype: ctypes[i],
                            compound_len: len as u8,
                        });
                        i += len;
                        found = true;
                        break;
                    }
                }
                if found {
                    continue;
                }
            }
        }

        // 単一文字
        seq.push(SourceEntry {
            cp: chars[i].1 as u32,
            cp2: 0,
            cp3: 0,
            orig_idx: chars[i].0 as u32,
            ctype: ctypes[i],
            compound_len: 1,
        });
        i += 1;
    }

    seq
}

/// ひらがな/カタカナ（基底文字、小書き含む）かどうか。
/// C++ 版 `is_base_kana()` と対応。
#[inline]
fn is_base_kana(c: char) -> bool {
    matches!(c, '\u{3041}'..='\u{3093}' | '\u{30A1}'..='\u{30F6}')
}

/// 位取り文字 (十・百・千・万・億・兆) か。
#[inline]
fn is_kurai_char(c: char) -> bool {
    matches!(c, '十' | '百' | '千' | '万' | '億' | '兆')
}

// ============================================================
// compute_source_features
// ============================================================

/// 各文字に対する [`FeatureKey`] のリストを計算する。
///
/// 戻り値の長さは `seq.len()` と等しい。
/// `result[i]` は `i` 番目の文字（またはユニット）に対応する特徴量キーのリスト。
///
/// **文脈特徴量** (bigram/trigram/char_p*/n*) は各エントリの **先頭コードポイント** (`cp`) を使用する。
/// 複合ユニット（例: "きゃ"）でも先頭文字 "き" のみを文脈に使う。
/// **char_s 特徴量**のみ複合ユニット全体を使用し、`compound_len` に応じて
/// `CharSelf` / `CharSelfCompound2` / `CharSelfCompound3` を選択する。
///
/// 学習時は window=4,5,7 が選べたが、推論時は学習時のモデルに合わせるしかない。
/// 本関数は **学習時 window=7** に相当する全特徴量を計算する。
/// 短い window のモデルでは、ここで生成した余分な特徴量は語彙テーブルに
/// 存在しないため、`vocab_find` で `None` になり自然に無視される。
pub(crate) fn compute_source_features(seq: &[SourceEntry]) -> Vec<Vec<FeatureKey>> {
    let n = seq.len();
    let mut result: Vec<Vec<FeatureKey>> = (0..n).map(|_| Vec::new()).collect();

    for i in 0..n {
        // 文脈特徴量は先頭コードポイント (cp) を使用。
        // char_s のみ複合ユニット全体を使う（下記 make_char_self_compound）。
        let c = seq[i].cp;
        let ctype = seq[i].ctype;

        let prev_c = if i > 0 { seq[i - 1].cp } else { 0 };
        let prev_ctype = if i > 0 {
            seq[i - 1].ctype
        } else {
            CharType::Other
        };
        let prev2_c = if i > 1 { seq[i - 2].cp } else { 0 };
        let prev2_ctype = if i > 1 {
            seq[i - 2].ctype
        } else {
            CharType::Other
        };
        let prev3_c = if i > 2 { seq[i - 3].cp } else { 0 };
        let prev3_ctype = if i > 2 {
            seq[i - 3].ctype
        } else {
            CharType::Other
        };
        let next_c = if i + 1 < n { seq[i + 1].cp } else { 0 };
        let next_ctype = if i + 1 < n {
            seq[i + 1].ctype
        } else {
            CharType::Other
        };
        let next2_c = if i + 2 < n { seq[i + 2].cp } else { 0 };
        let next2_ctype = if i + 2 < n {
            seq[i + 2].ctype
        } else {
            CharType::Other
        };
        let next3_c = if i + 3 < n { seq[i + 3].cp } else { 0 };
        let next3_ctype = if i + 3 < n {
            seq[i + 3].ctype
        } else {
            CharType::Other
        };

        let feats = &mut result[i];

        // --- bias, char_s（複合ユニット対応）, type_s ---
        feats.push(FeatureKey::no_payload(FeatureType::Bias));
        feats.push(make_char_self_compound(&seq[i]));
        feats.push(FeatureKey::type_1(FeatureType::TypeSelf, ctype));

        // --- 前方文脈 ---
        if i > 0 {
            feats.push(FeatureKey::char_1(FeatureType::CharPrev1, prev_c));
            feats.push(FeatureKey::type_1(FeatureType::TypePrev1, prev_ctype));
            feats.push(FeatureKey::char_2(FeatureType::BigramPrev1Self, prev_c, c));
            feats.push(FeatureKey::type_2(
                FeatureType::TypeTransition,
                prev_ctype,
                ctype,
            ));

            if i > 1 {
                feats.push(FeatureKey::char_1(FeatureType::CharPrev2, prev2_c));
                feats.push(FeatureKey::type_1(FeatureType::TypePrev2, prev2_ctype));
                feats.push(FeatureKey::char_2(
                    FeatureType::BigramPrev2Prev1,
                    prev2_c,
                    prev_c,
                ));
                // trigram: 前2-前1-対象
                feats.push(FeatureKey::char_3(
                    FeatureType::TrigramPrev2Prev1Self,
                    prev2_c,
                    prev_c,
                    c,
                ));
                feats.push(FeatureKey::type_3(
                    FeatureType::TypeTriPrev2Prev1Self,
                    prev2_ctype,
                    prev_ctype,
                    ctype,
                ));

                if i > 2 {
                    feats.push(FeatureKey::char_1(FeatureType::CharPrev3, prev3_c));
                    feats.push(FeatureKey::type_1(FeatureType::TypePrev3, prev3_ctype));
                    feats.push(FeatureKey::char_2(
                        FeatureType::BigramPrev3Prev2,
                        prev3_c,
                        prev2_c,
                    ));
                    // trigram: 前3-前2-前1
                    feats.push(FeatureKey::char_3(
                        FeatureType::TrigramPrev3Prev2Prev1,
                        prev3_c,
                        prev2_c,
                        prev_c,
                    ));
                    feats.push(FeatureKey::type_3(
                        FeatureType::TypeTriPrev3Prev2Prev1,
                        prev3_ctype,
                        prev2_ctype,
                        prev_ctype,
                    ));
                }
            }
        }

        // --- 後方文脈 ---
        if i + 1 < n {
            feats.push(FeatureKey::char_1(FeatureType::CharNext1, next_c));
            feats.push(FeatureKey::type_1(FeatureType::TypeNext1, next_ctype));
            feats.push(FeatureKey::char_2(FeatureType::BigramSelfNext1, c, next_c));

            if i + 2 < n {
                feats.push(FeatureKey::char_1(FeatureType::CharNext2, next2_c));
                feats.push(FeatureKey::type_1(FeatureType::TypeNext2, next2_ctype));
                feats.push(FeatureKey::char_2(
                    FeatureType::BigramNext1Next2,
                    next_c,
                    next2_c,
                ));
                // trigram: 対象-後1-後2
                feats.push(FeatureKey::char_3(
                    FeatureType::TrigramSelfNext1Next2,
                    c,
                    next_c,
                    next2_c,
                ));
                feats.push(FeatureKey::type_3(
                    FeatureType::TypeTriSelfNext1Next2,
                    ctype,
                    next_ctype,
                    next2_ctype,
                ));

                if i + 3 < n {
                    feats.push(FeatureKey::char_1(FeatureType::CharNext3, next3_c));
                    feats.push(FeatureKey::type_1(FeatureType::TypeNext3, next3_ctype));
                    feats.push(FeatureKey::char_2(
                        FeatureType::BigramNext2Next3,
                        next2_c,
                        next3_c,
                    ));
                    // trigram: 後1-後2-後3
                    feats.push(FeatureKey::char_3(
                        FeatureType::TrigramNext1Next2Next3,
                        next_c,
                        next2_c,
                        next3_c,
                    ));
                    feats.push(FeatureKey::type_3(
                        FeatureType::TypeTriNext1Next2Next3,
                        next_ctype,
                        next2_ctype,
                        next3_ctype,
                    ));
                }
            }
        }

        // --- trigram: 前1-対象-後1 ---
        if i > 0 && i + 1 < n {
            feats.push(FeatureKey::char_3(
                FeatureType::TrigramPrev1SelfNext1,
                prev_c,
                c,
                next_c,
            ));
            feats.push(FeatureKey::type_3(
                FeatureType::TypeTriPrev1SelfNext1,
                prev_ctype,
                ctype,
                next_ctype,
            ));
        }

        // --- 漢字連続長 ---
        if ctype == CharType::Kanji {
            let run = kanji_run_length(seq, i);
            feats.push(FeatureKey::u8_payload(
                FeatureType::KanjiRunLen,
                clamp_run(run),
            ));

            // 漢字連続の先頭か
            if i == 0 || seq[i - 1].ctype != CharType::Kanji {
                feats.push(FeatureKey::no_payload(FeatureType::KanjiPosFirst));
            }

            // 直前が JapaneseNumeric の連続なら、その長さを記録
            if i > 0 && seq[i - 1].ctype == CharType::JapaneseNumeric {
                let mut num_run = 0u32;
                let mut j = i;
                while j > 0 && seq[j - 1].ctype == CharType::JapaneseNumeric {
                    num_run += 1;
                    j -= 1;
                }
                feats.push(FeatureKey::u8_payload(
                    FeatureType::PrevJapaneseNumericRunLen,
                    clamp_run(num_run),
                ));
            }
        }

        // --- 漢数字連続長 ---
        if ctype == CharType::JapaneseNumeric {
            let run = numeric_run_length(seq, i);
            feats.push(FeatureKey::u8_payload(
                FeatureType::JapaneseNumericRunLen,
                clamp_run(run),
            ));
        }
    }

    result
}

// ============================================================
// ヘルパ関数
// ============================================================

/// compound_len に応じた char_s 特徴量を作る。
/// - 1文字: `CharSelf` (char32_t×1)
/// - 2文字: `CharSelfCompound2` (char32_t×2)
/// - 3文字: `CharSelfCompound3` (char32_t×3)
#[inline]
fn make_char_self_compound(e: &SourceEntry) -> FeatureKey {
    match e.compound_len {
        2 => FeatureKey::char_2(FeatureType::CharSelfCompound2, e.cp, e.cp2),
        3 => FeatureKey::char_3(FeatureType::CharSelfCompound3, e.cp, e.cp2, e.cp3),
        _ => FeatureKey::char_1(FeatureType::CharSelf, e.cp),
    }
}

/// 位置 `i` を含む漢字連続の長さ。
fn kanji_run_length(seq: &[SourceEntry], i: usize) -> u32 {
    let n = seq.len();
    let mut run = 1u32;
    // 前方
    let mut j = i + 1;
    while j < n && seq[j].ctype == CharType::Kanji {
        run += 1;
        j += 1;
    }
    // 後方 (逆向き走査)
    let mut j = i;
    while j > 0 && seq[j - 1].ctype == CharType::Kanji {
        run += 1;
        j -= 1;
    }
    run
}

/// 位置 `i` を含む漢数字連続の長さ。
fn numeric_run_length(seq: &[SourceEntry], i: usize) -> u32 {
    let n = seq.len();
    let mut run = 1u32;
    let mut j = i + 1;
    while j < n && seq[j].ctype == CharType::JapaneseNumeric {
        run += 1;
        j += 1;
    }
    let mut j = i;
    while j > 0 && seq[j - 1].ctype == CharType::JapaneseNumeric {
        run += 1;
        j -= 1;
    }
    run
}

/// 連続長を 1..=5 にクランプする。
///
/// 学習時の `_RUN_LEN_MAP = {"1": 1, "2": 2, "3": 3, "4": 4, "5+": 5}` に対応。
#[inline]
fn clamp_run(run: u32) -> u8 {
    if run <= 4 {
        run as u8
    } else {
        5
    }
}

/// 小書き仮名（拗音複合ユニット検出に使用）。
/// C++ 版 `is_small_kana_cp()` と対応。
#[inline]
fn is_small_kana(cp: u32) -> bool {
    matches!(
        cp,
        0x3041 | 0x3043 | 0x3045 | 0x3047 | 0x3049  // ぁぃぅぇぉ
            | 0x3063                                  // っ
            | 0x3083 | 0x3085 | 0x3087               // ゃゅょ
            | 0x308E                                  // ゎ
            | 0x30A1 | 0x30A3 | 0x30A5 | 0x30A7 | 0x30A9  // ァィゥェォ
            | 0x30C3                                  // ッ
            | 0x30E3 | 0x30E5 | 0x30E7               // ャュョ
            | 0x30EE                                  // ヮ
    )
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- to_source_seq ---

    #[test]
    fn to_source_seq_basic() {
        let seq = to_source_seq("漢字", None);
        assert_eq!(seq.len(), 2);
        assert_eq!(seq[0].cp, 0x6F22);
        assert_eq!(seq[0].orig_idx, 0);
        assert_eq!(seq[0].ctype, CharType::Kanji);
        assert_eq!(seq[0].compound_len, 1);
        assert_eq!(seq[1].cp, 0x5B57);
        assert_eq!(seq[1].orig_idx, 3); // 漢 は UTF-8 で 3 バイト
        assert_eq!(seq[1].ctype, CharType::Kanji);
    }

    #[test]
    fn to_source_seq_empty() {
        assert!(to_source_seq("", None).is_empty());
    }

    #[test]
    fn to_source_seq_ascii() {
        let seq = to_source_seq("abc", None);
        assert_eq!(seq.len(), 3);
        assert_eq!(seq[0].orig_idx, 0);
        assert_eq!(seq[1].orig_idx, 1);
        assert_eq!(seq[2].orig_idx, 2);
        assert!(seq.iter().all(|s| s.ctype == CharType::Alpha));
    }

    #[test]
    fn to_source_seq_youon_hiragana() {
        // 「きゃ」は拗音複合ユニット → 1エントリ
        let seq = to_source_seq("きゃ", None);
        assert_eq!(seq.len(), 1);
        assert_eq!(seq[0].cp, 'き' as u32);
        assert_eq!(seq[0].cp2, 'ゃ' as u32);
        assert_eq!(seq[0].compound_len, 2);
        assert_eq!(seq[0].orig_idx, 0);
        assert_eq!(seq[0].ctype, CharType::Hiragana);
    }

    #[test]
    fn to_source_seq_youon_katakana() {
        // 「キャ」は拗音複合ユニット → 1エントリ
        let seq = to_source_seq("キャ", None);
        assert_eq!(seq.len(), 1);
        assert_eq!(seq[0].cp, 'キ' as u32);
        assert_eq!(seq[0].cp2, 'ャ' as u32);
        assert_eq!(seq[0].compound_len, 2);
    }

    #[test]
    fn to_source_seq_youon_in_word() {
        // 「東京」(漢字2文字) の後に「キャ」が来ても問題なく処理
        let seq = to_source_seq("東京キャ", None);
        assert_eq!(seq.len(), 3); // 東・京・キャ
        assert_eq!(seq[2].compound_len, 2);
    }

    #[test]
    fn to_source_seq_compound_dict() {
        // 辞書に "今日" が入っていれば複合ユニットとして1エントリ
        let mut cu = HashSet::new();
        cu.insert("今日".to_string());
        let seq = to_source_seq("今日は", Some(&cu));
        assert_eq!(seq.len(), 2); // 今日・は
        assert_eq!(seq[0].compound_len, 2);
        assert_eq!(seq[0].cp, '今' as u32);
        assert_eq!(seq[0].cp2, '日' as u32);
        assert_eq!(seq[1].compound_len, 1);
    }

    #[test]
    fn to_source_seq_compound_dict_no_match() {
        // 辞書に "今日" がない場合は普通に1文字ずつ
        let seq = to_source_seq("今日は", None);
        assert_eq!(seq.len(), 3); // 今・日・は
    }

    #[test]
    fn kurai_promotion_san_man() {
        // 「三万」: 「万」が JapaneseNumeric に昇格する
        let seq = to_source_seq("三万", None);
        assert_eq!(seq[0].ctype, CharType::JapaneseNumeric); // 三
        assert_eq!(seq[1].ctype, CharType::JapaneseNumeric); // 万 ← 昇格
    }

    #[test]
    fn kurai_no_promotion_man_zen() {
        // 「万全」: 「万」は昇格しない (Kanji のまま)
        let seq = to_source_seq("万全", None);
        assert_eq!(seq[0].ctype, CharType::Kanji); // 万
        assert_eq!(seq[1].ctype, CharType::Kanji); // 全
    }

    #[test]
    fn kurai_promotion_juuichi() {
        // 「十一」: 「十」が右隣の「一」を見て昇格する (右→左パス)
        let seq = to_source_seq("十一", None);
        assert_eq!(seq[0].ctype, CharType::JapaneseNumeric); // 十 ← 昇格
        assert_eq!(seq[1].ctype, CharType::JapaneseNumeric); // 一
    }

    #[test]
    fn kurai_chain_promotion() {
        // 「三千百二十一」のような連鎖
        let seq = to_source_seq("三千百二十一", None);
        for s in &seq {
            assert_eq!(s.ctype, CharType::JapaneseNumeric);
        }
    }

    // --- compute_source_features ---

    #[test]
    fn features_length_matches() {
        let seq = to_source_seq("あいう", None);
        let feats = compute_source_features(&seq);
        assert_eq!(feats.len(), 3);
    }

    #[test]
    fn features_empty_input() {
        let feats = compute_source_features(&[]);
        assert!(feats.is_empty());
    }

    #[test]
    fn features_bias_present() {
        let seq = to_source_seq("あ", None);
        let feats = compute_source_features(&seq);
        let bias = FeatureKey::no_payload(FeatureType::Bias);
        assert!(feats[0].contains(&bias));
    }

    #[test]
    fn features_char_self_present() {
        let seq = to_source_seq("あ", None);
        let feats = compute_source_features(&seq);
        let key = FeatureKey::char_1(FeatureType::CharSelf, 'あ' as u32);
        assert!(feats[0].contains(&key));
    }

    #[test]
    fn features_char_self_compound2_present() {
        // 拗音「きゃ」は CHAR_SELF_COMPOUND_2 になる
        let seq = to_source_seq("きゃ", None);
        assert_eq!(seq.len(), 1);
        let feats = compute_source_features(&seq);
        let key = FeatureKey::char_2(FeatureType::CharSelfCompound2, 'き' as u32, 'ゃ' as u32);
        assert!(feats[0].contains(&key));
        // 通常の CharSelf は含まれない
        let plain = FeatureKey::char_1(FeatureType::CharSelf, 'き' as u32);
        assert!(!feats[0].contains(&plain));
    }

    #[test]
    fn features_char_self_compound3_present() {
        // 3文字複合ユニット
        let mut cu = HashSet::new();
        cu.insert("今日は".to_string());
        let seq = to_source_seq("今日は", Some(&cu));
        assert_eq!(seq[0].compound_len, 3);
        let feats = compute_source_features(&seq);
        let key = FeatureKey::char_3(
            FeatureType::CharSelfCompound3,
            '今' as u32,
            '日' as u32,
            'は' as u32,
        );
        assert!(feats[0].contains(&key));
    }

    #[test]
    fn features_type_self_present() {
        let seq = to_source_seq("あ", None);
        let feats = compute_source_features(&seq);
        let key = FeatureKey::type_1(FeatureType::TypeSelf, CharType::Hiragana);
        assert!(feats[0].contains(&key));
    }

    #[test]
    fn features_bigram_in_middle() {
        // 「あいう」の真ん中の「い」には BigramPrev1Self=あい, BigramSelfNext1=いう を含む
        let seq = to_source_seq("あいう", None);
        let feats = compute_source_features(&seq);
        let key1 = FeatureKey::char_2(FeatureType::BigramPrev1Self, 'あ' as u32, 'い' as u32);
        let key2 = FeatureKey::char_2(FeatureType::BigramSelfNext1, 'い' as u32, 'う' as u32);
        assert!(feats[1].contains(&key1));
        assert!(feats[1].contains(&key2));
    }

    #[test]
    fn features_bigram_uses_first_cp_of_compound() {
        // 拗音複合ユニット「きゃ」の前の文字のバイグラムは先頭コードポイント「き」を使う
        let seq = to_source_seq("あきゃ", None);
        assert_eq!(seq.len(), 2); // あ・きゃ
        let feats = compute_source_features(&seq);
        // [1] = きゃ: BigramPrev1Self は (あ, き) になるはず
        let key = FeatureKey::char_2(FeatureType::BigramPrev1Self, 'あ' as u32, 'き' as u32);
        assert!(feats[1].contains(&key));
    }

    #[test]
    fn features_kanji_run_len() {
        // 「漢字」: 両方とも kanji_run_len=2
        let seq = to_source_seq("漢字", None);
        let feats = compute_source_features(&seq);
        let key = FeatureKey::u8_payload(FeatureType::KanjiRunLen, 2);
        assert!(feats[0].contains(&key));
        assert!(feats[1].contains(&key));
    }

    #[test]
    fn features_kanji_run_clamp() {
        // 5 文字以上の漢字連続は run=5 にクランプ
        let seq = to_source_seq("漢字漢字漢字漢字", None);
        let feats = compute_source_features(&seq);
        let key5 = FeatureKey::u8_payload(FeatureType::KanjiRunLen, 5);
        for f in &feats {
            assert!(f.contains(&key5));
        }
    }

    #[test]
    fn features_kanji_pos_first() {
        // 「漢字」の最初の漢字には kanji_pos_first がある、次にはない
        let seq = to_source_seq("漢字", None);
        let feats = compute_source_features(&seq);
        let key = FeatureKey::no_payload(FeatureType::KanjiPosFirst);
        assert!(feats[0].contains(&key));
        assert!(!feats[1].contains(&key));
    }

    #[test]
    fn features_type_transition() {
        // 「あ字」の「字」には TypeTransition: Hiragana -> Kanji
        let seq = to_source_seq("あ字", None);
        let feats = compute_source_features(&seq);
        let key =
            FeatureKey::type_2(FeatureType::TypeTransition, CharType::Hiragana, CharType::Kanji);
        assert!(feats[1].contains(&key));
        // 最初の「あ」には TypeTransition がない (prev がないため)
        assert!(!feats[0]
            .iter()
            .any(|k| k.feature_type == FeatureType::TypeTransition));
    }

    #[test]
    fn features_prev_japanese_numeric_run_len() {
        // 「三万円」: 「円」(Kanji) の前に「三万」(JapaneseNumeric × 2) があるので
        // PrevJapaneseNumericRunLen=2 が付く
        let seq = to_source_seq("三万円", None);
        let feats = compute_source_features(&seq);
        let key = FeatureKey::u8_payload(FeatureType::PrevJapaneseNumericRunLen, 2);
        assert!(feats[2].contains(&key));
    }

    #[test]
    fn features_japanese_numeric_run_len() {
        // 「三万」(両方 JapaneseNumeric)
        let seq = to_source_seq("三万", None);
        let feats = compute_source_features(&seq);
        let key = FeatureKey::u8_payload(FeatureType::JapaneseNumericRunLen, 2);
        assert!(feats[0].contains(&key));
        assert!(feats[1].contains(&key));
    }

    // --- ヘルパ関数 ---

    #[test]
    fn clamp_run_works() {
        assert_eq!(clamp_run(1), 1);
        assert_eq!(clamp_run(4), 4);
        assert_eq!(clamp_run(5), 5);
        assert_eq!(clamp_run(10), 5);
        assert_eq!(clamp_run(100), 5);
    }

    #[test]
    fn is_kurai_works() {
        for c in "十百千万億兆".chars() {
            assert!(is_kurai_char(c));
        }
        assert!(!is_kurai_char('一'));
        assert!(!is_kurai_char('零'));
        assert!(!is_kurai_char('あ'));
    }

    #[test]
    fn is_small_kana_works() {
        for cp in [
            'ぁ' as u32,
            'ゃ' as u32,
            'っ' as u32,
            'ゎ' as u32,
            'ァ' as u32,
            'ャ' as u32,
            'ッ' as u32,
        ] {
            assert!(is_small_kana(cp));
        }
        assert!(!is_small_kana('あ' as u32));
        assert!(!is_small_kana('き' as u32));
    }

    #[test]
    fn is_base_kana_works() {
        assert!(is_base_kana('き'));
        assert!(is_base_kana('キ'));
        assert!(is_base_kana('ぁ'));
        assert!(!is_base_kana('漢'));
        assert!(!is_base_kana('a'));
    }
}
