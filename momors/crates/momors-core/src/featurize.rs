//! 特徴量計算モジュール。
//!
//! C++ 版の `momo_features.cpp` に対応する。
//! 入力テキストを [`SourceEntry`] 列に変換し、各文字に対する
//! [`FeatureKey`] 列を計算する。
//!
//! ## 主な公開関数
//!
//! - [`to_source_seq`]: テキスト → [`SourceEntry`] 列
//!   - 位取り文字 (十百千万億兆) の `JapaneseNumeric` 昇格判定もここで行う
//! - [`compute_source_features`]: [`SourceEntry`] 列 → 各文字の [`FeatureKey`] 列
//!
//! ## 原文位置の表現
//!
//! C++ 版は `orig_idx` を **コードポイント番目** として扱っているが、
//! Rust 版は **UTF-8 バイト位置** で統一する。`str::char_indices()` から
//! 自然に取れること、後段で原文スライスを取るときに変換が不要なこと、
//! Rust の文字列表現と整合的なことが理由。

use crate::char_type::{get_char_type, CharType};
use crate::feature::{FeatureKey, FeatureType};

// ============================================================
// SourceEntry
// ============================================================

/// 1 文字分のソース情報。
///
/// テキスト前処理 ([`to_source_seq`]) の出力で、特徴量計算
/// ([`compute_source_features`]) の入力でもある。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SourceEntry {
    /// Unicode コードポイント値 (`u32`)
    ///
    /// C++ 版の `char32_t` と互換。`0` をセンチネルとして使う場面があるため
    /// `char` ではなく `u32` で保持する。
    pub cp: u32,
    /// 原文中の UTF-8 バイト位置
    pub orig_idx: u32,
    /// 文字種 (位取り文字昇格後の最終値)
    pub ctype: CharType,
}

// ============================================================
// to_source_seq
// ============================================================

/// テキストを [`SourceEntry`] 列に変換する。
///
/// 位取り文字 (`十`, `百`, `千`, `万`, `億`, `兆`) の
/// [`JapaneseNumeric`] 昇格は隣接文脈で判定:
///
/// - 左隣が `JapaneseNumeric` なら昇格 (例: 「三万」の「万」)
/// - 右隣が `JapaneseNumeric` なら昇格 (例: 「十一」の「十」)
/// - どちらでもなければ昇格しない (例: 「万全」の「万」は `Kanji` のまま)
///
/// 左→右パスと右→左パスの 2 回走査することで、連鎖的な昇格にも対応する。
///
/// [`JapaneseNumeric`]: CharType::JapaneseNumeric
pub(crate) fn to_source_seq(text: &str) -> Vec<SourceEntry> {
    // char_indices(): (byte_idx, char) のイテレータ
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();

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
    if n >= 2 {
        for i in (0..n - 1).rev() {
            if is_kurai_char(chars[i].1) && ctypes[i + 1] == CharType::JapaneseNumeric {
                ctypes[i] = CharType::JapaneseNumeric;
            }
        }
    }

    // --- Step 4: SourceEntry を組み立てる ---
    chars
        .into_iter()
        .zip(ctypes)
        .map(|((byte_idx, c), ctype)| SourceEntry {
            cp: c as u32,
            orig_idx: byte_idx as u32,
            ctype,
        })
        .collect()
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
/// `result[i]` は `i` 番目の文字に対応する特徴量キーのリスト。
///
/// 学習時は window=4,5,7 が選べたが、推論時は学習時のモデルに合わせるしかない。
/// 本関数は **学習時 window=7** に相当する全特徴量を計算する。
/// 短い window のモデルでは、ここで生成した余分な特徴量は語彙テーブルに
/// 存在しないため、`vocab_find` で `None` になり自然に無視される。
pub(crate) fn compute_source_features(seq: &[SourceEntry]) -> Vec<Vec<FeatureKey>> {
    let n = seq.len();
    let mut result: Vec<Vec<FeatureKey>> = (0..n).map(|_| Vec::new()).collect();

    for i in 0..n {
        let c = seq[i].cp;
        let ctype = seq[i].ctype;

        // C++ 版に合わせ、範囲外は cp=0, ctype=Other で埋める
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

        // --- bias, char_s, type_s ---
        feats.push(FeatureKey::no_payload(FeatureType::Bias));
        feats.push(FeatureKey::char_1(FeatureType::CharSelf, c));
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

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- to_source_seq ---

    #[test]
    fn to_source_seq_basic() {
        let seq = to_source_seq("漢字");
        assert_eq!(seq.len(), 2);
        assert_eq!(seq[0].cp, 0x6F22);
        assert_eq!(seq[0].orig_idx, 0);
        assert_eq!(seq[0].ctype, CharType::Kanji);
        assert_eq!(seq[1].cp, 0x5B57);
        assert_eq!(seq[1].orig_idx, 3); // 漢 は UTF-8 で 3 バイト
        assert_eq!(seq[1].ctype, CharType::Kanji);
    }

    #[test]
    fn to_source_seq_empty() {
        assert!(to_source_seq("").is_empty());
    }

    #[test]
    fn to_source_seq_ascii() {
        let seq = to_source_seq("abc");
        assert_eq!(seq.len(), 3);
        assert_eq!(seq[0].orig_idx, 0);
        assert_eq!(seq[1].orig_idx, 1);
        assert_eq!(seq[2].orig_idx, 2);
        assert!(seq.iter().all(|s| s.ctype == CharType::Alpha));
    }

    #[test]
    fn kurai_promotion_san_man() {
        // 「三万」: 「万」が JapaneseNumeric に昇格する
        let seq = to_source_seq("三万");
        assert_eq!(seq[0].ctype, CharType::JapaneseNumeric); // 三
        assert_eq!(seq[1].ctype, CharType::JapaneseNumeric); // 万 ← 昇格
    }

    #[test]
    fn kurai_no_promotion_man_zen() {
        // 「万全」: 「万」は昇格しない (Kanji のまま)
        let seq = to_source_seq("万全");
        assert_eq!(seq[0].ctype, CharType::Kanji); // 万
        assert_eq!(seq[1].ctype, CharType::Kanji); // 全
    }

    #[test]
    fn kurai_promotion_juuichi() {
        // 「十一」: 「十」が右隣の「一」を見て昇格する (右→左パス)
        let seq = to_source_seq("十一");
        assert_eq!(seq[0].ctype, CharType::JapaneseNumeric); // 十 ← 昇格
        assert_eq!(seq[1].ctype, CharType::JapaneseNumeric); // 一
    }

    #[test]
    fn kurai_chain_promotion() {
        // 「三千百二十一」のような連鎖
        let seq = to_source_seq("三千百二十一");
        for s in &seq {
            assert_eq!(s.ctype, CharType::JapaneseNumeric);
        }
    }

    // --- compute_source_features ---

    #[test]
    fn features_length_matches() {
        let seq = to_source_seq("あいう");
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
        let seq = to_source_seq("あ");
        let feats = compute_source_features(&seq);
        let bias = FeatureKey::no_payload(FeatureType::Bias);
        assert!(feats[0].contains(&bias));
    }

    #[test]
    fn features_char_self_present() {
        let seq = to_source_seq("あ");
        let feats = compute_source_features(&seq);
        let key = FeatureKey::char_1(FeatureType::CharSelf, 'あ' as u32);
        assert!(feats[0].contains(&key));
    }

    #[test]
    fn features_type_self_present() {
        let seq = to_source_seq("あ");
        let feats = compute_source_features(&seq);
        let key = FeatureKey::type_1(FeatureType::TypeSelf, CharType::Hiragana);
        assert!(feats[0].contains(&key));
    }

    #[test]
    fn features_bigram_in_middle() {
        // 「あいう」の真ん中の「い」には BigramPrev1Self=あい, BigramSelfNext1=いう を含む
        let seq = to_source_seq("あいう");
        let feats = compute_source_features(&seq);
        let key1 = FeatureKey::char_2(FeatureType::BigramPrev1Self, 'あ' as u32, 'い' as u32);
        let key2 = FeatureKey::char_2(FeatureType::BigramSelfNext1, 'い' as u32, 'う' as u32);
        assert!(feats[1].contains(&key1));
        assert!(feats[1].contains(&key2));
    }

    #[test]
    fn features_kanji_run_len() {
        // 「漢字」: 両方とも kanji_run_len=2
        let seq = to_source_seq("漢字");
        let feats = compute_source_features(&seq);
        let key = FeatureKey::u8_payload(FeatureType::KanjiRunLen, 2);
        assert!(feats[0].contains(&key));
        assert!(feats[1].contains(&key));
    }

    #[test]
    fn features_kanji_run_clamp() {
        // 5 文字以上の漢字連続は run=5 にクランプ
        let seq = to_source_seq("漢字漢字漢字漢字");
        let feats = compute_source_features(&seq);
        let key5 = FeatureKey::u8_payload(FeatureType::KanjiRunLen, 5);
        for f in &feats {
            assert!(f.contains(&key5));
        }
    }

    #[test]
    fn features_kanji_pos_first() {
        // 「漢字」の最初の漢字には kanji_pos_first がある、次にはない
        let seq = to_source_seq("漢字");
        let feats = compute_source_features(&seq);
        let key = FeatureKey::no_payload(FeatureType::KanjiPosFirst);
        assert!(feats[0].contains(&key));
        assert!(!feats[1].contains(&key));
    }

    #[test]
    fn features_type_transition() {
        // 「あ字」の「字」には TypeTransition: Hiragana -> Kanji
        let seq = to_source_seq("あ字");
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
        let seq = to_source_seq("三万円");
        let feats = compute_source_features(&seq);
        let key = FeatureKey::u8_payload(FeatureType::PrevJapaneseNumericRunLen, 2);
        assert!(feats[2].contains(&key));
    }

    #[test]
    fn features_japanese_numeric_run_len() {
        // 「三万」(両方 JapaneseNumeric)
        let seq = to_source_seq("三万");
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
}
