//! 点訳入力の**幅正規化**（半角カナ → 全角カタカナ、全角英数字 → ASCII）。
//!
//! `JapaneseTranslator` のテーブルは全角カタカナ・ASCII をキーにしているので、
//! 半角カナ（`ｶﾞ` 等）や全角英数字（`Ａ１` 等）はそのままでは引けない。この
//! モジュールが入口で正典形へ畳んでから変換器に渡す。
//!
//! ## 半角カナの合成（2→1）
//!
//! 半角の濁点・半濁点は**合成済みコードポイントを持たない独立文字**で、`ｶﾞ` は
//! `ｶ`(U+FF76) + `ﾞ`(U+FF9E) の2文字。日本語点字の濁点は前置（`ガ` = `⠐⠡`）なので、
//! `ﾞ` を単独で点字化すると順序が壊れる。よって**先に `ｶﾞ → ガ` と合成**してから
//! テーブルを引く。合成できる組は Unicode の正準合成が存在するものだけ（`ｦﾞ → ヺ` は
//! 在る、`ｱﾞ`・`ﾝﾞ`・`ｶﾟ` は無い）。合成できない濁点・半濁点、および裸のマークは
//! 独立形 `゛`(U+309B) / `゜`(U+309C) へ落とす（点字テーブル側で扱う）。
//!
//! テーブルは Unicode の正準合成表の転記（凍結データ）。生成:
//! ```text
//! python -c "import unicodedata as u; V=chr(0xFF9E); S=chr(0xFF9F); \
//!   [print(c,m,u.normalize('NFKC',c+m)) for cp in range(0xFF61,0xFFA0) \
//!    for c in [chr(cp)] for m in (V,S) if len(u.normalize('NFKC',c+m))==1]"
//! ```
//!
//! 将来的に「一般的な文字処理」クレートへ抽出しやすいよう独立モジュールにしている。

use std::collections::HashMap;
use std::sync::LazyLock;

/// 半角base + 濁点/半濁点 → 合成済み全角カタカナ（2→1）。Unicode 正準合成が存在する組のみ。
const VOICED_PAIRS: &[(char, char, char)] = &[
    ('\u{FF66}', '\u{FF9E}', 'ヺ'), // ｦﾞ
    ('\u{FF73}', '\u{FF9E}', 'ヴ'), // ｳﾞ
    ('\u{FF76}', '\u{FF9E}', 'ガ'), // ｶﾞ
    ('\u{FF77}', '\u{FF9E}', 'ギ'), // ｷﾞ
    ('\u{FF78}', '\u{FF9E}', 'グ'), // ｸﾞ
    ('\u{FF79}', '\u{FF9E}', 'ゲ'), // ｹﾞ
    ('\u{FF7A}', '\u{FF9E}', 'ゴ'), // ｺﾞ
    ('\u{FF7B}', '\u{FF9E}', 'ザ'), // ｻﾞ
    ('\u{FF7C}', '\u{FF9E}', 'ジ'), // ｼﾞ
    ('\u{FF7D}', '\u{FF9E}', 'ズ'), // ｽﾞ
    ('\u{FF7E}', '\u{FF9E}', 'ゼ'), // ｾﾞ
    ('\u{FF7F}', '\u{FF9E}', 'ゾ'), // ｿﾞ
    ('\u{FF80}', '\u{FF9E}', 'ダ'), // ﾀﾞ
    ('\u{FF81}', '\u{FF9E}', 'ヂ'), // ﾁﾞ
    ('\u{FF82}', '\u{FF9E}', 'ヅ'), // ﾂﾞ
    ('\u{FF83}', '\u{FF9E}', 'デ'), // ﾃﾞ
    ('\u{FF84}', '\u{FF9E}', 'ド'), // ﾄﾞ
    ('\u{FF8A}', '\u{FF9E}', 'バ'), // ﾊﾞ
    ('\u{FF8A}', '\u{FF9F}', 'パ'), // ﾊﾟ
    ('\u{FF8B}', '\u{FF9E}', 'ビ'), // ﾋﾞ
    ('\u{FF8B}', '\u{FF9F}', 'ピ'), // ﾋﾟ
    ('\u{FF8C}', '\u{FF9E}', 'ブ'), // ﾌﾞ
    ('\u{FF8C}', '\u{FF9F}', 'プ'), // ﾌﾟ
    ('\u{FF8D}', '\u{FF9E}', 'ベ'), // ﾍﾞ
    ('\u{FF8D}', '\u{FF9F}', 'ペ'), // ﾍﾟ
    ('\u{FF8E}', '\u{FF9E}', 'ボ'), // ﾎﾞ
    ('\u{FF8E}', '\u{FF9F}', 'ポ'), // ﾎﾟ
    ('\u{FF9C}', '\u{FF9E}', 'ヷ'), // ﾜﾞ
];

/// 半角単体 → 全角（1→1）。半角カナ・半角記号・長音、および余りマーク
/// `ﾞ`→`゛`(U+309B) / `ﾟ`→`゜`(U+309C) を含む。
const SINGLE_MAP: &[(char, char)] = &[
    ('\u{FF61}', '。'), // ｡
    ('\u{FF62}', '「'), // ｢
    ('\u{FF63}', '」'), // ｣
    ('\u{FF64}', '、'), // ､
    ('\u{FF65}', '・'), // ･
    ('\u{FF66}', 'ヲ'), // ｦ
    ('\u{FF67}', 'ァ'), // ｧ
    ('\u{FF68}', 'ィ'), // ｨ
    ('\u{FF69}', 'ゥ'), // ｩ
    ('\u{FF6A}', 'ェ'), // ｪ
    ('\u{FF6B}', 'ォ'), // ｫ
    ('\u{FF6C}', 'ャ'), // ｬ
    ('\u{FF6D}', 'ュ'), // ｭ
    ('\u{FF6E}', 'ョ'), // ｮ
    ('\u{FF6F}', 'ッ'), // ｯ
    ('\u{FF70}', 'ー'), // ｰ
    ('\u{FF71}', 'ア'), // ｱ
    ('\u{FF72}', 'イ'), // ｲ
    ('\u{FF73}', 'ウ'), // ｳ
    ('\u{FF74}', 'エ'), // ｴ
    ('\u{FF75}', 'オ'), // ｵ
    ('\u{FF76}', 'カ'), // ｶ
    ('\u{FF77}', 'キ'), // ｷ
    ('\u{FF78}', 'ク'), // ｸ
    ('\u{FF79}', 'ケ'), // ｹ
    ('\u{FF7A}', 'コ'), // ｺ
    ('\u{FF7B}', 'サ'), // ｻ
    ('\u{FF7C}', 'シ'), // ｼ
    ('\u{FF7D}', 'ス'), // ｽ
    ('\u{FF7E}', 'セ'), // ｾ
    ('\u{FF7F}', 'ソ'), // ｿ
    ('\u{FF80}', 'タ'), // ﾀ
    ('\u{FF81}', 'チ'), // ﾁ
    ('\u{FF82}', 'ツ'), // ﾂ
    ('\u{FF83}', 'テ'), // ﾃ
    ('\u{FF84}', 'ト'), // ﾄ
    ('\u{FF85}', 'ナ'), // ﾅ
    ('\u{FF86}', 'ニ'), // ﾆ
    ('\u{FF87}', 'ヌ'), // ﾇ
    ('\u{FF88}', 'ネ'), // ﾈ
    ('\u{FF89}', 'ノ'), // ﾉ
    ('\u{FF8A}', 'ハ'), // ﾊ
    ('\u{FF8B}', 'ヒ'), // ﾋ
    ('\u{FF8C}', 'フ'), // ﾌ
    ('\u{FF8D}', 'ヘ'), // ﾍ
    ('\u{FF8E}', 'ホ'), // ﾎ
    ('\u{FF8F}', 'マ'), // ﾏ
    ('\u{FF90}', 'ミ'), // ﾐ
    ('\u{FF91}', 'ム'), // ﾑ
    ('\u{FF92}', 'メ'), // ﾒ
    ('\u{FF93}', 'モ'), // ﾓ
    ('\u{FF94}', 'ヤ'), // ﾔ
    ('\u{FF95}', 'ユ'), // ﾕ
    ('\u{FF96}', 'ヨ'), // ﾖ
    ('\u{FF97}', 'ラ'), // ﾗ
    ('\u{FF98}', 'リ'), // ﾘ
    ('\u{FF99}', 'ル'), // ﾙ
    ('\u{FF9A}', 'レ'), // ﾚ
    ('\u{FF9B}', 'ロ'), // ﾛ
    ('\u{FF9C}', 'ワ'), // ﾜ
    ('\u{FF9D}', 'ン'), // ﾝ
    ('\u{FF9E}', '\u{309B}'), // ﾞ → ゛（独立濁点）
    ('\u{FF9F}', '\u{309C}'), // ﾟ → ゜（独立半濁点）
];

static VOICED: LazyLock<HashMap<(char, char), char>> =
    LazyLock::new(|| VOICED_PAIRS.iter().map(|&(b, m, c)| ((b, m), c)).collect());

static SINGLE: LazyLock<HashMap<char, char>> =
    LazyLock::new(|| SINGLE_MAP.iter().copied().collect());

/// テキストを幅正規化する。
///
/// 返すのは (正規化後の文字列, 正規化後の各文字が消費した原文文字数)。
/// `ｶﾞ → ガ` のような合成は原文2文字を1文字に畳むため、呼び出し側は
/// 2つ目の要素で原文インデックスへ展開できる（`src_len[k]` 個の原文文字が
/// 正規化後の `k` 番目の文字に対応する）。
pub fn normalize(text: &str) -> (Vec<char>, Vec<usize>) {
    let src: Vec<char> = text.chars().collect();
    let mut out = Vec::with_capacity(src.len());
    let mut src_len = Vec::with_capacity(src.len());
    let mut i = 0;
    while i < src.len() {
        let c = src[i];

        // 半角base + 濁点/半濁点 の合成（2→1）
        if i + 1 < src.len()
            && let Some(&composed) = VOICED.get(&(c, src[i + 1]))
        {
            out.push(composed);
            src_len.push(2);
            i += 2;
            continue;
        }

        // 半角単体 → 全角（1→1、余りマーク ﾞ→゛ 含む）
        if let Some(&full) = SINGLE.get(&c) {
            out.push(full);
            src_len.push(1);
            i += 1;
            continue;
        }

        // 全角英数字 → ASCII（momors-core がバイパスした分を畳む）
        let folded = match c as u32 {
            0xFF10..=0xFF19 => char::from_u32(c as u32 - 0xFF10 + 0x30).unwrap_or(c), // ０-９→0-9
            0xFF21..=0xFF3A => char::from_u32(c as u32 - 0xFF21 + 0x41).unwrap_or(c), // Ａ-Ｚ→A-Z
            0xFF41..=0xFF5A => char::from_u32(c as u32 - 0xFF41 + 0x61).unwrap_or(c), // ａ-ｚ→a-z
            _ => c,
        };
        out.push(folded);
        src_len.push(1);
        i += 1;
    }
    (out, src_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(s: &str) -> (String, Vec<usize>) {
        let (chars, src_len) = normalize(s);
        (chars.into_iter().collect(), src_len)
    }

    #[test]
    fn voiced_composes_two_into_one() {
        assert_eq!(norm("ｶﾞ"), ("ガ".to_string(), vec![2]));
        assert_eq!(norm("ﾊﾟ"), ("パ".to_string(), vec![2]));
        assert_eq!(norm("ｳﾞ"), ("ヴ".to_string(), vec![2]));
        // 手書きだと漏らしがちな組
        assert_eq!(norm("ｦﾞ"), ("ヺ".to_string(), vec![2]));
        assert_eq!(norm("ﾜﾞ"), ("ヷ".to_string(), vec![2]));
    }

    #[test]
    fn single_halfwidth_maps_to_fullwidth() {
        assert_eq!(norm("ｱｲｳｴｵ"), ("アイウエオ".to_string(), vec![1, 1, 1, 1, 1]));
        assert_eq!(norm("ｦ"), ("ヲ".to_string(), vec![1]));
        assert_eq!(norm("ｰ"), ("ー".to_string(), vec![1]));
        assert_eq!(norm("ｷｬ"), ("キャ".to_string(), vec![1, 1]));
    }

    #[test]
    fn halfwidth_punct_folds() {
        assert_eq!(norm("｡｢｣､･"), ("。「」、・".to_string(), vec![1, 1, 1, 1, 1]));
    }

    #[test]
    fn invalid_voiced_falls_through_to_standalone_mark() {
        // ｱﾞ: (ｱ,ﾞ) は合成表に無い → ア + ゛(U+309B)
        assert_eq!(norm("ｱﾞ"), ("ア\u{309B}".to_string(), vec![1, 1]));
        // ｶﾟ: 半濁点はハ行だけ → カ + ゜(U+309C)
        assert_eq!(norm("ｶﾟ"), ("カ\u{309C}".to_string(), vec![1, 1]));
    }

    #[test]
    fn lone_mark_becomes_standalone() {
        assert_eq!(norm("ﾞ"), ("\u{309B}".to_string(), vec![1]));
        assert_eq!(norm("ﾟ"), ("\u{309C}".to_string(), vec![1]));
    }

    #[test]
    fn fullwidth_alnum_still_folds() {
        assert_eq!(norm("Ａ１ｚ"), ("A1z".to_string(), vec![1, 1, 1]));
    }

    #[test]
    fn passthrough_untouched() {
        assert_eq!(norm("アガ。abc"), ("アガ。abc".to_string(), vec![1; 6]));
    }

    #[test]
    fn consecutive_voiced() {
        assert_eq!(norm("ｶﾞｷﾞ"), ("ガギ".to_string(), vec![2, 2]));
    }
}
