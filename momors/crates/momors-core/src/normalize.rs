//! 入力テキストの正規化モジュール。
//!
//! CJK部首補助・康煕部首・CJK互換漢字（および補助面）は、通常のCJK統合漢字
//! （U+4E00–U+9FFF 等）と見た目が同じか酷似しているが別の符号点に存在する。
//! 正規化せずに `char_type::get_char_type()` へ渡すと KANJI と判定されず
//! 記号・未定義文字として扱われてしまうため、対応する統合漢字へ畳み込む（1→1）。
//!
//! さらに欧文リガチャ（ﬃ→ffi 等）のような 1→N 展開カテゴリも正規化する。
//! 展開で原文とバイト位置がずれるため、[`normalize_input`] は正規化後の
//! 各バイト位置に対応する原文バイト位置のマップを返す。
//!
//! Python 版 `momo_py.utils.normalize_compat_ideographs()` /
//! `normalize_input()` と挙動を一致させる。
//! 変換テーブルは `tools/gen_normalize_table.py` で Python から自動生成される。

mod table;

/// [`normalize_input`] の結果。
pub(crate) struct NormalizedText {
    /// 正規化後テキスト。
    pub text: String,
    /// 正規化後テキストの各**バイト位置** → 原文の対応文字の先頭バイト位置。
    /// 正規化で変化がなかった場合は `None`（恒等対応）。
    pub byte_map: Option<Vec<usize>>,
}

/// 文字列中の互換漢字系コードポイントを正規のCJK統合漢字へ畳み込む。
///
/// 対象4レンジはいずれも1文字→1文字の変換であることが生成スクリプト側で
/// 保証されているため、文字数の対応はずれない（バイト位置はずれ得る）。
/// 辞書表層形などインデックスマップを持てない箇所の正規化に使う。
/// 推論入力の正規化は [`normalize_input`] を使うこと。
pub(crate) fn normalize_compat_ideographs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match table::normalize_lookup(c as u32).and_then(char::from_u32) {
            Some(mapped) => out.push(mapped),
            None => out.push(c),
        }
    }
    out
}

/// 推論入力テキストを正規化する（互換漢字の畳み込み + 欧文リガチャ等の展開
/// + かな繰り返し記号の解決）。
///
/// 1→N 展開された文字は、展開後の全バイトが原文の同じ文字（の先頭バイト位置）
/// を指す。Python 版 `normalize_input()` と対象カテゴリを一致させる（ただし
/// かな繰り返し記号の解決は Rust 独自。Python 推論は凍結のため）。
pub(crate) fn normalize_input(text: &str) -> NormalizedText {
    let mut out = String::with_capacity(text.len());
    let mut map: Vec<usize> = Vec::with_capacity(text.len());
    let mut changed = false;
    // 繰り返し記号（ゝゞヽヾ）解決用に、直前に出力したかな1文字を覚えておく。
    let mut prev_out: Option<char> = None;
    for (src_byte, c) in text.char_indices() {
        // かな繰り返し記号: 直前の出力かなを繰り返す（ゞ/ヾ は濁音化）。
        // 例「みすゞ」→「みすず」。々（漢字の同の字点）と違い読みでなくテキストで
        // 解決できるので、モデルに渡る前にここで畳む（rare な記号を既知のかなにする）。
        if let Some(resolved) = prev_out.and_then(|p| resolve_iteration(p, c)) {
            changed = true;
            out.push(resolved);
            map.extend(std::iter::repeat_n(src_byte, resolved.len_utf8()));
            prev_out = Some(resolved);
            continue;
        }

        let pushed_len = if let Some(expanded) = table::expand_lookup(c as u32) {
            changed = true;
            out.push_str(expanded);
            prev_out = expanded.chars().last();
            expanded.len()
        } else if let Some(mapped) = table::normalize_lookup(c as u32).and_then(char::from_u32) {
            changed = true;
            out.push(mapped);
            prev_out = Some(mapped);
            mapped.len_utf8()
        } else {
            out.push(c);
            prev_out = Some(c);
            c.len_utf8()
        };
        map.extend(std::iter::repeat_n(src_byte, pushed_len));
    }
    NormalizedText {
        text: out,
        byte_map: if changed { Some(map) } else { None },
    }
}

/// かな繰り返し記号を直前のかな `prev` で解決する。
/// `c` が繰り返し記号でなければ `None`。
///
/// - `ゝ`(U+309D) / `ヽ`(U+30FD): 直前かなをそのまま繰り返す
/// - `ゞ`(U+309E) / `ヾ`(U+30FE): 直前かなを濁音化して繰り返す（す→ず）
fn resolve_iteration(prev: char, c: char) -> Option<char> {
    match c {
        'ゝ' | 'ヽ' => Some(prev),
        'ゞ' | 'ヾ' => Some(voice(prev)),
        _ => None,
    }
}

/// かなに濁点を付けた形を返す。濁点を持たないかなはそのまま返す（best effort）。
/// ひらがな・カタカナとも か/さ/た/は 行は濁音形が +1、う/ウ のみ特殊。
fn voice(c: char) -> char {
    match c {
        'う' => 'ゔ',
        'ウ' => 'ヴ',
        'か' | 'き' | 'く' | 'け' | 'こ' | 'さ' | 'し' | 'す' | 'せ' | 'そ' | 'た' | 'ち'
        | 'つ' | 'て' | 'と' | 'は' | 'ひ' | 'ふ' | 'へ' | 'ほ' | 'カ' | 'キ' | 'ク' | 'ケ'
        | 'コ' | 'サ' | 'シ' | 'ス' | 'セ' | 'ソ' | 'タ' | 'チ' | 'ツ' | 'テ' | 'ト' | 'ハ'
        | 'ヒ' | 'フ' | 'ヘ' | 'ホ' => char::from_u32(c as u32 + 1).unwrap_or(c),
        _ => c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cjk_radicals_supplement() {
        assert_eq!(normalize_compat_ideographs("⺟"), "母");
        assert_eq!(normalize_compat_ideographs("⻳"), "龟");
    }

    #[test]
    fn iteration_mark_voiced_hiragana() {
        // 金子みすゞ: ゞ が直前「す」を濁音化して「ず」になる。
        let n = normalize_input("みすゞ");
        assert_eq!(n.text, "みすず");
        // 3文字とも3バイトで 1:1、byte_map は原文位置を指す。
        assert_eq!(n.byte_map.unwrap(), vec![0, 0, 0, 3, 3, 3, 6, 6, 6]);
    }

    #[test]
    fn iteration_mark_voiced_katakana() {
        // ヾ（カタカナ濁点繰り返し）: ス→ズ。
        assert_eq!(normalize_input("ミスヾ").text, "ミスズ");
    }

    #[test]
    fn iteration_mark_unvoiced() {
        // ゝ/ヽ はそのまま繰り返す。
        assert_eq!(normalize_input("こゝ").text, "ここ");
        assert_eq!(normalize_input("バナヽ").text, "バナナ");
    }

    #[test]
    fn iteration_mark_unvoiceable_prev_stays() {
        // 濁点を持たないかな（な）に ゞ が付いても best effort でそのまま繰り返す。
        assert_eq!(normalize_input("なゞ").text, "なな");
    }

    #[test]
    fn kangxi_radical() {
        assert_eq!(normalize_compat_ideographs("⼀"), "一");
    }

    #[test]
    fn cjk_compatibility_ideograph() {
        // U+F900 と U+8C48 は見た目が同じ「豈」だが別コードポイント。
        assert_eq!(normalize_compat_ideographs("\u{F900}"), "\u{8C48}");
    }

    #[test]
    fn cjk_compatibility_ideograph_supplement() {
        assert_eq!(normalize_compat_ideographs("\u{2F800}"), "丽");
    }

    #[test]
    fn passthrough_normal_kanji() {
        assert_eq!(normalize_compat_ideographs("漢字"), "漢字");
    }

    #[test]
    fn passthrough_non_kanji() {
        assert_eq!(
            normalize_compat_ideographs("あいうえおABC123"),
            "あいうえおABC123"
        );
    }

    #[test]
    fn compat_ideographs_does_not_expand_ligature() {
        // 1→1変換専用なので欧文リガチャは対象外（normalize_input が担当）
        assert_eq!(normalize_compat_ideographs("\u{FB03}"), "\u{FB03}");
    }

    #[test]
    fn normalize_input_identity() {
        let n = normalize_input("あいうABC123");
        assert_eq!(n.text, "あいうABC123");
        assert!(n.byte_map.is_none());
    }

    #[test]
    fn normalize_input_ligature_expansion() {
        // "oﬃce": o(1B) + ﬃ(3B, U+FB03) + c(1B) + e(1B)
        let n = normalize_input("o\u{FB03}ce");
        assert_eq!(n.text, "office");
        // 展開後の f/f/i（バイト1..4）はすべて原文のﬃ先頭バイト(1)を指す
        assert_eq!(n.byte_map.unwrap(), vec![0, 1, 1, 1, 4, 5]);
    }

    #[test]
    fn normalize_input_all_ligatures() {
        let n = normalize_input("\u{FB00}\u{FB01}\u{FB02}\u{FB03}\u{FB04}\u{FB05}\u{FB06}");
        assert_eq!(n.text, "fffiflffifflstst");
    }

    #[test]
    fn normalize_input_compat_ideograph() {
        // 互換漢字（1→1）も normalize_input で畳み込まれ、マップ付きで返る
        let n = normalize_input("⼀二");
        assert_eq!(n.text, "一二");
        // ⼀(3B)→一(3B): バイト対応は恒等だがマップは Some
        assert_eq!(n.byte_map.unwrap(), vec![0, 0, 0, 3, 3, 3]);
    }

    #[test]
    fn normalize_input_byte_length_change() {
        // U+2F800(4B) → 丽 U+4E3D(3B): 1→1変換でもバイト長が変わるケース
        let n = normalize_input("\u{2F800}あ");
        assert_eq!(n.text, "丽あ");
        assert_eq!(n.byte_map.unwrap(), vec![0, 0, 0, 4, 4, 4]);
    }

    #[test]
    fn normalize_input_empty() {
        let n = normalize_input("");
        assert_eq!(n.text, "");
        assert!(n.byte_map.is_none());
    }
}
