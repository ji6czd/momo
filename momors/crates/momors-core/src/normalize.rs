//! 漢字互換コードポイントの正規化モジュール。
//!
//! CJK部首補助・康煕部首・CJK互換漢字（および補助面）は、通常のCJK統合漢字
//! （U+4E00–U+9FFF 等）と見た目が同じか酷似しているが別の符号点に存在する。
//! 正規化せずに `char_type::get_char_type()` へ渡すと KANJI と判定されず
//! 記号・未定義文字として扱われてしまうため、対応する統合漢字へ畳み込む。
//!
//! Python 版 `momo_py.utils.normalize_compat_ideographs()` と挙動を一致させる。
//! 変換テーブルは `tools/gen_normalize_table.py` で Python から自動生成される。

mod table;

/// 文字列中の互換漢字系コードポイントを正規のCJK統合漢字へ畳み込む。
///
/// 対象4レンジはいずれも1文字→1文字の変換であることが生成スクリプト側で
/// 保証されているため、文字数・原文位置の対応はずれない。
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cjk_radicals_supplement() {
        assert_eq!(normalize_compat_ideographs("⺟"), "母");
        assert_eq!(normalize_compat_ideographs("⻳"), "龟");
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
}
