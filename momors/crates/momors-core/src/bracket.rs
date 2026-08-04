//! 括弧類の構造分類テーブル。
//!
//! 括弧まわりの分かち書きを当てるには、括弧を素通しの記号として他の記号と
//! 同列にモデルへ流すだけでは足りない。個々の括弧字形（「『""''【 など）は
//! bigram/trigram特徴量にとって希少すぎて汎化しないため、特徴量計算だけは
//! Role×Treatmentを3値に圧縮したトークンに置き換えたビューから取る（実処理は
//! [`crate::prediction`] にある）。出力・ctype分岐・境界判定は常に実文字の
//! まま ─ 読みは他の記号と同じくbypass、直後のマスあけは境界モデルに委ねる
//! （`CharType::skips_boundary_check` 参照）。
//!
//! # 中身を本文の流れに残してよいか（[`Treatment`]）
//!
//! 括弧種によって扱いが2通りに分かれる。判定基準は **中身が文の流れの一部か** :
//!
//! - 引用 `彼は『走れメロス』を読む`: 中身は文の項なので流れの中にいる。
//!   括弧ごとそのまま本文に残して推論しても自然文のままで、モデルへの問い
//!   （`朝|おはよう` を空けるか）も正しい問いになる → [`Inline`]。
//! - 注釈 `オケ（オーケストラ）の団員`: 中身は傍注で流れの外にいる。その場に
//!   残すと `オケオーケストラ` のような実在しない隣接ができ、特徴量ウィンドウ
//!   が偽の文脈をまたぐ。span ごと本文から抜いて独立に推論すれば、本文は
//!   `オケの団員` と自然なままで、偽の隣接が最初から生じない → [`Aside`]。
//!
//! 分類は少数なので TOML ではなく本モジュールに直接持つ（core を toml 非依存に
//! 保つ）。点字変換テーブル（momors-braille）側の `class` とは関心が別
//! （あちらはセル隣接のスペース、こちらは本文の再構成）であり、モジュール
//! 独立性を優先して意図的に別管理にしている。
//!
//! [`Inline`]: Treatment::Inline
//! [`Aside`]: Treatment::Aside

/// 括弧の食いつき方向。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    /// 開き括弧。直後の文字に密着する（「 → 「オ）。
    Open,
    /// 閉じ括弧。直前の文字に密着する（」 → ヨー」）。
    Close,
}

/// 括弧の中身を本文と一緒に推論してよいか。モジュール冒頭の説明を参照。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Treatment {
    /// 中身は文の流れの一部（引用系）。括弧文字も本文から外さず、実文字の
    /// まま推論に乗せる（特徴量計算だけトークン置換する）。
    Inline,
    /// 中身は流れの外の傍注（注釈系）。span ごと本文から抜いて独立に推論し、
    /// 前の語へ食いつけて合成する。
    Aside,
}

/// `c` が括弧なら (役割, 中身の扱い) を返す。該当しなければ `None`。
///
/// 分類が実運用で合わないと分かれば、ここの割り当てを直すだけで調整できる。
pub(crate) fn lookup(c: char) -> Option<(Role, Treatment)> {
    use Role::*;
    use Treatment::*;
    Some(match c {
        // 引用系: 中身は文の項
        '「' => (Open, Inline),
        '」' => (Close, Inline),
        '『' => (Open, Inline),
        '』' => (Close, Inline),
        '“' => (Open, Inline),
        '”' => (Close, Inline),
        '‘' => (Open, Inline),
        '’' => (Close, Inline),
        '【' => (Open, Inline),
        '】' => (Close, Inline),
        // 注釈・挿入系: 中身は傍注
        '（' => (Open, Aside),
        '）' => (Close, Aside),
        '(' => (Open, Aside),
        ')' => (Close, Aside),
        '｛' => (Open, Aside),
        '｝' => (Close, Aside),
        _ => return None,
    })
}

/// `open` と `close` が同じ括弧ペアか。
pub(crate) fn is_pair(open: char, close: char) -> bool {
    matches!(
        (open, close),
        ('「', '」')
            | ('『', '』')
            | ('“', '”')
            | ('‘', '’')
            | ('（', '）')
            | ('(', ')')
            | ('｛', '｝')
            | ('【', '】')
    )
}

/// 特徴量計算専用の圧縮アイデンティティ・トークン（Private Use Area）。
///
/// 個々の括弧字形（「『""''【 など）はbigram/trigram特徴量にとって希少すぎて
/// 汎化しない。そこで特徴量計算専用の系列（出力・ctype分岐・境界判定を決める
/// `text` 側とは別のビュー）でだけ、括弧の位置をこの3値の圧縮トークンに
/// 置き換える。ctypeは既存のSymbolOpen/SymbolClose（get_char_type）のままで
/// よい（役割はそちらで十分圧縮済み）。
///
/// `INLINE_OPEN_TOKEN`/`INLINE_CLOSE_TOKEN` は `text` 側にも同じ位置に実文字
/// （括弧そのもの）が1:1で存在する ─ Inline括弧は本文から外さないため、その
/// 行自身が実際に読み/境界の対象になる（読みはbypass、境界は境界モデル）。
/// 一方 `ASIDE_TOKEN` は span 丸ごと1文字に圧縮したプレースホルダで、`text`
/// 側には対応する実文字が無い（Aside の中身は独立した副文として本文から
/// 完全に抜かれる）。
///
/// momo-py/src/momo_py/bracket.py と同じコードポイントを使うこと。
/// ずれるとモデルの語彙（char_s等の特徴量キー）がPython/Rustで食い違う。
pub(crate) const INLINE_OPEN_TOKEN: char = '\u{E000}';
pub(crate) const INLINE_CLOSE_TOKEN: char = '\u{E001}';
pub(crate) const ASIDE_TOKEN: char = '\u{E002}';

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotation_is_inline() {
        for c in "「」『』“”‘’【】".chars() {
            assert_eq!(lookup(c).map(|(_, t)| t), Some(Treatment::Inline), "{c}");
        }
    }

    #[test]
    fn annotation_is_aside() {
        for c in "（）()｛｝".chars() {
            assert_eq!(lookup(c).map(|(_, t)| t), Some(Treatment::Aside), "{c}");
        }
    }

    #[test]
    fn roles() {
        for c in "「『“‘（(｛【".chars() {
            assert_eq!(lookup(c).map(|(r, _)| r), Some(Role::Open), "{c}");
        }
        for c in "」』”’）)｝】".chars() {
            assert_eq!(lookup(c).map(|(r, _)| r), Some(Role::Close), "{c}");
        }
    }

    #[test]
    fn pairs_match() {
        assert!(is_pair('（', '）'));
        assert!(is_pair('「', '」'));
        assert!(!is_pair('（', '」'));
        assert!(!is_pair('「', '）'));
    }

    #[test]
    fn non_brackets_are_none() {
        assert_eq!(lookup('あ'), None);
        assert_eq!(lookup('。'), None);
        assert_eq!(lookup('、'), None);
        assert_eq!(lookup('…'), None);
        assert_eq!(lookup('A'), None);
    }

    #[test]
    fn identity_tokens_match_python_bracket_py() {
        // momo-py/src/momo_py/bracket.py の INLINE_OPEN_TOKEN/INLINE_CLOSE_TOKEN/
        // ASIDE_TOKEN と同じコードポイントであること（ずれるとモデルの語彙が
        // Python/Rustで食い違う）。
        assert_eq!(INLINE_OPEN_TOKEN as u32, 0xE000);
        assert_eq!(INLINE_CLOSE_TOKEN as u32, 0xE001);
        assert_eq!(ASIDE_TOKEN as u32, 0xE002);
    }
}
