//! 括弧類の構造分類テーブル。
//!
//! 括弧まわりの分かち書きを当てるには、括弧を bypass としてモデルに流すのでは
//! 足りない。特徴量ウィンドウが括弧で分断されるうえ、周囲のマスあけも境界モデル
//! 任せになるためである。そこで推論前に括弧を本文から外し、推論後にルールで
//! 書き戻す（strip / reinsert）。実処理は [`crate::prediction`] にある。
//!
//! # 中身を本文と混ぜてよいか（[`Treatment`]）
//!
//! 外し方は括弧種で2通りに分かれる。判定基準は **中身が文の流れの一部か** :
//!
//! | | span ごと抜いた本文 | 括弧だけ抜いた本文 |
//! |---|---|---|
//! | 引用 `彼は『走れメロス』を読む` | `彼はを読む` — 壊れる | `彼は走れメロスを読む` — 自然 |
//! | 注釈 `オケ（オーケストラ）の団員` | `オケの団員` — 自然 | `オケオーケストラの団員` — 非語 |
//!
//! 引用の中身は文の項なので流れの中にいる。括弧だけ外せば自然文になり、モデルへの
//! 問い（`朝|おはよう` を空けるか）も正しい問いになる → [`Inline`]。
//!
//! 注釈の中身は傍注で流れの外にいる。括弧だけ外すと `オケオーケストラ` のような
//! 実在しない隣接ができ、特徴量ウィンドウが偽の文脈をまたぐ。しかも「モデルが
//! それを1語だと誤解している」ことに正しさが依存してしまい、モデルが賢くなるほど
//! 壊れる。span ごと抜けば本文は `オケの団員` と自然なままで、偽の隣接が最初から
//! 生じない → [`Aside`]。
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

/// 括弧の中身を本文と一緒に推論してよいか。モジュール冒頭の表を参照。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Treatment {
    /// 中身は文の流れの一部（引用系）。括弧だけ外して本文と一緒に推論する。
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
/// 汎化しない。かといって完全にstripすると、隣接文字のkanji_run/kanji_pos_first/
/// bigram/trigramが「括弧がそこにあった」ことを一切知りえず、区切りとして
/// 認識できない（実測: momo-inspect trace で「、」が読み分けに効く主因は
/// bi_p1_s/type_tri_p2_p1_s のような隣接ベースの特徴量だった）。
///
/// そこで、特徴量計算専用の系列（出力・マスあけを決めるstrip/Atom/splice経路とは
/// 別）でだけ、括弧の位置をこの3値の圧縮トークンに置き換えて残す。ctypeは
/// 既存のSymbolOpen/SymbolClose（get_char_type）のままでよい（役割はそちらで
/// 十分圧縮済み）。この位置自体には読み/境界の予測はさせない。
///
/// momo-py/src/momo_py/bracket.py と同じコードポイントを使うこと。
/// ずれるとモデルの語彙（char_s等の特徴量キー）がPython/Rustで食い違う。
pub(crate) const INLINE_OPEN_TOKEN: char = '\u{E000}';
pub(crate) const INLINE_CLOSE_TOKEN: char = '\u{E001}';
pub(crate) const ASIDE_TOKEN: char = '\u{E002}';

/// `cp` が上記3つの圧縮アイデンティティ・トークンのいずれかか。
/// 特徴量専用ビューからプレースホルダ行をフィルタして除くのに使う。
pub(crate) fn is_identity_token(cp: u32) -> bool {
    cp == INLINE_OPEN_TOKEN as u32 || cp == INLINE_CLOSE_TOKEN as u32 || cp == ASIDE_TOKEN as u32
}

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
    fn is_identity_token_recognizes_all_three_and_nothing_else() {
        assert!(is_identity_token(INLINE_OPEN_TOKEN as u32));
        assert!(is_identity_token(INLINE_CLOSE_TOKEN as u32));
        assert!(is_identity_token(ASIDE_TOKEN as u32));
        assert!(!is_identity_token('「' as u32));
        assert!(!is_identity_token('あ' as u32));
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
