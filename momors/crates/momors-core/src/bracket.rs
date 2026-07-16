//! 括弧類の構造分類テーブル。
//!
//! 括弧は文中の任意位置に挿入でき、囲む内容の分かち書きをほぼ変えない
//! 「オーバーレイ」である。そこで推論前に本文から括弧を除去し、推論後に
//! ルールで書き戻す（strip / reinsert）。狙いは2点:
//!
//! 1. 特徴量ウィンドウが括弧で分断されず、素の本文としてモデルに渡る
//! 2. 括弧まわりのマスあけを、括弧種ごとの規則で決められる
//!
//! 実際の strip / reinsert は [`crate::prediction`] にある。
//!
//! 分類は少数なので TOML ではなく本モジュールに直接持つ（core を toml 非依存に
//! 保つ）。点字変換テーブル（momors-braille）側の `class`（open/close 等）とは
//! 関心が別（あちらはセル隣接のスペース、こちらは本文の再構成）であり、
//! モジュール独立性を優先して意図的に別管理にしている。

/// 括弧の食いつき方向。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    /// 開き括弧。直後の文字に密着する（「 → 「オ）。
    Open,
    /// 閉じ括弧。直前の文字に密着する（」 → ヨー」）。
    Close,
}

/// 括弧の「外エッジ」（開きの左側／閉じの右側）のマスあけ挙動。
///
/// 内エッジ（囲む内容に接する側）は常に密着なので表に持たない。
///
/// # 開きと閉じの非対称性
///
/// 括弧を除去すると、括弧が隔てていた2つの語が地続きになる。そこでモデルが
/// 下す判断を信じてよいかは、開きと閉じで違う:
///
/// - **開きの左**（[前の語]|[中身の先頭]）: 括弧はこの関係を**書き換える**。
///   注釈は前の語に食いつくべきなのに、素のテキストではただ隣接した2語に
///   見えるのでモデルは空けてしまう（`週末３連休` → `シューマツ␣3レンキュー`）。
///   よってモデルを信じられず、[`Attach`] のルールで上書きする。
/// - **閉じの右**（[中身の末尾]|[次の語]）: 括弧はこの関係を**書き換えない**。
///   助詞なら続き（`）の`）、新しい語なら空ける（`）カラオケ`）——答えは括弧の
///   有無で変わらない。よってモデルの判断が信頼でき、[`Defer`] でよい。
///
/// [`Attach`]: Outer::Attach
/// [`Defer`]: Outer::Defer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outer {
    /// 周囲の分かち書き（境界モデルの判定）をそのまま使う。
    Defer,
    /// 外側のマスあけを消す。注釈系の開き括弧が前の語に食いつくときに使う。
    Attach,
    /// 外側を必ずあける。現状どの括弧もこの値を取らないが、規則の三値目として
    /// reinsert 側で完全に扱えるようにしてある。
    #[allow(dead_code)]
    Space,
}

/// `c` が括弧なら (役割, 外エッジ挙動) を返す。該当しなければ `None`。
///
/// 分類が実運用で合わないと分かれば、ここの割り当てを直すだけで調整できる。
///
/// 閉じ括弧が一律 [`Outer::Defer`] なのは偶然ではなく、閉じの右側の分かち書きが
/// 括弧の有無で変わらないため（[`Outer`] の非対称性の項を参照）。ルールで
/// 上書きする必要があるのは、前の語との関係を書き換える注釈系の開き括弧だけ。
pub(crate) fn lookup(c: char) -> Option<(Role, Outer)> {
    use Outer::*;
    use Role::*;
    Some(match c {
        // 引用系: 引用は独立した句なので、開きの左も語境界の判断に委ねる
        '「' => (Open, Defer),
        '」' => (Close, Defer),
        '『' => (Open, Defer),
        '』' => (Close, Defer),
        '“' => (Open, Defer),
        '”' => (Close, Defer),
        '‘' => (Open, Defer),
        '’' => (Close, Defer),
        // 注釈・挿入系: 開きだけ前の語へ詰める（週末（３連休） / 今日（とりあえず））
        '（' => (Open, Attach),
        '）' => (Close, Defer),
        '(' => (Open, Attach),
        ')' => (Close, Defer),
        '｛' => (Open, Attach),
        '｝' => (Close, Defer),
        '【' => (Open, Attach),
        '】' => (Close, Defer),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotation_brackets_defer() {
        assert_eq!(lookup('「'), Some((Role::Open, Outer::Defer)));
        assert_eq!(lookup('」'), Some((Role::Close, Outer::Defer)));
        assert_eq!(lookup('『'), Some((Role::Open, Outer::Defer)));
        assert_eq!(lookup('』'), Some((Role::Close, Outer::Defer)));
    }

    #[test]
    fn annotation_open_attaches_close_defers() {
        // 開きだけ前の語へ詰める。閉じの右は語境界の判断が括弧の有無で変わらない
        // ので Defer（モデルが当てた `）カラオケ` の空きを潰さない）。
        assert_eq!(lookup('（'), Some((Role::Open, Outer::Attach)));
        assert_eq!(lookup('）'), Some((Role::Close, Outer::Defer)));
        assert_eq!(lookup('('), Some((Role::Open, Outer::Attach)));
        assert_eq!(lookup(')'), Some((Role::Close, Outer::Defer)));
        assert_eq!(lookup('【'), Some((Role::Open, Outer::Attach)));
        assert_eq!(lookup('】'), Some((Role::Close, Outer::Defer)));
    }

    #[test]
    fn all_closers_defer() {
        for c in "」』”’）)｝】".chars() {
            assert_eq!(lookup(c).map(|(_, o)| o), Some(Outer::Defer), "{c}");
        }
    }

    #[test]
    fn non_brackets_are_none() {
        assert_eq!(lookup('あ'), None);
        assert_eq!(lookup('。'), None);
        assert_eq!(lookup('、'), None);
        assert_eq!(lookup('…'), None);
        assert_eq!(lookup('A'), None);
    }
}
