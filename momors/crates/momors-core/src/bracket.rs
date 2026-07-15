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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outer {
    /// 周囲の分かち書き（境界モデルの判定）をそのまま使う。引用符系。
    Defer,
    /// 外側のマスあけを消す（オケ（オーケストラ）で継ぎ目を詰める）。注釈系。
    Attach,
    /// 外側を必ずあける。現状どの括弧もこの値を取らないが、規則の三値目として
    /// reinsert 側で完全に扱えるようにしてある。
    #[allow(dead_code)]
    Space,
}

/// `c` が括弧なら (役割, 外エッジ挙動) を返す。該当しなければ `None`。
///
/// 分類が実運用で合わないと分かれば、ここの割り当てを直すだけで調整できる。
pub(crate) fn lookup(c: char) -> Option<(Role, Outer)> {
    use Outer::*;
    use Role::*;
    Some(match c {
        // 引用系: 外エッジは周囲の語境界に従う（Defer）
        '「' => (Open, Defer),
        '」' => (Close, Defer),
        '『' => (Open, Defer),
        '』' => (Close, Defer),
        '“' => (Open, Defer),
        '”' => (Close, Defer),
        '‘' => (Open, Defer),
        '’' => (Close, Defer),
        // 注釈・挿入系: 外エッジを詰める（Attach）
        '（' => (Open, Attach),
        '）' => (Close, Attach),
        '(' => (Open, Attach),
        ')' => (Close, Attach),
        '｛' => (Open, Attach),
        '｝' => (Close, Attach),
        '【' => (Open, Attach),
        '】' => (Close, Attach),
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
    fn parenthesis_brackets_attach() {
        assert_eq!(lookup('（'), Some((Role::Open, Outer::Attach)));
        assert_eq!(lookup('）'), Some((Role::Close, Outer::Attach)));
        assert_eq!(lookup('('), Some((Role::Open, Outer::Attach)));
        assert_eq!(lookup(')'), Some((Role::Close, Outer::Attach)));
        assert_eq!(lookup('【'), Some((Role::Open, Outer::Attach)));
        assert_eq!(lookup('】'), Some((Role::Close, Outer::Attach)));
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
