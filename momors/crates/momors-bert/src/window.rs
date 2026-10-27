//! 長い行を、BERT の入力長に収まる窓に区切る。
//!
//! 窓どうしは前後に `overlap` 文字ずつ重ね、各窓では重なりを除いた中央（採用範囲）の
//! 結果だけを使う。採用範囲は重なりも欠けもなく行全体を覆う。窓の端の文字は片側の
//! 文脈しか見られないので、その位置の結果は隣の窓（そこでは中央寄り）から取る。

use std::ops::Range;

/// 1 つの窓。位置は行の中の文字番号。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Window {
    /// BERT に入れる範囲。
    pub span: Range<usize>,
    /// 結果を採用する範囲（`span` の内側）。
    pub keep: Range<usize>,
}

/// `n` 文字の行を、`max_len` 文字以下の窓に区切る。`n <= max_len` なら窓は 1 つ。
pub fn plan(n: usize, max_len: usize, overlap: usize) -> Vec<Window> {
    assert!(max_len > 2 * overlap, "窓が重なりより短い");
    if n <= max_len {
        return vec![Window {
            span: 0..n,
            keep: 0..n,
        }];
    }
    let mut out = Vec::new();
    let mut keep_start: usize = 0;
    loop {
        let start = keep_start.saturating_sub(overlap);
        let end = (start + max_len).min(n);
        let keep_end = if end == n { n } else { end - overlap };
        out.push(Window {
            span: start..end,
            keep: keep_start..keep_end,
        });
        if keep_end == n {
            return out;
        }
        keep_start = keep_end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_covers(ws: &[Window], n: usize, max_len: usize) {
        let mut next = 0;
        for w in ws {
            assert!(w.span.len() <= max_len, "{w:?}");
            assert!(
                w.span.start <= w.keep.start && w.keep.end <= w.span.end,
                "{w:?}"
            );
            assert_eq!(w.keep.start, next, "{ws:?}");
            assert!(!w.keep.is_empty(), "{w:?}");
            next = w.keep.end;
        }
        assert_eq!(next, n);
    }

    #[test]
    fn short_line_is_one_window() {
        assert_eq!(
            plan(10, 510, 64),
            vec![Window {
                span: 0..10,
                keep: 0..10
            }]
        );
        assert_eq!(plan(510, 510, 64).len(), 1);
    }

    #[test]
    fn long_lines_are_covered_exactly_once() {
        for n in [511, 700, 1000, 1020, 1021, 5000] {
            let ws = plan(n, 510, 64);
            assert!(ws.len() >= 2);
            assert_covers(&ws, n, 510);
        }
    }

    #[test]
    fn inner_windows_keep_context_on_both_sides() {
        let ws = plan(2000, 510, 64);
        for w in &ws[1..ws.len() - 1] {
            assert_eq!(w.keep.start - w.span.start, 64);
            assert_eq!(w.span.end - w.keep.end, 64);
        }
    }
}
