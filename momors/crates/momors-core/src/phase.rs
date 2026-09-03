//! `predict` の段階別所要時間を積算する診断用タイマー。
//!
//! cargo feature `diagnostics` が無効なら全て no-op で、本番の推論経路には
//! コストを残さない（`Guard` は零サイズ、`start`/`add` は空関数）。
//! 有効時はスレッドローカルに段階ごとのナノ秒を積算し、[`take`] で取り出してリセットする。

/// 計測する段階。[`PHASE_NAMES`] と同じ順序。
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum Phase {
    /// `predict` 全体（正規化・括弧処理・座標写像を含む）。
    Total = 0,
    /// 入力正規化 (`normalize_input`)。
    Normalize,
    /// 文字列 → `SourceEntry` 列 + 人名辞書照合。
    Source,
    /// 特徴量キー生成 (`compute_source_features`)。
    Featurize,
    /// 語彙引き (`resolve_all`)。
    Resolve,
    /// 1 文字ごとの推論ループ全体（Read/Boundary を含む）。
    Loop,
    /// 読みモデルのスコア計算 + argmax (`read_argmax`)。
    Read,
    /// 境界モデルのスコア計算 (`boundary_has_split`)。
    Boundary,
    /// 末尾整形 + 原文→かなインデックス構築。
    Finalize,
}

/// 段階名（[`Phase`] の順）。
pub const PHASE_NAMES: [&str; 9] = [
    "total",
    "normalize",
    "source",
    "featurize",
    "resolve",
    "loop",
    "read",
    "boundary",
    "finalize",
];

#[cfg(feature = "diagnostics")]
mod imp {
    use super::Phase;
    use std::cell::RefCell;
    use std::time::Instant;

    thread_local! {
        static ACC: RefCell<[u128; 9]> = const { RefCell::new([0; 9]) };
    }

    #[inline]
    pub fn start() -> Instant {
        Instant::now()
    }

    #[inline]
    pub fn add(phase: Phase, t: Instant) {
        let ns = t.elapsed().as_nanos();
        ACC.with(|a| a.borrow_mut()[phase as usize] += ns);
    }

    /// 積算値を取り出してリセットする。
    pub fn take() -> [u128; 9] {
        ACC.with(|a| std::mem::take(&mut *a.borrow_mut()))
    }

    /// スコープを抜けるときに経過時間を積算するガード。
    pub struct Guard(Phase, Instant);

    impl Guard {
        #[inline]
        pub fn new(phase: Phase) -> Self {
            Guard(phase, Instant::now())
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            add(self.0, self.1);
        }
    }
}

#[cfg(not(feature = "diagnostics"))]
mod imp {
    use super::Phase;

    #[inline(always)]
    pub fn start() {}

    #[inline(always)]
    pub fn add(_phase: Phase, _t: ()) {}

    pub fn take() -> [u128; 9] {
        [0; 9]
    }

    pub struct Guard;

    impl Guard {
        #[inline(always)]
        pub fn new(_phase: Phase) -> Self {
            Guard
        }
    }
}

pub use imp::{Guard, add, start, take};
