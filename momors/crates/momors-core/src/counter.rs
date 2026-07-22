//! 助数詞（「N日」「N人」…）の**多桁**の読みをルールで守る。
//!
//! ## 役割は「多桁の保護」だけ
//!
//! 一桁の読み（2日→ふつか・1人→ひとり・3つ→みっつ）は**モデル＝データが正本**。
//! ルールは触らない（読みが変ならデータを足す）。ルールの仕事は、**二桁以上の数が
//! 一桁の特殊読みで上書きされるのを防ぐ**こと：文字単位モデルは「21人→2ヒトリ」
//! 「22日→2ミッカ」のように下一桁へ特殊読みを漏らすので、多桁ランは数字＋接尾で
//! 確定させる（21人→21ニン・22日→22ニチ）。
//!
//! ## 多桁の例外（[`CounterSpec::exceptions`]）
//!
//! 数符（算用数字）で正しく読めない値だけを例外に持つ。点字の読み分けをそのまま表す：
//!
//! - `Spell`  … 数符では誤読するので訓読みを綴る。`数符10か`は「じゅうか」に読める
//!   ため 10日は **トオカ** と綴る（20日=ハツカ も同様）。
//! - `Suffix` … 数符で読めるので数字は保ち接尾だけ替える。`数符14か`は「じゅうよっか」
//!   と読めるため 14日は **14＋カ**（下一桁4＝よっか）。24日も同じ。
//! - 例外に無い多桁値 … 数字＋`default_suffix`（21日→21ニチ・21人→21ニン）。
//!
//! ## 発火ゲート（[`CounterSpec::gate_readings`]）
//!
//! 空でなければ、助数詞文字のモデル読みがこの集合のときだけ発火する。「2024日本」の
//! 日→ニ を誤って日付にしない安全弁（日のみ）。数字直後がまず助数詞になる 人 は空。
//!
//! ## 対象外
//!
//! - 一桁（モデル）・漢数字ラン（既存の [`crate::numeric`] 経由）。
//! - つ（多桁が実在しない＝ルール不関与。1〜9つ は全てモデル）。
//! - 本・匹・回 等、連濁/促音の音韻規則で決まる助数詞（値の表にならない）。

use crate::char_type::CharType;
use crate::featurize::SourceEntry;

/// 多桁の例外読みの種類。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Multi {
    /// 数符では誤読するので訓読みを綴る（10→"トオカ"）。
    Spell(&'static str),
    /// 数字は保ち接尾だけ差し替える（14→ +"カ" ＝ 14カ）。
    Suffix(&'static str),
}

/// 助数詞パスの発火結果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CounterAction {
    /// 数字ラン＋助数詞を丸ごとこの訓読みで置き換える（10日→トオカ）。
    Spell(&'static str),
    /// 原文の数字はそのまま出し、助数詞をこの接尾で出す（21日→"2""1"＋"ニチ"）。
    DigitsPlus(&'static str),
}

/// 1つの助数詞のルール定義。
struct CounterSpec {
    /// 助数詞のコードポイント（`'日' as u32` 等）。
    counter_cp: u32,
    /// 例外に無い多桁値の接尾（日→"ニチ"・人→"ニン"）。
    default_suffix: &'static str,
    /// 発火ゲート：空でなければ助数詞文字のモデル読みがこの集合のときだけ発火（日本よけ）。
    gate_readings: &'static [&'static str],
    /// 多桁の例外（数符で正しく読めない値）。
    exceptions: &'static [(u32, Multi)],
}

impl CounterSpec {
    fn needs_gate(&self) -> bool {
        !self.gate_readings.is_empty()
    }

    fn gate_ok(&self, model_reading: &str) -> bool {
        self.gate_readings.is_empty() || self.gate_readings.contains(&model_reading)
    }

    /// 多桁値 `value` に対する動作。
    fn action(&self, value: u32) -> CounterAction {
        match self.exceptions.iter().find(|(v, _)| *v == value) {
            Some((_, Multi::Spell(k))) => CounterAction::Spell(k),
            Some((_, Multi::Suffix(s))) => CounterAction::DigitsPlus(s),
            None => CounterAction::DigitsPlus(self.default_suffix),
        }
    }
}

/// 助数詞テーブル。多桁の保護と例外を持つ助数詞だけを登録する。
const COUNTERS: &[CounterSpec] = &[
    // 日: 10/20 は綴る（数符で誤読）・14/24 は下一桁4＝カ・他は数字＋ニチ。
    CounterSpec {
        counter_cp: '日' as u32,
        default_suffix: "ニチ",
        gate_readings: &["ニチ", "カ", "タチ"],
        exceptions: &[
            (10, Multi::Spell("トオカ")),
            (14, Multi::Suffix("カ")),
            (20, Multi::Spell("ハツカ")),
            (24, Multi::Suffix("カ")),
        ],
    },
    // 人: 特殊な多桁読みは無い。多桁は一律 数字＋ニン（21人→21ニン で漏れを断つ）。
    CounterSpec {
        counter_cp: '人' as u32,
        default_suffix: "ニン",
        gate_readings: &[],
        exceptions: &[],
    },
];

fn find_counter(cp: u32) -> Option<&'static CounterSpec> {
    COUNTERS.iter().find(|c| c.counter_cp == cp)
}

/// コードポイント `counter_cp` の助数詞が発火ゲート（モデル読み確認）を要するか。
///
/// `false`（人）ならモデル読みを見ずに [`resolve_multi`] を呼んでよい。
pub(crate) fn counter_needs_gate(counter_cp: u32) -> bool {
    find_counter(counter_cp).is_some_and(CounterSpec::needs_gate)
}

/// **多桁**ラン（値 `value`）＋助数詞 `counter_cp` の動作を返す。一桁では呼ばない。
///
/// `model_reading` は助数詞文字のモデル読み（ゲート判定用。ゲートが空の助数詞では
/// 参照されない）。未登録の助数詞・ゲート不通過なら `None`（＝通常経路）。
pub(crate) fn resolve_multi(
    counter_cp: u32,
    value: u32,
    model_reading: &str,
) -> Option<CounterAction> {
    let spec = find_counter(counter_cp)?;
    if !spec.gate_ok(model_reading) {
        return None;
    }
    Some(spec.action(value))
}

/// 算用数字（半角/全角）1文字のコードポイントを 0..=9 に変換。数字でなければ `None`。
fn ascii_digit_value(cp: u32) -> Option<u32> {
    match cp {
        0x30..=0x39 => Some(cp - 0x30),       // 0-9
        0xFF10..=0xFF19 => Some(cp - 0xFF10), // ０-９（全角）
        _ => None,
    }
}

/// `seq[start]` から始まる算用数字ランの 10進値とラン終端 `end`（排他）を返す。
///
/// - `seq[start]` が算用数字でなければ `None`。
/// - 桁が非現実的に長く値が `u32` を溢れる場合は `None`。
///
/// 呼び出し側は `start` がランの先頭（直前が算用数字でない）であることを保証する。
/// 一桁か多桁かは `end - start` で判定する。
pub(crate) fn arabic_run(seq: &[SourceEntry], start: usize) -> Option<(u32, usize)> {
    ascii_digit_value(seq.get(start)?.cp)?;

    let mut end = start;
    let mut val: u64 = 0;
    while let Some(e) = seq.get(end) {
        match ascii_digit_value(e.cp) {
            Some(d) if e.ctype == CharType::Numeric => {
                val = val * 10 + u64::from(d);
                if val > u64::from(u32::MAX) {
                    return None;
                }
                end += 1;
            }
            _ => break,
        }
    }
    Some((val as u32, end))
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::featurize::to_source_seq;

    // --- 日 ---

    #[test]
    fn day_spell_exceptions() {
        // 数符で誤読する 10/20 は綴る
        assert_eq!(
            resolve_multi('日' as u32, 10, "カ"),
            Some(CounterAction::Spell("トオカ"))
        );
        assert_eq!(
            resolve_multi('日' as u32, 20, "カ"),
            Some(CounterAction::Spell("ハツカ"))
        );
    }

    #[test]
    fn day_suffix_yon() {
        // 下一桁4（14/24）は数字＋カ
        assert_eq!(
            resolve_multi('日' as u32, 14, "カ"),
            Some(CounterAction::DigitsPlus("カ"))
        );
        assert_eq!(
            resolve_multi('日' as u32, 24, "カ"),
            Some(CounterAction::DigitsPlus("カ"))
        );
    }

    #[test]
    fn day_default_protect() {
        // 例外以外の多桁は 数字＋ニチ（漏れ防止）
        for v in [11, 12, 13, 15, 19, 21, 22, 25, 30, 31, 100] {
            assert_eq!(
                resolve_multi('日' as u32, v, "ニチ"),
                Some(CounterAction::DigitsPlus("ニチ")),
                "v={v}"
            );
        }
    }

    #[test]
    fn day_gate_blocks_non_counter() {
        // 「2024日本」の 日→ニ はゲート外 → 発火しない
        assert_eq!(resolve_multi('日' as u32, 2024, "ニ"), None);
        assert!(counter_needs_gate('日' as u32));
    }

    // --- 人 ---

    #[test]
    fn nin_protect_no_gate() {
        // 多桁は一律 数字＋ニン・ゲート無し
        assert_eq!(
            resolve_multi('人' as u32, 10, ""),
            Some(CounterAction::DigitsPlus("ニン"))
        );
        assert_eq!(
            resolve_multi('人' as u32, 21, ""),
            Some(CounterAction::DigitsPlus("ニン"))
        );
        assert!(!counter_needs_gate('人' as u32));
    }

    #[test]
    fn unknown_counter() {
        // つ・匹 は未登録（多桁が無い/音韻系）→ 発火しない
        assert_eq!(resolve_multi('つ' as u32, 3, ""), None);
        assert_eq!(resolve_multi('匹' as u32, 3, ""), None);
    }

    // --- arabic_run（一桁/多桁の判定込み） ---

    #[test]
    fn arabic_run_len() {
        // 一桁: end-start==1
        let s = to_source_seq("5日");
        assert_eq!(arabic_run(&s, 0), Some((5, 1)));
        // 多桁: end-start==2
        let s = to_source_seq("21日");
        assert_eq!(arabic_run(&s, 0), Some((21, 2)));
        // 全角
        let s = to_source_seq("２０人");
        assert_eq!(arabic_run(&s, 0), Some((20, 2)));
    }

    #[test]
    fn arabic_run_stops_and_none() {
        let s = to_source_seq("3月5日");
        assert_eq!(arabic_run(&s, 0), Some((3, 1)));
        let s = to_source_seq("日");
        assert_eq!(arabic_run(&s, 0), None);
    }
}
