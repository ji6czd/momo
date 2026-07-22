//! 人名辞書（gazetteer）の照合とフラグ計算。
//!
//! Python 版 `momo_py/name_dict.py` の `build_name_index` /
//! `compute_name_flags` に対応する。`.mbm` (version 0x03) 末尾の
//! 人名辞書テーブルから構築され、推論時に各ユニットへ B/I/O フラグを
//! 付与して `NameFlag*` 特徴量の入力にする。
//!
//! Python 側と同じく、照合は音節ユニット単位（拗音複合ユニットは1単位）の
//! 最長一致・左から右への貪欲マッチで行う。学習時（Python）と推論時（Rust）
//! で完全に同一の手順になるよう、変更時は必ず両実装を同期させること。

use std::collections::HashMap;

use crate::featurize::{SourceEntry, to_source_seq};

// ============================================================
// フラグ値
// ============================================================
// exporter.py の `_NAME_FLAG_MAP = {"B": 1, "I": 2}` と対応する。
// 0 (OUT) は特徴量を発火させない。

/// 人名スパン外。
pub(crate) const NAME_FLAG_OUT: u8 = 0;
/// 人名スパンの先頭 (B)。
pub(crate) const NAME_FLAG_BEGIN: u8 = 1;
/// 人名スパンの継続 (I)。
pub(crate) const NAME_FLAG_INSIDE: u8 = 2;

// ============================================================
// インデックス構造
// ============================================================

/// 人名1エントリを構成する1ユニット分のコードポイント。
///
/// [`SourceEntry`] と同じユニット分割（拗音複合対応）で、照合は
/// cp/cp2/cp3 の完全一致で行う（ctype は Python 版同様、比較しない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NameUnit {
    pub cp: u32,
    pub cp2: u32,
    pub cp3: u32,
}

impl NameUnit {
    #[inline]
    fn matches(&self, e: &SourceEntry) -> bool {
        self.cp == e.cp && self.cp2 == e.cp2 && self.cp3 == e.cp3
    }
}

/// 人名辞書の1エントリ（照合用に前処理済み）。
#[derive(Debug, Clone)]
pub struct NameDictEntry {
    /// ユニット列（拗音複合対応、cp/cp2/cp3 の完全一致で照合）
    pub units: Vec<NameUnit>,
    /// ユニット別の固定読み（カタカナ）。`units` と同じ長さ。
    /// 読み未登録のエントリは `None`（フラグ特徴量のみに使われる）。
    pub readings: Option<Vec<String>>,
}

/// 人名辞書の照合用インデックス。
///
/// キーは先頭ユニットの先頭コードポイント、値はエントリのリスト
/// （ユニット数の降順ソート済み = 最長一致用）。
pub type NameIndex = HashMap<u32, Vec<NameDictEntry>>;

/// 人名エントリ（表層形, ユニット別読み）のリストから照合用インデックスを構築する。
///
/// Python 版 `build_name_index()` と対応。表層形は [`to_source_seq`] で
/// ユニット分解するため、入力テキスト側と同じ分割規則が適用される。
/// 読みのユニット数が表層形のユニット数と一致しない場合
/// （Python 側とのユニット分割差異など）は、安全側に倒して読みを捨てる。
pub(crate) fn build_name_index(names: &[(String, Option<Vec<String>>)]) -> NameIndex {
    let mut index: NameIndex = HashMap::new();
    for (name, readings) in names {
        let units: Vec<NameUnit> = to_source_seq(name)
            .iter()
            .map(|e| NameUnit {
                cp: e.cp,
                cp2: e.cp2,
                cp3: e.cp3,
            })
            .collect();
        let Some(first) = units.first() else {
            continue;
        };
        let readings = readings
            .as_ref()
            .filter(|r| r.len() == units.len())
            .cloned();
        index
            .entry(first.cp)
            .or_default()
            .push(NameDictEntry { units, readings });
    }
    for lists in index.values_mut() {
        lists.sort_by(|a, b| b.units.len().cmp(&a.units.len()));
    }
    index
}

/// ソース系列に人名辞書を最長一致で照合する。
///
/// Python 版 `compute_name_matches()` と同一の手順:
/// 左から右へ走査し、マッチしたら B, I, I, ... を立てて末尾の次へ跳ぶ。
///
/// 戻り値: (B/I/O フラグ列, ユニット別辞書読み列)。
/// 読みは辞書に読みが登録されているスパン内の位置のみ `Some`。
pub(crate) fn compute_name_matches(
    seq: &[SourceEntry],
    index: &NameIndex,
) -> (Vec<u8>, Vec<Option<String>>) {
    let n = seq.len();
    let mut flags = vec![NAME_FLAG_OUT; n];
    let mut readings_out: Vec<Option<String>> = vec![None; n];
    if index.is_empty() {
        return (flags, readings_out);
    }

    let mut i = 0;
    while i < n {
        let mut matched = 0usize;
        let mut matched_readings: Option<&Vec<String>> = None;
        if let Some(candidates) = index.get(&seq[i].cp) {
            for entry in candidates {
                let len = entry.units.len();
                if i + len > n {
                    continue;
                }
                if (0..len).all(|j| entry.units[j].matches(&seq[i + j])) {
                    matched = len;
                    matched_readings = entry.readings.as_ref();
                    break;
                }
            }
        }
        if matched > 0 {
            flags[i] = NAME_FLAG_BEGIN;
            for flag in &mut flags[i + 1..i + matched] {
                *flag = NAME_FLAG_INSIDE;
            }
            if let Some(readings) = matched_readings {
                for j in 0..matched {
                    readings_out[i + j] = Some(readings[j].clone());
                }
            }
            i += matched;
        } else {
            i += 1;
        }
    }
    (flags, readings_out)
}

/// ソース系列に人名辞書を照合し、位置ごとの B/I/O フラグのみ返す。
#[cfg(test)]
pub(crate) fn compute_name_flags(seq: &[SourceEntry], index: &NameIndex) -> Vec<u8> {
    compute_name_matches(seq, index).0
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<(String, Option<Vec<String>>)> {
        v.iter().map(|s| (s.to_string(), None)).collect()
    }

    fn with_readings(surface: &str, readings: &[&str]) -> (String, Option<Vec<String>>) {
        (
            surface.to_string(),
            Some(readings.iter().map(|r| r.to_string()).collect()),
        )
    }

    #[test]
    fn surname_and_given_name() {
        // 佐=B 藤=I 太=B 郎=I 以降 O（Python 版 test_surname_and_given_name と対応）
        let index = build_name_index(&names(&["佐藤", "太郎"]));
        let seq = to_source_seq("佐藤太郎さんにあった");
        let flags = compute_name_flags(&seq, &index);
        assert_eq!(&flags[..4], &[1, 2, 1, 2]);
        assert!(flags[4..].iter().all(|&f| f == NAME_FLAG_OUT));
    }

    #[test]
    fn no_match() {
        let index = build_name_index(&names(&["佐藤"]));
        let seq = to_source_seq("鈴木さん");
        let flags = compute_name_flags(&seq, &index);
        assert!(flags.iter().all(|&f| f == NAME_FLAG_OUT));
    }

    #[test]
    fn longest_match_wins() {
        // 「佐藤原」と「佐藤」の両方が登録されている場合は長い方を採用
        let index = build_name_index(&names(&["佐藤", "佐藤原"]));
        let seq = to_source_seq("佐藤原さん");
        let flags = compute_name_flags(&seq, &index);
        assert_eq!(&flags[..3], &[1, 2, 2]);
    }

    #[test]
    fn compound_unit_name() {
        // 拗音を含むひらがな名は複合ユニット単位で照合される
        // りょうさん → りょ / う / さ / ん
        let index = build_name_index(&names(&["りょう"]));
        let seq = to_source_seq("りょうさん");
        assert_eq!(seq.len(), 4);
        let flags = compute_name_flags(&seq, &index);
        assert_eq!(flags, vec![1, 2, 0, 0]);
    }

    #[test]
    fn compound_unit_must_match_whole() {
        // 辞書「りよ」（2ユニット）は入力「りょ」（1複合ユニット）にマッチしない
        let index = build_name_index(&names(&["りよ"]));
        let seq = to_source_seq("りょう");
        let flags = compute_name_flags(&seq, &index);
        assert!(flags.iter().all(|&f| f == NAME_FLAG_OUT));
    }

    #[test]
    fn empty_index() {
        let seq = to_source_seq("佐藤さん");
        let flags = compute_name_flags(&seq, &NameIndex::new());
        assert!(flags.iter().all(|&f| f == NAME_FLAG_OUT));
    }

    #[test]
    fn empty_seq() {
        let index = build_name_index(&names(&["佐藤"]));
        assert!(compute_name_flags(&[], &index).is_empty());
    }

    #[test]
    fn readings_assigned_per_unit() {
        // 読み登録済みスパンだけ readings が入る（Python 版
        // test_readings_assigned_per_unit と対応）
        let entries = vec![
            with_readings("渡辺", &["ワタ", "ナベ"]),
            ("太郎".to_string(), None),
        ];
        let index = build_name_index(&entries);
        let seq = to_source_seq("渡辺太郎さん");
        let (flags, readings) = compute_name_matches(&seq, &index);
        assert_eq!(&flags[..4], &[1, 2, 1, 2]);
        assert_eq!(readings[0].as_deref(), Some("ワタ"));
        assert_eq!(readings[1].as_deref(), Some("ナベ"));
        assert_eq!(readings[2], None);
        assert_eq!(readings[3], None);
        assert_eq!(readings[4], None);
    }

    #[test]
    fn reading_unit_mismatch_is_dropped() {
        // 読みのユニット数が表層形と合わない場合は読みを捨てる（フラグは有効）
        let entries = vec![with_readings("佐藤", &["サトー"])];
        let index = build_name_index(&entries);
        let seq = to_source_seq("佐藤さん");
        let (flags, readings) = compute_name_matches(&seq, &index);
        assert_eq!(&flags[..2], &[1, 2]);
        assert!(readings.iter().all(|r| r.is_none()));
    }
}
