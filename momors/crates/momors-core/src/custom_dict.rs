//! カスタム辞書（フレーズ辞書）の読み込みと照合。
//!
//! `momopy predict --custom-dict` 相当の Rust 実装。エントリの読みは
//! `docs/dataset_raw_rule.md` §8 と同じ記法（`/` 区切り、空白ブロック
//! `/ /` が内部境界マーカー）で書く。エントリにマッチした範囲は読みを
//! 完全に上書きし、内部境界も辞書が確定する。エントリ両端の境界だけは
//! 辞書が持てず（先頭・末尾の `/ /` はロード時エラー）、通常の境界モデルに
//! 委ねる ── 直前の境界は「エントリ直前のユニットの通常の境界判定」に
//! 自然に委ねられ、直後の境界はエントリ最後のユニット位置で通常どおり
//! 境界モデルを呼ぶだけでよい（`prediction.rs` 側の呼び出し箇所を参照）。
//!
//! `name_dict.rs` と同じ「先頭コードポイントで引く HashMap、長さ降順
//! ソートで最長一致」構造を使う（`NameUnit` をそのまま再利用する）。

use std::collections::HashMap;
use std::path::Path;

use crate::Error;
use crate::Result;
use crate::featurize::{SourceEntry, to_source_seq};
use crate::name_dict::NameUnit;

/// カスタム辞書の1エントリ（照合用に前処理済み）。
#[derive(Debug, Clone)]
pub struct CustomDictEntry {
    /// ユニット列（拗音複合対応、cp/cp2/cp3 の完全一致で照合）。
    pub units: Vec<NameUnit>,
    /// ユニット別の読み。`units` と同じ長さ。
    pub readings: Vec<String>,
    /// `forced_split_after[k]` はユニット k の直後を強制的に区切るか。
    /// 長さは `readings.len() - 1`（末尾の境界は境界モデルに委ねるため
    /// ここには含まれない）。
    pub forced_split_after: Vec<bool>,
}

/// カスタム辞書の照合用インデックス。
///
/// キーは先頭ユニットの先頭コードポイント、値はエントリのリスト
/// （ユニット数の降順ソート済み = 最長一致用）。
pub type CustomDictIndex = HashMap<u32, Vec<CustomDictEntry>>;

/// 読み列を `/` で分割し、(ユニット別読み, forced_split_after) に変換する。
///
/// 空白1文字だけのブロック（`.../ /...`）が内部境界マーカー。先頭・末尾の
/// ブロックが境界マーカーなのはエラー（エントリ両端の境界は辞書が持てない
/// ─ 次に来る語に依存するため境界モデルに委ねる）。空文字ブロック
/// （`//` と続けて書いてしまった入力ミス）もエラーにする。
fn parse_reading_blocks(reading: &str) -> Result<(Vec<String>, Vec<bool>)> {
    let blocks: Vec<&str> = reading.split('/').collect();

    if blocks.first() == Some(&" ") || blocks.last() == Some(&" ") {
        return Err(Error::InvalidCustomDict {
            reason: format!(
                "先頭・末尾の空白ブロック（/ /）は指定できません（エントリ両端の境界は境界モデルに委ねます）: {reading:?}"
            ),
        });
    }

    let mut readings: Vec<String> = Vec::new();
    let mut forced_split_after: Vec<bool> = Vec::new();
    for block in blocks {
        if block == " " {
            // 直前に push した読み（ユニット）の直後を強制的に区切る。
            // 先頭ブロックが " " ではないことは確認済みなので readings は
            // 必ず1つ以上ある。
            let gap_idx = readings.len() - 1;
            if forced_split_after.len() <= gap_idx {
                forced_split_after.resize(gap_idx + 1, false);
            }
            forced_split_after[gap_idx] = true;
            continue;
        }
        if block.is_empty() {
            return Err(Error::InvalidCustomDict {
                reason: format!(
                    "空の読みブロックがあります（'//' の入力ミスの可能性）: {reading:?}"
                ),
            });
        }
        readings.push(block.to_string());
    }
    forced_split_after.resize(readings.len().saturating_sub(1), false);
    Ok((readings, forced_split_after))
}

/// カスタム辞書エントリ（表層形, 読み列）のリストから照合用インデックスを構築する。
pub(crate) fn build_custom_dict_index(entries: &[(String, String)]) -> Result<CustomDictIndex> {
    let mut index: CustomDictIndex = HashMap::new();
    for (surface, reading) in entries {
        let (readings, forced_split_after) = parse_reading_blocks(reading)?;

        let units: Vec<NameUnit> = to_source_seq(surface)
            .iter()
            .map(|e| NameUnit {
                cp: e.cp,
                cp2: e.cp2,
                cp3: e.cp3,
            })
            .collect();

        if units.is_empty() {
            return Err(Error::InvalidCustomDict {
                reason: format!("表層形が空です（読み: {reading:?}）"),
            });
        }
        if units.len() != readings.len() {
            return Err(Error::InvalidCustomDict {
                reason: format!(
                    "表層形「{surface}」のユニット数({}) と読みのブロック数({}) が一致しません: {reading:?}",
                    units.len(),
                    readings.len()
                ),
            });
        }

        let first = units[0].cp;
        index.entry(first).or_default().push(CustomDictEntry {
            units,
            readings,
            forced_split_after,
        });
    }
    for lists in index.values_mut() {
        lists.sort_by(|a, b| b.units.len().cmp(&a.units.len()));
    }
    Ok(index)
}

/// カスタム辞書 TSV ファイルを読み込む。
///
/// フォーマット: `表層形 [TAB] 読み`（`#` で始まる行・空行はコメントとして
/// スキップ）。読みの記法は [`parse_reading_blocks`] を参照。
pub fn load_custom_dict(path: &Path) -> Result<CustomDictIndex> {
    let content = std::fs::read_to_string(path).map_err(|e| Error::CustomDictIo {
        path: path.to_path_buf(),
        source: e,
    })?;

    let mut entries = Vec::new();
    for line in content.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(2, '\t');
        let surface = parts.next().unwrap_or("");
        let reading = parts.next().ok_or_else(|| Error::InvalidCustomDict {
            reason: format!("タブ区切りの読み列がありません: {line:?}"),
        })?;
        entries.push((surface.to_string(), reading.to_string()));
    }
    build_custom_dict_index(&entries)
}

/// ソース系列の位置 `i` から始まる最長一致エントリを返す（無ければ `None`）。
pub(crate) fn match_at<'a>(
    seq: &[SourceEntry],
    index: &'a CustomDictIndex,
    i: usize,
) -> Option<&'a CustomDictEntry> {
    let candidates = index.get(&seq[i].cp)?;
    for entry in candidates {
        let len = entry.units.len();
        if i + len > seq.len() {
            continue;
        }
        let matches = (0..len).all(|j| {
            let u = &entry.units[j];
            let e = &seq[i + j];
            u.cp == e.cp && u.cp2 == e.cp2 && u.cp3 == e.cp3
        });
        if matches {
            return Some(entry);
        }
    }
    None
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(entries: &[(&str, &str)]) -> CustomDictIndex {
        let v: Vec<(String, String)> = entries
            .iter()
            .map(|(s, r)| (s.to_string(), r.to_string()))
            .collect();
        build_custom_dict_index(&v).expect("valid entries")
    }

    #[test]
    fn single_word_no_internal_split() {
        // 電子計算機 → デン/シ/ケイ/サン/キ (内部で区切らない)
        let index = idx(&[("電子計算機", "デン/シ/ケイ/サン/キ")]);
        let seq = to_source_seq("電子計算機を使う");
        let m = match_at(&seq, &index, 0).expect("match");
        assert_eq!(m.units.len(), 5);
        assert_eq!(m.readings, vec!["デン", "シ", "ケイ", "サン", "キ"]);
        assert_eq!(m.forced_split_after, vec![false, false, false, false]);
    }

    #[test]
    fn internal_split_at_marked_position() {
        // 佐藤太郎 → サ/トー/ /タ/ロー (姓名間で区切る)
        let index = idx(&[("佐藤太郎", "サ/トー/ /タ/ロー")]);
        let seq = to_source_seq("佐藤太郎さん");
        let m = match_at(&seq, &index, 0).expect("match");
        assert_eq!(m.readings, vec!["サ", "トー", "タ", "ロー"]);
        assert_eq!(m.forced_split_after, vec![false, true, false]);
    }

    #[test]
    fn user_examples_taharazaka_and_haebaru() {
        let index = idx(&[
            ("田原坂", "タ/バル/ザカ"),
            ("南風原町", "ハ/エ/バル/ /チョー"),
        ]);

        let seq1 = to_source_seq("田原坂へ行く");
        let m1 = match_at(&seq1, &index, 0).expect("match");
        assert_eq!(m1.readings, vec!["タ", "バル", "ザカ"]);
        assert_eq!(m1.forced_split_after, vec![false, false]);

        let seq2 = to_source_seq("南風原町議会");
        let m2 = match_at(&seq2, &index, 0).expect("match");
        assert_eq!(m2.readings, vec!["ハ", "エ", "バル", "チョー"]);
        assert_eq!(m2.forced_split_after, vec![false, false, true]);
    }

    #[test]
    fn leading_boundary_marker_is_error() {
        let err =
            build_custom_dict_index(&[("南風原町".to_string(), " /ハ/エ/バル/チョー".to_string())]);
        assert!(err.is_err());
    }

    #[test]
    fn trailing_boundary_marker_is_error() {
        let err =
            build_custom_dict_index(&[("南風原町".to_string(), "ハ/エ/バル/チョー/ ".to_string())]);
        assert!(err.is_err());
    }

    #[test]
    fn double_slash_is_error() {
        let err =
            build_custom_dict_index(&[("佐藤太郎".to_string(), "サ/トー//タ/ロー".to_string())]);
        assert!(err.is_err());
    }

    #[test]
    fn unit_count_mismatch_is_error() {
        // 佐藤 (2ユニット) に対して読みは3ブロック
        let err = build_custom_dict_index(&[("佐藤".to_string(), "サ/トー/ロー".to_string())]);
        assert!(err.is_err());
    }

    #[test]
    fn no_match() {
        let index = idx(&[("田原坂", "タ/バル/ザカ")]);
        let seq = to_source_seq("鈴木さん");
        assert!(match_at(&seq, &index, 0).is_none());
    }

    #[test]
    fn longest_match_wins() {
        let index = idx(&[("佐藤", "サ/トー"), ("佐藤原", "サ/トー/バラ")]);
        let seq = to_source_seq("佐藤原さん");
        let m = match_at(&seq, &index, 0).expect("match");
        assert_eq!(m.units.len(), 3);
        assert_eq!(m.readings, vec!["サ", "トー", "バラ"]);
    }
}
