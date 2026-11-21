//! 読みの別推論器（BERT など）を差し込む口。
//!
//! 読みモデル（LR）は 1 文字ずつ局所窓の特徴量で読みを選ぶ。ここに差し込む推論器は
//! 1 行を丸ごと見て、漢字・漢数字の読みの候補を先にまとめて付ける。推論ループは
//! その位置では LR の代わりにこの候補から読みを選ぶ（候補の絞り込み・「々」の
//! 連濁 2 択・カスタム辞書・助数詞パスの優先順位は従来どおり）。
//!
//! 推論器は切れ目ごとのマスあけの確率も返せる。返した切れ目では、境界モデルの
//! 代わりにその確率でマスあけを決める（英数字ラン内部を割らない保証など、境界の
//! 規則は従来どおり先に効く。推論ループが境界を問わない位置では使わない）。
//!
//! 推論器の実装（ONNX Runtime など重い依存を持つもの）は別クレートに置き、
//! `momors-core` 自体は依存を増やさない。

use crate::char_type::CharType;
use crate::featurize::SourceEntry;

/// 1 文字ぶんの読みの候補。
#[derive(Debug, Clone)]
pub struct SlotScores {
    /// 入力行の中での、その文字の **UTF-8 バイト位置**。
    pub byte_pos: usize,
    /// (ラベル番号, 確率) を確率の降順に並べたもの。ラベル番号は
    /// [`SlotReader::labels`] の添字。
    pub candidates: Vec<(u32, f32)>,
}

/// 1 つの切れ目のマスあけの確率。
#[derive(Debug, Clone, Copy)]
pub struct SplitScore {
    /// 切れ目の **直後** の文字の、入力行の中での UTF-8 バイト位置。
    ///
    /// 直前の文字ではなく直後の文字で表すのは、括弧の注釈を抜いた本文でも同じ
    /// 切れ目を指せるようにするため（「語（注釈）本文」の本文側では「語」の後ろの
    /// 切れ目は「本文」の直前、つまり「）」の後ろに当たる）。
    pub byte_pos: usize,
    /// マスをあける確率。
    pub prob: f32,
}

/// 1 行ぶんの推論結果。
#[derive(Debug, Clone, Default)]
pub struct LineScores {
    /// 読みの候補を付けた文字（`byte_pos` の昇順）。
    pub slots: Vec<SlotScores>,
    /// マスあけの確率を付けた切れ目（`byte_pos` の昇順）。境界を推論しない推論器は空。
    pub splits: Vec<SplitScore>,
}

/// 1 行を丸ごと見て、漢字・漢数字の読みの候補（と、切れ目ごとのマスあけの確率）を
/// 付ける推論器。
pub trait SlotReader: Send + Sync {
    /// ラベル表。継続（前の文字と一続きの読み）は `"---"`。
    fn labels(&self) -> &[String];

    /// 正規化済みの 1 行を読む。
    ///
    /// 読みの対象は漢字・漢数字だが、推論器が読めない文字（語彙外など）は含めなくてよい
    /// （その文字は LR が読む）。切れ目も、確率を出せないもの（空白に接する切れ目など）は
    /// 含めなくてよい（その切れ目は境界モデルが決める）。
    fn read_line(&self, text: &str) -> std::result::Result<LineScores, String>;
}

/// [`SlotReader`] と、そのラベルを読みモデルのクラス id に解決した対応表。
pub(crate) struct AttachedReader {
    pub reader: Box<dyn SlotReader>,
    /// ラベル番号 → 読みモデルのクラス id（クラスに無いラベルは `None`）。
    pub label_map: Vec<Option<u32>>,
}

impl std::fmt::Debug for AttachedReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttachedReader")
            .field("labels", &self.label_map.len())
            .finish()
    }
}

impl AttachedReader {
    /// 1 行を読み、候補をクラス id に直した表を返す。クラスに無いラベルの候補は落とす。
    pub fn read_line(&self, text: &str) -> crate::Result<LineReadings> {
        let line = self
            .reader
            .read_line(text)
            .map_err(|reason| crate::Error::Prediction { reason })?;
        let entries = line
            .slots
            .into_iter()
            .map(|s| {
                let cands = s
                    .candidates
                    .into_iter()
                    .filter_map(|(label, p)| {
                        Some((self.label_map.get(label as usize).copied().flatten()?, p))
                    })
                    .collect::<Vec<_>>();
                (s.byte_pos, cands)
            })
            .filter(|(_, cands)| !cands.is_empty())
            .collect();
        let splits = line.splits.iter().map(|s| (s.byte_pos, s.prob)).collect();
        Ok(LineReadings { entries, splits })
    }
}

/// 1 行ぶんの推論器の結果（読みの候補はクラス id に解決済み）。
#[derive(Debug, Default)]
pub(crate) struct LineReadings {
    /// (行の中のバイト位置, [(クラス id, 確率)] 確率の降順)。バイト位置の昇順。
    entries: Vec<(usize, Vec<(u32, f32)>)>,
    /// (切れ目の直後の文字のバイト位置, マスをあける確率)。バイト位置の昇順。
    splits: Vec<(usize, f32)>,
}

/// 推論ループから見た [`LineReadings`]。括弧の注釈を抜いた本文や、注釈の中身を
/// 再帰的に推論するときに、そのテキストの座標を行の座標へ写す。
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ReadingView<'a> {
    line: Option<&'a LineReadings>,
    /// テキストの先頭が行のどこにあるか（バイト位置）。
    base: usize,
    /// テキストのバイト位置 → `base` から数えたバイト位置（注釈を抜いた本文用）。
    map: Option<&'a [usize]>,
}

impl<'a> ReadingView<'a> {
    pub fn new(line: Option<&'a LineReadings>) -> Self {
        Self {
            line,
            base: 0,
            map: None,
        }
    }

    /// 部分文字列（`base` からの相対位置 `offset` で始まる）を見るビュー。
    pub fn shifted(self, offset: usize) -> Self {
        Self {
            base: self.base + offset,
            map: None,
            ..self
        }
    }

    /// 文字を抜いたテキスト（`map` はそのテキストのバイト位置 → 元のバイト位置）を見るビュー。
    pub fn mapped(self, map: &'a [usize]) -> Self {
        Self {
            map: Some(map),
            ..self
        }
    }

    /// テキストのバイト位置 → 行のバイト位置。
    fn line_pos(&self, local: usize) -> usize {
        self.base + self.map.map_or(local, |m| m[local])
    }

    /// 推論器の候補が付いた文字なら、その候補（確率の降順）を返す。
    ///
    /// 対象は 1 文字の漢字・漢数字だけ（拗音などの複合ユニットは対象外）。
    pub fn get(&self, entry: &SourceEntry) -> Option<&'a [(u32, f32)]> {
        let line = self.line?;
        if entry.compound_len != 1
            || !matches!(entry.ctype, CharType::Kanji | CharType::JapaneseNumeric)
        {
            return None;
        }
        let pos = self.line_pos(entry.orig_idx as usize);
        let k = line.entries.binary_search_by_key(&pos, |(p, _)| *p).ok()?;
        Some(&line.entries[k].1)
    }

    /// `next` の直前の切れ目に推論器が確率を付けていれば、マスをあける確率を返す。
    pub fn split_before(&self, next: &SourceEntry) -> Option<f32> {
        let line = self.line?;
        let pos = self.line_pos(next.orig_idx as usize);
        let k = line.splits.binary_search_by_key(&pos, |(p, _)| *p).ok()?;
        Some(line.splits[k].1)
    }
}
