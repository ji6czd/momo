//! `.mbmf`（量子化前 float32 サイドカー）から読み込むモデルデータ構造。
//!
//! [`crate::model::MomoModel`] とほぼ同じフィールド構成だが、読みモデル重み・
//! 境界モデル重みが int8 量子化されておらず、`f32` のまま保持する（scale も持たない）。
//! `momo_py.exporter.export_float()` が書き出す `.mbmf` を [`crate::float_loader`]
//! で読み込んで構築する。
//!
//! `.mbm`（量子化済み）と `.mbmf`（量子化前）を見比べることで、量子化による
//! 誤差がテキスト推論の結果にどれだけ影響するかを検証できる
//! （`momo-compare-quant` CLI が用途）。

use crate::Result;
use crate::boundary::FloatBoundary;
use crate::feature::FeatureKey;
use crate::model::VocabEntry;
use crate::name_dict::NameIndex;
use crate::weight_model::WeightModel;

/// `.mbmf` から読み込まれたモデルデータ。
///
/// フィールドの意味は [`crate::model::MomoModel`] と対応するが、重み部分が
/// 量子化されていない点だけが異なる。
///
/// `pub` だが `float_model` モジュール自体は非公開 (`mod float_model;`) なので、
/// クレート外からは [`crate::FloatPredictor`] 型エイリアス経由でしか到達できない
/// （フィールドはすべて `pub(crate)` のままなので、外部から直接構築・参照はできない）。
#[derive(Debug)]
pub struct FloatMomoModel {
    pub(crate) vocab: Vec<VocabEntry>,
    pub(crate) read_classes: Vec<String>,

    // --- 読みモデル重み (CSC・float32・量子化なし) ---
    /// CSC 列ポインタ (size: `n_features + 1`)
    pub(crate) csc_colptr: Vec<u32>,
    /// CSC 行インデックス = クラスID (size: `n_nonzero`)
    pub(crate) csc_rowind: Vec<u16>,
    /// CSC 非ゼロ値 (size: `n_nonzero`)。量子化なしの実値そのもの。
    pub(crate) csc_data: Vec<f32>,

    /// 各クラスの intercept (size: `n_classes`)
    pub(crate) intercept_read: Vec<f32>,

    // --- 境界モデル ---
    /// 線形（float32・量子化なし）または木のアンサンブル（GBDT）。[`crate::boundary`] 参照。
    pub(crate) boundary: FloatBoundary,

    pub(crate) name_dict: NameIndex,
    pub(crate) single_char_dict: Vec<(char, Vec<String>)>,

    pub(crate) n_classes: u32,
    pub(crate) n_features: u32,
}

impl FloatMomoModel {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// 語彙テーブルから `key` に対応する `feature_id` を引く。
    ///
    /// 事前条件: `vocab` は Rust の `Ord` でソート済みであること。
    #[inline]
    fn vocab_find(&self, key: &FeatureKey) -> Option<u32> {
        self.vocab
            .binary_search_by(|entry| entry.key.cmp(key))
            .ok()
            .map(|idx| self.vocab[idx].feature_id)
    }

    /// 統合語彙テーブルを引いて [`crate::boundary::VocabRef`] を返す。
    /// [`crate::model::MomoModel::resolve`] の float 版。
    #[inline]
    fn resolve(&self, key: &FeatureKey) -> Option<crate::boundary::VocabRef> {
        self.vocab
            .binary_search_by(|entry| entry.key.cmp(key))
            .ok()
            .map(|idx| {
                let e = &self.vocab[idx];
                crate::boundary::VocabRef {
                    feature_id: e.feature_id,
                    cat: e.cat(),
                }
            })
    }
}

impl Default for FloatMomoModel {
    fn default() -> Self {
        Self {
            vocab: Vec::new(),
            read_classes: Vec::new(),
            csc_colptr: Vec::new(),
            csc_rowind: Vec::new(),
            csc_data: Vec::new(),
            intercept_read: Vec::new(),
            boundary: FloatBoundary::default(),
            name_dict: NameIndex::new(),
            single_char_dict: Vec::new(),
            n_classes: 0,
            n_features: 0,
        }
    }
}

// ============================================================
// WeightModel
// ============================================================

impl WeightModel for FloatMomoModel {
    /// 量子化なしで直接 f32 のまま `scores` に加算するため、
    /// int32 スクラッチのような中間バッファは不要。
    type Scratch = ();

    fn load(path: &std::path::Path) -> Result<Self> {
        crate::float_loader::load(path)
    }

    fn load_from_bytes(bytes: &[u8]) -> Result<Self> {
        crate::float_loader::load_from_bytes(bytes)
    }

    fn new_scratch(&self) {}

    fn n_classes(&self) -> u32 {
        self.n_classes
    }

    fn n_features(&self) -> u32 {
        self.n_features
    }

    fn read_class(&self, class_id: u32) -> Option<&str> {
        self.read_classes.get(class_id as usize).map(String::as_str)
    }

    fn read_classes(&self) -> &[String] {
        &self.read_classes
    }

    fn vocab_find(&self, key: &FeatureKey) -> Option<u32> {
        self.vocab_find(key)
    }

    fn name_dict(&self) -> &NameIndex {
        &self.name_dict
    }

    fn take_single_char_dict(&mut self) -> Vec<(char, Vec<String>)> {
        std::mem::take(&mut self.single_char_dict)
    }

    fn compute_read_scores(&self, feat_ids: &[u32], _scratch: &mut (), scores: &mut [f32]) {
        scores.copy_from_slice(&self.intercept_read);
        for &feat_id in feat_ids {
            if feat_id >= self.n_features {
                continue;
            }
            let col_start = self.csc_colptr[feat_id as usize] as usize;
            let col_end = self.csc_colptr[feat_id as usize + 1] as usize;
            for j in col_start..col_end {
                let cls = self.csc_rowind[j] as usize;
                scores[cls] += self.csc_data[j];
            }
        }
    }

    fn compute_boundary_score(&self, feat_keys: &[FeatureKey]) -> f32 {
        self.boundary.compute_score(feat_keys, |k| self.resolve(k))
    }

    fn read_feature_column(&self, feat_id: u32) -> Vec<(u32, f32)> {
        if feat_id >= self.n_features {
            return Vec::new();
        }
        let col_start = self.csc_colptr[feat_id as usize] as usize;
        let col_end = self.csc_colptr[feat_id as usize + 1] as usize;
        (col_start..col_end)
            .map(|j| (self.csc_rowind[j] as u32, self.csc_data[j]))
            .collect()
    }
}
