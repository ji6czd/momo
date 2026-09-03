//! モデルデータ構造。
//!
//! C++ 版の `model.hpp` の `MomoModel` 構造体に対応する。
//! `.mbm` ファイルから読み込まれた推論用パラメータをすべて保持する。
//! 一度読み込んだら不変として扱い、[`Predictor`] が共有参照で利用する。
//!
//! [`Predictor`]: crate::Predictor

use crate::Result;
use crate::boundary::Boundary;
use crate::feature::FeatureKey;
use crate::name_dict::NameIndex;
use crate::weight_model::WeightModel;
use std::sync::Mutex;

// ============================================================
// 語彙テーブル
// ============================================================

/// カテゴリカル列なしを表す番兵（統合語彙テーブル version 0x07）。
///
/// GBDT 境界モデルのカテゴリカル `(column, code)` を持たないキー（Bias や、
/// そもそも境界モデルが線形のモデル）で `cat_column` に入る。
pub(crate) const NO_CAT_COLUMN: u32 = u32::MAX;

/// 統合語彙テーブルのエントリ（version 0x07）。
///
/// 読みモデルの one-hot `feature_id` と、GBDT 境界モデルのカテゴリカル
/// `(cat_column, cat_code)` を1つのキーにまとめて持つ。0x06 までは読み語彙
/// (`FeatureKey → feature_id`) と GBDT の `cat_vocab` (`FeatureKey → (col, code)`)
/// が別テーブルで、同じキー集合のペイロードを二重に格納していた。
///
/// - `feature_id` はファイル内の並び順（0..n_features）で暗黙に決まる。CSC 重み
///   行列の列インデックスと一致させるため、ファイルは feature_id 昇順で格納する。
/// - `cat_column == NO_CAT_COLUMN` はカテゴリカル列を持たないことを表す。
///
/// `Vec<VocabEntry>` を `key` でソートしてバイナリサーチで使う。C++ 版の
/// `operator<` とソート順が異なる可能性があるため、loader が読み込み後に Rust の
/// `Ord` で必ず再ソートすること。
#[derive(Debug, Clone)]
pub struct VocabEntry {
    pub(crate) key: FeatureKey,
    pub(crate) feature_id: u32,
    pub(crate) cat_column: u32,
    pub(crate) cat_code: u32,
}

impl VocabEntry {
    /// GBDT カテゴリカル `(column, code)`。列を持たないキーでは `None`。
    #[inline]
    pub(crate) fn cat(&self) -> Option<(u32, u32)> {
        (self.cat_column != NO_CAT_COLUMN).then_some((self.cat_column, self.cat_code))
    }
}

// ============================================================
// MomoModel
// ============================================================

/// モデルデータ。
///
/// `.mbm` ファイルから読み込まれた以下を保持する:
///
/// - **語彙テーブル**: `FeatureKey` → `feature_id` の検索表
/// - **読みラベル**: `class_id` → カナ文字列
/// - **読みモデル重み**: CSC 形式 + int8 量子化のスパース行列
/// - **境界モデル重み**: 単語境界判定用の int8 量子化重み
///
/// ## CSC (Compressed Sparse Column) 形式について
///
/// 読みモデルの重み行列 `(n_classes × n_features)` は、特徴量 (列) 方向で
/// 圧縮されている:
///
/// - `csc_colptr[f]..csc_colptr[f+1]` が 特徴量 `f` の非ゼロ要素のインデックス範囲
/// - `csc_rowind[j]` が 非ゼロ要素の行インデックス (= class_id)
/// - `csc_data[j]` が int8 量子化された重み値
///   （実値は `csc_data[j] * read_scale[csc_rowind[j]]`。scale はクラスごと）
///
/// 推論時は「アクティブな特徴量ID列」が分かっているので、その列だけを
/// 走査して全クラスのスコアに加算する。
#[derive(Debug)]
pub struct MomoModel {
    // --- 語彙テーブル ---
    /// 特徴量キー → feature_id のルックアップテーブル。
    /// `binary_search` で検索するため、Rust の `Ord` でソート済みであること。
    pub(crate) vocab: Vec<VocabEntry>,

    // --- 読みラベルテーブル ---
    /// クラスID → 読みラベル (UTF-8)
    pub(crate) read_classes: Vec<String>,

    // --- 読みモデル重み (CSC・int8 量子化) ---
    /// クラスごとの量子化スケール係数 (size: `n_classes`)
    /// 実値 = `csc_data[j] * read_scale[csc_rowind[j]]`
    pub(crate) read_scale: Vec<f32>,
    /// CSC 列ポインタ (size: `n_features + 1`)
    pub(crate) csc_colptr: Vec<u32>,
    /// CSC 行インデックス = クラスID (size: `n_nonzero`)
    ///
    /// クラスID は `u16` で持つ。非ゼロ要素1個あたり `csc_data` (1 byte) と対で
    /// 常駐するため、ここが `u32` だと行インデックスだけで重み値の4倍を占め、
    /// int8 量子化の効果を打ち消してしまう。`n_classes <= MAX_CLASSES` は
    /// loader がヘッダ検証時に保証する。
    pub(crate) csc_rowind: Vec<u16>,
    /// CSC 非ゼロ値 (size: `n_nonzero`)
    pub(crate) csc_data: Vec<i8>,

    // --- 読みモデル intercept ---
    /// 各クラスの intercept (size: `n_classes`)
    pub(crate) intercept_read: Vec<f32>,

    // --- 境界モデル ---
    /// 線形（int8量子化）または木のアンサンブル（GBDT）。[`crate::boundary`] 参照。
    pub(crate) boundary: Boundary,

    // --- 人名辞書 (version 0x03 で追加) ---
    /// 人名辞書の照合用インデックス。辞書なしモデルでは空。
    /// 推論時に [`compute_name_matches`] で B/I フラグと固定読みを計算し、
    /// `NameFlag*` 特徴量と読みフォールバックの入力にする。
    ///
    /// [`compute_name_matches`]: crate::name_dict::compute_name_matches
    pub(crate) name_dict: NameIndex,

    // --- 単一文字辞書 (version 0x04 途中で追加) ---
    /// 読みモデルの候補制約に使う単一文字辞書（漢字・数字1文字→既知の読みリスト）。
    /// char でソート済み（binary_search 用）。
    /// `PredictorConfig::single_char_dict_path` の明示指定があればそちらが優先される。
    pub(crate) single_char_dict: Vec<(char, Vec<String>)>,

    // --- サイズ情報 ---
    pub(crate) n_classes: u32,
    pub(crate) n_features: u32,

    // --- 密な列の組み合わせ和キャッシュ（DenseSumCache 参照） ---
    pub(crate) dense_cache: Mutex<DenseSumCache>,
}

// ============================================================
// 密な列の組み合わせ和キャッシュ
// ============================================================
//
// 読みモデルの CSC 列は大半が疎（平均 3 要素）だが、文字種など「どの文字にも現れる」
// 汎用特徴の列はクラスのほぼ全部に非ゼロを持つ（w4 モデルで 67 列、全非ゼロの 9%）。
// ところが 1 文字あたりに触る非ゼロ要素の約 96% はこの密な列で占められる
// （1 文字 ≈ 11,400 要素のうち疎な列は ≈ 400）。
//
// 密な列の組み合わせは文脈の文字種パターンで決まるため種類が少なく、同じ組み合わせが
// 繰り返し現れる。そこで「ヒットした密な列の id 集合」をキーに、その列だけを足し合わせた
// int32 ベクトルをキャッシュし、ヒット時は疎な列だけを足す。整数加算なので順序に依らず
// 結果は完全に一致する。
//
// ESP32-P4（PSRAM 常駐・360MHz）で読みスコア計算が 1 文字 549 µs → 約 100 µs を狙う。
// PC でも同じ比率で効く。

/// 1 文字分のキーに含めることができる密な列の最大数。超えた分は疎として扱う（結果は同じ）。
const DENSE_KEY_MAX: usize = 16;

/// キャッシュのスロット数。1 スロット = n_classes × 4 バイト（w4 で約 6KB）。
const DENSE_CACHE_SLOTS: usize = 64;

#[derive(Debug, Default)]
struct DenseSlot {
    key: [u32; DENSE_KEY_MAX],
    key_len: u8,
    /// 最終利用時刻（LRU 用）。0 = 空きスロット。
    stamp: u64,
    sum: Vec<i32>,
}

/// 密な列の組み合わせ和キャッシュ本体。[`MomoModel::dense_cache`]。
#[derive(Debug, Default)]
pub(crate) struct DenseSumCache {
    slots: Vec<DenseSlot>,
    clock: u64,
}

impl DenseSumCache {
    /// `key`（昇順ソート済み）に対応する和ベクトルがあれば `dst` にコピーして true。
    fn get(&mut self, key: &[u32], dst: &mut [i32]) -> bool {
        self.clock += 1;
        let clock = self.clock;
        for slot in &mut self.slots {
            if slot.stamp != 0
                && slot.key_len as usize == key.len()
                && slot.key[..key.len()] == *key
            {
                slot.stamp = clock;
                dst.copy_from_slice(&slot.sum);
                return true;
            }
        }
        false
    }

    /// `key` → `sum` を登録する。満杯なら最も長く使われていないスロットを置き換える。
    fn put(&mut self, key: &[u32], sum: &[i32]) {
        self.clock += 1;
        let clock = self.clock;
        let slot = if self.slots.len() < DENSE_CACHE_SLOTS {
            self.slots.push(DenseSlot::default());
            self.slots.last_mut().unwrap()
        } else {
            self.slots.iter_mut().min_by_key(|s| s.stamp).unwrap()
        };
        slot.key[..key.len()].copy_from_slice(key);
        slot.key_len = key.len() as u8;
        slot.stamp = clock;
        slot.sum.clear();
        slot.sum.extend_from_slice(sum);
    }
}

impl MomoModel {
    /// 空のモデルを作成する。
    ///
    /// loader 側でファイル内容を読み込んで各フィールドを埋める用途。
    /// 直接外部から使うことは想定していない。
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// 特徴量 `feat_id` の CSC 列を `acc` に加算する（`feat_id < n_features` は呼び出し側で保証）。
    #[inline]
    fn add_column(&self, feat_id: u32, acc: &mut [i32]) {
        let col_start = self.csc_colptr[feat_id as usize] as usize;
        let col_end = self.csc_colptr[feat_id as usize + 1] as usize;
        // 列の範囲は loader が colptr の単調性・上限を検証済み（loader.rs read_csc_structure）。
        // rowind < n_classes も loader が検証済みで、acc の長さは n_classes（new_scratch）。
        // 組み込み（ESP32-P4）で 1 文字あたり約 1.1 万要素をこのループが処理するため、
        // 3 つの境界チェックを外す。
        debug_assert!(col_end <= self.csc_rowind.len() && col_end <= self.csc_data.len());
        let rows = &self.csc_rowind[col_start..col_end];
        let vals = &self.csc_data[col_start..col_end];
        for (&cls, &v) in rows.iter().zip(vals) {
            debug_assert!((cls as usize) < acc.len());
            // SAFETY: loader が rowind の各要素 < n_classes を保証し、acc.len() == n_classes。
            unsafe {
                *acc.get_unchecked_mut(cls as usize) += v as i32;
            }
        }
    }

    /// 語彙テーブルから `key` に対応する `feature_id` を引く。
    ///
    /// 事前条件: `vocab` は Rust の `Ord` でソート済みであること。
    /// 見つからない場合は `None`。
    #[inline]
    pub(crate) fn vocab_find(&self, key: &FeatureKey) -> Option<u32> {
        self.vocab
            .binary_search_by(|entry| entry.key.cmp(key))
            .ok()
            .map(|idx| self.vocab[idx].feature_id)
    }

    /// 統合語彙テーブルを1回のバイナリサーチで引き、境界モデルが必要とする
    /// [`crate::boundary::VocabRef`]（`feature_id` と カテゴリカル `(column, code)`）を返す。
    /// 線形境界は `feature_id`、GBDT は `cat` を使う。
    #[inline]
    pub(crate) fn resolve(&self, key: &FeatureKey) -> Option<crate::boundary::VocabRef> {
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

    /// 特徴量次元数。
    #[inline]
    pub fn n_features(&self) -> u32 {
        self.n_features
    }

    /// 読みラベル (クラス) 数。
    #[inline]
    pub fn n_classes(&self) -> u32 {
        self.n_classes
    }

    /// クラスIDから読みラベル文字列を引く。
    ///
    /// 範囲外の `class_id` は `None`。
    pub fn read_class(&self, class_id: u32) -> Option<&str> {
        self.read_classes.get(class_id as usize).map(String::as_str)
    }
}

impl Default for MomoModel {
    /// C++ 版と同じ初期値で構築する。
    ///
    /// `boundary` は [`Boundary::default`] の安全側の初期値（線形・scale=1.0）で
    /// 初期化する。実際には loader がファイルから値を読み込んで上書きする。
    /// `read_scale` はクラスごとの配列のため、loader が必ず `n_classes` 件で
    /// 埋めることを前提に空 `Vec` で初期化する。
    fn default() -> Self {
        Self {
            vocab: Vec::new(),
            read_classes: Vec::new(),
            read_scale: Vec::new(),
            csc_colptr: Vec::new(),
            csc_rowind: Vec::new(),
            csc_data: Vec::new(),
            intercept_read: Vec::new(),
            boundary: Boundary::default(),
            name_dict: NameIndex::new(),
            single_char_dict: Vec::new(),
            n_classes: 0,
            n_features: 0,
            dense_cache: Mutex::new(DenseSumCache::default()),
        }
    }
}

// ============================================================
// WeightModel
// ============================================================

impl WeightModel for MomoModel {
    /// int32 で加算してから、最後にクラスごとの scale を掛ける
    /// （量子化前の挙動・性能を変えないため、既存のロジックをそのまま維持する）。
    type Scratch = Vec<i32>;

    fn load(path: &std::path::Path) -> Result<Self> {
        crate::loader::load(path)
    }

    fn load_from_bytes(bytes: &[u8]) -> Result<Self> {
        crate::loader::load_from_bytes(bytes)
    }

    fn new_scratch(&self) -> Vec<i32> {
        vec![0i32; self.n_classes as usize]
    }

    fn n_classes(&self) -> u32 {
        self.n_classes()
    }

    fn n_features(&self) -> u32 {
        self.n_features()
    }

    fn read_class(&self, class_id: u32) -> Option<&str> {
        self.read_class(class_id)
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

    fn compute_read_scores(&self, feat_ids: &[u32], int_scores: &mut Vec<i32>, scores: &mut [f32]) {
        let acc = &mut int_scores[..];

        // 密な列（非ゼロがクラス数の半分以上）と疎な列に分ける。密な列の id は
        // キャッシュのキーになるので昇順に整列する（挿入ソート、高々 DENSE_KEY_MAX 個）。
        let dense_threshold = self.n_classes.div_ceil(2);
        let mut dense_key = [0u32; DENSE_KEY_MAX];
        let mut n_dense = 0usize;
        let mut sparse_ids = [0u32; 64];
        let mut n_sparse = 0usize;
        let mut overflow: Vec<u32> = Vec::new();
        for &feat_id in feat_ids {
            if feat_id >= self.n_features {
                continue;
            }
            let nnz = self.csc_colptr[feat_id as usize + 1] - self.csc_colptr[feat_id as usize];
            if nnz >= dense_threshold && n_dense < DENSE_KEY_MAX {
                let mut i = n_dense;
                while i > 0 && dense_key[i - 1] > feat_id {
                    dense_key[i] = dense_key[i - 1];
                    i -= 1;
                }
                dense_key[i] = feat_id;
                n_dense += 1;
            } else if n_sparse < sparse_ids.len() {
                sparse_ids[n_sparse] = feat_id;
                n_sparse += 1;
            } else {
                overflow.push(feat_id);
            }
        }
        let dense_key = &dense_key[..n_dense];

        // 密な列: キャッシュにあればコピー、無ければ計算して登録する。
        let cached = if n_dense > 0 {
            let mut cache = self.dense_cache.lock().unwrap_or_else(|e| e.into_inner());
            cache.get(dense_key, acc)
        } else {
            false
        };
        if !cached {
            acc.fill(0);
            for &feat_id in dense_key {
                self.add_column(feat_id, acc);
            }
            if n_dense > 0 {
                let mut cache = self.dense_cache.lock().unwrap_or_else(|e| e.into_inner());
                cache.put(dense_key, acc);
            }
        }

        // 疎な列
        for &feat_id in sparse_ids[..n_sparse].iter().chain(&overflow) {
            self.add_column(feat_id, acc);
        }

        for ((score, &acc), (&intercept, &scale)) in scores
            .iter_mut()
            .zip(acc.iter())
            .zip(self.intercept_read.iter().zip(&self.read_scale))
        {
            *score = intercept + (acc as f32) * scale;
        }
    }

    fn compute_boundary_score(&self, feat_keys: &[FeatureKey]) -> f32 {
        self.boundary.compute_score(feat_keys, |k| self.resolve(k))
    }

    fn resolve(&self, key: &FeatureKey) -> Option<crate::boundary::VocabRef> {
        self.resolve(key)
    }

    fn compute_boundary_score_resolved(&self, refs: &[crate::boundary::VocabRef]) -> f32 {
        self.boundary.compute_score_resolved(refs)
    }

    fn read_feature_column(&self, feat_id: u32) -> Vec<(u32, f32)> {
        if feat_id >= self.n_features {
            return Vec::new();
        }
        let col_start = self.csc_colptr[feat_id as usize] as usize;
        let col_end = self.csc_colptr[feat_id as usize + 1] as usize;
        (col_start..col_end)
            .map(|j| {
                let cls = self.csc_rowind[j] as u32;
                let weight = (self.csc_data[j] as f32) * self.read_scale[cls as usize];
                (cls, weight)
            })
            .collect()
    }
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::char_type::CharType;
    use crate::feature::FeatureType;

    #[test]
    fn default_matches_cpp_initial_values() {
        let m = MomoModel::default();
        assert_eq!(m.n_classes, 0);
        assert_eq!(m.n_features, 0);
        // read_scale はクラスごとの配列。loader が埋める前は空。
        assert!(m.read_scale.is_empty());
        match &m.boundary {
            crate::boundary::Boundary::Linear {
                scale, intercept, ..
            } => {
                assert_eq!(*scale, 1.0);
                assert_eq!(*intercept, [0.0, 0.0]);
            }
            crate::boundary::Boundary::Tree(_) => panic!("既定値は線形であるべき"),
        }
        assert!(m.vocab.is_empty());
        assert!(m.read_classes.is_empty());
    }

    #[test]
    fn vocab_find_on_empty() {
        let m = MomoModel::default();
        let key = FeatureKey::char_1(FeatureType::CharSelf, 0x4E00);
        assert_eq!(m.vocab_find(&key), None);
    }

    #[test]
    fn vocab_find_basic() {
        let mut m = MomoModel::default();

        // Rust の Ord でソート済みになるように構築する。
        // (実装としては loader 側で sort() を呼ぶことになる)
        let ve = |key, feature_id| VocabEntry {
            key,
            feature_id,
            cat_column: NO_CAT_COLUMN,
            cat_code: 0,
        };
        m.vocab = vec![
            ve(FeatureKey::no_payload(FeatureType::Bias), 0),
            ve(
                FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji),
                1,
            ),
            ve(FeatureKey::char_1(FeatureType::CharSelf, 0x4E00), 2),
            ve(FeatureKey::char_1(FeatureType::CharSelf, 0x4E01), 3),
        ];
        // 念のため明示的にソート
        m.vocab.sort_by(|a, b| a.key.cmp(&b.key));

        // 存在するキー
        let k0 = FeatureKey::no_payload(FeatureType::Bias);
        assert_eq!(m.vocab_find(&k0), Some(0));

        let k1 = FeatureKey::char_1(FeatureType::CharSelf, 0x4E00);
        assert_eq!(m.vocab_find(&k1), Some(2));

        // 存在しないキー
        let k_missing = FeatureKey::char_1(FeatureType::CharSelf, 0x9999);
        assert_eq!(m.vocab_find(&k_missing), None);
    }

    #[test]
    fn read_class_access() {
        let mut m = MomoModel::default();
        m.read_classes = vec!["カ".to_string(), "キ".to_string(), "ク".to_string()];
        m.n_classes = 3;

        assert_eq!(m.read_class(0), Some("カ"));
        assert_eq!(m.read_class(2), Some("ク"));
        assert_eq!(m.read_class(3), None); // 範囲外
        assert_eq!(m.read_class(100), None);
    }

    #[test]
    fn size_accessors() {
        let mut m = MomoModel::default();
        m.n_classes = 256;
        m.n_features = 100_000;
        assert_eq!(m.n_classes(), 256);
        assert_eq!(m.n_features(), 100_000);
    }
}
