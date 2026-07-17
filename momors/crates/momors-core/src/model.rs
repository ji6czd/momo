//! モデルデータ構造。
//!
//! C++ 版の `model.hpp` の `MomoModel` 構造体に対応する。
//! `.mbm` ファイルから読み込まれた推論用パラメータをすべて保持する。
//! 一度読み込んだら不変として扱い、[`Predictor`] が共有参照で利用する。
//!
//! [`Predictor`]: crate::Predictor

use crate::feature::FeatureKey;
use crate::name_dict::NameIndex;
use crate::weight_model::WeightModel;
use crate::Result;

// ============================================================
// 語彙テーブル
// ============================================================

/// 語彙テーブルのエントリ。
///
/// `(FeatureKey, feature_id)` のタプル。
/// `Vec<VocabEntry>` をキーでソートしてバイナリサーチで使う。
///
/// 注意: C++ 版の `operator<` とソート順が異なる可能性があるため、
/// loader が読み込み後に Rust の `Ord` で必ず再ソートすること。
pub type VocabEntry = (FeatureKey, u32);

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

    // --- 境界モデル重み (int8 量子化) ---
    /// 量子化スケール係数
    pub(crate) boundary_scale: f32,
    /// 境界モデル (二値分類) のクラス 1 重みベクトル (size: `n_features`)
    pub(crate) boundary_data: Vec<i8>,
    /// 境界モデルの intercept `[class 0, class 1]`
    pub(crate) boundary_intercept: [f32; 2],

    // --- 人名辞書 (version 0x03 で追加) ---
    /// 人名辞書の照合用インデックス。辞書なしモデルでは空。
    /// 推論時に [`compute_name_matches`] で B/I フラグと固定読みを計算し、
    /// `NameFlag*` 特徴量と読みフォールバックの入力にする。
    ///
    /// [`compute_name_matches`]: crate::name_dict::compute_name_matches
    pub(crate) name_dict: NameIndex,

    // --- 単一漢字辞書 (version 0x04 途中で追加) ---
    /// 読みモデルの候補制約に使う単一漢字辞書（漢字→既知の読みリスト）。
    /// char でソート済み（binary_search 用）。
    /// `PredictorConfig::kanji_dict_path` の明示指定があればそちらが優先される。
    pub(crate) kanji_dict: Vec<(char, Vec<String>)>,

    // --- サイズ情報 ---
    pub(crate) n_classes: u32,
    pub(crate) n_features: u32,
}

impl MomoModel {
    /// 空のモデルを作成する。
    ///
    /// loader 側でファイル内容を読み込んで各フィールドを埋める用途。
    /// 直接外部から使うことは想定していない。
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// 語彙テーブルから `key` に対応する `feature_id` を引く。
    ///
    /// 事前条件: `vocab` は Rust の `Ord` でソート済みであること。
    /// 見つからない場合は `None`。
    #[inline]
    pub(crate) fn vocab_find(&self, key: &FeatureKey) -> Option<u32> {
        // `binary_search_by` でキーだけを比較する。
        // 第一要素 (FeatureKey) で比較し、第二要素 (feature_id) は無視。
        self.vocab
            .binary_search_by(|entry| entry.0.cmp(key))
            .ok()
            .map(|idx| self.vocab[idx].1)
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
    /// 特に `boundary_scale` は `1.0` で初期化する
    /// (C++ 版の `float read_scale = 1.0f;` と一致)。
    /// 実際には loader がファイルから値を読み込んで上書きするが、
    /// 初期値が `0.0` だと万一上書きされない場合に全スコアが消失するため
    /// 安全側の初期値として `1.0` を採用する。
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
            boundary_scale: 1.0,
            boundary_data: Vec::new(),
            boundary_intercept: [0.0, 0.0],
            name_dict: NameIndex::new(),
            kanji_dict: Vec::new(),
            n_classes: 0,
            n_features: 0,
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

    fn take_kanji_dict(&mut self) -> Vec<(char, Vec<String>)> {
        std::mem::take(&mut self.kanji_dict)
    }

    fn compute_read_scores(&self, feat_ids: &[u32], int_scores: &mut Vec<i32>, scores: &mut [f32]) {
        int_scores.fill(0);
        for &feat_id in feat_ids {
            if feat_id >= self.n_features {
                continue;
            }
            let col_start = self.csc_colptr[feat_id as usize] as usize;
            let col_end = self.csc_colptr[feat_id as usize + 1] as usize;
            for j in col_start..col_end {
                let cls = self.csc_rowind[j] as usize;
                int_scores[cls] += self.csc_data[j] as i32;
            }
        }
        for cls in 0..scores.len() {
            scores[cls] =
                self.intercept_read[cls] + (int_scores[cls] as f32) * self.read_scale[cls];
        }
    }

    fn compute_boundary_score(&self, feat_ids: &[u32]) -> f32 {
        let mut score = self.boundary_intercept[1];
        let scale = self.boundary_scale;
        for &feat_id in feat_ids {
            if feat_id < self.n_features {
                score += (self.boundary_data[feat_id as usize] as f32) * scale;
            }
        }
        score
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
        assert_eq!(m.boundary_scale, 1.0);
        assert_eq!(m.boundary_intercept, [0.0, 0.0]);
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
        m.vocab = vec![
            (FeatureKey::no_payload(FeatureType::Bias), 0),
            (
                FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji),
                1,
            ),
            (FeatureKey::char_1(FeatureType::CharSelf, 0x4E00), 2),
            (FeatureKey::char_1(FeatureType::CharSelf, 0x4E01), 3),
        ];
        // 念のため明示的にソート
        m.vocab.sort_by(|a, b| a.0.cmp(&b.0));

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
