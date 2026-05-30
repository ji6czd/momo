//! モデルデータ構造。
//!
//! C++ 版の `model.hpp` の `MomoModel` 構造体に対応する。
//! `.mbm` ファイルから読み込まれた推論用パラメータをすべて保持する。
//! 一度読み込んだら不変として扱い、[`Predictor`] が共有参照で利用する。
//!
//! [`Predictor`]: crate::Predictor

use crate::feature::FeatureKey;

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
/// - `csc_data[j]` が int8 量子化された重み値（実値は `csc_data[j] * read_scale`）
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
    /// 量子化スケール係数 (実値 = `csc_data[j] * read_scale`)
    pub(crate) read_scale: f32,
    /// CSC 列ポインタ (size: `n_features + 1`)
    pub(crate) csc_colptr: Vec<u32>,
    /// CSC 行インデックス = クラスID (size: `n_nonzero`)
    pub(crate) csc_rowind: Vec<u32>,
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
    /// 特に `read_scale` と `boundary_scale` は `1.0` で初期化する
    /// (C++ 版の `float read_scale = 1.0f;` と一致)。
    /// 実際には loader がファイルから値を読み込んで上書きするが、
    /// 初期値が `0.0` だと万一上書きされない場合に全スコアが消失するため
    /// 安全側の初期値として `1.0` を採用する。
    fn default() -> Self {
        Self {
            vocab: Vec::new(),
            read_classes: Vec::new(),
            read_scale: 1.0,
            csc_colptr: Vec::new(),
            csc_rowind: Vec::new(),
            csc_data: Vec::new(),
            intercept_read: Vec::new(),
            boundary_scale: 1.0,
            boundary_data: Vec::new(),
            boundary_intercept: [0.0, 0.0],
            n_classes: 0,
            n_features: 0,
        }
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
        // C++ 版の `float read_scale = 1.0f;` と一致
        assert_eq!(m.read_scale, 1.0);
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
            (FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji), 1),
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
