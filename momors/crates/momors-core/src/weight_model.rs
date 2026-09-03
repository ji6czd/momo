//! [`crate::prediction::Predictor`] が扱う「重みの持ち方」を抽象化するトレイト。
//!
//! `.mbm`（量子化済み int8 + クラスごと scale）を読み込む [`crate::model::MomoModel`]
//! と、`.mbmf`（量子化前の float32、`momo_py.exporter.export_float()` が書き出す
//! サイドカーファイル）を読み込む [`crate::float_model::FloatMomoModel`] の両方が
//! このトレイトを実装する。
//!
//! `Predictor<M: WeightModel>` をジェネリックにすることで、bypass・々フォールバック・
//! 小書き仮名・人名辞書フォールバック・漢数字ルールベース変換などの重み表現に
//! 依存しない推論ロジック（`prediction.rs`）を一切複製せずに、量子化版・非量子化版の
//! 両方で `predict()` を実行できる。量子化前後のテキスト推論パリティを比較する
//! `momo-compare-quant` はこの仕組みの上に成り立っている。

use std::path::Path;

use crate::Result;
use crate::boundary::VocabRef;
use crate::feature::FeatureKey;
use crate::name_dict::NameIndex;

/// [`Predictor`](crate::prediction::Predictor) が推論に使う重みモデルの共通インタフェース。
///
/// `pub` だが `weight_model` モジュール自体は非公開 (`mod weight_model;`) なので、
/// クレート外からは実質到達不能（`Predictor<M: WeightModel>` の境界として
/// 内部的に必要なだけで、外部に実装させることは想定していない）。
pub trait WeightModel: Sized {
    /// `compute_read_scores` が使うスクラッチバッファの型。
    ///
    /// 量子化モデル (`MomoModel`) は int32 で加算してから最後にクラスごとの
    /// scale を掛けるため `Vec<i32>` を持つ。非量子化モデル (`FloatMomoModel`)
    /// は f32 のまま `scores` に直接加算できるためスクラッチが不要 (`()`)。
    type Scratch;

    /// ファイルパスからモデルを読み込む。
    fn load(path: &Path) -> Result<Self>;

    /// バイト列からモデルを読み込む (WASM / インメモリ用)。
    fn load_from_bytes(bytes: &[u8]) -> Result<Self>;

    /// `compute_read_scores` 用のスクラッチバッファを新規作成する。
    fn new_scratch(&self) -> Self::Scratch;

    fn n_classes(&self) -> u32;
    fn n_features(&self) -> u32;

    /// クラスIDから読みラベル文字列を引く。
    fn read_class(&self, class_id: u32) -> Option<&str>;

    /// 読みラベルの全体（クラスID順）。
    fn read_classes(&self) -> &[String];

    /// 語彙テーブルから特徴量キーに対応する feature_id を引く。
    fn vocab_find(&self, key: &FeatureKey) -> Option<u32>;

    /// 人名辞書の照合用インデックス。
    fn name_dict(&self) -> &NameIndex;

    /// 単一文字辞書を所有権ごと取り出す（`Predictor::load` が一度だけ呼ぶ）。
    fn take_single_char_dict(&mut self) -> Vec<(char, Vec<String>)>;

    /// `scores[cls]` を「intercept + 特徴量重みの合計」で上書きする。
    ///
    /// `scores.len()` は `n_classes()` と一致していること。
    fn compute_read_scores(
        &self,
        feat_ids: &[u32],
        scratch: &mut Self::Scratch,
        scores: &mut [f32],
    );

    /// 境界モデルの生スコア (sigmoid 前) を計算する。
    ///
    /// 読みモデルのスコア計算 (`compute_read_scores`) と違い、事前に解決した
    /// 語彙IDではなく生の `FeatureKey` を受け取る。線形実装は内部で
    /// `vocab_find` を都度呼ぶが、GBDT（カテゴリカル特徴量）実装は読みモデルと
    /// 別の語彙空間を持つため、そもそも共有 `feat_ids` に解決する意味がない
    /// （[`crate::boundary`] 参照）。
    fn compute_boundary_score(&self, feat_keys: &[FeatureKey]) -> f32;

    /// 特徴量キーを統合語彙で引き、読みモデル用の `feature_id` と境界モデル用の
    /// カテゴリカル情報を一度に返す（二分探索は 1 回）。
    fn resolve(&self, key: &FeatureKey) -> Option<VocabRef>;

    /// [`Self::compute_boundary_score`] の、[`Self::resolve`] 済みの列を受け取る版。
    fn compute_boundary_score_resolved(&self, refs: &[VocabRef]) -> f32;

    /// 読みモデルの1特徴量（列）の非ゼロ重みを `(class_id, 実重み)` の列で返す。
    ///
    /// 実重みは量子化・scale を解決した後の値（`.mbm` は `int8 * scale`、
    /// `.mbmf` はそのまま）。範囲外 `feat_id` は空 Vec。診断（explain / label
    /// スキャナ）専用で、推論のホットパスでは使わない。
    fn read_feature_column(&self, feat_id: u32) -> Vec<(u32, f32)>;
}
