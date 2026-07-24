//! 境界モデル（マスあけ判定）のデータ構造・パース・スコア計算。
//!
//! 線形モデル（SGDClassifier、既定）と GBDT（LightGBM、カテゴリカル特徴量）の
//! 両方式を1つのモジュールにまとめる。`model.rs`/`float_model.rs` は
//! [`Boundary`]/[`FloatBoundary`] を1個持ち、スコア計算を委譲するだけの薄い形にする
//! （線形だけをフィールドに埋め込み、木だけを別モジュールに切り出すと非対称になる
//! ため、両方式をここに集約する）。
//!
//! `.mbm`/`.mbmf` の境界モデルセクションは `algo_tag` プレフィックス付き
//! （`momo_py.exporter` version 0x06）:
//!   - `0x00` = 線形（`.mbm` は int8 量子化、`.mbmf` は float32 のまま）
//!   - `0x01` = 木のアンサンブル（`.mbm`/`.mbmf` で完全に同一バイト列、量子化しない）
//!
//! 線形のスコア計算は読みモデルと共有する vocab テーブルへの参照が要るが、
//! このモジュール自体は vocab テーブルを持たない（`model.rs`/`float_model.rs` が
//! `vocab_find` をクロージャで渡す）。

use std::io::Read;
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::feature::FeatureKey;
use crate::loader::{MAX_REASONABLE_COUNT, io_err, read_f32_vec, read_feature_key, read_i8_vec};
use crate::{Error, Result};

/// カテゴリカル境界列数の妥当性上限（スコア計算時の固定長スタック配列のサイズ）。
/// 現状の列数（momo_py.categorical、window=7で38列）に対して十分な余裕を持たせる。
/// これを超える列数のモデルファイルは `CorruptModel` として拒否する。
const MAX_BOUNDARY_CAT_COLUMNS: usize = 64;

/// 木の再帰の深さの妥当性上限。壊れたファイルによる過大な再帰を防ぐ。
const MAX_TREE_DEPTH: u32 = 256;

// ============================================================
// データ構造
// ============================================================

/// 境界モデル（int8 量子化モデル = [`crate::model::MomoModel`] 用）。
#[derive(Debug)]
pub(crate) enum Boundary {
    Linear {
        scale: f32,
        data: Vec<i8>,
        intercept: [f32; 2],
    },
    Tree(TreeEnsemble),
}

/// 境界モデル（float32 モデル = [`crate::float_model::FloatMomoModel`] 用）。
/// 線形は量子化しないため `scale` を持たない。
#[derive(Debug)]
pub(crate) enum FloatBoundary {
    Linear { data: Vec<f32>, intercept: [f32; 2] },
    Tree(TreeEnsemble),
}

/// GBDT境界モデルの木のアンサンブル。`.mbm`/`.mbmf` で完全に同一のデータ
/// （量子化しない）なので `Boundary`/`FloatBoundary` の両方で共有する。
#[derive(Debug)]
pub(crate) struct TreeEnsemble {
    /// (FeatureKey, column_index, code) を FeatureKey でソートした語彙表。
    /// `column_index` は LightGBM の `split_feature` と同じ空間（0-based）。
    cat_vocab: Vec<(FeatureKey, u32, u32)>,
    trees: Vec<TreeNode>,
}

#[derive(Debug)]
enum TreeNode {
    Leaf {
        value: f32,
    },
    Split {
        /// カテゴリカル列インデックス（`cat_vocab` の column_index と同じ空間）。
        column: u32,
        /// この列に対応する特徴量がこの位置に存在しない（欠損）ときの行き先。
        default_left: bool,
        /// 昇順ソート済み。この集合に列のコードが含まれれば左の子へ進む。
        cats: Vec<u32>,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

// ============================================================
// スコア計算
// ============================================================

impl Boundary {
    /// 境界モデルの生スコア（sigmoid 前）を計算する。
    ///
    /// 線形の場合は `vocab_find` で読みモデルと共有する語彙テーブルを引く
    /// （呼び出し側の `model.rs` が `|k| self.vocab_find(k)` を渡す）。
    pub(crate) fn compute_score(
        &self,
        feat_keys: &[FeatureKey],
        vocab_find: impl Fn(&FeatureKey) -> Option<u32>,
    ) -> f32 {
        match self {
            Boundary::Linear {
                scale,
                data,
                intercept,
            } => {
                let mut score = intercept[1];
                for key in feat_keys {
                    if let Some(id) = vocab_find(key)
                        && (id as usize) < data.len()
                    {
                        score += (data[id as usize] as f32) * scale;
                    }
                }
                score
            }
            Boundary::Tree(tree) => tree.compute_score(feat_keys),
        }
    }
}

impl FloatBoundary {
    /// 境界モデルの生スコア（sigmoid 前）を計算する。[`Boundary::compute_score`] の
    /// float32 版（線形は量子化なしでそのまま加算）。
    pub(crate) fn compute_score(
        &self,
        feat_keys: &[FeatureKey],
        vocab_find: impl Fn(&FeatureKey) -> Option<u32>,
    ) -> f32 {
        match self {
            FloatBoundary::Linear { data, intercept } => {
                let mut score = intercept[1];
                for key in feat_keys {
                    if let Some(id) = vocab_find(key)
                        && (id as usize) < data.len()
                    {
                        score += data[id as usize];
                    }
                }
                score
            }
            FloatBoundary::Tree(tree) => tree.compute_score(feat_keys),
        }
    }
}

impl TreeEnsemble {
    /// 全木のリーフ値の合計（追加の intercept はない）。
    fn compute_score(&self, feat_keys: &[FeatureKey]) -> f32 {
        // 列ごとのコードを解決する。同じ列に複数の FeatureKey が対応することは
        // 通常ないが、あれば最後に見つかったものが勝つ（vocab_find と同じ扱い）。
        let mut col_codes = [None::<u32>; MAX_BOUNDARY_CAT_COLUMNS];
        for key in feat_keys {
            if let Ok(idx) = self.cat_vocab.binary_search_by(|entry| entry.0.cmp(key)) {
                let (_, column, code) = self.cat_vocab[idx];
                if let Some(slot) = col_codes.get_mut(column as usize) {
                    *slot = Some(code);
                }
            }
        }
        self.trees
            .iter()
            .map(|t| eval_tree_node(t, &col_codes))
            .sum()
    }
}

fn eval_tree_node(node: &TreeNode, col_codes: &[Option<u32>; MAX_BOUNDARY_CAT_COLUMNS]) -> f32 {
    match node {
        TreeNode::Leaf { value } => *value,
        TreeNode::Split {
            column,
            default_left,
            cats,
            left,
            right,
        } => {
            let go_left = match col_codes.get(*column as usize).copied().flatten() {
                Some(code) => cats.binary_search(&code).is_ok(),
                None => *default_left,
            };
            eval_tree_node(if go_left { left } else { right }, col_codes)
        }
    }
}

// ============================================================
// パース
// ============================================================

/// `.mbm`（int8量子化）の境界モデルセクションを読む。`algo_tag` で分岐する。
pub(crate) fn parse_int8<R: Read>(
    reader: &mut R,
    n_features: usize,
    path: &Path,
) -> Result<Boundary> {
    match read_algo_tag(reader, path)? {
        AlgoTag::Linear => {
            let scale = reader.read_f32::<LittleEndian>().map_err(io_err(path))?;
            let data = read_i8_vec(reader, n_features, path)?;
            let intercept = read_intercept(reader, path)?;
            Ok(Boundary::Linear {
                scale,
                data,
                intercept,
            })
        }
        AlgoTag::Tree => Ok(Boundary::Tree(parse_tree_ensemble(reader, path)?)),
    }
}

/// `.mbmf`（float32・量子化なし）の境界モデルセクションを読む。`algo_tag` で分岐する。
pub(crate) fn parse_float<R: Read>(
    reader: &mut R,
    n_features: usize,
    path: &Path,
) -> Result<FloatBoundary> {
    match read_algo_tag(reader, path)? {
        AlgoTag::Linear => {
            let data = read_f32_vec(reader, n_features, path)?;
            let intercept = read_intercept(reader, path)?;
            Ok(FloatBoundary::Linear { data, intercept })
        }
        AlgoTag::Tree => Ok(FloatBoundary::Tree(parse_tree_ensemble(reader, path)?)),
    }
}

enum AlgoTag {
    Linear,
    Tree,
}

fn read_algo_tag<R: Read>(reader: &mut R, path: &Path) -> Result<AlgoTag> {
    match reader.read_u8().map_err(io_err(path))? {
        0x00 => Ok(AlgoTag::Linear),
        0x01 => Ok(AlgoTag::Tree),
        other => Err(Error::CorruptModel {
            reason: format!("未知の境界モデル algo_tag です: 0x{other:02X}"),
        }),
    }
}

fn read_intercept<R: Read>(reader: &mut R, path: &Path) -> Result<[f32; 2]> {
    Ok([
        reader.read_f32::<LittleEndian>().map_err(io_err(path))?,
        reader.read_f32::<LittleEndian>().map_err(io_err(path))?,
    ])
}

/// カテゴリカル語彙テーブルを読む。戻り値: `(語彙表, n_cat_columns)`。
fn parse_cat_vocab<R: Read>(
    reader: &mut R,
    path: &Path,
) -> Result<(Vec<(FeatureKey, u32, u32)>, u32)> {
    let n_columns = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;
    let n_entries = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;
    if n_entries > MAX_REASONABLE_COUNT {
        return Err(Error::CorruptModel {
            reason: format!(
                "境界モデルのカテゴリカル語彙エントリ数={n_entries}が大きすぎます（上限 {MAX_REASONABLE_COUNT}）"
            ),
        });
    }
    if n_columns as usize > MAX_BOUNDARY_CAT_COLUMNS {
        return Err(Error::CorruptModel {
            reason: format!(
                "境界モデルのカテゴリカル列数={n_columns}が上限 {MAX_BOUNDARY_CAT_COLUMNS} を超えています"
            ),
        });
    }

    let mut cat_vocab = Vec::with_capacity(n_entries as usize);
    for _ in 0..n_entries {
        let key = read_feature_key(reader, path)?;
        let column = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;
        let code = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;
        if column >= n_columns {
            return Err(Error::CorruptModel {
                reason: format!(
                    "境界モデルのcolumn_index={column}がn_cat_columns={n_columns}以上です"
                ),
            });
        }
        cat_vocab.push((key, column, code));
    }
    // FeatureKey でソートし、後で binary_search できるようにする
    // （読みモデル語彙テーブルの後処理ソートと同じ理由）。
    cat_vocab.sort_by(|a, b| a.0.cmp(&b.0));
    Ok((cat_vocab, n_columns))
}

fn parse_tree_ensemble<R: Read>(reader: &mut R, path: &Path) -> Result<TreeEnsemble> {
    let (cat_vocab, n_columns) = parse_cat_vocab(reader, path)?;

    let n_trees = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;
    if n_trees > MAX_REASONABLE_COUNT {
        return Err(Error::CorruptModel {
            reason: format!(
                "境界モデルのn_trees={n_trees}が大きすぎます（上限 {MAX_REASONABLE_COUNT}）"
            ),
        });
    }
    let mut trees = Vec::with_capacity(n_trees as usize);
    for _ in 0..n_trees {
        trees.push(parse_tree_node(reader, path, n_columns, 0)?);
    }
    Ok(TreeEnsemble { cat_vocab, trees })
}

fn parse_tree_node<R: Read>(
    reader: &mut R,
    path: &Path,
    n_columns: u32,
    depth: u32,
) -> Result<TreeNode> {
    if depth > MAX_TREE_DEPTH {
        return Err(Error::CorruptModel {
            reason: format!(
                "境界モデルの木が深すぎます（上限 {MAX_TREE_DEPTH}）。壊れたファイルの可能性があります。"
            ),
        });
    }

    match reader.read_u8().map_err(io_err(path))? {
        0 => {
            let value = reader.read_f32::<LittleEndian>().map_err(io_err(path))?;
            Ok(TreeNode::Leaf { value })
        }
        1 => {
            let column = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;
            if column >= n_columns {
                return Err(Error::CorruptModel {
                    reason: format!(
                        "境界モデルのsplit_feature={column}がn_cat_columns={n_columns}以上です"
                    ),
                });
            }
            let default_left = reader.read_u8().map_err(io_err(path))? != 0;
            let n_cats = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;
            if n_cats > MAX_REASONABLE_COUNT {
                return Err(Error::CorruptModel {
                    reason: format!(
                        "境界モデルのn_cats={n_cats}が大きすぎます（上限 {MAX_REASONABLE_COUNT}）"
                    ),
                });
            }
            let mut cats = Vec::with_capacity(n_cats as usize);
            for _ in 0..n_cats {
                cats.push(reader.read_u32::<LittleEndian>().map_err(io_err(path))?);
            }
            let left = Box::new(parse_tree_node(reader, path, n_columns, depth + 1)?);
            let right = Box::new(parse_tree_node(reader, path, n_columns, depth + 1)?);
            Ok(TreeNode::Split {
                column,
                default_left,
                cats,
                left,
                right,
            })
        }
        other => Err(Error::CorruptModel {
            reason: format!("境界モデルの木の node_tag が不正です: {other}"),
        }),
    }
}

// ============================================================
// Default
// ============================================================

impl Default for Boundary {
    /// C++ 版・旧実装と同じ安全側の初期値（`scale=1.0`、線形・全ゼロ）で構築する。
    /// loader が実際の値で上書きする前提だが、万一上書きされない場合に
    /// スコアが消失しないようにする。
    fn default() -> Self {
        Boundary::Linear {
            scale: 1.0,
            data: Vec::new(),
            intercept: [0.0, 0.0],
        }
    }
}

impl Default for FloatBoundary {
    fn default() -> Self {
        FloatBoundary::Linear {
            data: Vec::new(),
            intercept: [0.0, 0.0],
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

    fn leaf(value: f32) -> TreeNode {
        TreeNode::Leaf { value }
    }

    fn ensemble(mut cat_vocab: Vec<(FeatureKey, u32, u32)>, trees: Vec<TreeNode>) -> TreeEnsemble {
        // 本番の parse_cat_vocab と同じく、binary_search できるよう FeatureKey でソートする。
        cat_vocab.sort_by(|a, b| a.0.cmp(&b.0));
        TreeEnsemble { cat_vocab, trees }
    }

    #[test]
    fn tree_score_sums_across_trees() {
        let vocab = vec![(
            FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji),
            0,
            7,
        )];
        let t = ensemble(vocab, vec![leaf(0.5), leaf(-0.25), leaf(1.0)]);
        let keys = vec![FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji)];
        assert!((t.compute_score(&keys) - 1.25).abs() < 1e-6);
    }

    #[test]
    fn tree_split_routes_by_category_membership() {
        // 列0の値が {7, 9} なら左(leaf=1.0)、それ以外なら右(leaf=-1.0)。
        let vocab = vec![
            (
                FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji),
                0,
                7,
            ),
            (
                FeatureKey::type_1(FeatureType::TypeSelf, CharType::Hiragana),
                0,
                9,
            ),
            (
                FeatureKey::type_1(FeatureType::TypeSelf, CharType::Katakana),
                0,
                99,
            ),
        ];
        let split = TreeNode::Split {
            column: 0,
            default_left: false,
            cats: vec![7, 9],
            left: Box::new(leaf(1.0)),
            right: Box::new(leaf(-1.0)),
        };
        let t = ensemble(vocab, vec![split]);

        let kanji = vec![FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji)];
        assert_eq!(t.compute_score(&kanji), 1.0);

        let katakana = vec![FeatureKey::type_1(
            FeatureType::TypeSelf,
            CharType::Katakana,
        )];
        assert_eq!(t.compute_score(&katakana), -1.0);
    }

    #[test]
    fn tree_split_missing_feature_uses_default_left() {
        let vocab = vec![(
            FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji),
            0,
            7,
        )];

        let default_left_true = TreeNode::Split {
            column: 0,
            default_left: true,
            cats: vec![7],
            left: Box::new(leaf(1.0)),
            right: Box::new(leaf(-1.0)),
        };
        let t = ensemble(vocab.clone(), vec![default_left_true]);
        // 空 = この列に対応する FeatureKey がこの位置に存在しない（欠損）。
        assert_eq!(t.compute_score(&[]), 1.0);

        let default_left_false = TreeNode::Split {
            column: 0,
            default_left: false,
            cats: vec![7],
            left: Box::new(leaf(1.0)),
            right: Box::new(leaf(-1.0)),
        };
        let t = ensemble(vocab, vec![default_left_false]);
        assert_eq!(t.compute_score(&[]), -1.0);
    }

    #[test]
    fn tree_score_unknown_category_code_treated_as_missing() {
        // 訓練時に見なかった値（cat_vocabに載っていない FeatureKey）は
        // vocab_find が None を返すのと同じ扱い＝欠損として default_left に従う。
        let vocab = vec![(
            FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji),
            0,
            7,
        )];
        let split = TreeNode::Split {
            column: 0,
            default_left: true,
            cats: vec![7],
            left: Box::new(leaf(1.0)),
            right: Box::new(leaf(-1.0)),
        };
        let t = ensemble(vocab, vec![split]);

        // Hiragana は語彙表に無い。
        let unknown = vec![FeatureKey::type_1(
            FeatureType::TypeSelf,
            CharType::Hiragana,
        )];
        assert_eq!(t.compute_score(&unknown), 1.0);
    }

    #[test]
    fn linear_boundary_compute_score_matches_dot_product() {
        let b = Boundary::Linear {
            scale: 0.5,
            data: vec![10, -20, 30],
            intercept: [0.1, -0.2],
        };
        let keys = vec![
            FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji),
            FeatureKey::type_1(FeatureType::TypeSelf, CharType::Hiragana),
        ];
        // key0 -> id 0 (10*0.5=5.0), key1 -> id 2 (30*0.5=15.0), intercept[1]=-0.2
        let score = b.compute_score(&keys, |k| {
            if *k == keys[0] {
                Some(0)
            } else if *k == keys[1] {
                Some(2)
            } else {
                None
            }
        });
        assert!((score - (5.0 + 15.0 - 0.2)).abs() < 1e-6);
    }

    #[test]
    fn default_boundary_is_safe_linear() {
        match Boundary::default() {
            Boundary::Linear {
                scale,
                data,
                intercept,
            } => {
                assert_eq!(scale, 1.0);
                assert!(data.is_empty());
                assert_eq!(intercept, [0.0, 0.0]);
            }
            Boundary::Tree(_) => panic!("既定値は線形であるべき"),
        }
    }
}
