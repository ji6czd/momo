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
//! ## 統合語彙（version 0x07）
//!
//! このモジュールは vocab テーブルを一切持たない。線形も GBDT も、特徴量キー →
//! 情報の解決は `model.rs`/`float_model.rs` が持つ**統合語彙テーブル**に委譲し、
//! [`VocabRef`] を返すクロージャ (`resolve`) で受け取る:
//!   - 線形は `feature_id`（one-hot 列）を使う。
//!   - GBDT は `cat`（カテゴリカル列と code）を使う。
//!
//! version 0x06 までは GBDT が自前の `cat_vocab`（読み語彙とほぼ同一のキー集合を
//! `(column, code)` に写す表）を丸ごと持っており、読み語彙とペイロードが二重に
//! 格納されていた。0x07 で読み語彙に `(column, code)` を吸収し、GBDT
//! セクションは木だけを持つようにした。前提として**カテゴリカルキーは読み語彙の
//! 部分集合**（exporter が保証。cat 専用キーがあると解決できず欠損扱いになる）。

use std::io::Read;
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::feature::FeatureKey;
use crate::loader::{MAX_REASONABLE_COUNT, io_err, read_f32_vec, read_i8_vec};
use crate::{Error, Result};

/// カテゴリカル境界列数の妥当性上限（スコア計算時の固定長スタック配列のサイズ）。
/// 現状の列数（momo_py.categorical、window=7で38列）に対して十分な余裕を持たせる。
/// これを超える列数のモデルファイルは `CorruptModel` として拒否する。
pub(crate) const MAX_BOUNDARY_CAT_COLUMNS: usize = 64;

/// 統合語彙テーブル（version 0x07）1エントリのうち、境界モデルのスコア計算が
/// 必要とする部分。`model.rs`/`float_model.rs` の `resolve` クロージャが返す。
#[derive(Debug, Clone, Copy)]
pub struct VocabRef {
    /// one-hot 特徴量ID（線形境界モデル・読みモデル用）。
    pub(crate) feature_id: u32,
    /// GBDT カテゴリカル `(column, code)`。カテゴリカル列を持たないキー（Bias 等）
    /// や、そもそもカテゴリカル情報を持たない（線形境界）モデルでは `None`。
    pub(crate) cat: Option<(u32, u32)>,
}

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
///
/// version 0x07 で自前の `cat_vocab` を廃止した。カテゴリカル `(column, code)` は
/// 統合語彙テーブル（`model.rs`/`float_model.rs`）が持ち、スコア計算時に
/// `resolve` クロージャ経由で受け取る。
#[derive(Debug)]
pub(crate) struct TreeEnsemble {
    /// 全木のノードを連結した圧縮表現（下記のワード配置）。
    words: Vec<u32>,
    /// 各木の根の `words` 内インデックス。
    roots: Vec<u32>,
    /// split ノードの総数（メンバーシップのビット集合の大きさ）。
    n_splits: usize,
    /// 列ごとの反転索引: `(コード, split id)` をコード順に並べたもの。
    columns: Vec<ColumnIndex>,
}

/// 1 列ぶんの反転索引。`codes[i]` を cats に含む split が `splits[i]`（コード昇順）。
#[derive(Debug, Default)]
struct ColumnIndex {
    codes: Vec<u32>,
    splits: Vec<u32>,
}

// ============================================================
// 木の圧縮表現と反転索引
// ============================================================
//
// ファイル形式（`Box<TreeNode>` 再帰列・split ごとの cats 配列）は変えず、ロード時に
// 次の 2 つへ詰め直す。
//
// 1. ノード列 `words`（前順。左の子は親の直後、右の子の位置だけを持つ）:
//      leaf : [ LEAF_FLAG                         ][ value (f32 bits) ]
//      split: [ column | dl<<8 | split_id<<9      ][ right index      ]
//    cats を持たないので 1 ノード 2 ワード。w4 モデル（split 2,200）で全木 36KB。
//
// 2. 列ごとの反転索引 `columns`: 「(列, コード) → そのコードを cats に含む split の id」。
//
// 推論時は 1 文字につき、列ごとに 1 回だけ反転索引を引いて「左へ進む split」のビット集合を
// 作り、木を辿るときは `cats.binary_search(&code)` の代わりにビットを見る。
// 元の判定（コードがその split の cats に含まれるか）と厳密に同じ結果になる。
//
// 動機は ESP32-P4（PSRAM 常駐・360MHz）: 1 文字あたり 100 本 × 約 10 段のノード訪問で、
// それぞれ平均 29 個の cats を二分探索していた（境界 ≈ 280 µs/文字）。ノード 2 ワード化で
// 木全体が L1/L2 に収まり、二分探索は列数（≤ 21）回だけになる。
// 2026-08-16 に撤回した「共有 cats プール」方式とは違い、cats はそもそも辿らない。

const LEAF_FLAG: u32 = 1 << 31;
const COLUMN_BITS: u32 = 8;
const DEFAULT_LEFT_BIT: u32 = 1 << COLUMN_BITS;
const SPLIT_ID_SHIFT: u32 = COLUMN_BITS + 1;

/// メンバーシップのビット集合をスタックに置ける split 数の上限（これを超えると Vec）。
const STACK_BITSET_WORDS: usize = 256;

/// [`TreeEnsemble::stats`] の結果（診断用）。
#[cfg(feature = "diagnostics")]
#[derive(Debug, Clone)]
pub struct TreeStats {
    pub n_trees: usize,
    pub n_splits: usize,
    pub n_leaves: usize,
    pub cats_total: usize,
    pub max_depth: usize,
    pub splits_per_column: [usize; MAX_BOUNDARY_CAT_COLUMNS],
}

#[cfg(feature = "diagnostics")]
impl Default for TreeStats {
    fn default() -> Self {
        Self {
            n_trees: 0,
            n_splits: 0,
            n_leaves: 0,
            cats_total: 0,
            max_depth: 0,
            splits_per_column: [0; MAX_BOUNDARY_CAT_COLUMNS],
        }
    }
}

impl TreeEnsemble {
    /// パース済みの再帰木を圧縮表現＋反転索引へ詰め直す。
    pub(crate) fn new(trees: Vec<TreeNode>) -> Self {
        let mut words = Vec::new();
        let mut roots = Vec::with_capacity(trees.len());
        let mut columns: Vec<ColumnIndex> = (0..MAX_BOUNDARY_CAT_COLUMNS)
            .map(|_| ColumnIndex::default())
            .collect();
        let mut n_splits = 0u32;
        for t in &trees {
            roots.push(words.len() as u32);
            Self::emit(t, &mut words, &mut n_splits, &mut columns);
        }
        for col in &mut columns {
            // コード順（同一コード内は split id 順）に整列する
            let mut pairs: Vec<(u32, u32)> = col.codes.iter().copied().zip(col.splits.iter().copied()).collect();
            pairs.sort_unstable();
            col.codes = pairs.iter().map(|p| p.0).collect();
            col.splits = pairs.iter().map(|p| p.1).collect();
        }
        Self {
            words,
            roots,
            n_splits: n_splits as usize,
            columns,
        }
    }

    fn emit(node: &TreeNode, words: &mut Vec<u32>, n_splits: &mut u32, columns: &mut [ColumnIndex]) {
        match node {
            TreeNode::Leaf { value } => {
                words.push(LEAF_FLAG);
                words.push(value.to_bits());
            }
            TreeNode::Split {
                column,
                default_left,
                cats,
                left,
                right,
            } => {
                debug_assert!(*column < (1 << COLUMN_BITS));
                let split_id = *n_splits;
                *n_splits += 1;
                let col = &mut columns[*column as usize];
                for &code in cats {
                    col.codes.push(code);
                    col.splits.push(split_id);
                }
                let header = (*column & ((1 << COLUMN_BITS) - 1))
                    | if *default_left { DEFAULT_LEFT_BIT } else { 0 }
                    | (split_id << SPLIT_ID_SHIFT);
                words.push(header);
                let right_slot = words.len();
                words.push(0); // 右の子の位置は後で埋める
                Self::emit(left, words, n_splits, columns);
                words[right_slot] = words.len() as u32;
                Self::emit(right, words, n_splits, columns);
            }
        }
    }

    /// 木の統計（診断用）。
    #[cfg(feature = "diagnostics")]
    pub(crate) fn stats(&self) -> TreeStats {
        let mut st = TreeStats {
            n_trees: self.roots.len(),
            n_splits: self.n_splits,
            ..Default::default()
        };
        for (c, col) in self.columns.iter().enumerate() {
            st.cats_total += col.codes.len();
            // 列ごとの split 数: この列に現れる split id の種類数
            let mut ids: Vec<u32> = col.splits.clone();
            ids.sort_unstable();
            ids.dedup();
            st.splits_per_column[c] = ids.len();
        }
        for &r in &self.roots {
            self.walk_stats(r as usize, 0, &mut st);
        }
        st
    }

    #[cfg(feature = "diagnostics")]
    fn walk_stats(&self, i: usize, depth: usize, st: &mut TreeStats) {
        let header = self.words[i];
        st.max_depth = st.max_depth.max(depth);
        if header & LEAF_FLAG != 0 {
            st.n_leaves += 1;
            return;
        }
        self.walk_stats(i + 2, depth + 1, st);
        self.walk_stats(self.words[i + 1] as usize, depth + 1, st);
    }

    /// 列ごとのコードから「左へ進む split」のビット集合を `bits` に作る。
    fn build_membership(&self, col_codes: &[Option<u32>; MAX_BOUNDARY_CAT_COLUMNS], bits: &mut [u32]) {
        bits.fill(0);
        for (col, code) in self.columns.iter().zip(col_codes) {
            let Some(code) = *code else { continue };
            if col.codes.is_empty() {
                continue;
            }
            // コード順なので、最初の一致位置から同じコードが続く範囲を舐める
            let start = col.codes.partition_point(|&c| c < code);
            for (&c, &split) in col.codes[start..].iter().zip(&col.splits[start..]) {
                if c != code {
                    break;
                }
                bits[(split >> 5) as usize] |= 1 << (split & 31);
            }
        }
    }

    /// 1 本の木を根から葉まで辿ってリーフ値を返す。
    #[inline]
    fn eval_tree(
        &self,
        root: usize,
        col_present: &[bool; MAX_BOUNDARY_CAT_COLUMNS],
        bits: &[u32],
    ) -> f32 {
        let w = &self.words;
        let mut i = root;
        loop {
            let header = w[i];
            if header & LEAF_FLAG != 0 {
                return f32::from_bits(w[i + 1]);
            }
            let column = (header & ((1 << COLUMN_BITS) - 1)) as usize;
            let split = (header >> SPLIT_ID_SHIFT) & ((1 << (31 - SPLIT_ID_SHIFT)) - 1);
            let go_left = if col_present[column] {
                bits[(split >> 5) as usize] & (1 << (split & 31)) != 0
            } else {
                header & DEFAULT_LEFT_BIT != 0
            };
            i = if go_left { i + 2 } else { w[i + 1] as usize };
        }
    }

    /// 全木のリーフ値の合計を、列コード表から計算する。
    fn sum_trees(&self, col_codes: &[Option<u32>; MAX_BOUNDARY_CAT_COLUMNS]) -> f32 {
        let n_words = self.n_splits.div_ceil(32);
        let mut stack_buf = [0u32; STACK_BITSET_WORDS];
        let mut heap_buf: Vec<u32>;
        let bits: &mut [u32] = if n_words <= STACK_BITSET_WORDS {
            &mut stack_buf[..n_words]
        } else {
            heap_buf = vec![0u32; n_words];
            &mut heap_buf[..]
        };
        self.build_membership(col_codes, bits);
        let mut col_present = [false; MAX_BOUNDARY_CAT_COLUMNS];
        for (p, c) in col_present.iter_mut().zip(col_codes) {
            *p = c.is_some();
        }
        self.roots
            .iter()
            .map(|&r| self.eval_tree(r as usize, &col_present, bits))
            .sum()
    }
}

#[derive(Debug)]
pub(crate) enum TreeNode {
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
    /// `resolve` は特徴量キーを統合語彙テーブル（`model.rs`）で引くクロージャ。
    /// 線形は [`VocabRef::feature_id`]、GBDT は [`VocabRef::cat`] を使う。
    pub(crate) fn compute_score(
        &self,
        feat_keys: &[FeatureKey],
        resolve: impl Fn(&FeatureKey) -> Option<VocabRef>,
    ) -> f32 {
        self.compute_score_iter(feat_keys.iter().filter_map(resolve))
    }

    /// [`Self::compute_score`] の、語彙引き済み [`VocabRef`] 列を受け取る版。
    ///
    /// 推論ループは読みモデル用に各キーを一度語彙で引いており、その結果を
    /// 境界モデルにも使い回すことで二分探索を二重に行わずに済む。
    pub(crate) fn compute_score_resolved(&self, refs: &[VocabRef]) -> f32 {
        self.compute_score_iter(refs.iter().copied())
    }

    fn compute_score_iter(&self, refs: impl Iterator<Item = VocabRef>) -> f32 {
        match self {
            Boundary::Linear {
                scale,
                data,
                intercept,
            } => {
                let mut score = intercept[1];
                for vr in refs {
                    if (vr.feature_id as usize) < data.len() {
                        score += (data[vr.feature_id as usize] as f32) * scale;
                    }
                }
                score
            }
            Boundary::Tree(tree) => tree.compute_score_iter(refs),
        }
    }
}

impl FloatBoundary {
    /// 境界モデルの生スコア（sigmoid 前）を計算する。[`Boundary::compute_score`] の
    /// float32 版（線形は量子化なしでそのまま加算）。
    pub(crate) fn compute_score(
        &self,
        feat_keys: &[FeatureKey],
        resolve: impl Fn(&FeatureKey) -> Option<VocabRef>,
    ) -> f32 {
        self.compute_score_iter(feat_keys.iter().filter_map(resolve))
    }

    /// [`Boundary::compute_score_resolved`] の float 版。
    pub(crate) fn compute_score_resolved(&self, refs: &[VocabRef]) -> f32 {
        self.compute_score_iter(refs.iter().copied())
    }

    fn compute_score_iter(&self, refs: impl Iterator<Item = VocabRef>) -> f32 {
        match self {
            FloatBoundary::Linear { data, intercept } => {
                let mut score = intercept[1];
                for vr in refs {
                    if (vr.feature_id as usize) < data.len() {
                        score += data[vr.feature_id as usize];
                    }
                }
                score
            }
            FloatBoundary::Tree(tree) => tree.compute_score_iter(refs),
        }
    }
}

impl TreeEnsemble {
    /// 全木のリーフ値の合計（追加の intercept はない）。
    ///
    /// アクティブな各特徴量キーを `resolve` で統合語彙に引き、カテゴリカル
    /// `(column, code)` を持つものだけを列コード表に反映する。
    #[cfg(test)]
    fn compute_score(
        &self,
        feat_keys: &[FeatureKey],
        resolve: impl Fn(&FeatureKey) -> Option<VocabRef>,
    ) -> f32 {
        self.compute_score_iter(feat_keys.iter().filter_map(resolve))
    }

    fn compute_score_iter(&self, refs: impl Iterator<Item = VocabRef>) -> f32 {
        // 列ごとのコードを解決する。同じ列に複数の FeatureKey が対応することは
        // 通常ないが、あれば最後に見つかったものが勝つ。
        let mut col_codes = [None::<u32>; MAX_BOUNDARY_CAT_COLUMNS];
        for vr in refs {
            if let Some((column, code)) = vr.cat
                && let Some(slot) = col_codes.get_mut(column as usize)
            {
                *slot = Some(code);
            }
        }
        self.sum_trees(&col_codes)
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

fn parse_tree_ensemble<R: Read>(reader: &mut R, path: &Path) -> Result<TreeEnsemble> {
    // version 0x07: 木のアンサンブルは cat_vocab を持たず、n_columns と木だけ。
    // カテゴリカル `(column, code)` は統合語彙テーブル（loader.rs の read_vocab）が持つ。
    let n_columns = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;
    if n_columns as usize > MAX_BOUNDARY_CAT_COLUMNS {
        return Err(Error::CorruptModel {
            reason: format!(
                "境界モデルのカテゴリカル列数={n_columns}が上限 {MAX_BOUNDARY_CAT_COLUMNS} を超えています"
            ),
        });
    }

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
    Ok(TreeEnsemble::new(trees))
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

    fn tree(trees: Vec<TreeNode>) -> TreeEnsemble {
        TreeEnsemble::new(trees)
    }

    /// テスト用: `(FeatureKey, column, code)` の一覧から、本番の統合語彙が返すのと
    /// 同じ `resolve` クロージャを作る（feature_id は木のスコアでは使わないのでダミー）。
    fn resolver(
        cat_vocab: Vec<(FeatureKey, u32, u32)>,
    ) -> impl Fn(&FeatureKey) -> Option<VocabRef> {
        move |k: &FeatureKey| {
            cat_vocab
                .iter()
                .find(|(key, _, _)| key == k)
                .map(|&(_, col, code)| VocabRef {
                    feature_id: 0,
                    cat: Some((col, code)),
                })
        }
    }

    #[test]
    fn tree_score_sums_across_trees() {
        let r = resolver(vec![(
            FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji),
            0,
            7,
        )]);
        let t = tree(vec![leaf(0.5), leaf(-0.25), leaf(1.0)]);
        let keys = vec![FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji)];
        assert!((t.compute_score(&keys, &r) - 1.25).abs() < 1e-6);
    }

    #[test]
    fn tree_split_routes_by_category_membership() {
        // 列0の値が {7, 9} なら左(leaf=1.0)、それ以外なら右(leaf=-1.0)。
        let r = resolver(vec![
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
        ]);
        let t = tree(vec![TreeNode::Split {
            column: 0,
            default_left: false,
            cats: vec![7, 9],
            left: Box::new(leaf(1.0)),
            right: Box::new(leaf(-1.0)),
        }]);

        let kanji = vec![FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji)];
        assert_eq!(t.compute_score(&kanji, &r), 1.0);

        let katakana = vec![FeatureKey::type_1(
            FeatureType::TypeSelf,
            CharType::Katakana,
        )];
        assert_eq!(t.compute_score(&katakana, &r), -1.0);
    }

    #[test]
    fn tree_split_missing_feature_uses_default_left() {
        let r = resolver(vec![(
            FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji),
            0,
            7,
        )]);

        let t = tree(vec![TreeNode::Split {
            column: 0,
            default_left: true,
            cats: vec![7],
            left: Box::new(leaf(1.0)),
            right: Box::new(leaf(-1.0)),
        }]);
        // 空 = この列に対応する FeatureKey がこの位置に存在しない（欠損）。
        assert_eq!(t.compute_score(&[], &r), 1.0);

        let t = tree(vec![TreeNode::Split {
            column: 0,
            default_left: false,
            cats: vec![7],
            left: Box::new(leaf(1.0)),
            right: Box::new(leaf(-1.0)),
        }]);
        assert_eq!(t.compute_score(&[], &r), -1.0);
    }

    #[test]
    fn tree_score_unknown_category_code_treated_as_missing() {
        // 訓練時に見なかった値（統合語彙に載っていない FeatureKey）は
        // resolve が None を返すのと同じ扱い＝欠損として default_left に従う。
        let r = resolver(vec![(
            FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji),
            0,
            7,
        )]);
        let t = tree(vec![TreeNode::Split {
            column: 0,
            default_left: true,
            cats: vec![7],
            left: Box::new(leaf(1.0)),
            right: Box::new(leaf(-1.0)),
        }]);

        // Hiragana は語彙表に無い。
        let unknown = vec![FeatureKey::type_1(
            FeatureType::TypeSelf,
            CharType::Hiragana,
        )];
        assert_eq!(t.compute_score(&unknown, &r), 1.0);
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
            let id = if *k == keys[0] {
                Some(0)
            } else if *k == keys[1] {
                Some(2)
            } else {
                None
            };
            id.map(|feature_id| VocabRef {
                feature_id,
                cat: None,
            })
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
