//! `.mbm` モデルファイル読み込み。
//!
//! C++ 版 `loader.cpp` に対応する。
//! Python 側 `momopy/exporter.py` が書き出すバイナリフォーマットを
//! ストリーミング読み込みして [`MomoModel`] を構築する。
//!
//! ## バイナリフォーマット
//!
//! ```text
//! [ファイルヘッダ]          16 bytes
//!   magic        : u8[4]   "MOMO"
//!   version      : u8      0x02
//!   _reserved    : u8[3]   0x00 × 3
//!   n_classes    : u32 LE  読みラベル数
//!   n_features   : u32 LE  特徴量次元数
//!
//! [語彙テーブル]            n_features エントリ
//!   feature_type : u8
//!   chartype[N]  : u8 × N    N = chartype_count(feature_type)
//!   char32[M]    : u32 × M   M = char32_count(feature_type)
//!   uint8_val    : u8        is_uint8_payload(feature_type) のときのみ
//!   feature_id   : u32 LE
//!
//! [読みラベルテーブル]      n_classes エントリ
//!   len          : u8
//!   utf8         : u8[len]
//!
//! [読みモデル重み (CSR・int8 量子化・クラスごとscale)]
//!   quant_scale  : f32 × n_classes  クラス(行)ごとの量子化スケール
//!   n_nonzero    : u32 LE
//!   indptr       : u32 × (n_classes + 1)
//!   indices      : u32 × n_nonzero
//!   data         : i8  × n_nonzero
//!
//! [読みモデル intercept]
//!   intercept    : f32 × n_classes
//!
//! [境界モデル重み (int8 量子化)]
//!   quant_scale  : f32
//!   data         : i8 × n_features
//!   intercept    : f32 × 2
//!
//! ```
//!
//! ## CSR → CSC 変換
//!
//! ファイルは CSR (Compressed Sparse Row) 形式で保存されているが、
//! 推論時のアクセスパターン (特徴量列で走査) に合わせて [`MomoModel`]
//! では CSC (Compressed Sparse Column) で保持する。この変換は本モジュール内で行う。

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use byteorder::{LittleEndian, ReadBytesExt};

use crate::char_type::CharType;
use crate::feature::{FeatureKey, FeatureType};
use crate::model::MomoModel;
use crate::{Error, Result};

// ============================================================
// 定数
// ============================================================

const MAGIC: [u8; 4] = *b"MOMO";

// ============================================================
// 公開エントリポイント
// ============================================================

/// `.mbm` ファイルを読み込んで [`MomoModel`] を構築する。
pub(crate) fn load(path: impl AsRef<Path>) -> Result<MomoModel> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|e| Error::ModelIo {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut reader = BufReader::new(file);
    load_from_reader(&mut reader, path)
}

/// バイト列から [`MomoModel`] を構築する (WASM / インメモリ用)。
pub(crate) fn load_from_bytes(bytes: &[u8]) -> Result<MomoModel> {
    let mut cursor = std::io::Cursor::new(bytes);
    load_from_reader(&mut cursor, Path::new("<memory>"))
}

// ============================================================
// メインロジック
// ============================================================

fn load_from_reader<R: Read>(reader: &mut R, path: &Path) -> Result<MomoModel> {
    // ---- ヘッダ ----
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(io_err(path))?;
    if magic != MAGIC {
        return Err(Error::InvalidMagic {
            path: path.to_path_buf(),
        });
    }

    let version = reader.read_u8().map_err(io_err(path))?;
    if version != 0x02 {
        return Err(Error::UnsupportedVersion { version });
    }

    // reserved 3 bytes をスキップ
    let mut reserved = [0u8; 3];
    reader.read_exact(&mut reserved).map_err(io_err(path))?;

    let n_classes = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;
    let n_features = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;

    // ---- モデル本体を構築 ----
    let mut model = MomoModel::new();
    model.n_classes = n_classes;
    model.n_features = n_features;

    // ---- 語彙テーブル ----
    model.vocab = read_vocab(reader, n_features, path)?;

    // ---- 読みラベルテーブル ----
    model.read_classes = read_labels(reader, n_classes, path)?;

    // ---- 読みモデル重み (CSR → CSC) ----
    let (read_scale, csr_indptr, csr_indices, csr_data) =
        read_csr_weights(reader, n_classes, path)?;
    model.read_scale = read_scale;

    let (colptr, rowind, data_csc) = csr_to_csc(
        n_classes as usize,
        n_features as usize,
        &csr_indptr,
        &csr_indices,
        &csr_data,
    );
    model.csc_colptr = colptr;
    model.csc_rowind = rowind;
    model.csc_data = data_csc;

    // ---- 読みモデル intercept ----
    model.intercept_read = read_f32_vec(reader, n_classes as usize, path)?;

    // ---- 境界モデル重み ----
    model.boundary_scale = reader.read_f32::<LittleEndian>().map_err(io_err(path))?;
    model.boundary_data = read_i8_vec(reader, n_features as usize, path)?;
    model.boundary_intercept = [
        reader.read_f32::<LittleEndian>().map_err(io_err(path))?,
        reader.read_f32::<LittleEndian>().map_err(io_err(path))?,
    ];

    // ---- 後処理: vocab を Rust の Ord で再ソート ----
    // C++ 版とソート順が異なるため、binary_search できるように改めて整列する。
    model.vocab.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(model)
}

// ============================================================
// セクション別読み込み
// ============================================================

/// 語彙テーブルを読む。
fn read_vocab<R: Read>(
    reader: &mut R,
    n_features: u32,
    path: &Path,
) -> Result<Vec<(FeatureKey, u32)>> {
    let mut vocab = Vec::with_capacity(n_features as usize);
    for _ in 0..n_features {
        let ft_byte = reader.read_u8().map_err(io_err(path))?;
        let feature_type =
            FeatureType::from_u8(ft_byte).ok_or(Error::InvalidFeatureType { value: ft_byte })?;

        let mut key = FeatureKey {
            feature_type,
            ..FeatureKey::default()
        };

        // ペイロード読み込み
        let nct = feature_type.chartype_count();
        for i in 0..nct {
            let ct_byte = reader.read_u8().map_err(io_err(path))?;
            let ct = CharType::from_u8(ct_byte).ok_or(Error::InvalidCharType { value: ct_byte })?;
            key.ct[i] = ct;
        }

        let ncp = feature_type.char32_count();
        for i in 0..ncp {
            key.cp[i] = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;
        }

        if feature_type.is_uint8_payload() {
            key.u8val = reader.read_u8().map_err(io_err(path))?;
        }

        let feature_id = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;
        vocab.push((key, feature_id));
    }
    Ok(vocab)
}

/// 読みラベルテーブルを読む。
fn read_labels<R: Read>(reader: &mut R, n_classes: u32, path: &Path) -> Result<Vec<String>> {
    let mut labels = Vec::with_capacity(n_classes as usize);
    let mut buf = Vec::new();
    for _ in 0..n_classes {
        let len = reader.read_u8().map_err(io_err(path))? as usize;
        buf.clear();
        buf.resize(len, 0u8);
        reader.read_exact(&mut buf).map_err(io_err(path))?;
        let label =
            String::from_utf8(buf.clone()).map_err(|e| Error::InvalidLabelUtf8 { source: e })?;
        labels.push(label);
    }
    Ok(labels)
}

/// 読みモデル重み (CSR 形式) を読む。
/// 戻り値: `(quant_scale[n_classes], indptr, indices, data)`
fn read_csr_weights<R: Read>(
    reader: &mut R,
    n_classes: u32,
    path: &Path,
) -> Result<(Vec<f32>, Vec<u32>, Vec<u32>, Vec<i8>)> {
    let scales = read_f32_vec(reader, n_classes as usize, path)?;
    let n_nonzero = reader.read_u32::<LittleEndian>().map_err(io_err(path))? as usize;

    let indptr_len = n_classes as usize + 1;
    let mut indptr = vec![0u32; indptr_len];
    for slot in &mut indptr {
        *slot = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;
    }

    // 整合性チェック: indptr の最後の値は n_nonzero と一致するはず
    if *indptr.last().unwrap() as usize != n_nonzero {
        return Err(Error::CorruptModel {
            reason: format!(
                "CSR indptr[last]={} と n_nonzero={} が一致しません",
                indptr.last().unwrap(),
                n_nonzero
            ),
        });
    }

    let mut indices = vec![0u32; n_nonzero];
    for slot in &mut indices {
        *slot = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;
    }

    let data = read_i8_vec(reader, n_nonzero, path)?;

    Ok((scales, indptr, indices, data))
}

/// f32 ベクタを読む。
fn read_f32_vec<R: Read>(reader: &mut R, len: usize, path: &Path) -> Result<Vec<f32>> {
    let mut v = vec![0f32; len];
    for slot in &mut v {
        *slot = reader.read_f32::<LittleEndian>().map_err(io_err(path))?;
    }
    Ok(v)
}

/// i8 ベクタを読む。
fn read_i8_vec<R: Read>(reader: &mut R, len: usize, path: &Path) -> Result<Vec<i8>> {
    let mut v = vec![0i8; len];
    // i8 は単純な符号付きバイトなので、まず u8 として読み、transmute する。
    // read_exact は &mut [u8] を取るので、安全に変換するため bytemuck 等は使わず
    // unsafe で as_mut_ptr 経由か、または個別読み出し。シンプルさのため個別読み出し。
    for slot in &mut v {
        *slot = reader.read_i8().map_err(io_err(path))?;
    }
    Ok(v)
}

// ============================================================
// CSR → CSC 変換
// ============================================================

/// CSR を CSC に変換する。
///
/// CSR は `(indptr[n_classes+1], indices[n_nonzero], data[n_nonzero])` で、
/// 行 (クラス) ごとの非ゼロエントリを表す。CSC は列 (特徴量) ごとに整理する。
fn csr_to_csc(
    n_classes: usize,
    n_features: usize,
    indptr: &[u32],
    indices: &[u32],
    data: &[i8],
) -> (Vec<u32>, Vec<u32>, Vec<i8>) {
    let n_nonzero = data.len();

    // --- Step 1: 各列の非ゼロエントリ数をカウント ---
    let mut col_count = vec![0u32; n_features];
    for &col in indices {
        col_count[col as usize] += 1;
    }

    // --- Step 2: 累積和で colptr を構築 ---
    let mut colptr = vec![0u32; n_features + 1];
    for i in 0..n_features {
        colptr[i + 1] = colptr[i] + col_count[i];
    }

    // --- Step 3: 各エントリを CSC の正しい位置に書き込む ---
    let mut rowind = vec![0u32; n_nonzero];
    let mut data_csc = vec![0i8; n_nonzero];
    // 次の挿入位置を追跡 (colptr のコピーを使い回す)
    let mut next_pos: Vec<u32> = colptr[..n_features].to_vec();

    for row in 0..n_classes {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        for j in start..end {
            let col = indices[j] as usize;
            let pos = next_pos[col] as usize;
            rowind[pos] = row as u32;
            data_csc[pos] = data[j];
            next_pos[col] += 1;
        }
    }

    (colptr, rowind, data_csc)
}

// ============================================================
// ヘルパ
// ============================================================

/// `std::io::Error` を `Error::ModelIo` に変換するクロージャを作る。
///
/// `?` 演算子と `map_err` で簡潔にエラー変換するために使用する。
fn io_err(path: &Path) -> impl Fn(std::io::Error) -> Error + '_ {
    move |e| Error::ModelIo {
        path: path.to_path_buf(),
        source: e,
    }
}

// `'_` ライフタイムが警告される場合に備えた alternative（未使用、参考）
#[allow(dead_code)]
fn _io_err_owned(path: PathBuf) -> impl Fn(std::io::Error) -> Error {
    move |e| Error::ModelIo {
        path: path.clone(),
        source: e,
    }
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// dummy.mbm のパスを返す。
    /// テストは crate ルートから実行される (`cargo test`) ことを前提とする。
    fn dummy_path() -> PathBuf {
        // crates/momors-core から見たプロジェクトルートの testdata
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/dummy.mbm")
    }

    #[test]
    fn load_dummy_header() {
        let model = load(dummy_path()).expect("dummy.mbm が読めること");
        assert_eq!(model.n_classes(), 3);
        assert_eq!(model.n_features(), 5);
    }

    #[test]
    fn load_dummy_labels() {
        let model = load(dummy_path()).unwrap();
        assert_eq!(model.read_class(0), Some("カ"));
        assert_eq!(model.read_class(1), Some("キ"));
        assert_eq!(model.read_class(2), Some("ク"));
        assert_eq!(model.read_class(3), None);
    }

    #[test]
    fn load_dummy_vocab() {
        let model = load(dummy_path()).unwrap();

        // bias
        let k = FeatureKey::no_payload(FeatureType::Bias);
        assert_eq!(model.vocab_find(&k), Some(0));

        // char_s=漢
        let k = FeatureKey::char_1(FeatureType::CharSelf, 0x6F22);
        assert_eq!(model.vocab_find(&k), Some(1));

        // char_s=字
        let k = FeatureKey::char_1(FeatureType::CharSelf, 0x5B57);
        assert_eq!(model.vocab_find(&k), Some(2));

        // type_s=KANJI
        let k = FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji);
        assert_eq!(model.vocab_find(&k), Some(3));

        // kanji_run=2
        let k = FeatureKey::u8_payload(FeatureType::KanjiRunLen, 2);
        assert_eq!(model.vocab_find(&k), Some(4));

        // 存在しないキー
        let k = FeatureKey::char_1(FeatureType::CharSelf, 0x9999);
        assert_eq!(model.vocab_find(&k), None);
    }

    #[test]
    fn load_dummy_read_weights() {
        let model = load(dummy_path()).unwrap();

        // QUANT_SCALES_READ = [0.01, 0.02, 0.005]（クラスごと）
        assert_eq!(model.read_scale.len(), 3);
        assert!((model.read_scale[0] - 0.01).abs() < 1e-6);
        assert!((model.read_scale[1] - 0.02).abs() < 1e-6);
        assert!((model.read_scale[2] - 0.005).abs() < 1e-6);

        // CSC 形式の検証。
        // 期待される CSR:
        //   カ: (0,50), (1,80), (3,30)
        //   キ: (0,40), (2,70), (3,20)
        //   ク: (0,10), (4,90)
        // 期待される CSC:
        //   col 0: rows [0,1,2], vals [50,40,10]
        //   col 1: rows [0],     vals [80]
        //   col 2: rows [1],     vals [70]
        //   col 3: rows [0,1],   vals [30,20]
        //   col 4: rows [2],     vals [90]
        //   colptr = [0, 3, 4, 5, 7, 8]
        assert_eq!(model.csc_colptr, vec![0, 3, 4, 5, 7, 8]);
        assert_eq!(model.csc_rowind, vec![0, 1, 2, 0, 1, 0, 1, 2]);
        assert_eq!(model.csc_data, vec![50, 40, 10, 80, 70, 30, 20, 90]);
    }

    #[test]
    fn load_dummy_intercept() {
        let model = load(dummy_path()).unwrap();
        assert_eq!(model.intercept_read.len(), 3);
        assert!((model.intercept_read[0] - 0.1).abs() < 1e-6);
        assert!((model.intercept_read[1] - 0.05).abs() < 1e-6);
        assert!((model.intercept_read[2] - (-0.05)).abs() < 1e-6);
    }

    #[test]
    fn load_dummy_boundary() {
        let model = load(dummy_path()).unwrap();

        assert!((model.boundary_scale - 0.005).abs() < 1e-6);
        assert_eq!(model.boundary_data, vec![10, -5, 20, 15, -3]);
        assert!((model.boundary_intercept[0] - 0.2).abs() < 1e-6);
        assert!((model.boundary_intercept[1] - (-0.2)).abs() < 1e-6);
    }

    // --- エラー系のテスト (in-memory バイト列で検証) ---

    #[test]
    fn invalid_magic_returns_error() {
        let bad_data = b"XXXX\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let mut cursor = std::io::Cursor::new(&bad_data[..]);
        let result = load_from_reader(&mut cursor, Path::new("test"));
        assert!(matches!(result, Err(Error::InvalidMagic { .. })));
    }

    #[test]
    fn invalid_version_returns_error() {
        let bad_data = b"MOMO\x99\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let mut cursor = std::io::Cursor::new(&bad_data[..]);
        let result = load_from_reader(&mut cursor, Path::new("test"));
        assert!(matches!(
            result,
            Err(Error::UnsupportedVersion { version: 0x99 })
        ));
    }

    // --- CSR → CSC 変換の独立テスト ---

    #[test]
    fn csr_to_csc_simple() {
        // 3 行 × 5 列、ダミー .mbm と同じパターン
        let indptr = vec![0u32, 3, 6, 8];
        let indices = vec![0u32, 1, 3, 0, 2, 3, 0, 4];
        let data = vec![50i8, 80, 30, 40, 70, 20, 10, 90];

        let (colptr, rowind, data_csc) = csr_to_csc(3, 5, &indptr, &indices, &data);
        assert_eq!(colptr, vec![0, 3, 4, 5, 7, 8]);
        assert_eq!(rowind, vec![0, 1, 2, 0, 1, 0, 1, 2]);
        assert_eq!(data_csc, vec![50, 40, 10, 80, 70, 30, 20, 90]);
    }

    #[test]
    fn csr_to_csc_empty() {
        // 非ゼロエントリがない場合
        let indptr = vec![0u32, 0, 0, 0];
        let indices: Vec<u32> = vec![];
        let data: Vec<i8> = vec![];

        let (colptr, rowind, data_csc) = csr_to_csc(3, 5, &indptr, &indices, &data);
        assert_eq!(colptr, vec![0, 0, 0, 0, 0, 0]);
        assert!(rowind.is_empty());
        assert!(data_csc.is_empty());
    }
}
