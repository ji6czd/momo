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
//!   version      : u8      0x03
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
//! [人名辞書テーブル]        version 0x03 で追加、0x04 で読みを追加
//!   n_names      : u32 LE  人名エントリ数（辞書なしモデルは 0）
//!   以下 n_names エントリ:
//!     len        : u8
//!     utf8       : u8[len] 表層形 (UTF-8)
//!     n_readings : u8      ユニット別読みの個数（0 = 読みなし）
//!     以下 n_readings 個:
//!       len      : u8
//!       utf8     : u8[len] ユニット読み (カタカナ、UTF-8)
//!
//! [単一漢字辞書テーブル]    version 0x04 の途中（アルファ期間）で追加
//!   n_entries    : u32 LE  エントリ数
//!   以下 n_entries エントリ:
//!     len        : u8
//!     utf8       : u8[len] 漢字 (1文字、UTF-8)
//!     n_readings : u8      既知の読みの個数
//!     以下 n_readings 個:
//!       len      : u8
//!       utf8     : u8[len] 読み (カタカナ、UTF-8)
//! ```
//!
//! version 0x03 以前は読めない。フォーマット互換性を装って誤動作するより
//! 明示的にエラーにする方針（人名特徴量・読みの有無が精度に直結するため）。
//! 単一漢字辞書テーブル追加前の旧 0x04 ファイルは、テーブル読み込み時に
//! EOF となり分かりやすいエラーメッセージを出す（再エクスポートが必要）。
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

/// ヘッダ由来のカウント値（n_classes / n_features / n_nonzero）の妥当性上限。
/// これを超える値をそのまま `Vec::with_capacity` 等に渡すと、壊れた/不正な
/// `.mbm` ファイル1つで巨大メモリ確保・OOM を引き起こしうるため、
/// 本クレートが現実的に扱う規模を大幅に超える値は早期に `CorruptModel` で弾く。
const MAX_REASONABLE_COUNT: u32 = 50_000_000;

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
    if version != 0x04 {
        return Err(Error::UnsupportedVersion { version });
    }

    // reserved 3 bytes をスキップ
    let mut reserved = [0u8; 3];
    reader.read_exact(&mut reserved).map_err(io_err(path))?;

    let n_classes = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;
    let n_features = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;

    if n_classes == 0 {
        return Err(Error::CorruptModel {
            reason: "n_classes が 0 です（読みラベルが1つも無いモデルは不正）".to_string(),
        });
    }
    if n_classes > MAX_REASONABLE_COUNT || n_features > MAX_REASONABLE_COUNT {
        return Err(Error::CorruptModel {
            reason: format!(
                "n_classes={n_classes} または n_features={n_features} が大きすぎます（上限 {MAX_REASONABLE_COUNT}）"
            ),
        });
    }

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
        read_csr_weights(reader, n_classes, n_features, path)?;
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

    // ---- 人名辞書テーブル (version 0x04: 表層形 + ユニット別読み) ----
    let names = read_name_dict(reader, path)?;
    model.name_dict = crate::name_dict::build_name_index(&names);

    // ---- 単一漢字辞書テーブル (version 0x04 途中で追加) ----
    model.kanji_dict = read_kanji_dict(reader, path)?;

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
///
/// `indptr`/`indices` はファイルから読んだ生の値であり、`csr_to_csc` で
/// そのまま配列添字として使われる。壊れたファイルによる範囲外アクセス
/// panic を防ぐため、ここで整合性を検証してから返す。
fn read_csr_weights<R: Read>(
    reader: &mut R,
    n_classes: u32,
    n_features: u32,
    path: &Path,
) -> Result<(Vec<f32>, Vec<u32>, Vec<u32>, Vec<i8>)> {
    let scales = read_f32_vec(reader, n_classes as usize, path)?;
    let n_nonzero = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;
    if n_nonzero > MAX_REASONABLE_COUNT {
        return Err(Error::CorruptModel {
            reason: format!("n_nonzero={n_nonzero} が大きすぎます（上限 {MAX_REASONABLE_COUNT}）"),
        });
    }
    let n_nonzero = n_nonzero as usize;

    let indptr_len = n_classes as usize + 1;
    let mut indptr = vec![0u32; indptr_len];
    for slot in &mut indptr {
        *slot = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;
    }

    // 整合性チェック: indptr は単調非減少で、各要素は n_nonzero 以下であること
    // （csr_to_csc がこれを前提に範囲外チェック無しで添字アクセスするため）。
    let mut prev = 0u32;
    for (row, &p) in indptr.iter().enumerate() {
        if p < prev || p as usize > n_nonzero {
            return Err(Error::CorruptModel {
                reason: format!(
                    "CSR indptr[{row}]={p} が不正です（直前の値={prev}, n_nonzero={n_nonzero}）"
                ),
            });
        }
        prev = p;
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
    // 整合性チェック: 各特徴量IDは n_features 未満であること
    // （csc_colptr/csc_rowind の構築時にこの範囲を前提に添字アクセスするため）。
    if let Some(&bad) = indices.iter().find(|&&col| col >= n_features) {
        return Err(Error::CorruptModel {
            reason: format!(
                "CSR indices に不正な特徴量ID {bad} があります（n_features={n_features}）"
            ),
        });
    }

    let data = read_i8_vec(reader, n_nonzero, path)?;

    Ok((scales, indptr, indices, data))
}

/// 人名辞書テーブルを読む。
///
/// 表層形は Python 側 exporter が正規化済みだが、入力テキストの正規化
/// （[`normalize_compat_ideographs`]）と確実に揃えるためここでも適用する。
///
/// [`normalize_compat_ideographs`]: crate::normalize::normalize_compat_ideographs
fn read_name_dict<R: Read>(
    reader: &mut R,
    path: &Path,
) -> Result<Vec<(String, Option<Vec<String>>)>> {
    let n_names = reader.read_u32::<LittleEndian>().map_err(io_err(path))?;
    if n_names > MAX_REASONABLE_COUNT {
        return Err(Error::CorruptModel {
            reason: format!("n_names={n_names} が大きすぎます（上限 {MAX_REASONABLE_COUNT}）"),
        });
    }

    let mut names = Vec::with_capacity(n_names as usize);
    let mut buf = Vec::new();
    let read_str = |reader: &mut R, buf: &mut Vec<u8>| -> Result<String> {
        let len = reader.read_u8().map_err(io_err(path))? as usize;
        buf.clear();
        buf.resize(len, 0u8);
        reader.read_exact(buf).map_err(io_err(path))?;
        String::from_utf8(buf.clone()).map_err(|e| Error::InvalidLabelUtf8 { source: e })
    };
    for _ in 0..n_names {
        let surface = read_str(reader, &mut buf)?;
        let surface = crate::normalize::normalize_compat_ideographs(&surface);
        let n_readings = reader.read_u8().map_err(io_err(path))? as usize;
        let readings = if n_readings == 0 {
            None
        } else {
            let mut readings = Vec::with_capacity(n_readings);
            for _ in 0..n_readings {
                readings.push(read_str(reader, &mut buf)?);
            }
            Some(readings)
        };
        names.push((surface, readings));
    }
    Ok(names)
}

/// 単一漢字辞書テーブルを読む。
///
/// 読みモデルの候補制約に使う必須データ。旧 0x04 ファイル（テーブル追加前）は
/// ここで EOF になるため、再エクスポートを促すエラーメッセージに変換する。
fn read_kanji_dict<R: Read>(reader: &mut R, path: &Path) -> Result<Vec<(char, Vec<String>)>> {
    let n_entries = reader.read_u32::<LittleEndian>().map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            Error::CorruptModel {
                reason: "単一漢字辞書テーブルがありません（同テーブル追加前の旧 0x04 ファイル\
                         の可能性があります。モデルを再エクスポートしてください）"
                    .to_string(),
            }
        } else {
            Error::ModelIo {
                path: path.to_path_buf(),
                source: e,
            }
        }
    })?;
    if n_entries > MAX_REASONABLE_COUNT {
        return Err(Error::CorruptModel {
            reason: format!(
                "単一漢字辞書の n_entries={n_entries} が大きすぎます（上限 {MAX_REASONABLE_COUNT}）"
            ),
        });
    }

    let mut dict: Vec<(char, Vec<String>)> = Vec::with_capacity(n_entries as usize);
    let mut buf = Vec::new();
    let read_str = |reader: &mut R, buf: &mut Vec<u8>| -> Result<String> {
        let len = reader.read_u8().map_err(io_err(path))? as usize;
        buf.clear();
        buf.resize(len, 0u8);
        reader.read_exact(buf).map_err(io_err(path))?;
        String::from_utf8(buf.clone()).map_err(|e| Error::InvalidLabelUtf8 { source: e })
    };
    for _ in 0..n_entries {
        let surface = read_str(reader, &mut buf)?;
        let n_readings = reader.read_u8().map_err(io_err(path))? as usize;
        let mut readings = Vec::with_capacity(n_readings);
        for _ in 0..n_readings {
            readings.push(read_str(reader, &mut buf)?);
        }
        // キーは1文字の漢字。複数文字や空のキーは安全側に倒してスキップする。
        let mut chars = surface.chars();
        match (chars.next(), chars.next()) {
            (Some(kanji), None) if !readings.is_empty() => dict.push((kanji, readings)),
            _ => {}
        }
    }
    dict.sort_unstable_by_key(|(k, _)| *k);
    Ok(dict)
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

    #[test]
    fn old_version_v2_returns_error() {
        // 旧バージョンは明示的にエラー（人名辞書セクションが無く、黙って読めると
        // 人名特徴量・読みなしで誤動作するため）
        let bad_data = b"MOMO\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let mut cursor = std::io::Cursor::new(&bad_data[..]);
        let result = load_from_reader(&mut cursor, Path::new("test"));
        assert!(matches!(
            result,
            Err(Error::UnsupportedVersion { version: 0x02 })
        ));
    }

    #[test]
    fn old_version_v3_returns_error() {
        // v3（読みなし人名テーブル）も読めない
        let bad_data = b"MOMO\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let mut cursor = std::io::Cursor::new(&bad_data[..]);
        let result = load_from_reader(&mut cursor, Path::new("test"));
        assert!(matches!(
            result,
            Err(Error::UnsupportedVersion { version: 0x03 })
        ));
    }

    #[test]
    fn load_dummy_name_dict() {
        let model = load(dummy_path()).unwrap();
        // gen_dummy_mbm.py の NAME_DICT = [("佐藤", ["サ","トー"]), ("太郎", None)]
        // インデックスは先頭コードポイント引き
        let sa = model
            .name_dict
            .get(&('佐' as u32))
            .expect("佐藤 が載っていること");
        assert_eq!(sa.len(), 1);
        assert_eq!(sa[0].units.len(), 2); // 佐・藤 の2ユニット
        assert_eq!(
            sa[0].readings.as_deref(),
            Some(&["サ".to_string(), "トー".to_string()][..])
        );
        // 太郎 は読みなしエントリ
        let ta = model
            .name_dict
            .get(&('太' as u32))
            .expect("太郎 が載っていること");
        assert_eq!(ta[0].readings, None);
        assert!(!model.name_dict.contains_key(&('鈴' as u32)));
    }

    #[test]
    fn load_dummy_kanji_dict() {
        let model = load(dummy_path()).unwrap();
        // gen_dummy_mbm.py の KANJI_DICT = [("漢", ["カン"]), ("字", ["ジ", "アザ"])]
        // char でソート済み（字 U+5B57 < 漢 U+6F22）
        assert_eq!(model.kanji_dict.len(), 2);
        assert_eq!(model.kanji_dict[0].0, '字');
        assert_eq!(model.kanji_dict[0].1, vec!["ジ", "アザ"]);
        assert_eq!(model.kanji_dict[1].0, '漢');
        assert_eq!(model.kanji_dict[1].1, vec!["カン"]);
    }

    #[test]
    fn missing_kanji_dict_section_returns_corrupt_model() {
        // 単一漢字辞書テーブル追加前の旧 0x04 ファイルを模擬:
        // dummy.mbm から同テーブル（gen_dummy_mbm.py の build_kanji_dict = 32 bytes）
        // を末尾から削る。
        let bytes = std::fs::read(dummy_path()).unwrap();
        let truncated = &bytes[..bytes.len() - 32];
        let result = load_from_bytes(truncated);
        match result {
            Err(Error::CorruptModel { reason }) => {
                assert!(reason.contains("再エクスポート"), "reason: {reason}");
            }
            other => panic!("CorruptModel になるべきところ: {other:?}"),
        }
    }

    /// ヘッダのみ (16 bytes) を組み立てるヘルパー。
    fn build_header_bytes(n_classes: u32, n_features: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"MOMO");
        bytes.push(0x04);
        bytes.extend_from_slice(&[0, 0, 0]); // reserved
        bytes.extend_from_slice(&n_classes.to_le_bytes());
        bytes.extend_from_slice(&n_features.to_le_bytes());
        bytes
    }

    #[test]
    fn n_classes_zero_returns_corrupt_model_error() {
        let bytes = build_header_bytes(0, 5);
        let mut cursor = std::io::Cursor::new(bytes);
        let result = load_from_reader(&mut cursor, Path::new("test"));
        assert!(matches!(result, Err(Error::CorruptModel { .. })));
    }

    #[test]
    fn n_features_too_large_returns_corrupt_model_error() {
        let bytes = build_header_bytes(3, MAX_REASONABLE_COUNT + 1);
        let mut cursor = std::io::Cursor::new(bytes);
        let result = load_from_reader(&mut cursor, Path::new("test"));
        assert!(matches!(result, Err(Error::CorruptModel { .. })));
    }

    #[test]
    fn csr_index_out_of_range_returns_error() {
        // n_classes=2, n_features=3, n_nonzero=1
        // indices[0] = 5 は n_features=3 の範囲外
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0.01f32.to_le_bytes()); // scales[0]
        bytes.extend_from_slice(&0.02f32.to_le_bytes()); // scales[1]
        bytes.extend_from_slice(&1u32.to_le_bytes()); // n_nonzero
        bytes.extend_from_slice(&0u32.to_le_bytes()); // indptr[0]
        bytes.extend_from_slice(&0u32.to_le_bytes()); // indptr[1]
        bytes.extend_from_slice(&1u32.to_le_bytes()); // indptr[2]
        bytes.extend_from_slice(&5u32.to_le_bytes()); // indices[0] = 5 (範囲外)
        bytes.push(10u8); // data[0]
        let mut cursor = std::io::Cursor::new(bytes);
        let result = read_csr_weights(&mut cursor, 2, 3, Path::new("test"));
        assert!(matches!(result, Err(Error::CorruptModel { .. })));
    }

    #[test]
    fn csr_indptr_non_monotonic_returns_error() {
        // n_classes=2, n_features=3, n_nonzero=3
        // indptr = [0, 3, 1] は単調非減少ではない
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0.01f32.to_le_bytes()); // scales[0]
        bytes.extend_from_slice(&0.02f32.to_le_bytes()); // scales[1]
        bytes.extend_from_slice(&3u32.to_le_bytes()); // n_nonzero
        bytes.extend_from_slice(&0u32.to_le_bytes()); // indptr[0]
        bytes.extend_from_slice(&3u32.to_le_bytes()); // indptr[1]
        bytes.extend_from_slice(&1u32.to_le_bytes()); // indptr[2] = 1 < 直前の 3
        let mut cursor = std::io::Cursor::new(bytes);
        let result = read_csr_weights(&mut cursor, 2, 3, Path::new("test"));
        assert!(matches!(result, Err(Error::CorruptModel { .. })));
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
