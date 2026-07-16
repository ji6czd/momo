//! `.mbmf` モデルファイル読み込み（量子化前 float32 サイドカー）。
//!
//! Python 側 `momo_py/exporter.py` の `export_float()` が書き出すバイナリを
//! ストリーミング読み込みして [`FloatMomoModel`] を構築する。`.mbm`
//! (`loader.rs`) とセクション構成はほぼ同一で、語彙テーブル・読みラベル
//! テーブル・人名辞書テーブル・単一漢字辞書テーブルは完全に同一のバイト列
//! フォーマットのため、そのセクション読み込みは `loader.rs` の
//! `pub(crate)` ヘルパー (`read_vocab` / `read_labels` / `read_name_dict` /
//! `read_kanji_dict` / `read_f32_vec` / `csr_to_csc` / `io_err`) をそのまま
//! 再利用する。差分は読みモデル重み・境界モデル重みの2セクションのみ:
//!
//! ```text
//! [ファイルヘッダ]          16 bytes
//!   magic        : u8[4]   "MBMF"
//!   version      : u8      0x01
//!   _reserved    : u8[3]   0x00 × 3
//!   n_classes    : u32 LE
//!   n_features   : u32 LE
//!
//! [語彙テーブル]            .mbm と同一
//! [読みラベルテーブル]      .mbm と同一
//!
//! [読みモデル重み (CSR・float32・量子化なし)]
//!   n_nonzero    : u32 LE
//!   indptr       : u32 × (n_classes + 1)
//!   indices      : u32 × n_nonzero
//!   data         : f32 × n_nonzero   ← quant_scale なし、実値そのもの
//!
//! [読みモデル intercept]    .mbm と同一
//!
//! [境界モデル重み (float32・量子化なし)]
//!   data         : f32 × n_features  ← quant_scale なし、実値そのもの
//!   intercept    : f32 × 2
//!
//! [人名辞書テーブル]        .mbm と同一
//! [単一漢字辞書テーブル]    .mbm と同一
//! ```

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::float_model::FloatMomoModel;
use crate::loader::{csr_to_csc, io_err, read_f32_vec, read_kanji_dict, read_labels, read_name_dict, read_vocab};
use crate::{Error, Result};

const MAGIC: [u8; 4] = *b"MBMF";

/// ヘッダ由来のカウント値の妥当性上限。`loader.rs` と同じ値。
const MAX_REASONABLE_COUNT: u32 = 50_000_000;

/// `.mbmf` ファイルを読み込んで [`FloatMomoModel`] を構築する。
pub(crate) fn load(path: impl AsRef<Path>) -> Result<FloatMomoModel> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|e| Error::ModelIo {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut reader = BufReader::new(file);
    load_from_reader(&mut reader, path)
}

/// バイト列から [`FloatMomoModel`] を構築する (WASM / インメモリ用)。
pub(crate) fn load_from_bytes(bytes: &[u8]) -> Result<FloatMomoModel> {
    let mut cursor = std::io::Cursor::new(bytes);
    load_from_reader(&mut cursor, Path::new("<memory>"))
}

fn load_from_reader<R: Read>(reader: &mut R, path: &Path) -> Result<FloatMomoModel> {
    // ---- ヘッダ ----
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(io_err(path))?;
    if magic != MAGIC {
        return Err(Error::InvalidMagic {
            path: path.to_path_buf(),
        });
    }

    let version = reader.read_u8().map_err(io_err(path))?;
    if version != 0x01 {
        return Err(Error::UnsupportedVersion { version });
    }

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

    let mut model = FloatMomoModel::new();
    model.n_classes = n_classes;
    model.n_features = n_features;

    // ---- 語彙テーブル / 読みラベルテーブル (.mbm と共通実装) ----
    model.vocab = read_vocab(reader, n_features, path)?;
    model.read_classes = read_labels(reader, n_classes, path)?;

    // ---- 読みモデル重み (CSR・float32 → CSC) ----
    let (csr_indptr, csr_indices, csr_data) =
        read_csr_weights_float(reader, n_classes, n_features, path)?;

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

    // ---- 読みモデル intercept (.mbm と共通実装) ----
    model.intercept_read = read_f32_vec(reader, n_classes as usize, path)?;

    // ---- 境界モデル重み (float32・量子化なし) ----
    model.boundary_data = read_f32_vec(reader, n_features as usize, path)?;
    model.boundary_intercept = [
        reader.read_f32::<LittleEndian>().map_err(io_err(path))?,
        reader.read_f32::<LittleEndian>().map_err(io_err(path))?,
    ];

    // ---- 人名辞書テーブル / 単一漢字辞書テーブル (.mbm と共通実装) ----
    let names = read_name_dict(reader, path)?;
    model.name_dict = crate::name_dict::build_name_index(&names);
    model.kanji_dict = read_kanji_dict(reader, path)?;

    // ---- 後処理: vocab を Rust の Ord で再ソート (.mbm の loader と同じ理由) ----
    model.vocab.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(model)
}

/// 読みモデル重み (CSR 形式・float32・量子化なし) を読む。
///
/// 戻り値: `(indptr, indices, data)`。`loader.rs::read_csr_weights` と
/// 同じ整合性チェック（indptr 単調非減少、indices の範囲）を行う。
fn read_csr_weights_float<R: Read>(
    reader: &mut R,
    n_classes: u32,
    n_features: u32,
    path: &Path,
) -> Result<(Vec<u32>, Vec<u32>, Vec<f32>)> {
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
    if let Some(&bad) = indices.iter().find(|&&col| col >= n_features) {
        return Err(Error::CorruptModel {
            reason: format!(
                "CSR indices に不正な特徴量ID {bad} があります（n_features={n_features}）"
            ),
        });
    }

    let data = read_f32_vec(reader, n_nonzero, path)?;

    Ok((indptr, indices, data))
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::{FeatureKey, FeatureType};
    use crate::weight_model::WeightModel;
    use std::path::PathBuf;

    /// fixture.mbmf のパスを返す。
    fn fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/fixture.mbmf")
    }

    #[test]
    fn load_fixture_header() {
        let model = load(fixture_path()).expect("fixture.mbmf が読めること");
        assert_eq!(model.n_classes(), 3);
        assert_eq!(model.n_features(), 5);
    }

    #[test]
    fn load_fixture_labels() {
        let model = load(fixture_path()).unwrap();
        assert_eq!(model.read_class(0), Some("カ"));
        assert_eq!(model.read_class(1), Some("キ"));
        assert_eq!(model.read_class(2), Some("ク"));
        assert_eq!(model.read_class(3), None);
    }

    #[test]
    fn load_fixture_vocab() {
        let model = load(fixture_path()).unwrap();

        let k = FeatureKey::no_payload(FeatureType::Bias);
        assert_eq!(model.vocab_find(&k), Some(0));

        let k = FeatureKey::char_1(FeatureType::CharSelf, 0x6F22);
        assert_eq!(model.vocab_find(&k), Some(1));

        let k = FeatureKey::char_1(FeatureType::CharSelf, 0x9999);
        assert_eq!(model.vocab_find(&k), None);
    }

    #[test]
    fn load_fixture_read_weights_are_dequantized() {
        let model = load(fixture_path()).unwrap();

        // fixture.mbm の QUANT_SCALES_READ=[0.01, 0.02, 0.005] と CSR_ROWS から
        // 導出した実値 (int8 * scale) と厳密に一致するはず。
        // CSC 列順: col0: rows[0,1,2] vals[50*0.01, 40*0.02, 10*0.005]
        assert_eq!(model.csc_colptr, vec![0, 3, 4, 5, 7, 8]);
        assert_eq!(model.csc_rowind, vec![0, 1, 2, 0, 1, 0, 1, 2]);
        let expected: Vec<f32> = vec![0.5, 0.8, 0.05, 0.8, 1.4, 0.3, 0.4, 0.45];
        for (a, b) in model.csc_data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "a={a} b={b}");
        }
    }

    #[test]
    fn invalid_magic_returns_error() {
        let bad_data = b"XXXX\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let mut cursor = std::io::Cursor::new(&bad_data[..]);
        let result = load_from_reader(&mut cursor, Path::new("test"));
        assert!(matches!(result, Err(Error::InvalidMagic { .. })));
    }

    #[test]
    fn invalid_version_returns_error() {
        let bad_data = b"MBMF\x99\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let mut cursor = std::io::Cursor::new(&bad_data[..]);
        let result = load_from_reader(&mut cursor, Path::new("test"));
        assert!(matches!(
            result,
            Err(Error::UnsupportedVersion { version: 0x99 })
        ));
    }
}
