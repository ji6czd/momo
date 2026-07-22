//! `.mbmf` モデルファイル読み込み（量子化前 float32 サイドカー）。
//!
//! Python 側 `momo_py/exporter.py` の `export_float()` が書き出すバイナリを
//! ストリーミング読み込みして [`FloatMomoModel`] を構築する。`.mbm`
//! (`loader.rs`) とセクション構成はほぼ同一で、語彙テーブル・読みラベル
//! テーブル・人名辞書テーブル・単一漢字辞書テーブルは完全に同一のバイト列
//! フォーマットのため、そのセクション読み込みは `loader.rs` の
//! `pub(crate)` ヘルパー (`read_vocab` / `read_labels` / `read_name_dict` /
//! `read_kanji_dict` / `read_f32_vec` / `read_csc_structure` / `io_err`) を
//! そのまま再利用する。ヘッダ検証の定数 (`VERSION` / `MAX_CLASSES` /
//! `MAX_REASONABLE_COUNT`) も `loader.rs` の定義を共有し、ここでは複製しない
//! ―― 複製すると次のフォーマット変更で片方だけ更新してしまう。
//! 差分は読みモデル重み・境界モデル重みの2セクションのみ:
//!
//! ```text
//! [ファイルヘッダ]          16 bytes
//!   magic        : u8[4]   "MBMF"
//!   version      : u8      0x05    ← .mbm と同じ番号を共有する
//!   _reserved    : u8[3]   0x00 × 3
//!   n_classes    : u32 LE
//!   n_features   : u32 LE
//!
//! [語彙テーブル]            .mbm と同一
//! [読みラベルテーブル]      .mbm と同一
//!
//! [読みモデル重み (CSC・float32・量子化なし)]
//!   n_nonzero    : u32 LE       ← 疎構造は .mbm と同一（read_csc_structure で共有）
//!   colptr       : u32 × (n_features + 1)
//!   rowind       : u16 × n_nonzero
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
use crate::loader::{
    MAX_CLASSES, MAX_REASONABLE_COUNT, VERSION, io_err, read_csc_structure, read_f32_vec,
    read_kanji_dict, read_labels, read_name_dict, read_vocab,
};
use crate::{Error, Result};

/// ファイル識別情報。magic は `.mbm` と区別するため独自だが、バージョン番号は
/// `.mbm` と共有する（[`crate::loader::VERSION`] を import している）。セクション
/// 構成を共通に保つ設計なので、採番を分けると「どちらの 0x02 か」を常に意識する
/// 羽目になる。
const MAGIC: [u8; 4] = *b"MBMF";

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
    if version != VERSION {
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
    if n_classes > MAX_CLASSES {
        return Err(Error::CorruptModel {
            reason: format!(
                "n_classes={n_classes} が上限 {MAX_CLASSES} を超えています（CSC 行インデックスが u16 のため）"
            ),
        });
    }

    let mut model = FloatMomoModel::new();
    model.n_classes = n_classes;
    model.n_features = n_features;

    // ---- 語彙テーブル / 読みラベルテーブル (.mbm と共通実装) ----
    model.vocab = read_vocab(reader, n_features, path)?;
    model.read_classes = read_labels(reader, n_classes, path)?;

    // ---- 読みモデル重み (CSC・float32・量子化なし) ----
    let (colptr, rowind, n_nonzero) = read_csc_structure(reader, n_classes, n_features, path)?;
    model.csc_colptr = colptr;
    model.csc_rowind = rowind;
    model.csc_data = read_f32_vec(reader, n_nonzero, path)?;

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
    fn n_classes_over_u16_returns_error() {
        // .mbm 側と同じ理由 (csc_rowind が u16) で 65537 は弾かれる。
        let bad_data = b"MBMF\x05\x00\x00\x00\x01\x00\x01\x00\x05\x00\x00\x00";
        let mut cursor = std::io::Cursor::new(&bad_data[..]);
        let result = load_from_reader(&mut cursor, Path::new("test"));
        assert!(matches!(result, Err(Error::CorruptModel { .. })));
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
