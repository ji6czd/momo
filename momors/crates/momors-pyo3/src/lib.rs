use momors_braille::BrailleConverter as RustBrailleConverter;
use momors_core::{
    PredictionResult as RustPredictionResult, Predictor as RustPredictor, PredictorConfig,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;

// ============================================================
// PredictionResult
// ============================================================

/// 予測結果。`Predictor.predict()` の戻り値。
#[pyclass(get_all)]
struct PredictionResult {
    /// 入力文字列（原文）
    source: String,
    /// 変換後のカナ文字列
    kana: String,
    /// 各カナ文字の自信度（0.0〜1.0）
    confidences: Vec<f32>,
    /// 分かち書きされたカナ（`ワガハイ/ワ/ネコ/デ/アル` 形式）
    segmented: String,
    /// 分かち書きされた原文（`吾輩/は/猫/で/ある` 形式）
    source_segmented: String,
    /// かな文字インデックス → 原文文字インデックス
    kana_to_source: Vec<usize>,
    /// 原文文字インデックス → かな文字インデックスのリスト
    source_to_kana: Vec<Vec<usize>>,
}

impl PredictionResult {
    fn from_rust(r: RustPredictionResult) -> Self {
        let kana_to_source = r.kana_to_source_char();
        let source_to_kana = r.source_to_kana_char();
        let segmented = r.format_segmented();
        let source_segmented = r.format_source_segmented();
        let confidences: Vec<f32> = r
            .kana_text()
            .char_indices()
            .map(|(b, _)| r.confidences()[b])
            .collect();
        Self {
            source: r.source_text().to_string(),
            kana: r.kana_text().to_string(),
            confidences,
            segmented,
            source_segmented,
            kana_to_source,
            source_to_kana,
        }
    }
}

#[pymethods]
impl PredictionResult {
    fn __repr__(&self) -> String {
        format!(
            "PredictionResult(source={:?}, kana={:?})",
            self.source, self.kana
        )
    }
}

// ============================================================
// Predictor
// ============================================================

/// 日本語テキスト → カナ変換器。
///
/// ```python
/// import momors_py
/// p = momors_py.Predictor("basic_data_5.mbm")
/// r = p.predict("吾輩は猫である")
/// print(r.kana)           # カタカナ読み
/// print(r.source_segmented)  # 分かち書き原文
/// ```
#[pyclass]
struct Predictor {
    inner: RustPredictor,
}

#[pymethods]
impl Predictor {
    /// モデルファイルを読み込んで予測器を作成する。
    ///
    /// Args:
    ///     model_path: `.mbm` モデルファイルのパス
    ///     kanji_dict: 漢字辞書 TSV のパス（省略可）
    ///     numeric_threshold: 数字ルールベース変換を発動させる自信度の上限（デフォルト 0.5）
    #[new]
    #[pyo3(signature = (model_path, *, kanji_dict=None, numeric_threshold=0.5))]
    fn new(model_path: &str, kanji_dict: Option<&str>, numeric_threshold: f32) -> PyResult<Self> {
        let mut config =
            PredictorConfig::new(model_path).with_numeric_confidence_threshold(numeric_threshold);
        if let Some(path) = kanji_dict {
            config = config.with_kanji_dict_path(path);
        }
        let inner =
            RustPredictor::load(config).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// テキストをカナに変換する。
    fn predict(&self, text: &str) -> PyResult<PredictionResult> {
        let result = self
            .inner
            .predict(text)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PredictionResult::from_rust(result))
    }

    /// 同梱モデルから予測器を作成する。
    ///
    /// Args:
    ///     window: コンテキストウィンドウサイズ（4, 5, 7）。デフォルトは 7。
    ///     kanji_dict: 漢字辞書 TSV のパス（省略可）
    ///     numeric_threshold: 数字ルールベース変換を発動させる自信度の上限
    #[classmethod]
    #[pyo3(signature = (window=7, *, kanji_dict=None, numeric_threshold=0.5))]
    fn from_bundled(
        cls: &Bound<'_, PyType>,
        window: u32,
        kanji_dict: Option<&str>,
        numeric_threshold: f32,
    ) -> PyResult<Self> {
        if !matches!(window, 4 | 5 | 7) {
            return Err(PyValueError::new_err(format!(
                "window must be 4, 5, or 7, got {window}"
            )));
        }
        let module = cls.py().import("momors_py")?;
        let file: String = module.getattr("__file__")?.extract()?;
        let dir = std::path::Path::new(&file)
            .parent()
            .ok_or_else(|| PyRuntimeError::new_err("Cannot determine module directory"))?;
        let model_path = dir.join(format!("basic_data_{window}.mbm"));
        if !model_path.exists() {
            return Err(PyRuntimeError::new_err(format!(
                "Bundled model not found: {}",
                model_path.display()
            )));
        }
        let mut config =
            PredictorConfig::new(&model_path).with_numeric_confidence_threshold(numeric_threshold);
        if let Some(path) = kanji_dict {
            config = config.with_kanji_dict_path(path);
        }
        let inner =
            RustPredictor::load(config).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "Predictor(model={:?})",
            self.inner.config().model_path().display()
        )
    }
}

// ============================================================
// BrailleResult
// ============================================================

/// 点字変換結果。`BrailleConverter.convert()` の戻り値。
#[pyclass(get_all)]
struct BrailleResult {
    /// 変換後の点字文字列
    braille: String,
    /// かな文字インデックス → 点字先頭セルインデックス
    kana_to_braille: Vec<usize>,
}

#[pymethods]
impl BrailleResult {
    fn __repr__(&self) -> String {
        format!("BrailleResult(braille={:?})", self.braille)
    }
}

// ============================================================
// BrailleConverter
// ============================================================

/// カナ文字列 → 日本語点字変換器。
///
/// ```python
/// import momors_py
/// c = momors_py.BrailleConverter()
/// r = c.convert("ワガハイワネコデアル")
/// print(r.braille)
/// ```
#[pyclass]
struct BrailleConverter {
    inner: RustBrailleConverter,
}

#[pymethods]
impl BrailleConverter {
    /// 組み込みテーブルで変換器を作成する。
    #[new]
    fn new() -> PyResult<Self> {
        let inner = RustBrailleConverter::from_embedded()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// カナ文字列を点字に変換する。
    fn convert(&self, kana: &str) -> PyResult<BrailleResult> {
        let result = self
            .inner
            .convert(kana)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(BrailleResult {
            braille: result.braille_text().to_string(),
            kana_to_braille: result.kana_to_braille().to_vec(),
        })
    }
}

// ============================================================
// モジュール登録
// ============================================================

#[pymodule]
fn momors_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Predictor>()?;
    m.add_class::<PredictionResult>()?;
    m.add_class::<BrailleConverter>()?;
    m.add_class::<BrailleResult>()?;
    Ok(())
}
