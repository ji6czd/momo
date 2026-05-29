//! 予測器の公開 API。
//!
//! C++ 版の `Predictor` / `PredictorConfig` / `PredictionResult` に対応する。
//! Rust らしさのために以下のように再設計している:
//!
//! - `Predictor::new` + `load()` の二段階を `Predictor::load(config)` に統合
//! - 失敗は例外ではなく [`Result`] で返す
//! - 入力は `&str`（UTF-8 が型で保証されている）
//! - [`PredictionResult`] のフィールドは直接公開せず、メソッド経由でアクセス
//!   （将来内部表現を変えても破壊的変更にならない）

use std::path::{Path, PathBuf};

use crate::Result;

// ============================================================
// PredictorConfig
// ============================================================

/// 予測器の設定。
///
/// ビルダーパターンで `.with_xxx(...)` を連鎖させて作る。
/// 必須引数のモデルパスだけ [`new`] で受け取り、その他はデフォルト値を持つ。
///
/// ```no_run
/// use momors_core::PredictorConfig;
///
/// let config = PredictorConfig::new("basic_data.mbm")
///     .with_confidence_threshold(0.3)
///     .with_numeric_confidence_threshold(0.5);
/// ```
///
/// [`new`]: PredictorConfig::new
#[derive(Debug, Clone)]
pub struct PredictorConfig {
    pub(crate) model_path: PathBuf,
    pub(crate) confidence_threshold: f32,
    pub(crate) numeric_confidence_threshold: f32,
}

impl PredictorConfig {
    /// モデルファイルのパスを指定して新規作成する。
    ///
    /// 他のパラメータは妥当なデフォルト値で初期化される。
    pub fn new(model_path: impl AsRef<Path>) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            confidence_threshold: 0.5,
            numeric_confidence_threshold: 0.5,
        }
    }

    /// KANJI フォールバックを発動させる自信度の上限を設定する。
    pub fn with_confidence_threshold(mut self, value: f32) -> Self {
        self.confidence_threshold = value;
        self
    }

    /// JAPANESE_NUMERIC ルールベース変換を発動させる自信度の上限を設定する。
    pub fn with_numeric_confidence_threshold(mut self, value: f32) -> Self {
        self.numeric_confidence_threshold = value;
        self
    }

    // --- getters（内部からの参照用） ---

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }

    pub fn numeric_confidence_threshold(&self) -> f32 {
        self.numeric_confidence_threshold
    }
}

// ============================================================
// PredictionResult
// ============================================================

/// 1 回の予測の結果。
///
/// フィールドは公開せず、アクセサ経由で参照する。
/// これにより内部表現を変更しても利用側のコードを壊さない。
#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub(crate) source_text: String,
    pub(crate) kana_text: String,
    pub(crate) confidences: Vec<f32>,
    /// かなのバイト位置 → 原文のコードポイント位置
    pub(crate) kana_to_src_index: Vec<usize>,
    /// 原文のコードポイント位置 → かなのバイト位置の列
    pub(crate) src_to_kana_index: Vec<Vec<usize>>,
}

impl PredictionResult {
    /// 入力された原文（UTF-8）。
    pub fn source_text(&self) -> &str {
        &self.source_text
    }

    /// 変換後のかな（UTF-8）。
    pub fn kana_text(&self) -> &str {
        &self.kana_text
    }

    /// 各かな文字の自信度のスライス。
    pub fn confidences(&self) -> &[f32] {
        &self.confidences
    }

    /// 「かなのバイト位置 → 原文のコードポイント位置」のマッピング。
    pub fn kana_to_source(&self) -> &[usize] {
        &self.kana_to_src_index
    }

    /// 「原文のコードポイント位置 → かなのバイト位置の列」のマッピング。
    /// 1 つの漢字が複数のかなに展開されるため、内側も配列になる。
    pub fn source_to_kana(&self) -> &[Vec<usize>] {
        &self.src_to_kana_index
    }
}

// ============================================================
// Predictor
// ============================================================

/// 予測器本体。
///
/// [`PredictorConfig`] を渡して [`load`] するとモデルを読み込んだ
/// 状態のインスタンスが得られる。読み込み済みなので即 [`predict`] できる。
///
/// [`load`]: Predictor::load
/// [`predict`]: Predictor::predict
#[derive(Debug)]
pub struct Predictor {
    config: PredictorConfig,
    // 将来:
    // model: crate::model::MomoModel,
}

impl Predictor {
    /// 設定からモデルを読み込んで予測器を構築する。
    ///
    /// モデルファイルのオープン・パース失敗時は [`Error`] を返す。
    ///
    /// [`Error`]: crate::Error
    pub fn load(config: PredictorConfig) -> Result<Self> {
        // TODO: ここで .mbm を読み込んで MomoModel を構築する。
        // とりあえずスケルトンとして config だけ保持する。
        Ok(Self { config })
    }

    /// 設定を参照する。
    pub fn config(&self) -> &PredictorConfig {
        &self.config
    }

    /// 文字列を予測してかなに変換する。
    pub fn predict(&self, _text: &str) -> Result<PredictionResult> {
        // TODO: 実装する
        todo!("予測ロジックは未実装です")
    }
}

// ============================================================
// テスト（スケルトン段階の最低限のものだけ）
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_builder_chains() {
        let config = PredictorConfig::new("dummy.mbm")
            .with_confidence_threshold(0.3)
            .with_numeric_confidence_threshold(0.4);

        assert_eq!(config.model_path(), Path::new("dummy.mbm"));
        assert!((config.confidence_threshold() - 0.3).abs() < 1e-6);
        assert!((config.numeric_confidence_threshold() - 0.4).abs() < 1e-6);
    }
}
