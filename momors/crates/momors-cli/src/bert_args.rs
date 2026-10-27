//! BERT 読みモードのオプション（`momo` と `momo-inspect` で共通。feature `bert`）。

use std::path::PathBuf;

use momors_bert::{BertOptions, BertReader};

#[derive(Debug, clap::Args)]
pub struct BertArgs {
    /// BERT 読みモード: 漢字・漢数字の読みを微調整 BERT で推論する。
    /// `export_onnx`（experimentals/bert）の出力ディレクトリを指定する
    #[arg(long, value_name = "DIR", global = true)]
    pub bert: Option<PathBuf>,

    /// BERT を float32 版で動かす（既定は int8 版。float32 は約 1.6 倍遅い）
    #[arg(long, global = true)]
    pub bert_f32: bool,

    /// BERT の演算スレッド数（0 は ONNX Runtime の既定＝物理コア数）
    #[arg(long, default_value_t = 0, value_name = "N", global = true)]
    pub bert_threads: usize,
}

impl BertArgs {
    /// `--bert` が指定されていれば推論器を読み込む。
    pub fn load(&self) -> Result<Option<BertReader>, String> {
        let Some(dir) = &self.bert else {
            return Ok(None);
        };
        let options = BertOptions {
            int8: !self.bert_f32,
            threads: self.bert_threads,
        };
        BertReader::load(dir, &options)
            .map(Some)
            .map_err(|e| format!("BERT モデルの読み込みエラー: {e}"))
    }
}

/// 推論器のラベルのうち、読みモデルのクラスに無かったものを警告する。
pub fn warn_unknown_labels(unknown: &[String]) {
    if !unknown.is_empty() {
        eprintln!(
            "警告: BERT のラベル {} 個が読みモデルに無いため使いません（例: {}）",
            unknown.len(),
            unknown
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
}
