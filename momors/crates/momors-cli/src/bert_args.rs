//! BERT 読みモードのオプション（`momo` と `momo-inspect` で共通。feature `bert`）。

use std::path::{Path, PathBuf};

use momors_bert::{BertOptions, BertReader, DEFAULT_DIR_NAME};

#[derive(Debug, clap::Args)]
pub struct BertArgs {
    /// BERT 読みモード: 漢字・漢数字の読みを微調整 BERT で推論する。
    /// モデルは .mbm と同じ既定のディレクトリの下の momo-bert/ から読む
    #[arg(long, global = true)]
    pub bert: bool,

    /// BERT モデルのディレクトリ（`export_onnx` の出力、または momo-bert-*.zip を展開したもの）。
    /// 指定すると --bert も有効になる
    #[arg(long, value_name = "DIR", global = true)]
    pub bert_dir: Option<PathBuf>,

    /// BERT を float32 版で動かす（既定は int8 版。float32 は約 1.6 倍遅い）
    #[arg(long, global = true)]
    pub bert_f32: bool,

    /// BERT の演算スレッド数（0 は ONNX Runtime の既定＝物理コア数）
    #[arg(long, default_value_t = 0, value_name = "N", global = true)]
    pub bert_threads: usize,
}

impl BertArgs {
    /// `--bert` か `--bert-dir` が指定されていれば推論器を読み込む。
    ///
    /// `--bert-dir` が無ければ `data_dir`（.mbm を探す既定のディレクトリ）の下の
    /// [`DEFAULT_DIR_NAME`] を使う。
    pub fn load(&self, data_dir: &Path) -> Result<Option<BertReader>, String> {
        let dir = match &self.bert_dir {
            Some(dir) => dir.clone(),
            None if self.bert => data_dir.join(DEFAULT_DIR_NAME),
            None => return Ok(None),
        };
        if !dir.is_dir() {
            return Err(format!(
                "BERT モデルのディレクトリが見つかりません: {}（momo-bert-*.zip を展開して置くか、--bert-dir で指定する）",
                dir.display()
            ));
        }
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
