//! `momo-compare-quant` — 量子化前 (`.mbmf`) と量子化後 (`.mbm`) の
//! テキスト推論結果を比較する CLI ツール。
//!
//! momo-py は学習ツールとしてのみメンテナンスされる方針になったため、
//! Rust 側で量子化による予測結果の劣化を直接検証できるようにする。
//! `.mbm` と `.mbmf` は同じ `.zip` から `momo_py.exporter.export()` /
//! `export_float()` で書き出されたペアであることを前提とする
//! （語彙・ラベル・人名辞書・単一漢字辞書は共通、重みだけが
//! 量子化されているかどうかが異なる）。
//!
//! 標準入力（または `--input`）から1行ずつ読み、量子化版・非量子化版
//! それぞれで `predict()` を実行して `kana_text()` を比較する。
//! 一致しない行だけを diff 表示し、最後に自信度の差分統計をまとめる。

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use momors_core::{FloatPredictor, PredictionResult, Predictor, PredictorConfig};

/// `.mbm`（量子化後）と `.mbmf`（量子化前）のテキスト推論パリティを比較する。
#[derive(Debug, Parser)]
#[command(name = "momo-compare-quant", version, about, long_about = None)]
struct Cli {
    /// 量子化済みモデル (.mbm) のパス
    #[arg(long)]
    mbm: PathBuf,

    /// 量子化前モデル (.mbmf) のパス
    #[arg(long)]
    mbmf: PathBuf,

    /// 入力ファイル（省略時は標準入力から1行ずつ読む）
    #[arg(long, short)]
    input: Option<PathBuf>,

    /// 一致した行も含めて全行を表示する（既定では不一致行のみ表示）
    #[arg(long)]
    show_all: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let quantized: Predictor = match Predictor::load(PredictorConfig::new(&cli.mbm)) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("量子化モデル読み込みエラー ({}): {e}", cli.mbm.display());
            return ExitCode::FAILURE;
        }
    };
    let float_predictor: FloatPredictor =
        match FloatPredictor::load(PredictorConfig::new(&cli.mbmf)) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("非量子化モデル読み込みエラー ({}): {e}", cli.mbmf.display());
                return ExitCode::FAILURE;
            }
        };

    if quantized.n_classes() != float_predictor.n_classes()
        || quantized.n_features() != float_predictor.n_features()
    {
        eprintln!(
            "警告: モデルの形状が一致しません（.mbm: n_classes={} n_features={}, \
             .mbmf: n_classes={} n_features={}）。同じ .zip から書き出されたペアか確認してください。",
            quantized.n_classes(),
            quantized.n_features(),
            float_predictor.n_classes(),
            float_predictor.n_features(),
        );
    }

    let lines: Vec<String> = match &cli.input {
        Some(path) => match std::fs::read_to_string(path) {
            Ok(text) => text.lines().map(str::to_owned).collect(),
            Err(e) => {
                eprintln!("入力ファイル読み込みエラー ({}): {e}", path.display());
                return ExitCode::FAILURE;
            }
        },
        None => {
            let stdin = io::stdin();
            let mut lines = Vec::new();
            for line in stdin.lock().lines() {
                match line {
                    Ok(l) => lines.push(l),
                    Err(e) => {
                        eprintln!("標準入力エラー: {e}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            lines
        }
    };

    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut stats = Stats {
        show_all: cli.show_all,
        ..Stats::default()
    };

    for text in &lines {
        if text.is_empty() {
            continue;
        }
        let q = match quantized.predict(text) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("予測エラー (.mbm, text={text:?}): {e}");
                continue;
            }
        };
        let f = match float_predictor.predict(text) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("予測エラー (.mbmf, text={text:?}): {e}");
                continue;
            }
        };

        stats.record(text, &q, &f, &mut out);
    }

    stats.print_summary(&mut out);
    ExitCode::SUCCESS
}

/// 比較結果の集計。
#[derive(Default)]
struct Stats {
    total_lines: usize,
    mismatched_lines: usize,
    /// kana_text が一致した行だけの自信度差分（バイトごと）。
    conf_abs_diffs: Vec<f32>,
    /// 一致した行も表示するかどうか（`--show-all`）。
    show_all: bool,
}

impl Stats {
    fn record(
        &mut self,
        text: &str,
        q: &PredictionResult,
        f: &PredictionResult,
        out: &mut impl Write,
    ) {
        self.total_lines += 1;

        let matches = q.kana_text() == f.kana_text();
        if !matches {
            self.mismatched_lines += 1;
            writeln!(out, "❌ 不一致: {text}").ok();
            writeln!(out, "   .mbm  : {}", q.kana_text()).ok();
            writeln!(out, "   .mbmf : {}", f.kana_text()).ok();
        } else {
            for (&cq, &cf) in q.confidences().iter().zip(f.confidences()) {
                self.conf_abs_diffs.push((cq - cf).abs());
            }
            if self.show_all {
                writeln!(out, "✅ 一致: {text} -> {}", q.kana_text()).ok();
            }
        }
    }

    fn print_summary(&self, out: &mut impl Write) {
        writeln!(out).ok();
        writeln!(out, "==== サマリ ====").ok();
        writeln!(out, "行数         : {}", self.total_lines).ok();
        writeln!(out, "不一致行数   : {}", self.mismatched_lines).ok();
        if self.total_lines > 0 {
            let rate = self.mismatched_lines as f64 / self.total_lines as f64 * 100.0;
            writeln!(out, "不一致率     : {rate:.2}%").ok();
        }
        if !self.conf_abs_diffs.is_empty() {
            let n = self.conf_abs_diffs.len() as f64;
            let mean = self.conf_abs_diffs.iter().map(|&d| d as f64).sum::<f64>() / n;
            let max = self
                .conf_abs_diffs
                .iter()
                .cloned()
                .fold(0f32, f32::max);
            writeln!(out, "自信度差分   : 平均={mean:.6} 最大={max:.6} (一致行のみ、{} バイト)", self.conf_abs_diffs.len()).ok();
        }
    }
}
