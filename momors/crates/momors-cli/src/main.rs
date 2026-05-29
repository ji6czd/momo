//! `momo` コマンドの実装。
//!
//! 現 C++ 版の `main.cpp` 相当。
//! 標準入力から 1 行ずつ読んで予測し、結果を標準出力に書く。

use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use clap::Parser;
use momors_core::{Predictor, PredictorConfig};

/// 日本語の漢字かな交じり文をかなに変換する。
#[derive(Debug, Parser)]
#[command(name = "momo", version, about, long_about = None)]
struct Cli {
    /// モデルファイル (.mbm) のパス
    #[arg(long, default_value = "basic_data.mbm")]
    model: String,

    /// 自信度スコアも一緒に出力する
    #[arg(long)]
    confidence: bool,

    /// 各行の処理時間を出力する
    #[arg(long)]
    profile: bool,

    /// KANJI フォールバック発動の自信度しきい値
    #[arg(long, default_value_t = 0.3)]
    threshold: f32,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    // 設定を組み立てる
    let config = PredictorConfig::new(&cli.model).with_confidence_threshold(cli.threshold);

    // 予測器を読み込む
    let predictor = match Predictor::load(config) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("モデル読み込みエラー: {e}");
            return ExitCode::FAILURE;
        }
    };

    // 標準入力ループ
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("入力エラー: {e}");
                return ExitCode::FAILURE;
            }
        };

        if line.is_empty() {
            writeln!(out).ok();
            continue;
        }

        let start = std::time::Instant::now();
        match predictor.predict(&line) {
            Ok(result) => {
                let elapsed = start.elapsed();
                if cli.profile {
                    writeln!(out, "予測時間: {:?}", elapsed).ok();
                }
                writeln!(out, "{}", result.kana_text()).ok();

                if cli.confidence {
                    for &c in result.confidences() {
                        write!(out, "{c} ").ok();
                    }
                    writeln!(out).ok();
                }
            }
            Err(e) => {
                eprintln!("予測エラー: {e}");
            }
        }
    }

    ExitCode::SUCCESS
}
