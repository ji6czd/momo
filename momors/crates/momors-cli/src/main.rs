//! `momo` コマンドの実装。
//!
//! 現 C++ 版の `main.cpp` 相当。
//! 標準入力から 1 行ずつ読んで予測し、結果を標準出力に書く。

use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use momors_core::{Predictor, PredictorConfig};

/// モデルサイズの選択肢
#[derive(Debug, Clone, ValueEnum)]
enum ModelSize {
    Small,
    Medium,
    Large,
}

impl ModelSize {
    fn file_path(&self) -> &'static str {
        match self {
            ModelSize::Small => "basic_data_4.mbm",
            ModelSize::Medium => "basic_data_5.mbm",
            ModelSize::Large => "basic_data_7.mbm",
        }
    }
}

/// 日本語の漢字かな交じり文をかなに変換する。
#[derive(Debug, Parser)]
#[command(name = "momo", version, about, long_about = None)]
struct Cli {
    /// モデルファイル (.mbm) のパス
    #[arg(long)]
    model_file: Option<String>,

    /// 使用するモデルのサイズ (small, medium, large)
    #[arg(long, default_value = "large")]
    model: ModelSize,

    /// 自信度スコアも一緒に出力する
    #[arg(long)]
    confidence: bool,

    /// 各行の処理時間を出力する
    #[arg(long)]
    profile: bool,

    /// KANJI フォールバック発動の自信度しきい値
    #[arg(long, default_value_t = 0.3)]
    threshold: f32,

    /// 予測結果を原文の文字ごとに分割して出力する
    #[arg(long)]
    segment: bool,

    /// 漢字辞書ファイル (.tsv) のパス（省略時は実行ファイルと同じ場所を自動検索）
    #[arg(long)]
    single_dict: Option<String>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    // モデルファイルのパスを決定 (--model-file が優先)
    let model_path_buf;
    let model_path: &str = if let Some(ref path) = cli.model_file {
        path.as_str()
    } else {
        // 実行ファイルと同じディレクトリのモデルファイルを使う
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()));
        model_path_buf = exe_dir
            .map(|d| d.join(cli.model.file_path()))
            .unwrap_or_else(|| std::path::PathBuf::from(cli.model.file_path()));
        model_path_buf
            .to_str()
            .unwrap_or_else(|| cli.model.file_path())
    };

    // 設定を組み立てる
    let mut config = PredictorConfig::new(model_path)
        .with_confidence_threshold(cli.threshold)
        .with_segment_output(cli.segment);
    let dict_path = if let Some(ref path) = cli.single_dict {
        let p = std::path::PathBuf::from(path);
        if !p.exists() {
            eprintln!("漢字辞書ファイルが見つかりません: {}", p.display());
            return ExitCode::FAILURE;
        }
        Some(p)
    } else {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        let candidate = exe_dir.join("single_character_dic.tsv");
        if candidate.exists() { Some(candidate) } else { None }
    };
    if let Some(ref path) = dict_path {
        config = config.with_kanji_dict_path(path);
    }

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
