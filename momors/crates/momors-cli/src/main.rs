//! `momo` コマンドの実装。
//!
//! 現 C++ 版の `main.cpp` 相当。
//! 標準入力から 1 行ずつ読んで予測し、結果を標準出力に書く。

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use momors_braille::BrailleConverter;
use momors_core::{Predictor, PredictorConfig};

fn data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("MOMO_DATASET_DIR") {
        return PathBuf::from(dir);
    }
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
}

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

    /// 予測結果を原文の文字ごとに分割して出力する
    #[arg(long)]
    segment: bool,

    /// かな変換結果をさらに日本語点字に変換して出力する
    #[arg(long)]
    braille: bool,

    /// 漢字辞書ファイル (.tsv) のパス（省略時は実行ファイルと同じ場所を自動検索）
    #[arg(long)]
    single_dict: Option<String>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    // モデルファイルのパスを決定 (--model-file が優先、次に MOMO_DATASET_DIR、最後に exe と同じ場所)
    let model_path = if let Some(ref path) = cli.model_file {
        PathBuf::from(path)
    } else {
        data_dir().join(cli.model.file_path())
    };

    // 設定を組み立てる
    let mut config = PredictorConfig::new(&model_path)
        .with_segment_output(cli.segment);
    let dict_path = if let Some(ref path) = cli.single_dict {
        let p = PathBuf::from(path);
        if !p.exists() {
            eprintln!("漢字辞書ファイルが見つかりません: {}", p.display());
            return ExitCode::FAILURE;
        }
        Some(p)
    } else {
        let candidate = data_dir().join("single_character_dic.tsv");
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

    // 点字変換器（--braille が指定されたときのみ初期化）
    // MOMO_DATASET_DIR または exe と同じ場所の toml を優先し、なければ埋め込みを使う
    let braille_converter: Option<BrailleConverter> = if cli.braille {
        let toml_path = data_dir().join("japanese_braille.toml");
        let result = if toml_path.exists() {
            BrailleConverter::from_file(&toml_path)
        } else {
            BrailleConverter::from_embedded()
        };
        match result {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("点字テーブル読み込みエラー: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
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
                if let Some(ref converter) = braille_converter {
                    match converter.convert(result.kana_text()) {
                        Ok(brl) => writeln!(out, "{}", brl.braille_text()).ok(),
                        Err(e) => {
                            eprintln!("点字変換エラー: {e}");
                            None
                        }
                    };
                } else {
                    writeln!(out, "{}", result.kana_text()).ok();
                }

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
