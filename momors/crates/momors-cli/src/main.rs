//! `momo` コマンドの実装。
//!
//! 標準入力から 1 行ずつ読んで予測し、結果を標準出力に書く（デフォルト動作）。
//! `--input` を指定するとファイル整形モードに切り替わる。

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use momors_braille::{BrailleConverter, BrailleFormatter, FormatterConfig, OutputFormat};
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

    /// 漢字辞書ファイル (.tsv) のパス（省略時は実行ファイルと同じ場所を自動検索）
    #[arg(long)]
    single_dict: Option<String>,

    // ---- 標準入力モード専用 ----------------------------------------
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

    // ---- ファイル整形モード専用 ------------------------------------
    /// 入力ファイル（指定するとファイル整形モードに切り替わる）
    #[arg(long, short)]
    input: Option<PathBuf>,

    /// 出力ファイル（省略時は標準出力）
    #[arg(long, short)]
    output: Option<PathBuf>,

    /// 1行あたりのマス数（ファイル整形モード）
    #[arg(long, default_value_t = 32)]
    line_width: usize,

    /// 1ページあたりの行数（ファイル整形モード）
    #[arg(long, default_value_t = 22)]
    lines_per_page: usize,

    /// ページヘッダのタイトル（日本語テキスト、ファイル整形モード）
    #[arg(long)]
    title: Option<String>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let model_path = if let Some(ref path) = cli.model_file {
        PathBuf::from(path)
    } else {
        data_dir().join(cli.model.file_path())
    };

    let mut config = PredictorConfig::new(&model_path).with_segment_output(cli.segment);
    let dict_path = if let Some(ref path) = cli.single_dict {
        let p = PathBuf::from(path);
        if !p.exists() {
            eprintln!("漢字辞書ファイルが見つかりません: {}", p.display());
            return ExitCode::FAILURE;
        }
        Some(p)
    } else {
        let candidate = data_dir().join("single_character_dic.tsv");
        if candidate.exists() {
            Some(candidate)
        } else {
            None
        }
    };
    if let Some(ref path) = dict_path {
        config = config.with_kanji_dict_path(path);
    }

    let predictor = match Predictor::load(config) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("モデル読み込みエラー: {e}");
            return ExitCode::FAILURE;
        }
    };

    if cli.input.is_some() {
        match run_format(&cli, &predictor) {
            Ok(()) => ExitCode::SUCCESS,
            Err(msg) => {
                eprintln!("{msg}");
                ExitCode::FAILURE
            }
        }
    } else {
        run_stdin(&cli, &predictor)
    }
}

// ============================================================
// ファイル整形モード
// ============================================================

fn run_format(cli: &Cli, predictor: &Predictor) -> Result<(), String> {
    let ext = cli.output.as_ref()
        .and_then(|p| p.extension())
        .and_then(|e| e.to_str());
    let braille_format = match ext {
        Some("brf") => Some(OutputFormat::BrailleText),
        Some("bse") => Some(OutputFormat::Base),
        _ => None,
    };

    let text = std::fs::read_to_string(cli.input.as_ref().unwrap())
        .map_err(|e| format!("ファイル読み込みエラー: {e}"))?;
    let paragraphs: Vec<String> = text.lines().map(str::to_owned).collect();

    let out = match braille_format {
        Some(format) => {
            let converter =
                make_braille_converter().map_err(|e| format!("点字テーブル読み込みエラー: {e}"))?;
            let braille_list: Vec<String> = paragraphs
                .iter()
                .map(|p| to_braille(p, predictor, &converter))
                .collect::<Result<_, _>>()?;
            let title_braille = cli
                .title
                .as_ref()
                .map(|t| to_braille(t, predictor, &converter))
                .transpose()?;
            let formatter = BrailleFormatter::new(FormatterConfig {
                line_width: cli.line_width,
                lines_per_page: cli.lines_per_page,
                page_header: true,
                title: title_braille,
            });
            format.write(&formatter.format(&braille_list))
        }
        None => {
            let kana_lines: Vec<String> = paragraphs
                .iter()
                .map(|p| to_kana(p, predictor))
                .collect::<Result<_, _>>()?;
            kana_lines.join("\n") + "\n"
        }
    };

    match &cli.output {
        Some(path) => std::fs::write(path, out).map_err(|e| format!("ファイル書き込みエラー: {e}")),
        None => {
            print!("{out}");
            Ok(())
        }
    }
}

/// テキストを仮名に変換する。空文字列は空文字列のまま返す。
fn to_kana(text: &str, predictor: &Predictor) -> Result<String, String> {
    if text.is_empty() {
        return Ok(String::new());
    }
    predictor
        .predict(text)
        .map(|r| r.kana_text().to_owned())
        .map_err(|e| format!("仮名変換エラー: {e}"))
}

/// テキストを仮名に変換してからさらに点字に変換する。空文字列は空文字列のまま返す。
fn to_braille(
    text: &str,
    predictor: &Predictor,
    converter: &BrailleConverter,
) -> Result<String, String> {
    if text.is_empty() {
        return Ok(String::new());
    }
    let kana = predictor
        .predict(text)
        .map_err(|e| format!("仮名変換エラー: {e}"))?;
    converter
        .convert(kana.kana_text())
        .map(|b| b.braille_text().to_owned())
        .map_err(|e| format!("点字変換エラー: {e}"))
}

fn make_braille_converter() -> momors_braille::Result<BrailleConverter> {
    let toml_path = data_dir().join("japanese_braille.toml");
    if toml_path.exists() {
        BrailleConverter::from_file(&toml_path)
    } else {
        BrailleConverter::from_embedded()
    }
}

// ============================================================
// 標準入力モード
// ============================================================

fn run_stdin(cli: &Cli, predictor: &Predictor) -> ExitCode {
    let braille_converter: Option<BrailleConverter> = if cli.braille {
        match make_braille_converter() {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("点字テーブル読み込みエラー: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };

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
