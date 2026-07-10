//! `momo` コマンドの実装。
//!
//! 標準入力から 1 行ずつ読んで予測し、結果を標準出力に書く（デフォルト動作）。
//! `--input` を指定するとファイル整形モードに切り替わる。

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use momors_braille::{
    BrailleConverter, BrailleDocument, DocumentConfig, LineTranslator, OutputFormat,
};
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

    /// 漢字辞書ファイル (.tsv) のパス（省略時はモデル .mbm に同梱された辞書を使用）
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

    /// 点字変換テーブル。.toml のパス、または埋め込みテーブル名
    /// (japanese_grade1 / japanese_noconversion / ueb_english)。
    /// `ueb_english` を指定したときだけ、行ごとに日本語／英語を判定して
    /// 英語行を UEB grade 2 で点訳する（省略時は日本語テーブルのみ）
    #[arg(long)]
    table: Option<String>,

    // 予測結果→ソースの文字ごとの対応関係を出力する
    #[arg(long)]
    brl_src_index: bool,

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

    let mut config = PredictorConfig::new(&model_path);
    // 単一漢字辞書は .mbm に同梱されたもの（学習時と同一）を既定とし、
    // --single-dict の明示指定があるときだけ外部ファイルで上書きする。
    if let Some(ref path) = cli.single_dict {
        let p = PathBuf::from(path);
        if !p.exists() {
            eprintln!("漢字辞書ファイルが見つかりません: {}", p.display());
            return ExitCode::FAILURE;
        }
        config = config.with_kanji_dict_path(&p);
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
    let ext = cli
        .output
        .as_ref()
        .and_then(|p| p.extension())
        .and_then(|e| e.to_str());
    let braille_format = match ext {
        Some("brf") => Some(OutputFormat::BrailleText),
        Some("bse") => Some(OutputFormat::Base),
        Some("bes") => Some(OutputFormat::Bes),
        Some("mbr") => Some(OutputFormat::Mbr),
        _ => None,
    };

    let text = std::fs::read_to_string(cli.input.as_ref().unwrap())
        .map_err(|e| format!("ファイル読み込みエラー: {e}"))?;
    let paragraphs: Vec<String> = text.lines().map(str::to_owned).collect();

    let out = match braille_format {
        Some(format) => {
            let (lt, routing) = make_line_translator(cli.table.as_ref())?;
            let braille_list: Vec<String> = paragraphs
                .iter()
                .map(|p| to_braille(p, predictor, &lt, routing))
                .collect::<Result<_, _>>()?;
            let title_braille = cli
                .title
                .as_ref()
                .map(|t| to_braille(t, predictor, &lt, routing))
                .transpose()?;
            let config = DocumentConfig {
                line_width: cli.line_width,
                lines_per_page: cli.lines_per_page,
                page_header: true,
                title: title_braille,
                ..DocumentConfig::default()
            };
            let doc = BrailleDocument::from_paragraphs(&braille_list, config);
            format.write(&doc)
        }
        None => {
            let kana_lines: Vec<String> = paragraphs
                .iter()
                .map(|p| to_kana(p, predictor))
                .collect::<Result<_, _>>()?;
            (kana_lines.join("\n") + "\n").into_bytes()
        }
    };

    match &cli.output {
        Some(path) => {
            std::fs::write(path, &out).map_err(|e| format!("ファイル書き込みエラー: {e}"))
        }
        None => {
            use std::io::Write;
            std::io::stdout()
                .write_all(&out)
                .map_err(|e| format!("標準出力エラー: {e}"))
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

/// 1行を点訳する。空文字列は空文字列のまま返す。
///
/// `routing` が true のときだけ言語を判定し、英語行を UEB grade 2 で点訳する。
/// false のときは従来どおり必ず日本語として点訳する（英字は外字符 `⠰` ＋無縮約）。
fn to_braille(
    text: &str,
    predictor: &Predictor,
    lt: &LineTranslator,
    routing: bool,
) -> Result<String, String> {
    if text.is_empty() {
        return Ok(String::new());
    }
    let result = if routing {
        lt.translate_line(text, predictor)
    } else {
        lt.translate_japanese(text, predictor)
    };
    result
        .map(|r| r.braille_text().to_owned())
        .map_err(|e| format!("点訳エラー: {e}"))
}

/// `--table` の指定で `ueb_english` を選ぶと、行ごとに日本語／英語を判定する。
const TABLE_UEB_ENGLISH: [&str; 2] = ["ueb_english", "ueb_english_grade2"];

/// `--table` の指定から、日本語点字変換器と「行単位の言語判定を行うか」を決める。
///
/// - `ueb_english` … 日本語行は既定テーブル、英語行は UEB grade 2（判定あり）
/// - 既存ファイルのパス … そのテーブル（判定なし）
/// - 埋め込みテーブル名 … そのテーブル（判定なし）
/// - 省略 … 既定テーブル（判定なし）
fn resolve_table(table: Option<&String>) -> Result<(BrailleConverter, bool), String> {
    let Some(spec) = table else {
        let c = BrailleConverter::from_embedded()
            .map_err(|e| format!("点字テーブル読み込みエラー: {e}"))?;
        return Ok((c, false));
    };

    if TABLE_UEB_ENGLISH.contains(&spec.as_str()) {
        let c = BrailleConverter::from_embedded()
            .map_err(|e| format!("点字テーブル読み込みエラー: {e}"))?;
        return Ok((c, true));
    }

    if Path::new(spec).exists() {
        let c = BrailleConverter::from_file(spec)
            .map_err(|e| format!("点字テーブル読み込みエラー: {e}"))?;
        return Ok((c, false));
    }

    match BrailleConverter::from_embedded_name(spec) {
        Ok(c) => Ok((c, false)),
        Err(_) => {
            let names: Vec<&str> = momors_braille::embedded_tables()
                .iter()
                .filter_map(|t| t.name.as_deref())
                .collect();
            Err(format!(
                "点字テーブル \"{spec}\" が見つかりません。\
                 ファイルパスか、埋め込みテーブル名 ({} / ueb_english) を指定してください",
                names.join(" / ")
            ))
        }
    }
}

/// ルーティングの有無を問わず使える点訳器を作る。
fn make_line_translator(table: Option<&String>) -> Result<(LineTranslator, bool), String> {
    let (japanese, routing) = resolve_table(table)?;
    let english = momors_braille::EnglishTranslator::from_embedded()
        .map_err(|e| format!("英語点字テーブル読み込みエラー: {e}"))?;
    Ok((LineTranslator::new(japanese, english), routing))
}

// ============================================================
// 標準入力モード
// ============================================================

fn run_stdin(cli: &Cli, predictor: &Predictor) -> ExitCode {
    // 点字モードのみ。`--table ueb_english` のときだけ行ごとに言語を判定する。
    let braille: Option<(LineTranslator, bool)> = if cli.braille {
        match make_line_translator(cli.table.as_ref()) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    // 1行だけ不正 (例: 不正な UTF-8 バイト列) でも、それ以降の正常な行の
    // 処理を継続する。ストリーム全体を打ち切らず、その行だけ警告してスキップする。
    let mut had_error = false;

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("入力エラー: {e}");
                had_error = true;
                continue;
            }
        };

        if line.is_empty() {
            writeln!(out).ok();
            continue;
        }

        // 点字モード: routing が有効なら、日本語を含まない行を UEB で点訳する（予測を通さない）。
        if let Some((ref lt, routing)) = braille {
            let start = std::time::Instant::now();
            let translated = if routing {
                lt.translate_line(&line, predictor)
            } else {
                lt.translate_japanese(&line, predictor)
            };
            match translated {
                Ok(result) => {
                    if cli.profile {
                        writeln!(
                            out,
                            "処理時間: {:?} ({:?})",
                            start.elapsed(),
                            result.language()
                        )
                        .ok();
                    }
                    writeln!(out, "{}", result.braille_text()).ok();
                    if cli.brl_src_index {
                        for i in result.braille_to_source() {
                            write!(out, "{i} ").ok();
                        }
                        writeln!(out).ok();
                    }
                    // 確信度は日本語行（予測を通った行）だけ出せる
                    if cli.confidence {
                        if let Some(pred) = result.prediction() {
                            for &c in pred.confidences() {
                                write!(out, "{c} ").ok();
                            }
                            writeln!(out).ok();
                        }
                    }
                }
                Err(e) => {
                    eprintln!("点訳エラー: {e}");
                    had_error = true;
                }
            }
            continue;
        }

        let start = std::time::Instant::now();
        match predictor.predict(&line) {
            Ok(result) => {
                let elapsed = start.elapsed();
                if cli.profile {
                    writeln!(out, "予測時間: {:?}", elapsed).ok();
                }
                if cli.segment {
                    writeln!(out, "{}", result.format_segmented()).ok();
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

    if had_error {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
