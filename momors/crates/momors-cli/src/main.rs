//! `momo` コマンドの実装。
//!
//! 標準入力から 1 行ずつ読んで予測し、結果を標準出力に書く（デフォルト動作）。
//! `--input` を指定するとファイル整形モードに切り替わる。

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use momors_braille::{
    BrailleDocument, BrailleResult, BrailleTranslator, DocumentConfig, EnglishTranslator,
    JapaneseTranslator, NabccCase, OutputFormat,
};
use momors_core::{PredictionResult, Predictor, PredictorConfig};

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

    /// 英語（UEB）点字テーブル。埋め込みテーブル名（english_ueb_grade1 / english_ueb_grade2）か、
    /// UEB スキーマの .toml のパス。指定すると行ごとに日本語／英語を判定し、英語行はこのテーブルで
    /// UEB 点訳、日本語行は既定の日本語テーブル（japanese_grade1）で点訳する。省略時は全行を
    /// 既定の日本語テーブルで点訳する（言語判定なし）
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

    /// .brf の NABCC 英字ケース（upper: 規格どおり／lower: 点字ディスプレイで直接読む用）。
    /// .brf 以外の出力形式に指定するとエラー
    #[arg(long, default_value = "upper")]
    nabcc_case: NabccCaseArg,
}

/// `--nabcc-case` の選択肢。
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum NabccCaseArg {
    Upper,
    Lower,
}

impl From<NabccCaseArg> for NabccCase {
    fn from(a: NabccCaseArg) -> Self {
        match a {
            NabccCaseArg::Upper => NabccCase::Upper,
            NabccCaseArg::Lower => NabccCase::Lower,
        }
    }
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
// 変換パイプライン（ファイル整形モード・標準入力モード共通）
// ============================================================
//
// 経路は常に2段: to_kana() → to_braille()。
// - to_kana(): 漢字かな交じり文 → かな（Predictor）。ASCII 文字はそのまま通る。
// - to_braille(): かな → 点字（BrailleTranslator）。行に日本語文字が含まれるかを
//   内部で判定し、日本語／英語テーブルへ振り分ける（`BrailleTranslator::translate`）。
// 呼び出し側は言語判定やテーブルの出し分けを行わない。

/// テキストを仮名に変換する（漢字かな交じり文 → かな）。空文字列も含め Predictor に委ねる。
fn to_kana(text: &str, predictor: &Predictor) -> Result<PredictionResult, String> {
    predictor
        .predict(text)
        .map_err(|e| format!("仮名変換エラー: {e}"))
}

/// 仮名変換結果を点訳する。日本語／英語の振り分けは `BrailleTranslator` が内部で行う。
fn to_braille(pred: &PredictionResult, lt: &BrailleTranslator) -> Result<BrailleResult, String> {
    lt.translate(pred.kana_text())
        .map_err(|e| format!("点訳エラー: {e}"))
}

/// 点字文字 → 原文文字の対応。
fn braille_to_source(result: &BrailleResult, pred: &PredictionResult) -> Vec<usize> {
    pred.braille_char_to_source(result.text_to_braille(), result.braille_char_count())
}

/// `--table` から点訳器を作る。
///
/// `--table` の役割は「英語（UEB）テーブルの指定」。指定すれば埋め込み名
/// （`english_ueb_grade1` / `english_ueb_grade2`）または UEB スキーマの .toml として読み込み、
/// `BrailleTranslator` に英語エンジンとして渡す（行ごとの言語判定・振り分けは
/// `BrailleTranslator::translate` が内部で行う）。省略時は英語エンジンを持たせない
/// （`None`）ため、全行が既定の日本語テーブルへ流れる。
///
/// 埋め込みカタログは日本語・英語のテーブルを名前だけで共有しているため、
/// `japanese_no_conversion` のような日本語テーブル名を渡してもエラーにはならず、
/// 全行がそのテーブル（英語行も無縮約でそのまま）で点訳される。これは未文書化の
/// 副作用だが実用上そのまま使えるため許容する。
fn make_line_translator(table: Option<&String>) -> Result<BrailleTranslator, String> {
    let japanese = JapaneseTranslator::from_embedded()
        .map_err(|e| format!("点字テーブル読み込みエラー: {e}"))?;

    let Some(spec) = table else {
        return Ok(BrailleTranslator::new(japanese, None));
    };

    let english = if Path::new(spec).exists() {
        EnglishTranslator::from_file(spec)
    } else {
        EnglishTranslator::from_embedded_name(spec)
    }
    .map_err(|_| {
        let names: Vec<&str> = momors_braille::embedded_tables()
            .iter()
            .filter_map(|t| t.name.as_deref())
            .collect();
        format!(
            "点字テーブル \"{spec}\" が見つかりません。\
             ファイルパスか、埋め込みテーブル名 ({}) を指定してください",
            names.join(" / ")
        )
    })?;

    Ok(BrailleTranslator::new(japanese, Some(english)))
}

/// 自信度スコアを1行に出力する。
fn write_confidences(out: &mut impl Write, confidences: &[f32]) {
    for &c in confidences {
        write!(out, "{c} ").ok();
    }
    writeln!(out).ok();
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
        Some("brf") => Some(OutputFormat::BrailleText {
            case: cli.nabcc_case.into(),
        }),
        Some("bse") => Some(OutputFormat::Base),
        Some("bes") => Some(OutputFormat::Bes),
        Some("mbr") => Some(OutputFormat::Mbr),
        _ => None,
    };

    // 効かない指定を黙って無視すると「指定したつもり」の事故になるので弾く。
    if cli.nabcc_case != NabccCaseArg::Upper
        && !matches!(braille_format, Some(OutputFormat::BrailleText { .. }))
    {
        return Err("--nabcc-case は .brf 出力にのみ指定できます".to_string());
    }

    let text = std::fs::read_to_string(cli.input.as_ref().unwrap())
        .map_err(|e| format!("ファイル読み込みエラー: {e}"))?;
    let paragraphs: Vec<String> = text.lines().map(str::to_owned).collect();

    let out = match braille_format {
        Some(format) => {
            let lt = make_line_translator(cli.table.as_ref())?;
            let to_braille_text = |p: &str| -> Result<String, String> {
                let pred = to_kana(p, predictor)?;
                to_braille(&pred, &lt).map(|r| r.braille_text().to_owned())
            };
            let braille_list: Vec<String> = paragraphs
                .iter()
                .map(|p| to_braille_text(p))
                .collect::<Result<_, _>>()?;
            let title_braille = cli.title.as_ref().map(|t| to_braille_text(t)).transpose()?;
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
                .map(|p| to_kana(p, predictor).map(|r| r.kana_text().to_owned()))
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

// ============================================================
// 標準入力モード
// ============================================================

fn run_stdin(cli: &Cli, predictor: &Predictor) -> ExitCode {
    let lt: Option<BrailleTranslator> = if cli.braille {
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

        let start = std::time::Instant::now();
        let pred = match to_kana(&line, predictor) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{e}");
                had_error = true;
                continue;
            }
        };

        if let Some(lt) = &lt {
            match to_braille(&pred, lt) {
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
                        for i in braille_to_source(&result, &pred) {
                            write!(out, "{i} ").ok();
                        }
                        writeln!(out).ok();
                    }
                    if cli.confidence {
                        write_confidences(&mut out, pred.confidences());
                    }
                }
                Err(e) => {
                    eprintln!("{e}");
                    had_error = true;
                }
            }
            continue;
        }

        if cli.profile {
            writeln!(out, "予測時間: {:?}", start.elapsed()).ok();
        }
        if cli.segment {
            writeln!(out, "{}", pred.format_segmented()).ok();
        } else {
            writeln!(out, "{}", pred.kana_text()).ok();
        }
        if cli.confidence {
            write_confidences(&mut out, pred.confidences());
        }
    }

    if had_error {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
