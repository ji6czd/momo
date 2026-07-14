//! `momo` コマンドの実装。
//!
//! 標準入力から 1 行ずつ読んで予測し、結果を標準出力に書く（デフォルト動作）。
//! `--input` を指定するとファイル整形モードに切り替わる。

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use momors_braille::{
    detect_language, BrailleDocument, BrailleResult, BrailleTranslator, DocumentConfig,
    JapaneseTranslator, Language, NabccCase, OutputFormat,
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

    /// 点字変換テーブル。埋め込みテーブル名（japanese_grade1 / japanese_no_conversion /
    /// english_ueb_grade1 / english_ueb_grade2）か、.toml のパス。名前は TOML のファイル名と
    /// 同じ。英語テーブルを指定したときだけ、行ごとに日本語／英語を判定して英語行を UEB で
    /// 点訳する（日本語行は既定の日本語テーブル）
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

/// 1行を点訳する。予測（漢字かな交じり → かな）は CLI 側で行い、点訳器にはかなを渡す
/// （momors-braille は漢字列を扱わない）。日本語行なら予測結果も一緒に返す。
///
/// `routing` が true のときだけ言語を判定し、日本語文字を含まない行を **予測を通さずに**
/// UEB で点訳する。false のときは従来どおり必ず日本語として点訳する
/// （英字は外字符 `⠰` ＋無縮約）。
fn translate_line(
    text: &str,
    predictor: &Predictor,
    lt: &BrailleTranslator,
    routing: bool,
) -> Result<(BrailleResult, Option<PredictionResult>), String> {
    if routing && detect_language(text) == Language::English {
        // 英語エンジンを持たない点訳器（no_conversion など）では日本語経路へ落ちる。
        if let Some(result) = lt.translate_english(text) {
            return Ok((result, None));
        }
    }
    let pred = predictor
        .predict(text)
        .map_err(|e| format!("仮名変換エラー: {e}"))?;
    let result = lt
        .translate_japanese(pred.kana_text())
        .map_err(|e| format!("点訳エラー: {e}"))?;
    Ok((result, Some(pred)))
}

/// 1行を点訳して点字だけ返す。空文字列は空文字列のまま返す。
fn to_braille(
    text: &str,
    predictor: &Predictor,
    lt: &BrailleTranslator,
    routing: bool,
) -> Result<String, String> {
    if text.is_empty() {
        return Ok(String::new());
    }
    translate_line(text, predictor, lt, routing).map(|(r, _)| r.braille_text().to_owned())
}

/// 点字文字 → 原文文字の対応。日本語行は 点字→かな→原文 を合成し、
/// 英語行は読み層が原文と同一なので 点字→テキスト をそのまま使う。
fn braille_to_source(result: &BrailleResult, pred: Option<&PredictionResult>) -> Vec<usize> {
    match pred {
        Some(pred) => {
            pred.braille_char_to_source(result.text_to_braille(), result.braille_char_count())
        }
        None => result.braille_to_text().to_vec(),
    }
}

/// `--table` から点訳器を作り、「行ごとに言語を判定するか」を返す。
///
/// - **英語テーブル**（埋め込み名 `english_*`、または UEB スキーマの .toml）… 英語行はそれで
///   点訳し、日本語行は既定の日本語テーブル。**行ごとの言語判定を行う**
/// - **日本語テーブル**（埋め込み名 `japanese_*`、または日本語スキーマの .toml）… 全行それで
///   点訳する（判定なし）
/// - 省略 … 既定の日本語テーブル（判定なし）
fn make_line_translator(table: Option<&String>) -> Result<(BrailleTranslator, bool), String> {
    let default_japanese = || {
        JapaneseTranslator::from_embedded().map_err(|e| format!("点字テーブル読み込みエラー: {e}"))
    };

    let Some(spec) = table else {
        let english = default_english()?;
        return Ok((
            BrailleTranslator::new(default_japanese()?, Some(english)),
            false,
        ));
    };

    // 英語テーブル（名前 or ファイル）なら、日本語は既定にして行ごとに判定する。
    let english = if Path::new(spec).exists() {
        momors_braille::EnglishTranslator::from_file(spec).ok()
    } else {
        momors_braille::EnglishTranslator::from_embedded_name(spec).ok()
    };
    if let Some(english) = english {
        return Ok((
            BrailleTranslator::new(default_japanese()?, Some(english)),
            true,
        ));
    }

    // 日本語テーブル（名前 or ファイル）。
    let japanese = if Path::new(spec).exists() {
        JapaneseTranslator::from_file(spec)
            .map_err(|e| format!("点字テーブル読み込みエラー: {e}"))?
    } else {
        JapaneseTranslator::from_embedded_name(spec).map_err(|_| {
            let names: Vec<&str> = momors_braille::embedded_tables()
                .iter()
                .filter_map(|t| t.name.as_deref())
                .collect();
            format!(
                "点字テーブル \"{spec}\" が見つかりません。\
                 ファイルパスか、埋め込みテーブル名 ({}) を指定してください",
                names.join(" / ")
            )
        })?
    };
    Ok((
        BrailleTranslator::new(japanese, Some(default_english()?)),
        false,
    ))
}

/// 既定の英語テーブル（UEB grade 2）。日本語テーブル指定時も英語行の点訳器として持っておく。
fn default_english() -> Result<momors_braille::EnglishTranslator, String> {
    momors_braille::EnglishTranslator::from_embedded()
        .map_err(|e| format!("英語点字テーブル読み込みエラー: {e}"))
}

// ============================================================
// 標準入力モード
// ============================================================

fn run_stdin(cli: &Cli, predictor: &Predictor) -> ExitCode {
    // 点字モードのみ。英語テーブル（`--table english_ueb_grade2` など）のときだけ
    // 行ごとに言語を判定する。
    let braille: Option<(BrailleTranslator, bool)> = if cli.braille {
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
            match translate_line(&line, predictor, lt, routing) {
                Ok((result, pred)) => {
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
                        for i in braille_to_source(&result, pred.as_ref()) {
                            write!(out, "{i} ").ok();
                        }
                        writeln!(out).ok();
                    }
                    // 確信度は日本語行（予測を通った行）だけ出せる
                    if let (true, Some(pred)) = (cli.confidence, &pred) {
                        for &c in pred.confidences() {
                            write!(out, "{c} ").ok();
                        }
                        writeln!(out).ok();
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
