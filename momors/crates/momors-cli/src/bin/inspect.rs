//! `momo-inspect` — momo の読みモデルを覗くための診断ツール。
//!
//! momo-py が学習専用ツールへ縮退した方針（`momo-compare-quant` 参照）に伴い、
//! Python CLI にあった診断系サブコマンド（`predict --trace` / `label`）を Rust へ移す。
//!
//! - `trace` : 標準入力の各行を推論し、原文文字ごとに「かな・決定根拠・自信度・
//!   上位特徴量寄与度」を色付きで表示する（Python `predict --trace [--explain N]` 相当）。
//! - `label` : 標準入力に特徴量名（または単一文字）を与えると、その特徴量に対する
//!   読みラベル別の重みを降順で表示する（Python `label` 相当）。単一文字は現行の
//!   特徴量名 `char_s=<文字>` として引く（Python 版は形骸化した `char=` を引いていた）。

use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use momors_core::{DecisionSource, FloatPredictor, Predictor, PredictorConfig, TraceResult};

fn data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("MOMO_DATASET_DIR") {
        return PathBuf::from(dir);
    }
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// モデルサイズの選択肢（`momo` 本体と同じ対応）。
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

/// momo の読みモデルを覗く診断ツール。
#[derive(Debug, Parser)]
#[command(name = "momo-inspect", version, about, long_about = None)]
struct Cli {
    /// モデルファイル (.mbm または量子化前 .mbmf) のパス
    #[arg(long, global = true)]
    model_file: Option<String>,

    /// 使用するモデルのサイズ (small, medium, large)
    #[arg(long, default_value = "large", global = true)]
    model: ModelSize,

    /// 単一文字辞書ファイル (.tsv) のパス（省略時はモデルに同梱された辞書を使用）
    #[arg(long, global = true)]
    single_dict: Option<String>,

    /// カスタム辞書（フレーズ辞書）ファイル (.tsv) のパス（省略時は無効）
    #[arg(long, global = true)]
    custom_dict: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// 標準入力の各行を推論し、文字ごとの決定根拠・自信度・特徴量寄与度を表示する。
    Trace {
        /// 各文字で表示する特徴量寄与度の上位件数（0 で寄与度を表示しない）。
        #[arg(long, default_value_t = 8, value_name = "N")]
        explain: usize,
    },
    /// 標準入力の特徴量名（または単一文字）に対する読みラベル別の重みを表示する。
    Label,
}

/// `.mbm`（量子化済み）と `.mbmf`（量子化前）のどちらでも読み込める薄いラッパー。
enum AnyPredictor {
    Quantized(Predictor),
    Float(FloatPredictor),
}

impl AnyPredictor {
    fn load(config: PredictorConfig) -> momors_core::Result<Self> {
        let is_float = config
            .model_path()
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("mbmf"));
        if is_float {
            FloatPredictor::load(config).map(AnyPredictor::Float)
        } else {
            Predictor::load(config).map(AnyPredictor::Quantized)
        }
    }

    fn trace(&self, text: &str, top_n: usize) -> momors_core::Result<TraceResult> {
        match self {
            AnyPredictor::Quantized(p) => p.trace(text, top_n),
            AnyPredictor::Float(p) => p.trace(text, top_n),
        }
    }

    fn feature_label_weights(&self, feature: &str) -> Vec<(String, f32)> {
        match self {
            AnyPredictor::Quantized(p) => p.feature_label_weights(feature),
            AnyPredictor::Float(p) => p.feature_label_weights(feature),
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
    if let Some(ref path) = cli.single_dict {
        let p = PathBuf::from(path);
        if !p.exists() {
            eprintln!("単一文字辞書ファイルが見つかりません: {}", p.display());
            return ExitCode::FAILURE;
        }
        config = config.with_single_char_dict_path(&p);
    }
    if let Some(ref path) = cli.custom_dict {
        let p = PathBuf::from(path);
        if !p.exists() {
            eprintln!("カスタム辞書ファイルが見つかりません: {}", p.display());
            return ExitCode::FAILURE;
        }
        config = config.with_custom_dict_path(&p);
    }

    let predictor = match AnyPredictor::load(config) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("モデル読み込みエラー: {e}");
            return ExitCode::FAILURE;
        }
    };

    match cli.command {
        Command::Trace { explain } => run_trace(&predictor, explain),
        Command::Label => run_label(&predictor),
    }
}

// ============================================================
// trace
// ============================================================

const BAR_WIDTH: usize = 12;
const ANSI_RESET: &str = "\x1b[0m";
const ANSI_DIM: &str = "\x1b[2m";

/// 決定根拠に対応する ANSI 色（Python `_ANSI` と対応）。
fn decision_color(dec: DecisionSource) -> &'static str {
    match dec {
        DecisionSource::Lr => "\x1b[32m",              // 緑
        DecisionSource::LrLow => "\x1b[33m",           // 黄
        DecisionSource::FallbackNumeric => "\x1b[36m", // シアン
        DecisionSource::FallbackCounter => "\x1b[96m", // 明るいシアン
        DecisionSource::FallbackRepeat => "\x1b[35m",  // マゼンタ
        DecisionSource::FallbackKana => "\x1b[96m",    // 明るいシアン
        DecisionSource::FallbackOrphan => "\x1b[31m",  // 赤
        DecisionSource::DictName => "\x1b[94m",        // 明るい青
        DecisionSource::DictCustom => "\x1b[95m",      // 明るいマゼンタ
        DecisionSource::Bypass => "\x1b[90m",          // グレー
    }
}

fn confidence_bar(conf: f32) -> String {
    let filled = (conf * BAR_WIDTH as f32).round() as usize;
    let filled = filled.min(BAR_WIDTH);
    let mut s = String::new();
    for _ in 0..filled {
        s.push('█');
    }
    for _ in 0..(BAR_WIDTH - filled) {
        s.push('░');
    }
    s
}

fn run_trace(predictor: &AnyPredictor, top_n: usize) -> ExitCode {
    let use_color = io::stdout().is_terminal();
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
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
        let text = line.trim();
        if text.is_empty() {
            continue;
        }

        let result = match predictor.trace(text, top_n) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("推論エラー: {e}");
                had_error = true;
                continue;
            }
        };

        writeln!(out, "{}", "─".repeat(48)).ok();
        write_trace(&mut out, &result, use_color);
        writeln!(out, "{}", "─".repeat(48)).ok();
    }

    if had_error {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn write_trace(out: &mut impl Write, result: &TraceResult, use_color: bool) {
    for row in &result.rows {
        // かなが無い文字はスキップ表示。
        let Some(dec) = row.decision else {
            writeln!(out, "  {}  →  (skip)", row.source).ok();
            continue;
        };
        let conf = row.confidence.unwrap_or(0.0);
        let tag = format!("[{:<13}]", dec.as_str());

        let line = if dec == DecisionSource::Bypass {
            format!("  {}  →  {:<6} {}", row.source, row.kana, tag)
        } else {
            format!(
                "  {}  →  {:<6} {} {} {:.3}",
                row.source,
                row.kana,
                tag,
                confidence_bar(conf),
                conf
            )
        };
        if use_color {
            writeln!(out, "{}{}{}", decision_color(dec), line, ANSI_RESET).ok();
        } else {
            writeln!(out, "{line}").ok();
        }

        // 特徴量寄与度（LR / LR_LOW の行のみ付いている）。
        for c in &row.contributions {
            let cline = format!("      {:<38} {:+.4}", c.feature, c.weight);
            if use_color {
                writeln!(out, "{ANSI_DIM}{cline}{ANSI_RESET}").ok();
            } else {
                writeln!(out, "{cline}").ok();
            }
        }
    }
}

// ============================================================
// label
// ============================================================

fn run_label(predictor: &AnyPredictor) -> ExitCode {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    writeln!(out, "AI 脳内スキャナー (Ctrl+D で終了)").ok();
    writeln!(out, "使い方: 見たい文字を1文字入力（例: 金 → char_s=金）").ok();
    writeln!(
        out,
        "応用: 特徴量名で直接検索（例: type_trans_p1_s=KANJI->HIRAGANA）"
    )
    .ok();
    writeln!(out, "{}", "-".repeat(40)).ok();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("入力エラー: {e}");
                continue;
            }
        };
        let feature = line.trim();
        if feature.is_empty() {
            continue;
        }

        let weights = predictor.feature_label_weights(feature);
        if weights.is_empty() {
            writeln!(out, "  -> (この特徴量は学習データに存在しません)").ok();
            writeln!(out, "{}", "-".repeat(40)).ok();
            continue;
        }

        writeln!(out, "\n検索対象: '{feature}'").ok();
        // 重みは降順。支持側（正の重み）のラベルだけを表示する
        // （Python 版が 0.0 で打ち切っていたのと同じ意図）。
        for (label, weight) in weights.iter().take_while(|(_, w)| *w > 0.0) {
            writeln!(out, "  ラベル: {label:6} | 点数: {weight:+.3}").ok();
        }
        writeln!(out, "{}", "-".repeat(40)).ok();
    }

    ExitCode::SUCCESS
}
