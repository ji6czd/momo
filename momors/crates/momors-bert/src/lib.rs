//! # momors-bert
//!
//! 微調整した BERT（`tohoku-nlp/bert-base-japanese-char-v3`）で漢字・漢数字の読みと
//! マスあけを推論し、[`momors_core::SlotReader`] として予測器に差し込む。
//!
//! モデルは `momo-bert-train` の `export_onnx` が書き出したディレクトリを読む:
//!
//! | ファイル | 中身 |
//! |---|---|
//! | `reading_bert.int8.onnx` / `reading_bert.onnx` | 入力 `input_ids`・`attention_mask` [B, T]、出力 `logits` [B, T, ラベル数]（読み）と、境界も学習したモデルなら `boundary_logits` [B, T]（その文字の後ろでマスをあける logit） |
//! | `labels.json` | 読みのラベル表（継続は `""`） |
//! | `cands.json` | `{"chars": {文字: [ラベル番号, …]}}`（学習データに出た読み） |
//! | `vocab.txt` | 文字の語彙 |
//!
//! 1 行を丸ごと（長ければ窓に区切って）BERT に通し、`cands.json` に載っている文字の
//! 位置ごとに、その文字の候補だけに絞った softmax を返す。候補に無い文字（学習データに
//! 出なかった漢字など）は返さないので、その文字は LR が読む。
//!
//! モデルが `boundary_logits` を持っていれば（`momo-bert-train` の `step9_finetune --boundary`）、
//! 空白に接しない切れ目ごとにマスあけの確率も返し、その切れ目は境界モデルの代わりにこれで
//! 決まる。マスあけは漢字の無い行にもあるので、そのときは全行を BERT に通す。
//!
//! ```no_run
//! use momors_bert::{BertOptions, BertReader};
//! use momors_core::{Predictor, PredictorConfig};
//!
//! let mut predictor: Predictor = Predictor::load(PredictorConfig::new("basic_data_7.mbm"))?;
//! let reader = BertReader::load("work/onnx/train", &BertOptions::default())?;
//! predictor.set_slot_reader(Box::new(reader));
//! println!("{}", predictor.predict("株式市場が反発した。")?.kana_text());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod vocab;
mod window;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use momors_core::{LineScores, SlotReader, SlotScores, SplitScore};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use crate::vocab::CharVocab;

/// モデルを置く既定のディレクトリ名。`.mbm` を探す既定のディレクトリの下に置く
/// （配布用の momo-bert-*.zip を展開すると、このディレクトリができる）。
pub const DEFAULT_DIR_NAME: &str = "momo-bert";

/// 読みのラベル表で、継続（前の文字と一続きの読み）を表すもの（momors-core と同じ）。
const LABEL_CONTINUE: &str = "---";

/// 読みのラベル表で、読まずに飛ばす文字を表すもの（momors-core と同じ）。
///
/// BERT の学習データは飛ばす文字（「三百二十」の「十」など）の正解を、その文字自身で
/// 表している。読み込み時に「ラベル＝候補を持つ文字そのもの」をこのラベルに読み替える。
const LABEL_SKIP: &str = "_";

/// BERT に入れる 1 窓の最大文字数（位置埋め込み 512 から [CLS] と [SEP] を除く）。
const MAX_CHARS: usize = 510;

/// 長い行を区切るとき、窓の前後に重ねる文字数。
const OVERLAP: usize = 64;

/// momors-bert のエラー。
#[derive(Debug, thiserror::Error)]
pub enum BertError {
    #[error("{path} を読めません: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path} の JSON が不正です: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("モデルの形式が不正です: {0}")]
    Format(String),
    #[error("ONNX Runtime: {0}")]
    Ort(#[from] ort::Error),
}

/// 読み込みの設定。
#[derive(Debug, Clone)]
pub struct BertOptions {
    /// int8 量子化版（`reading_bert.int8.onnx`）を使う。偽なら float32 版。
    pub int8: bool,
    /// ONNX Runtime の演算スレッド数。0 なら ONNX Runtime の既定（物理コア数）。
    pub threads: usize,
    /// モデルが境界の出力（`boundary_logits`）を持っていれば、マスあけも BERT で決める。
    /// 偽なら読みだけを返し、マスあけは境界モデルに任せる。
    pub boundary: bool,
}

impl Default for BertOptions {
    fn default() -> Self {
        Self {
            int8: true,
            threads: 0,
            boundary: true,
        }
    }
}

/// 境界の出力の名前。
const BOUNDARY_OUTPUT: &str = "boundary_logits";

/// 微調整 BERT による読み（とマスあけ）の推論器。
pub struct BertReader {
    /// ONNX Runtime のセッションは推論に `&mut` が要るのでロックで包む。
    session: Mutex<Session>,
    vocab: CharVocab,
    labels: Vec<String>,
    /// 文字 → その文字に許す読み（ラベル番号）。
    cands: HashMap<char, Vec<u32>>,
    /// マスあけも返すか（モデルが境界の出力を持ち、[`BertOptions::boundary`] が真）。
    boundary: bool,
}

impl std::fmt::Debug for BertReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BertReader")
            .field("labels", &self.labels.len())
            .field("chars", &self.cands.len())
            .field("boundary", &self.boundary)
            .finish()
    }
}

/// 窓をまとめて BERT に通した結果。
struct RunOutput {
    /// 読みの logit（[B, T, L] を平たくしたもの）。
    logits: Vec<f32>,
    /// 境界の logit（[B, T] を平たくしたもの）。マスあけを返さないときは `None`。
    boundary: Option<Vec<f32>>,
    /// 窓の長さ T（[CLS]・[SEP] と詰め物を含む）。
    width: usize,
    /// ラベル数 L。
    n_labels: usize,
}

/// プロセスに 1 つだけ作れる ONNX Runtime の環境（推論器を複数読み込んでも共有する）。
fn environment() -> Result<&'static ort::environment::Environment, BertError> {
    static ENV: OnceLock<ort::environment::Environment> = OnceLock::new();
    static INIT: Mutex<()> = Mutex::new(());
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    // 2 つのスレッドが同時に作ろうとすると片方が失敗するので、作るところだけ直列にする。
    let _guard = INIT.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    let env = ort::init().with_name("momo-bert").build()?;
    Ok(ENV.get_or_init(|| env))
}

fn read_text(path: &Path) -> Result<String, BertError> {
    std::fs::read_to_string(path).map_err(|source| BertError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn read_json(path: &Path) -> Result<serde_json::Value, BertError> {
    serde_json::from_str(&read_text(path)?).map_err(|source| BertError::Json {
        path: path.to_path_buf(),
        source,
    })
}

impl BertReader {
    /// `export_onnx` の出力ディレクトリから読み込む。
    pub fn load(dir: impl AsRef<Path>, options: &BertOptions) -> Result<Self, BertError> {
        let dir = dir.as_ref();
        let labels: Vec<String> = read_json(&dir.join("labels.json"))?
            .as_array()
            .ok_or_else(|| BertError::Format("labels.json が配列ではありません".into()))?
            .iter()
            .map(|l| match l.as_str() {
                Some("") => Ok(LABEL_CONTINUE.to_string()),
                Some(l) => Ok(l.to_string()),
                None => Err(BertError::Format(
                    "labels.json に文字列でない要素があります".into(),
                )),
            })
            .collect::<Result<_, _>>()?;

        let cands_json = read_json(&dir.join("cands.json"))?;
        let chars = cands_json
            .get("chars")
            .and_then(|c| c.as_object())
            .ok_or_else(|| BertError::Format("cands.json に chars がありません".into()))?;
        let mut cands = HashMap::with_capacity(chars.len());
        for (key, ids) in chars {
            let mut it = key.chars();
            let (Some(ch), None) = (it.next(), it.next()) else {
                return Err(BertError::Format(format!(
                    "cands.json の文字が 1 文字でない: {key}"
                )));
            };
            let ids: Vec<u32> = ids
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_u64())
                .map(|v| v as u32)
                .filter(|&v| (v as usize) < labels.len())
                .collect();
            if !ids.is_empty() {
                cands.insert(ch, ids);
            }
        }
        let mut labels = labels;
        for (ch, ids) in &cands {
            for &l in ids {
                let label = &mut labels[l as usize];
                if label.chars().eq(std::iter::once(*ch)) {
                    *label = LABEL_SKIP.to_string();
                }
            }
        }

        let vocab = CharVocab::parse(&read_text(&dir.join("vocab.txt"))?)?;

        let model = dir.join(if options.int8 {
            "reading_bert.int8.onnx"
        } else {
            "reading_bert.onnx"
        });
        let mut builder = Session::builder(environment()?)?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(ort::Error::from)?;
        if options.threads > 0 {
            builder = builder
                .with_intra_threads(options.threads)
                .map_err(ort::Error::from)?;
        }
        let session = builder.commit_from_file(&model)?;
        let boundary = options.boundary
            && session
                .outputs()
                .iter()
                .any(|o| o.name() == BOUNDARY_OUTPUT);

        Ok(Self {
            session: Mutex::new(session),
            vocab,
            labels,
            cands,
            boundary,
        })
    }

    /// マスあけも返すか（モデルが境界の出力を持ち、[`BertOptions::boundary`] が真）。
    pub fn has_boundary(&self) -> bool {
        self.boundary
    }

    /// 窓（行の中の文字範囲）をまとめて BERT に通す。
    fn run(&self, ids: &[Vec<i64>]) -> Result<RunOutput, BertError> {
        let batch = ids.len();
        let width = ids.iter().map(Vec::len).max().unwrap_or(0);
        let mut input = vec![self.vocab.pad; batch * width];
        let mut mask = vec![0i64; batch * width];
        for (b, row) in ids.iter().enumerate() {
            input[b * width..b * width + row.len()].copy_from_slice(row);
            mask[b * width..b * width + row.len()].fill(1);
        }
        let shape = [batch, width];
        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
        let outputs = session.run(ort::inputs![
            "input_ids" => TensorRef::from_array_view((shape, &input[..]))?,
            "attention_mask" => TensorRef::from_array_view((shape, &mask[..]))?,
        ])?;
        let (out_shape, data) = outputs["logits"].try_extract_tensor::<f32>()?;
        let n_labels = *out_shape
            .last()
            .ok_or_else(|| BertError::Format("logits の形が不正".into()))?
            as usize;
        if n_labels != self.labels.len() {
            return Err(BertError::Format(format!(
                "logits のラベル数 {n_labels} が labels.json の {} と合いません",
                self.labels.len()
            )));
        }
        let logits = data.to_vec();
        let boundary = if self.boundary {
            let (_, data) = outputs[BOUNDARY_OUTPUT].try_extract_tensor::<f32>()?;
            if data.len() != batch * width {
                return Err(BertError::Format(format!(
                    "{BOUNDARY_OUTPUT} の要素数 {} が入力の {batch}×{width} と合いません",
                    data.len()
                )));
            }
            Some(data.to_vec())
        } else {
            None
        };
        Ok(RunOutput {
            logits,
            boundary,
            width,
            n_labels,
        })
    }

    /// 1 行の読みの候補とマスあけの確率（[`SlotReader::read_line`] の本体）。
    pub fn read(&self, text: &str) -> Result<LineScores, BertError> {
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let has_cands = |i: usize| self.cands.contains_key(&chars[i].1);
        // 位置 i の後ろの切れ目にマスあけの確率を付けるか。学習と同じく、空白に接する
        // 切れ目と行末は除く。
        let has_gap = |i: usize| {
            self.boundary
                && i + 1 < chars.len()
                && !chars[i].1.is_whitespace()
                && !chars[i + 1].1.is_whitespace()
        };
        // 候補のある文字も、マスあけを問う切れ目も無い窓は流さない（境界を返すなら、
        // 2 文字以上の行は全部流れる）。
        let windows: Vec<_> = window::plan(chars.len(), MAX_CHARS, OVERLAP)
            .into_iter()
            .filter(|w| w.keep.clone().any(|i| has_cands(i) || has_gap(i)))
            .collect();
        if windows.is_empty() {
            return Ok(LineScores::default());
        }
        let ids: Vec<Vec<i64>> = windows
            .iter()
            .map(|w| {
                let mut v = Vec::with_capacity(w.span.len() + 2);
                v.push(self.vocab.cls);
                v.extend(
                    chars[w.span.clone()]
                        .iter()
                        .map(|&(_, c)| self.vocab.token_id(c)),
                );
                v.push(self.vocab.sep);
                v
            })
            .collect();
        let out = self.run(&ids)?;

        let mut line = LineScores::default();
        for (b, w) in windows.iter().enumerate() {
            for i in w.keep.clone() {
                // 窓 b の中のトークン位置（先頭の [CLS] の分ずらす）。
                let row = b * out.width + 1 + i - w.span.start;
                let (byte_pos, ch) = chars[i];
                if let Some(allowed) = self.cands.get(&ch) {
                    let l = out.n_labels;
                    line.slots.push(SlotScores {
                        byte_pos,
                        candidates: softmax_over(&out.logits[row * l..(row + 1) * l], allowed),
                    });
                }
                if let Some(boundary) = &out.boundary
                    && has_gap(i)
                {
                    line.splits.push(SplitScore {
                        byte_pos: chars[i + 1].0,
                        prob: sigmoid(boundary[row]),
                    });
                }
            }
        }
        Ok(line)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `allowed` のラベルだけで softmax を取り、(ラベル番号, 確率) を確率の降順に並べる。
fn softmax_over(logits: &[f32], allowed: &[u32]) -> Vec<(u32, f32)> {
    let max = allowed
        .iter()
        .map(|&l| logits[l as usize])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<(u32, f32)> = allowed
        .iter()
        .map(|&l| (l, (logits[l as usize] - max).exp()))
        .collect();
    let sum: f32 = out.iter().map(|(_, e)| e).sum();
    for (_, e) in &mut out {
        *e /= sum;
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

impl SlotReader for BertReader {
    fn labels(&self) -> &[String] {
        &self.labels
    }

    fn read_line(&self, text: &str) -> Result<LineScores, String> {
        self.read(text).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_over_candidates_sorted() {
        let logits = [0.0, 3.0, 1.0, 10.0];
        let out = softmax_over(&logits, &[0, 1, 2]);
        assert_eq!(
            out.iter().map(|(l, _)| *l).collect::<Vec<_>>(),
            vec![1, 2, 0]
        );
        let sum: f32 = out.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // 候補外（3）は大きくても入らない。
        assert!(out.iter().all(|(l, _)| *l != 3));
    }

    /// 1 行ずつ BERT だけを流して µs/字 を出す使い捨て計測（`MOMO_BERT_DIR` と
    /// `MOMO_BENCH_TEXT`（1 行 1 段落のテキスト）が要る。スレッド数は `MOMO_BERT_THREADS`）。
    ///
    /// `cargo test -p momors-bert --release -- --ignored --nocapture bench_read`
    #[test]
    #[ignore]
    fn bench_read() {
        let dir = std::env::var("MOMO_BERT_DIR").expect("MOMO_BERT_DIR");
        let text = read_text(Path::new(
            &std::env::var("MOMO_BENCH_TEXT").expect("MOMO_BENCH_TEXT"),
        ))
        .unwrap();
        let threads = std::env::var("MOMO_BERT_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let lines: Vec<&str> = text.lines().collect();
        let n_chars: usize = lines.iter().map(|l| l.chars().count()).sum();
        for int8 in [true, false] {
            let options = BertOptions {
                int8,
                threads,
                ..BertOptions::default()
            };
            let reader = BertReader::load(&dir, &options).unwrap();
            let t0 = std::time::Instant::now();
            for line in &lines {
                reader.read(line).unwrap();
            }
            let sec = t0.elapsed().as_secs_f64();
            eprintln!(
                "int8={int8} threads={threads}: {} 行 {n_chars} 字 {sec:.2} 秒 = {:.1} µs/字・{:.1} ms/行",
                lines.len(),
                sec * 1e6 / n_chars as f64,
                sec * 1e3 / lines.len() as f64
            );
        }
    }

    /// `MOMO_BERT_DIR`（`export_onnx` の出力）があれば、Python が書いた参照 logit と突き合わせる。
    ///
    /// float32 版は logit の最大差、int8 版は候補内 argmax の一致率を見る。境界の出力が
    /// あれば、同じく float32 版は logit の最大差、int8 版はマスあけの判定の一致率を見る。
    #[test]
    fn matches_python_reference() {
        let Ok(dir) = std::env::var("MOMO_BERT_DIR") else {
            eprintln!("MOMO_BERT_DIR が無いので省略");
            return;
        };
        let dir = PathBuf::from(dir);
        let refs: Vec<serde_json::Value> = read_text(&dir.join("reference.jsonl"))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        for int8 in [false, true] {
            let options = BertOptions {
                int8,
                threads: 0,
                ..BertOptions::default()
            };
            let reader = BertReader::load(&dir, &options).unwrap();
            assert_eq!(
                reader.has_boundary(),
                refs.iter().all(|r| r.get("boundary").is_some()),
                "境界の出力の有無が参照と合わない"
            );
            let (mut max_diff, mut hit, mut n) = (0f32, 0, 0);
            let (mut b_max_diff, mut b_hit, mut b_n) = (0f32, 0, 0);
            for r in &refs {
                let text = r["text"].as_str().unwrap();
                let ids: Vec<i64> = r["ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect();
                let chars: Vec<char> = text.chars().collect();
                // 語彙の引き方が Python と同じであること。
                let mine: Vec<i64> = std::iter::once(reader.vocab.cls)
                    .chain(chars.iter().map(|&c| reader.vocab.token_id(c)))
                    .chain(std::iter::once(reader.vocab.sep))
                    .collect();
                assert_eq!(mine, ids, "{text}");
                let out = reader.run(&[ids]).unwrap();
                let (logits, l) = (&out.logits, out.n_labels);
                if let (Some(got), Some(want)) = (&out.boundary, r.get("boundary")) {
                    // 参照は文字ごと（[CLS] を除く）。
                    for (t, w) in want.as_array().unwrap().iter().enumerate() {
                        let (a, b) = (got[1 + t], w.as_f64().unwrap() as f32);
                        b_max_diff = b_max_diff.max((a - b).abs());
                        b_hit += ((a >= 0.0) == (b >= 0.0)) as usize;
                        b_n += 1;
                    }
                }
                for (p, want) in r["pos"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(r["logits"].as_array().unwrap())
                {
                    let t = 1 + p.as_u64().unwrap() as usize;
                    let got = &logits[t * l..(t + 1) * l];
                    let want: Vec<f32> = want
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap() as f32)
                        .collect();
                    for (a, b) in got.iter().zip(&want) {
                        max_diff = max_diff.max((a - b).abs());
                    }
                    let ch = chars[p.as_u64().unwrap() as usize];
                    if let Some(allowed) = reader.cands.get(&ch) {
                        hit += (softmax_over(got, allowed)[0].0
                            == softmax_over(&want, allowed)[0].0)
                            as usize;
                        n += 1;
                    }
                }
            }
            eprintln!("int8={int8}: logit 最大差 {max_diff:.2e}、候補内 argmax 一致 {hit}/{n}");
            if b_n > 0 {
                eprintln!(
                    "int8={int8}: 境界 logit 最大差 {b_max_diff:.2e}、マスあけの判定の一致 {b_hit}/{b_n}"
                );
            }
            if int8 {
                assert!(
                    hit * 100 >= n * 95,
                    "int8 の argmax 一致が 95% 未満: {hit}/{n}"
                );
                assert!(
                    b_hit * 100 >= b_n * 95,
                    "int8 のマスあけの判定の一致が 95% 未満: {b_hit}/{b_n}"
                );
            } else {
                assert!(max_diff < 1e-3, "float32 の logit 最大差 {max_diff}");
                assert_eq!(hit, n);
                assert!(
                    b_max_diff < 1e-3,
                    "float32 の境界 logit 最大差 {b_max_diff}"
                );
            }
        }
    }
}
