//! コーパス全体を流して `predict` の段階別内訳を 1 行で出す使い捨て診断。
//!
//! [`char_latency_probe`](crate::char_latency_probe) は 1 文字ごとの生の値を出すが、
//! こちらは「実運用に近い流し方（行単位の `predict`）で、どの段階に何 µs/文字
//! 掛かっているか」を 1 行にまとめる。語彙データ構造を変えたときの before/after を
//! 同じ物差しで比べるために作った（`.mbm` v0x09 の作業）。
//!
//! 実行例:
//! ```text
//! MOMO_BENCH_MODEL=../../dataset/basic_data_4.mbm MOMO_BENCH_TEXT=source.txt \
//!   cargo test --release -p momors-core --features diagnostics \
//!   --lib phase_probe -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(all(test, feature = "diagnostics"))]

use std::time::Instant;

use crate::model::MomoModel;
use crate::prediction::{Predictor, PredictorConfig};
use crate::weight_model::WeightModel;

fn model_path() -> String {
    std::env::var("MOMO_BENCH_MODEL")
        .expect("MOMO_BENCH_MODEL 環境変数に .mbm ファイルのパスを指定してください")
}

fn load_text() -> String {
    let path = std::env::var("MOMO_BENCH_TEXT")
        .expect("MOMO_BENCH_TEXT 環境変数にテキストファイルのパスを指定してください");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("テキストファイル読み込み失敗 ({path}): {e}"))
}

#[test]
#[ignore]
fn phase_breakdown() {
    let path = model_path();

    let t_load = Instant::now();
    let predictor =
        Predictor::<MomoModel>::load(PredictorConfig::new(&path)).expect("モデル読み込み失敗");
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    let text = load_text();
    let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();

    let model = predictor.model();
    let file_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "model={path}  file={:.2} MiB  n_features={}  n_classes={}  load={load_ms:.0} ms",
        file_bytes as f64 / 1048576.0,
        model.n_features(),
        model.n_classes(),
    );
    eprintln!(
        "vocab heap = {:.2} MiB  ({} エントリ)",
        model.vocab_heap_bytes() as f64 / 1048576.0,
        model.vocab_len(),
    );

    // ウォームアップ: 常駐プロセスの定常状態に近づける。
    for l in lines.iter().take(300) {
        std::hint::black_box(predictor.predict(l).ok());
    }
    crate::phase::take();

    let t0 = Instant::now();
    let mut chars = 0usize;
    for l in &lines {
        chars += l.chars().count();
        std::hint::black_box(predictor.predict(l).ok());
    }
    let wall = t0.elapsed();
    let acc = crate::phase::take();

    let n = chars.max(1) as f64;
    println!(
        "lines={} chars={chars} wall={:.3} s",
        lines.len(),
        wall.as_secs_f64()
    );
    println!("wall {:.2} µs/char", wall.as_secs_f64() * 1e6 / n);
    for (name, ns) in crate::phase::PHASE_NAMES.iter().zip(&acc) {
        println!("{name:<10} {:8.2} µs/char", *ns as f64 / 1000.0 / n);
    }
}
