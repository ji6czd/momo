//! 読み推論と境界推論を分離して、1文字ずつ生の所要時間を出す使い捨て診断。
//!
//! 直近のモデルデータ構造変更（境界モデルのGBDT化・統合語彙テーブル化）以降、
//! 体感の推論速度が落ちたという報告を受けて作った。正規化・漢数字ルール・
//! 々救済・LABEL_CONTINUE/SKIP救済・人名辞書読みフォールバック・インデックス
//! 構築・かな文字列の組み立てなど、[`crate::prediction::Predictor::predict`] が
//! 持つ「推論そのものではない」処理を一切行わない。各文字位置について
//!
//! 1. 語彙引き (`vocab_find`) + 読みモデルのスコア計算 (`compute_read_scores`)
//! 2. 境界モデルのスコア計算 (`compute_boundary_score`)
//!
//! だけを行い、対象の文字と所要時間 (ns) を1文字1行ずつ
//! `文字\t読み推論ns\t境界推論ns` の形で標準出力へ流す。集計はしない ──
//! 分布や外れ値は出力を渡した先（Python等）で見る想定。
//!
//! 実行例:
//! ```text
//! MOMO_BENCH_MODEL=../../dataset/basic_data_4.mbm MOMO_BENCH_TEXT=sample.txt \
//!   cargo test --release -p momors-core --lib char_latency_probe -- --ignored --nocapture --test-threads=1 \
//!   > timings.tsv
//! ```
//! （`cargo test` の "running 1 test" 等の枠が標準出力の先頭・末尾に混じるので、
//! 保存したファイルから該当行を取り除いて使うこと。）

#![cfg(test)]

use std::time::Instant;

use crate::feature::FeatureKey;
use crate::featurize::SourceEntry;
use crate::model::MomoModel;
use crate::name_dict::compute_name_matches;
use crate::weight_model::WeightModel;

fn model_path() -> String {
    std::env::var("MOMO_BENCH_MODEL")
        .expect("MOMO_BENCH_MODEL 環境変数に .mbm ファイルのパスを指定してください")
}

/// 計測対象のテキスト。`MOMO_BENCH_TEXT` が指すファイルを読む。未設定なら標準入力。
fn load_text() -> String {
    match std::env::var("MOMO_BENCH_TEXT") {
        Ok(path) => std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("テキストファイル読み込み失敗 ({path}): {e}")),
        Err(_) => {
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .expect("標準入力からのテキスト読み込み失敗");
            buf
        }
    }
}

/// 特徴量キー列を読みモデルの feature_id 列へ変換する。
/// `prediction.rs::lookup_feature_ids` と同じ処理（private のためここに複製）。
fn read_feat_ids(model: &MomoModel, keys: &[FeatureKey]) -> Vec<u32> {
    keys.iter().filter_map(|k| model.vocab_find(k)).collect()
}

/// `SourceEntry` が表す原文の文字列（拗音・促音などの複合ユニットは2〜3文字）を
/// 表示用に組み立てる。出力に「どの文字の結果か」を添えるためだけの補助。
fn entry_text(entry: &SourceEntry) -> String {
    let cps = [entry.cp, entry.cp2, entry.cp3];
    cps[..entry.compound_len.max(1) as usize]
        .iter()
        .filter_map(|&cp| char::from_u32(cp))
        .collect()
}

#[test]
#[ignore]
fn char_latency_raw() {
    let model = MomoModel::load(std::path::Path::new(&model_path())).expect("モデル読み込み失敗");
    let text = load_text();

    // 特徴量計算は1文字あたりの推論そのものではない共通前処理なので、
    // 計測ループの外（ウォームアップより前）でまとめて行う。
    let source_seq = crate::featurize::to_source_seq(&text);
    let (name_flags, _) = compute_name_matches(&source_seq, model.name_dict());
    let all_feat_keys = crate::featurize::compute_source_features(&source_seq, &name_flags);

    eprintln!(
        "boundary={} chars={}",
        match &model.boundary {
            crate::boundary::Boundary::Linear { .. } => "Linear",
            crate::boundary::Boundary::Tree(_) => "Tree(GBDT)",
        },
        all_feat_keys.len(),
    );

    let mut scratch = model.new_scratch();
    let mut scores = vec![0f32; model.n_classes() as usize];

    // ウォームアップ: ページキャッシュ・分岐予測を常駐プロセスの定常状態に近づける
    // （初回だけ余計に遅い分を計測対象から外す）。
    for keys in &all_feat_keys {
        let feat_ids = read_feat_ids(&model, keys);
        model.compute_read_scores(&feat_ids, &mut scratch, &mut scores);
        std::hint::black_box(&scores);
        std::hint::black_box(model.compute_boundary_score(keys));
    }

    // 本計測: 1文字ずつ「文字」「読み推論 ns」「境界推論 ns」を1行に出す。
    for (entry, keys) in source_seq.iter().zip(&all_feat_keys) {
        let t0 = Instant::now();
        let feat_ids = read_feat_ids(&model, keys);
        model.compute_read_scores(&feat_ids, &mut scratch, &mut scores);
        std::hint::black_box(&scores);
        let read_ns = t0.elapsed().as_nanos();

        let t1 = Instant::now();
        let boundary_score = model.compute_boundary_score(keys);
        std::hint::black_box(boundary_score);
        let boundary_ns = t1.elapsed().as_nanos();

        println!("{}\t{read_ns}\t{boundary_ns}", entry_text(entry));
    }
}
