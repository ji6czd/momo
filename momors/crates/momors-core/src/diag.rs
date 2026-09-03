//! 推論速度の内訳を測るための診断 API（cargo feature `diagnostics` で有効）。
//!
//! [`char_latency_probe`](crate::char_latency_probe)（PC 上の `#[cfg(test)]` 版）と同じ
//! 計測を、組み込みターゲット（momors-esp32 など）から呼べる形にしたもの。
//! [`Predictor::predict`] が持つ正規化・救済・かな組み立てなどは行わず、各文字位置について
//!
//! 1. 語彙引き (`vocab_find`)
//! 2. 読みモデルのスコア計算 (`compute_read_scores`)
//! 3. 境界モデルのスコア計算 (`compute_boundary_score`。内部で語彙引きをもう一度行う)
//!
//! の所要時間をナノ秒で返す。本番コードからは使わない。

use std::time::Instant;

use crate::feature::FeatureKey;
use crate::model::MomoModel;
use crate::name_dict::compute_name_matches;
use crate::prediction::Predictor;
use crate::weight_model::WeightModel;

/// 1 文字位置ぶんの計測値。
#[derive(Debug, Clone)]
pub struct CharLatency {
    /// 対象の文字（複合ユニットは 2〜3 文字）。
    pub text: String,
    /// この位置の特徴量キー数。
    pub n_keys: usize,
    /// 語彙に載っていたキー数。
    pub n_hits: usize,
    /// ヒットした特徴の CSC 列の非ゼロ要素数の合計（読みスコア計算で触る要素数）。
    pub nnz: usize,
    /// 語彙引き（全キーの `vocab_find`）の所要時間。
    pub lookup_ns: u128,
    /// 読みモデルのスコア計算（語彙引きを除く）の所要時間。
    pub read_ns: u128,
    /// 境界モデルのスコア計算（内部の語彙引きを含む）の所要時間。
    pub boundary_ns: u128,
}

/// 読みモデル CSC の統計（[`Predictor::csc_stats`]）。
#[derive(Debug, Clone, Copy)]
pub struct CscStats {
    /// 列数（= 特徴量数）。
    pub n_cols: usize,
    /// 非ゼロ総数。
    pub total_nnz: usize,
    /// 1 列あたり非ゼロ数の最大。
    pub max_nnz: usize,
    /// 密な列（非ゼロがクラス数の半分以上）の数。
    pub dense_cols: usize,
    /// 密な列の非ゼロ総数。
    pub dense_nnz: usize,
}

/// [`Predictor::predict_phases`] の結果: 段階ごとの 1 文字あたり平均 µs。
#[derive(Debug, Clone)]
pub struct PhaseReport {
    /// 文字数 × 繰り返し回数。
    pub chars: usize,
    /// `crate::phase::PHASE_NAMES` の順で、1 文字あたりの平均 µs。
    pub us_per_char: [f64; 9],
}

impl PhaseReport {
    /// "total 1055.0 | normalize 3.2 ..." の 1 行にまとめる。
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        for (name, us) in crate::phase::PHASE_NAMES.iter().zip(&self.us_per_char) {
            parts.push(format!("{name} {us:.1}"));
        }
        // ループ内で読み・境界以外に掛かった分（ラベル出力・インデックス構築・argmax 以外）
        let loop_other = self.us_per_char[5] - self.us_per_char[6] - self.us_per_char[7];
        // loop は read/boundary を含むので、内訳の合計は loop を 1 回だけ数える
        let accounted: f64 = self.us_per_char[1] + self.us_per_char[2] + self.us_per_char[3]
            + self.us_per_char[4] + self.us_per_char[5] + self.us_per_char[8];
        parts.push(format!("| loop_other {loop_other:.1}"));
        parts.push(format!(
            "| unaccounted {:.1}",
            self.us_per_char[0] - accounted
        ));
        parts.join(" ")
    }
}

impl Predictor<MomoModel> {
    /// `text` を `repeat` 回 `predict` し、段階別の 1 文字あたり平均所要時間を返す。
    ///
    /// 1 回目はウォームアップとして計測から外す。
    pub fn predict_phases(&self, text: &str, repeat: usize) -> PhaseReport {
        let chars = text.chars().count().max(1);
        std::hint::black_box(self.predict(text).ok());
        crate::phase::take();
        for _ in 0..repeat.max(1) {
            std::hint::black_box(self.predict(text).ok());
        }
        let acc = crate::phase::take();
        let n = (chars * repeat.max(1)) as f64;
        let mut us_per_char = [0f64; 9];
        for (dst, &ns) in us_per_char.iter_mut().zip(&acc) {
            *dst = ns as f64 / 1000.0 / n;
        }
        PhaseReport {
            chars: chars * repeat.max(1),
            us_per_char,
        }
    }

    /// 読みモデル CSC の全体統計。
    pub fn csc_stats(&self) -> CscStats {
        let model = self.model();
        let colptr = &model.csc_colptr;
        let n_classes = model.n_classes() as usize;
        let total = *colptr.last().unwrap_or(&0) as usize;
        let n_cols = colptr.len().saturating_sub(1);
        let mut max = 0;
        let mut dense_cols = 0;
        let mut dense_nnz = 0;
        for w in colptr.windows(2) {
            let nnz = (w[1] - w[0]) as usize;
            max = max.max(nnz);
            // 半分以上のクラスに非ゼロを持つ列を「密」とみなす
            if nnz * 2 >= n_classes {
                dense_cols += 1;
                dense_nnz += nnz;
            }
        }
        CscStats {
            n_cols,
            total_nnz: total,
            max_nnz: max,
            dense_cols,
            dense_nnz,
        }
    }

    /// `text` の各文字位置について、語彙引き／読み推論／境界推論の所要時間を測る。
    ///
    /// 計測前に全位置を一度走らせてウォームアップする（キャッシュを定常状態に近づける）。
    pub fn char_latency(&self, text: &str) -> Vec<CharLatency> {
        self.char_latency_with_featurize(text).1
    }

    /// [`Self::char_latency`] に加えて、特徴量生成（`to_source_seq` + 人名辞書照合 +
    /// `compute_source_features`）の合計所要時間 (ns) も返す。
    pub fn char_latency_with_featurize(&self, text: &str) -> (u128, Vec<CharLatency>) {
        let model = self.model();
        let tf = Instant::now();
        let source_seq = crate::featurize::to_source_seq(text);
        let (name_flags, _) = compute_name_matches(&source_seq, model.name_dict());
        let all_keys = crate::featurize::compute_source_features(&source_seq, &name_flags);
        let featurize_ns = tf.elapsed().as_nanos();

        let mut scratch = model.new_scratch();
        let mut scores = vec![0f32; model.n_classes() as usize];
        let mut feat_ids: Vec<u32> = Vec::new();

        let lookup = |model: &MomoModel, keys: &[FeatureKey], out: &mut Vec<u32>| {
            out.clear();
            out.extend(keys.iter().filter_map(|k| model.vocab_find(k)));
        };

        // ウォームアップ
        for keys in &all_keys {
            lookup(model, keys, &mut feat_ids);
            model.compute_read_scores(&feat_ids, &mut scratch, &mut scores);
            std::hint::black_box(&scores);
            std::hint::black_box(model.compute_boundary_score(keys));
        }

        let rows = source_seq
            .iter()
            .zip(&all_keys)
            .map(|(entry, keys)| {
                let t0 = Instant::now();
                lookup(model, keys, &mut feat_ids);
                let lookup_ns = t0.elapsed().as_nanos();

                let t1 = Instant::now();
                model.compute_read_scores(&feat_ids, &mut scratch, &mut scores);
                std::hint::black_box(&scores);
                let read_ns = t1.elapsed().as_nanos();

                let t2 = Instant::now();
                std::hint::black_box(model.compute_boundary_score(keys));
                let boundary_ns = t2.elapsed().as_nanos();

                let nnz = feat_ids
                    .iter()
                    .map(|&id| {
                        (model.csc_colptr[id as usize + 1] - model.csc_colptr[id as usize]) as usize
                    })
                    .sum();

                let cps = [entry.cp, entry.cp2, entry.cp3];
                let text: String = cps[..entry.compound_len.max(1) as usize]
                    .iter()
                    .filter_map(|&cp| char::from_u32(cp))
                    .collect();
                CharLatency {
                    text,
                    n_keys: keys.len(),
                    n_hits: feat_ids.len(),
                    nnz,
                    lookup_ns,
                    read_ns,
                    boundary_ns,
                }
            })
            .collect();
        (featurize_ns, rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PC 上で段階別プロファイルを出す（`MOMO_BENCH_MODEL` と `MOMO_BENCH_TEXT` を指定）。
    /// `cargo test --release -p momors-core --features diagnostics -- --ignored --nocapture profile_predict_phases`
    #[test]
    #[ignore]
    fn profile_predict_phases() {
        let model = std::env::var("MOMO_BENCH_MODEL").expect("MOMO_BENCH_MODEL");
        let text = std::fs::read_to_string(std::env::var("MOMO_BENCH_TEXT").expect("MOMO_BENCH_TEXT"))
            .expect("read text");
        let p = Predictor::load(crate::PredictorConfig::new(&model)).expect("load");
        let r = p.predict_phases(text.trim(), 50);
        println!("phases (us/char, {} chars): {}", r.chars, r.summary());
    }
}
