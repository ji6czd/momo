//! 予測器の公開 API と推論ロジック本体。
//!
//! C++ 版の `predictor.hpp` / `predictor.cpp` に対応する。
//!
//! ## Rust らしさの設計
//!
//! - `Predictor::new` + `load()` の二段階を `Predictor::load(config)` に統合
//! - 失敗は例外ではなく [`Result`] で返す
//! - 入力は `&str`(UTF-8 が型で保証されている)
//! - [`PredictionResult`] のフィールドは直接公開せず、メソッド経由でアクセス
//! - 原文位置は **UTF-8 バイト位置** で統一 (C++ 版はコードポイント位置)

use std::path::{Path, PathBuf};

use crate::char_type::CharType;
use crate::feature::FeatureKey;
use crate::featurize::{compute_source_features, to_source_seq, SourceEntry};
use crate::model::MomoModel;
use crate::numeric::{convert_japanese_numeric, NumericFallback};
use crate::Result;

// ============================================================
// 定数
// ============================================================

/// このラベルが付いた文字は、直前のラベルに含まれる扱い。
/// 出力時にはスキップする (ただし救済対象あり)。
const LABEL_CONTINUE: &str = "---";

/// このラベルが付いた文字は、スキップする (ただし NUMERIC は救済)。
const LABEL_SKIP: &str = "_";

/// 々 (同の字点) の Unicode コードポイント。
const CHAR_NOMA: u32 = 0x3005;

// ============================================================
// PredictorConfig
// ============================================================

/// 予測器の設定。
///
/// ビルダーパターンで `.with_xxx(...)` を連鎖させて作る。
/// 必須引数のモデルパスだけ [`new`] で受け取り、その他はデフォルト値を持つ。
///
/// ```no_run
/// use momors_core::PredictorConfig;
///
/// let config = PredictorConfig::new("basic_data.mbm")
///     .with_confidence_threshold(0.3)
///     .with_numeric_confidence_threshold(0.5);
/// ```
///
/// [`new`]: PredictorConfig::new
#[derive(Debug, Clone)]
pub struct PredictorConfig {
    pub(crate) model_path: PathBuf,
    pub(crate) confidence_threshold: f32,
    pub(crate) numeric_confidence_threshold: f32,
}

impl PredictorConfig {
    /// モデルファイルのパスを指定して新規作成する。
    pub fn new(model_path: impl AsRef<Path>) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            confidence_threshold: 0.5,
            numeric_confidence_threshold: 0.5,
        }
    }

    /// KANJI フォールバック (々など) を発動させる自信度の上限を設定する。
    pub fn with_confidence_threshold(mut self, value: f32) -> Self {
        self.confidence_threshold = value;
        self
    }

    /// JAPANESE_NUMERIC ルールベース変換を発動させる自信度の上限を設定する。
    pub fn with_numeric_confidence_threshold(mut self, value: f32) -> Self {
        self.numeric_confidence_threshold = value;
        self
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }

    pub fn numeric_confidence_threshold(&self) -> f32 {
        self.numeric_confidence_threshold
    }
}

// ============================================================
// PredictionResult
// ============================================================

/// 1 回の予測の結果。
///
/// フィールドは公開せず、アクセサ経由で参照する。
/// これにより内部表現を変更しても利用側のコードを壊さない。
#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub(crate) source_text: String,
    pub(crate) kana_text: String,
    pub(crate) confidences: Vec<f32>,
    /// かなのバイト位置 → 原文の **UTF-8 バイト位置**
    pub(crate) kana_to_src_index: Vec<usize>,
    /// 原文の **UTF-8 バイト位置** → かなのバイト位置のリスト
    pub(crate) src_to_kana_index: Vec<Vec<usize>>,
}

impl PredictionResult {
    /// 入力された原文 (UTF-8)。
    pub fn source_text(&self) -> &str {
        &self.source_text
    }

    /// 変換後のかな (UTF-8)。
    pub fn kana_text(&self) -> &str {
        &self.kana_text
    }

    /// 各かなバイトの自信度のスライス。
    /// `len()` は `kana_text().len()` と同じ。
    pub fn confidences(&self) -> &[f32] {
        &self.confidences
    }

    /// 「かなのバイト位置 → 原文の UTF-8 バイト位置」のマッピング。
    pub fn kana_to_source(&self) -> &[usize] {
        &self.kana_to_src_index
    }

    /// 「原文の UTF-8 バイト位置 → かなのバイト位置の列」のマッピング。
    pub fn source_to_kana(&self) -> &[Vec<usize>] {
        &self.src_to_kana_index
    }
}

// ============================================================
// Predictor
// ============================================================

/// 予測器本体。
///
/// [`PredictorConfig`] を渡して [`load`] するとモデルを読み込んだ
/// 状態のインスタンスが得られる。読み込み済みなので即 [`predict`] できる。
///
/// [`load`]: Predictor::load
/// [`predict`]: Predictor::predict
#[derive(Debug)]
pub struct Predictor {
    config: PredictorConfig,
    model: MomoModel,
}

impl Predictor {
    /// 設定からモデルを読み込んで予測器を構築する。
    pub fn load(config: PredictorConfig) -> Result<Self> {
        let model = crate::loader::load(config.model_path())?;
        Ok(Self { config, model })
    }

    /// 設定を参照する。
    pub fn config(&self) -> &PredictorConfig {
        &self.config
    }

    /// モデルの特徴量次元数。
    pub fn n_features(&self) -> u32 {
        self.model.n_features()
    }

    /// モデルの読みラベル数。
    pub fn n_classes(&self) -> u32 {
        self.model.n_classes()
    }

    /// 文字列を予測してかなに変換する。
    ///
    /// C++ 版 `Predictor::predict()` とフィーチャーパリティ:
    ///
    /// - bypass (記号系・ALPHA)
    /// - JAPANESE_NUMERIC ルールベース変換 (`digit_table` / `kurai_fallback`)
    /// - 々フォールバック (直前漢字の読みを繰り返す)
    /// - LABEL_CONTINUE 救済 (孤立、NUMERIC、小書き仮名)
    /// - LABEL_SKIP 救済 (NUMERIC)
    /// - 境界モデルによるスペース挿入
    pub fn predict(&self, text: &str) -> Result<PredictionResult> {
        // --- 初期化 ---
        let mut result = PredictionResult {
            source_text: text.to_string(),
            kana_text: String::new(),
            confidences: Vec::new(),
            kana_to_src_index: Vec::new(),
            src_to_kana_index: vec![Vec::new(); text.len()],
        };

        if text.is_empty() {
            return Ok(result);
        }

        // --- 前処理 ---
        let cu_opt = if self.model.compound_units.is_empty() {
            None
        } else {
            Some(&self.model.compound_units)
        };
        let source_seq = to_source_seq(text, cu_opt);
        let n = source_seq.len();
        if n == 0 {
            return Ok(result);
        }

        let all_feat_keys = compute_source_features(&source_seq);
        let all_feat_ids: Vec<Vec<u32>> = all_feat_keys
            .iter()
            .map(|keys| lookup_feature_ids(keys, &self.model))
            .collect();

        let n_cls = self.model.n_classes() as usize;
        let mut int_scores = vec![0i32; n_cls];
        let mut scores = vec![0f32; n_cls];

        // --- 複合ユニット後続文字の src_to_kana_index 補完用 ---
        // (src_byte_idx, kana_byte_begin, kana_byte_end) のリスト
        let mut compound_extras: Vec<(usize, usize, usize)> = Vec::new();

        // --- 状態変数 ---
        // 々フォールバック用の「直前漢字の読み」記憶。
        // KANJI 出力時に更新、それ以外で clear する。
        let mut last_fallback = String::new();

        // 親追跡: 小書き仮名 + CONTINUE 救済で親ラベルに後付け挿入するために必要。
        // 「親」とは「現在の文字が CONTINUE のときに、その読みを統合される側」のこと。
        let mut parent_src_idx: Option<usize> = None;
        let mut parent_is_bypass = false;
        let mut parent_has_small_kana = false;
        // 親ラベル出力直後の kana_text / kana_to_src_index 末尾位置。
        // 小書き仮名はこの位置に挿入する。
        let mut parent_kana_byte_end: usize = 0;

        // --- 各文字を処理 ---
        for i in 0..n {
            let entry = &source_seq[i];

            // ============================================================
            // bypass (素通し)
            // ============================================================
            if entry.ctype.is_bypass() {
                self.emit_char_passthrough(entry, 1.0, &mut result);
                parent_src_idx = Some(i);
                parent_is_bypass = true;
                parent_has_small_kana = false;
                parent_kana_byte_end = result.kana_text.len();
                last_fallback.clear();
                continue;
            }

            // ============================================================
            // スコア計算 + argmax
            // ============================================================
            int_scores.fill(0);
            self.compute_read_scores(&all_feat_ids[i], &mut int_scores);

            let scale = self.model.read_scale;
            for cls in 0..n_cls {
                scores[cls] = self.model.intercept_read[cls] + (int_scores[cls] as f32) * scale;
            }

            let (best_cls, &best_score) = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .expect("n_classes >= 1");

            let conf = sigmoid(best_score);
            let label = self.model.read_class(best_cls as u32).unwrap_or("");

            // ============================================================
            // JAPANESE_NUMERIC ルールベース変換
            // ============================================================
            if entry.ctype == CharType::JapaneseNumeric
                && conf < self.config.numeric_confidence_threshold
            {
                match convert_japanese_numeric(i, &source_seq) {
                    NumericFallback::Skip => {
                        // 出力せずにスキップ (「三百二十」の「十」など)
                        continue;
                    }
                    NumericFallback::Output(fallback) => {
                        let has_split =
                            sigmoid(self.compute_boundary_score(&all_feat_ids[i])) >= 0.5;
                        self.emit_label(entry, fallback, conf, &mut result);
                        parent_src_idx = Some(i);
                        parent_is_bypass = false;
                        parent_has_small_kana = false;
                        parent_kana_byte_end = result.kana_text.len();
                        last_fallback.clear();
                        if has_split {
                            result.kana_text.push(' ');
                            result.kana_to_src_index.push(entry.orig_idx as usize);
                            result.confidences.push(conf);
                        }
                        continue;
                    }
                }
            }

            // ============================================================
            // LABEL_CONTINUE 救済
            // ============================================================
            if label == LABEL_CONTINUE {
                // 1. 孤立 CONTINUE 救済: 親がない or 親が bypass の場合は
                //    現在の文字を「親」にして、文字をそのまま出力する。
                if parent_src_idx.is_none() || parent_is_bypass {
                    self.emit_char_passthrough(entry, conf, &mut result);
                    parent_src_idx = Some(i);
                    parent_is_bypass = false;
                    parent_has_small_kana = false;
                    parent_kana_byte_end = result.kana_text.len();
                    let has_split =
                        sigmoid(self.compute_boundary_score(&all_feat_ids[i])) >= 0.5;
                    if has_split {
                        result.kana_text.push(' ');
                        result.kana_to_src_index.push(entry.orig_idx as usize);
                        result.confidences.push(conf);
                    }
                    continue;
                }

                // 2. NUMERIC + CONTINUE 救済: 数字 (1, 2, ...) を素通し
                if entry.ctype == CharType::Numeric {
                    self.emit_char_passthrough(entry, conf, &mut result);
                    continue;
                }

                // 3. 小書き仮名 + CONTINUE 救済: 親ラベル末尾に追記
                if is_small_kana(entry.cp) && !parent_has_small_kana {
                    self.insert_small_kana_into_parent(
                        entry.cp,
                        conf,
                        &source_seq,
                        parent_src_idx.unwrap(),
                        &mut parent_kana_byte_end,
                        &mut result,
                    );
                    parent_has_small_kana = true;
                }
                // それ以外の CONTINUE は単純スキップ (親情報は更新しない)
                continue;
            }

            // ============================================================
            // LABEL_SKIP 救済
            // ============================================================
            if label == LABEL_SKIP {
                // NUMERIC + SKIP 救済: 数字を素通し
                if entry.ctype == CharType::Numeric {
                    self.emit_char_passthrough(entry, conf, &mut result);
                    continue;
                }
                // それ以外の SKIP は単純スキップ (親情報は更新しない)
                continue;
            }

            // ============================================================
            // 通常ラベル出力 (々フォールバックも含む)
            // ============================================================

            // 境界判定
            let has_split = sigmoid(self.compute_boundary_score(&all_feat_ids[i])) >= 0.5;

            // 々フォールバック: 低自信度の「々」は直前漢字の読みを繰り返す
            //
            // 借用の都合上、`effective_label` は `&str` ではなく所有値の `String` で持つ。
            // - 々発動時: `last_fallback` から複製
            // - 通常時:   `label` から複製
            // これにより、後段の `last_fallback` 更新時に借用が競合しない。
            // 文字列は通常短い (数バイト) のでコピーのオーバーヘッドは小さい。
            let effective_label: String = if entry.cp == CHAR_NOMA
                && !last_fallback.is_empty()
                && conf < self.config.confidence_threshold
            {
                last_fallback.clone()
            } else {
                label.to_string()
            };

            let kana_before = result.kana_text.len();
            self.emit_label(entry, &effective_label, conf, &mut result);
            let kana_after = result.kana_text.len();

            // 複合ユニット後続文字（orig_idx+1, +2）の src_to_kana_index 補完用に記録。
            // C++ 版 compound_extra と対応。orig_idx は UTF-8 バイト位置なので
            // 先頭コードポイントのバイト長を加算して後続文字のバイト位置を求める。
            if entry.compound_len >= 2 {
                let cp1_bytes = char::from_u32(entry.cp).map_or(1, |c| c.len_utf8());
                let orig2 = entry.orig_idx as usize + cp1_bytes;
                compound_extras.push((orig2, kana_before, kana_after));
            }
            if entry.compound_len >= 3 {
                let cp1_bytes = char::from_u32(entry.cp).map_or(1, |c| c.len_utf8());
                let cp2_bytes = char::from_u32(entry.cp2).map_or(1, |c| c.len_utf8());
                let orig3 = entry.orig_idx as usize + cp1_bytes + cp2_bytes;
                compound_extras.push((orig3, kana_before, kana_after));
            }

            // 親情報更新
            parent_src_idx = Some(i);
            parent_is_bypass = false;
            // 親ラベルに小書き仮名が含まれているか確認（C++ 版と同じ判定）
            parent_has_small_kana = effective_label.chars().any(|c| is_small_kana(c as u32));
            parent_kana_byte_end = result.kana_text.len();

            // last_fallback の更新: KANJI のラベルだけ記憶する
            // (LR が高自信度でも、々のために伝播させる)
            if entry.ctype == CharType::Kanji {
                // 所有権を移譲することで余計な clone を避ける
                last_fallback = effective_label;
            } else {
                last_fallback.clear();
            }

            if has_split {
                result.kana_text.push(' ');
                result.kana_to_src_index.push(entry.orig_idx as usize);
                result.confidences.push(conf);
            }
        }

        // --- src_to_kana_index 構築 (kana_to_src_index から逆引き) ---
        let src_size = text.len();
        for (j, &src_pos) in result.kana_to_src_index.iter().enumerate() {
            if src_pos < src_size {
                result.src_to_kana_index[src_pos].push(j);
            }
        }
        // 複合ユニットの後続文字（orig_idx+1, +2）も同じ kana 範囲に登録。
        // C++ 版 compound_extra 処理と対応。
        for (src_byte_idx, kana_begin, kana_end) in &compound_extras {
            if *src_byte_idx < src_size {
                for j in *kana_begin..*kana_end {
                    result.src_to_kana_index[*src_byte_idx].push(j);
                }
            }
        }

        Ok(result)
    }

    // ============================================================
    // private helpers
    // ============================================================

    /// 整数スコアに対して CSC 行列の特徴量列を加算する。
    fn compute_read_scores(&self, feat_ids: &[u32], int_scores: &mut [i32]) {
        for &feat_id in feat_ids {
            if feat_id >= self.model.n_features() {
                continue;
            }
            let col_start = self.model.csc_colptr[feat_id as usize] as usize;
            let col_end = self.model.csc_colptr[feat_id as usize + 1] as usize;
            for j in col_start..col_end {
                let cls = self.model.csc_rowind[j] as usize;
                int_scores[cls] += self.model.csc_data[j] as i32;
            }
        }
    }

    /// 境界モデルの生スコア (sigmoid 前) を計算する。
    fn compute_boundary_score(&self, feat_ids: &[u32]) -> f32 {
        let mut score = self.model.boundary_intercept[1];
        let scale = self.model.boundary_scale;
        for &feat_id in feat_ids {
            if feat_id < self.model.n_features() {
                score += (self.model.boundary_data[feat_id as usize] as f32) * scale;
            }
        }
        score
    }

    /// 原文の `entry.cp` をそのまま結果に書き出す (bypass / 救済共通)。
    fn emit_char_passthrough(
        &self,
        entry: &SourceEntry,
        conf: f32,
        result: &mut PredictionResult,
    ) {
        let ch = char::from_u32(entry.cp).unwrap_or('\u{FFFD}');
        let mut buf = [0u8; 4];
        let ch_utf8 = ch.encode_utf8(&mut buf);
        result.kana_text.push_str(ch_utf8);
        for _ in 0..ch_utf8.len() {
            result.kana_to_src_index.push(entry.orig_idx as usize);
            result.confidences.push(conf);
        }
    }

    /// ラベル文字列を結果に書き出す (通常ラベル / フォールバック共通)。
    fn emit_label(
        &self,
        entry: &SourceEntry,
        label: &str,
        conf: f32,
        result: &mut PredictionResult,
    ) {
        result.kana_text.push_str(label);
        for _ in 0..label.len() {
            result.kana_to_src_index.push(entry.orig_idx as usize);
            result.confidences.push(conf);
        }
    }

    /// 親ラベル末尾に小書き仮名を挿入する (LABEL_CONTINUE 救済の 3 番目)。
    ///
    /// `parent_kana_byte_end` を呼び出し側で更新する必要がある (`&mut` 経由)。
    fn insert_small_kana_into_parent(
        &self,
        small_cp: u32,
        conf: f32,
        source_seq: &[SourceEntry],
        parent_idx: usize,
        parent_kana_byte_end: &mut usize,
        result: &mut PredictionResult,
    ) {
        // ひらがな小書きはカタカナ小書きに変換 (+0x60)、カタカナはそのまま
        let kana_cp = small_kana_to_kana(small_cp);
        let kana_ch = char::from_u32(kana_cp).unwrap_or('\u{FFFD}');
        let mut buf = [0u8; 4];
        let kana_utf8 = kana_ch.encode_utf8(&mut buf).to_string();

        // 親の orig_idx を取得 (この文字の orig_idx ではなく親のものを使う)
        let parent_orig_idx = source_seq[parent_idx].orig_idx as usize;

        // 挿入: kana_text の parent_kana_byte_end バイト目に kana_utf8 を入れ、
        //       kana_to_src_index / confidences も対応する位置に挿入する。
        // (kana_to_src_index の要素数 = kana_text のバイト数なので、
        //  parent_kana_byte_end は両方のインデックスとして使える)
        result.kana_text.insert_str(*parent_kana_byte_end, &kana_utf8);
        for offset in 0..kana_utf8.len() {
            result
                .kana_to_src_index
                .insert(*parent_kana_byte_end + offset, parent_orig_idx);
            result
                .confidences
                .insert(*parent_kana_byte_end + offset, conf);
        }
        *parent_kana_byte_end += kana_utf8.len();
    }
}

// ============================================================
// 自由関数
// ============================================================

/// シグモイド関数 σ(x) = 1 / (1 + exp(-x))
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 特徴量キー列をモデルの語彙テーブルで引いて feature_id 列に変換する。
fn lookup_feature_ids(keys: &[FeatureKey], model: &MomoModel) -> Vec<u32> {
    let mut ids = Vec::with_capacity(keys.len());
    for k in keys {
        if let Some(id) = model.vocab_find(k) {
            ids.push(id);
        }
    }
    ids
}

/// 小書き仮名 (拗音・促音など) の判定。
///
/// C++ 版 `SMALL_KANA_LIST` および Python 版 `_SMALL_KANA` と一致:
/// ひらがな: ぁぃぅぇぉ っ ゃゅょ ゎ
/// カタカナ: ァィゥェォ ッ ャュョ ヮ
#[inline]
fn is_small_kana(cp: u32) -> bool {
    matches!(
        cp,
        0x3041 | 0x3043 | 0x3045 | 0x3047 | 0x3049  // ぁぃぅぇぉ
            | 0x3063                                 // っ
            | 0x3083 | 0x3085 | 0x3087               // ゃゅょ
            | 0x308E                                 // ゎ
            | 0x30A1 | 0x30A3 | 0x30A5 | 0x30A7 | 0x30A9  // ァィゥェォ
            | 0x30C3                                  // ッ
            | 0x30E3 | 0x30E5 | 0x30E7                // ャュョ
            | 0x30EE                                  // ヮ
    )
}

/// ひらがな小書き → カタカナ小書き変換 (+0x60)。
/// ひらがな範囲外はそのまま返す。
///
/// C++ 版 `small_kana_to_kana()` と一致。
/// `parent_idx` の親ラベルがカタカナで書かれている前提で、追記する小書きも
/// カタカナ化する。
#[inline]
fn small_kana_to_kana(cp: u32) -> u32 {
    // ぁ (U+3041) 〜 ん (U+3093) の範囲ならカタカナ化
    if (0x3041..=0x3093).contains(&cp) {
        cp + 0x60
    } else {
        cp
    }
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_builder_chains() {
        let config = PredictorConfig::new("dummy.mbm")
            .with_confidence_threshold(0.3)
            .with_numeric_confidence_threshold(0.4);

        assert_eq!(config.model_path(), Path::new("dummy.mbm"));
        assert!((config.confidence_threshold() - 0.3).abs() < 1e-6);
        assert!((config.numeric_confidence_threshold() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_basics() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn is_small_kana_works() {
        // 小書きひらがな
        for c in "ぁぃぅぇぉっゃゅょゎ".chars() {
            assert!(is_small_kana(c as u32), "{c} should be small kana");
        }
        // 小書きカタカナ
        for c in "ァィゥェォッャュョヮ".chars() {
            assert!(is_small_kana(c as u32), "{c} should be small kana");
        }
        // 通常仮名は false
        for c in "あいうえおかきくけこアイウエオ".chars() {
            assert!(!is_small_kana(c as u32), "{c} should NOT be small kana");
        }
    }

    #[test]
    fn small_kana_to_kana_converts() {
        // ぁ (U+3041) → ァ (U+30A1)
        assert_eq!(small_kana_to_kana(0x3041), 0x30A1);
        // っ (U+3063) → ッ (U+30C3)
        assert_eq!(small_kana_to_kana(0x3063), 0x30C3);
        // ゎ (U+308E) → ヮ (U+30EE)
        assert_eq!(small_kana_to_kana(0x308E), 0x30EE);
        // 既にカタカナならそのまま
        assert_eq!(small_kana_to_kana(0x30A1), 0x30A1);
        // 範囲外もそのまま
        assert_eq!(small_kana_to_kana('A' as u32), 'A' as u32);
    }

    // --- predict() の動作確認テスト (ダミーモデルで) ---

    fn dummy_model_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/dummy.mbm")
    }

    #[test]
    fn predict_empty_string() {
        let config = PredictorConfig::new(dummy_model_path());
        let predictor = Predictor::load(config).expect("dummy.mbm が読めること");

        let result = predictor.predict("").unwrap();
        assert_eq!(result.source_text(), "");
        assert_eq!(result.kana_text(), "");
        assert!(result.confidences().is_empty());
    }

    #[test]
    fn predict_bypass_symbol() {
        let config = PredictorConfig::new(dummy_model_path());
        let predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("、").unwrap();
        assert_eq!(result.kana_text(), "、");
        assert!(result.confidences().iter().all(|&c| (c - 1.0).abs() < 1e-6));
        assert_eq!(result.source_to_kana().len(), "、".len());
    }

    #[test]
    fn predict_bypass_alpha() {
        let config = PredictorConfig::new(dummy_model_path());
        let predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("abc").unwrap();
        assert_eq!(result.kana_text(), "abc");
    }

    #[test]
    fn predict_mixed_bypass() {
        let config = PredictorConfig::new(dummy_model_path());
        let predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("a、b").unwrap();
        assert_eq!(result.kana_text(), "a、b");
    }

    #[test]
    fn predict_source_to_kana_indices() {
        let config = PredictorConfig::new(dummy_model_path());
        let predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("a").unwrap();
        assert_eq!(result.kana_to_source(), &[0usize]);
        assert_eq!(result.source_to_kana()[0], vec![0usize]);
    }

    #[test]
    fn predict_does_not_crash_on_complex_input() {
        // dummy.mbm では意味のあるカナは出ないが、ロジックが落ちないことを確認
        let config = PredictorConfig::new(dummy_model_path());
        let predictor = Predictor::load(config).unwrap();

        // 小書き仮名、々、漢数字、句読点 - 全ロジックパスを通す
        let inputs = ["きょう", "学校々", "三百二十一円。", "abc、漢字!"];
        for text in &inputs {
            let _ = predictor.predict(text).expect("crash しないこと");
        }
    }
}
