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

use crate::Error;
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
///     .with_numeric_confidence_threshold(0.5);
/// ```
///
/// [`new`]: PredictorConfig::new
#[derive(Debug, Clone)]
pub struct PredictorConfig {
    pub(crate) model_path: PathBuf,
    pub(crate) numeric_confidence_threshold: f32,
    pub(crate) kanji_dict_path: Option<PathBuf>,
}

impl PredictorConfig {
    /// モデルファイルのパスを指定して新規作成する。
    pub fn new(model_path: impl AsRef<Path>) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            numeric_confidence_threshold: 0.5,
            kanji_dict_path: None,
        }
    }

    /// JAPANESE_NUMERIC ルールベース変換を発動させる自信度の上限を設定する。
    pub fn with_numeric_confidence_threshold(mut self, value: f32) -> Self {
        self.numeric_confidence_threshold = value;
        self
    }

    /// 予測結果を原文の文字ごとに分割して出力するかどうかを設定する。
    pub fn with_segment_output(self, value: bool) -> Self {
        if value {
            // segment 出力を有効にするための追加設定があればここで行う。
            // 現状は特に追加の設定はないが、将来的に必要になった場合はこのメソッド内で対応する。
        }
        self
    }

    /// 漢字辞書ファイル (.tsv) のパスを設定する。
    /// 設定すると推論時に辞書の読みに候補が制約される。
    pub fn with_kanji_dict_path(mut self, path: impl AsRef<Path>) -> Self {
        self.kanji_dict_path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
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

    // ============================================================
    // Python 互換: セグメント分割 / フォーマット
    // ============================================================

    /// 境界スペースで原文を分割したスライスのリストを返す。
    ///
    /// 境界がまったくない場合は原文全体を含む 1 要素のスライスを返す。
    /// Python の `get_source_segments()` に相当する。
    pub fn get_source_segments(&self) -> Vec<&str> {
        if self.source_text.is_empty() {
            return Vec::new();
        }

        let mut segments: Vec<&str> = Vec::new();
        let mut current_start: usize = 0;

        for (k, ch) in self.kana_text.char_indices() {
            if ch == ' ' {
                let src_byte = self.kana_to_src_index[k];
                let char_end = src_byte
                    + self.source_text[src_byte..]
                        .chars()
                        .next()
                        .map_or(0, |c| c.len_utf8());
                let seg = &self.source_text[current_start..char_end];
                if !seg.is_empty() {
                    segments.push(seg);
                }
                current_start = char_end;
            }
        }

        let remaining = &self.source_text[current_start..];
        if !remaining.is_empty() {
            segments.push(remaining);
        }

        segments
    }

    /// 点字の分かち書きに沿って原文を `/` で区切って返す。
    ///
    /// Python の `format_source_segmented()` に相当する。
    pub fn format_source_segmented(&self) -> String {
        self.get_source_segments().join("/")
    }

    /// 各ソース文字に対応するかな部分を `/` で区切って返す。
    ///
    /// Python の `format_segmented()` に相当する。
    pub fn format_segmented(&self) -> String {
        // かな列: バイト位置 → コードポイントインデックス の逆引き
        let kana_char_starts: std::collections::HashMap<usize, usize> = self
            .kana_text
            .char_indices()
            .enumerate()
            .map(|(ci, (byte, _))| (byte, ci))
            .collect();
        let kana_chars: Vec<char> = self.kana_text.chars().collect();

        let mut segments: Vec<String> = Vec::new();

        for (src_byte, _) in self.source_text.char_indices() {
            let kana_bytes = &self.src_to_kana_index[src_byte];
            if kana_bytes.is_empty() {
                continue;
            }

            let mut current = String::new();
            for &kb in kana_bytes {
                // バイト位置がコードポイント先頭のときのみ処理（重複排除）
                if let Some(&ci) = kana_char_starts.get(&kb) {
                    let ch = kana_chars[ci];
                    if ch == ' ' {
                        if !current.is_empty() {
                            segments.push(std::mem::take(&mut current));
                        }
                        segments.push(" ".to_string());
                    } else {
                        current.push(ch);
                    }
                }
            }
            if !current.is_empty() {
                segments.push(current);
            }
        }

        segments.join("/")
    }

    // ============================================================
    // Python 互換: コードポイント単位のインデックス変換
    // ============================================================

    /// かなのコードポイント位置 → 原文のコードポイント位置 のマッピング。
    ///
    /// Python の `kana_to_src_index` と互換性がある。
    /// 内部では UTF-8 バイト位置を使っているためこのメソッドで変換する。
    pub fn kana_to_source_char(&self) -> Vec<usize> {
        let src_char_starts: Vec<usize> = self.source_text.char_indices().map(|(b, _)| b).collect();

        let mut result = Vec::with_capacity(self.kana_text.chars().count());
        let mut kana_byte = 0;
        for ch in self.kana_text.chars() {
            let src_byte = self.kana_to_src_index[kana_byte];
            let char_idx = src_char_starts.binary_search(&src_byte).unwrap_or(0);
            result.push(char_idx);
            kana_byte += ch.len_utf8();
        }
        result
    }

    /// 原文のコードポイント位置 → かなのコードポイント位置のリスト。
    ///
    /// Python の `src_to_kana_index` と互換性がある。
    pub fn source_to_kana_char(&self) -> Vec<Vec<usize>> {
        let kana_byte_to_char: std::collections::HashMap<usize, usize> = self
            .kana_text
            .char_indices()
            .enumerate()
            .map(|(ci, (byte, _))| (byte, ci))
            .collect();

        self.source_text
            .char_indices()
            .map(|(src_byte, _)| {
                self.src_to_kana_index[src_byte]
                    .iter()
                    .filter_map(|&kb| kana_byte_to_char.get(&kb).copied())
                    .collect()
            })
            .collect()
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
    /// ソート済み漢字辞書。binary_search でルックアップする。
    kanji_dict: Vec<(char, Vec<String>)>,
}

impl Predictor {
    /// 設定からモデルを読み込んで予測器を構築する。
    pub fn load(config: PredictorConfig) -> Result<Self> {
        let model = crate::loader::load(config.model_path())?;
        let kanji_dict = if let Some(ref path) = config.kanji_dict_path {
            load_kanji_dict(path)?
        } else {
            Vec::new()
        };
        Ok(Self {
            config,
            model,
            kanji_dict,
        })
    }

    /// バイト列からモデルを読み込んで予測器を構築する (WASM / インメモリ用)。
    ///
    /// デフォルト設定 (numeric_confidence_threshold=0.5、漢字辞書なし) を使用する。
    pub fn from_model_bytes(bytes: &[u8]) -> Result<Self> {
        let model = crate::loader::load_from_bytes(bytes)?;
        let config = PredictorConfig {
            model_path: PathBuf::from("<memory>"),
            numeric_confidence_threshold: 0.5,
            kanji_dict_path: None,
        };
        Ok(Self {
            config,
            model,
            kanji_dict: Vec::new(),
        })
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
        let source_seq = to_source_seq(text);
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

        // --- 状態変数 ---
        // 々フォールバック用の「直前漢字の読み」記憶。
        // KANJI 出力時に更新、それ以外で clear する。
        let mut last_fallback = String::new();

        // 親追跡: 小書き仮名 + CONTINUE 救済で親ラベルに後付け挿入するために必要。
        // 「親」とは「現在の文字が CONTINUE のときに、その読みを統合される側」のこと。
        let mut parent_src_idx: Option<usize> = None;
        let mut parent_is_bypass = false;
        let mut parent_is_kanji = false;
        let mut parent_has_small_kana = false;
        // 親ラベル出力直後の kana_text / kana_to_src_index 末尾位置。
        // 小書き仮名はこの位置に挿入する。
        let mut parent_kana_byte_end: usize = 0;

        // --- 各文字を処理 ---
        for i in 0..n {
            let entry = &source_seq[i];

            // ============================================================
            // skip（仮名に現れない、インデックスのみ保持）
            // ============================================================
            if entry.ctype.is_skip() {
                continue;
            }

            // ============================================================
            // bypass (素通し)
            // ============================================================
            if entry.ctype.is_bypass() {
                self.emit_char_passthrough(entry, 1.0, &mut result);
                parent_src_idx = Some(i);
                parent_is_bypass = true;
                parent_is_kanji = false;
                parent_has_small_kana = false;
                parent_kana_byte_end = result.kana_text.len();
                last_fallback.clear();
                continue;
            }

            // ============================================================
            // カタカナ: 原文のまま出力（モデル不要）
            // ヵ (U+30F5) / ヶ (U+30F6) は助数詞用の小書きカタカナで拗音ではないため
            // パススルーせずモデルに委ねる。
            // ============================================================
            if entry.ctype == CharType::Katakana && !is_counter_small_kana(entry.cp) {
                let direct = katakana_passthrough(entry);
                self.emit_label(entry, &direct, 1.0, &mut result);
                parent_src_idx = Some(i);
                parent_is_bypass = false;
                parent_is_kanji = false;
                parent_has_small_kana = entry.compound_len >= 2 && is_small_kana(entry.cp2);
                parent_kana_byte_end = result.kana_text.len();
                last_fallback.clear();
                let has_split = sigmoid(self.compute_boundary_score(&all_feat_ids[i])) >= 0.5;
                if has_split {
                    result.kana_text.push(' ');
                    result.kana_to_src_index.push(entry.orig_idx as usize);
                    result.confidences.push(1.0);
                }
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

            // 漢字辞書制約付き argmax
            let (best_cls, best_score): (usize, f32) = if entry.ctype == CharType::Kanji
                && !self.kanji_dict.is_empty()
            {
                if let Some(ch) = char::from_u32(entry.cp) {
                    if let Some(readings) = lookup_kanji_dict(&self.kanji_dict, ch) {
                        let mut best = 0usize;
                        let mut best_s = f32::NEG_INFINITY;
                        for cls in 0..n_cls {
                            let lbl = self.model.read_class(cls as u32).unwrap_or("");
                            if readings.iter().any(|r| r.as_str() == lbl) && scores[cls] > best_s {
                                best_s = scores[cls];
                                best = cls;
                            }
                        }
                        (best, best_s)
                    } else {
                        unconstrained_argmax(&scores)
                    }
                } else {
                    unconstrained_argmax(&scores)
                }
            } else {
                unconstrained_argmax(&scores)
            };

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
                        parent_is_kanji = false;
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
                // 1. 孤立 CONTINUE 救済: 親がない・bypass・KANJI以外が親の場合は
                //    現在の文字を「親」にして、文字をそのまま出力する。
                if parent_src_idx.is_none() || parent_is_bypass || !parent_is_kanji {
                    self.emit_char_passthrough(entry, conf, &mut result);
                    parent_src_idx = Some(i);
                    parent_is_bypass = false;
                    parent_is_kanji = entry.ctype == CharType::Kanji;
                    parent_has_small_kana = false;
                    parent_kana_byte_end = result.kana_text.len();
                    let has_split = sigmoid(self.compute_boundary_score(&all_feat_ids[i])) >= 0.5;
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
                && !is_valid_repeat(label, &last_fallback)
            {
                last_fallback.clone()
            } else if entry.ctype == CharType::Hiragana {
                let direct = hiragana_direct(entry);
                if is_valid_kana_prediction(entry, label, &direct) {
                    label.to_string()
                } else {
                    direct
                }
            } else {
                label.to_string()
            };

            self.emit_label(entry, &effective_label, conf, &mut result);

            // 親情報更新
            parent_src_idx = Some(i);
            parent_is_bypass = false;
            parent_is_kanji = entry.ctype == CharType::Kanji;
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
    fn emit_char_passthrough(&self, entry: &SourceEntry, conf: f32, result: &mut PredictionResult) {
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
        result
            .kana_text
            .insert_str(*parent_kana_byte_end, &kana_utf8);
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

/// スコア配列の argmax を返す。
fn unconstrained_argmax(scores: &[f32]) -> (usize, f32) {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &s)| (i, s))
        .expect("n_classes >= 1")
}

/// 漢字辞書 TSV を読み込み、char でソートされた Vec を返す。
///
/// フォーマット: 漢字[TAB]読み1[TAB]読み2[TAB]...
/// ロード後に sort_unstable_by_key でソートするので、TSV の行順は問わない。
fn load_kanji_dict(path: &Path) -> Result<Vec<(char, Vec<String>)>> {
    let content = std::fs::read_to_string(path).map_err(|e| Error::DictIo {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut dict: Vec<(char, Vec<String>)> = content
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| {
            let mut parts = l.splitn(2, '\t');
            let kanji_str = parts.next()?;
            let kanji = kanji_str.chars().next()?;
            let readings: Vec<String> = parts
                .next()
                .unwrap_or("")
                .split('\t')
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect();
            if readings.is_empty() {
                return None;
            }
            Some((kanji, readings))
        })
        .collect();
    dict.sort_unstable_by_key(|(k, _)| *k);
    Ok(dict)
}

/// ソート済み漢字辞書から kanji をバイナリサーチで引く。
fn lookup_kanji_dict<'a>(dict: &'a [(char, Vec<String>)], kanji: char) -> Option<&'a [String]> {
    dict.binary_search_by_key(&kanji, |(k, _)| *k)
        .ok()
        .map(|i| dict[i].1.as_slice())
}

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

/// ひらがなコードポイントをカタカナに変換する (+0x60)。
/// ひらがな範囲 (U+3041..=U+3096) 以外はそのまま返す。
#[inline]
fn hiragana_to_katakana(cp: u32) -> u32 {
    if (0x3041..=0x3096).contains(&cp) {
        cp + 0x60
    } else {
        cp
    }
}

/// ひらがなユニットをカタカナに直接変換して返す。
fn hiragana_direct(entry: &SourceEntry) -> String {
    let mut s = String::new();
    if let Some(c) = char::from_u32(hiragana_to_katakana(entry.cp)) {
        s.push(c);
    }
    if entry.compound_len >= 2 {
        if let Some(c) = char::from_u32(hiragana_to_katakana(entry.cp2)) {
            s.push(c);
        }
    }
    s
}

/// カタカナユニットを原文のまま文字列化する。
fn katakana_passthrough(entry: &SourceEntry) -> String {
    let mut s = String::new();
    if let Some(c) = char::from_u32(entry.cp) {
        s.push(c);
    }
    if entry.compound_len >= 2 {
        if let Some(c) = char::from_u32(entry.cp2) {
            s.push(c);
        }
    }
    s
}

/// モデル予測がひらがなユニットに対して合法的かどうかを判定する。
///
/// 合法的な逸脱:
/// - は (U+306F) → ワ  (助詞)
/// - へ (U+3078) → エ  (助詞)
/// - う (U+3046) → ー  (長音)
/// 々の繰り返し判定: 先頭音節の連濁マップ（カタカナ）。
fn rendaku_first(c: char) -> Option<char> {
    match c {
        'カ' => Some('ガ'),
        'キ' => Some('ギ'),
        'ク' => Some('グ'),
        'ケ' => Some('ゲ'),
        'コ' => Some('ゴ'),
        'サ' => Some('ザ'),
        'シ' => Some('ジ'),
        'ス' => Some('ズ'),
        'セ' => Some('ゼ'),
        'ソ' => Some('ゾ'),
        'タ' => Some('ダ'),
        'チ' => Some('ヂ'),
        'ツ' => Some('ヅ'),
        'テ' => Some('デ'),
        'ト' => Some('ド'),
        'ハ' => Some('バ'),
        'ヒ' => Some('ビ'),
        'フ' => Some('ブ'),
        'ヘ' => Some('ベ'),
        'ホ' => Some('ボ'),
        _ => None,
    }
}

/// ライマンの法則チェック用: 有声阻害音（濁音）の判定。
fn is_voiced_obstruent(c: char) -> bool {
    matches!(
        c,
        'ガ' | 'ギ'
            | 'グ'
            | 'ゲ'
            | 'ゴ'
            | 'ザ'
            | 'ジ'
            | 'ズ'
            | 'ゼ'
            | 'ゾ'
            | 'ダ'
            | 'ヂ'
            | 'ヅ'
            | 'デ'
            | 'ド'
            | 'バ'
            | 'ビ'
            | 'ブ'
            | 'ベ'
            | 'ボ'
    )
}

/// 々の予測として妥当かどうかを検証する。
///
/// `label` が `last_fallback` そのまま、または連濁形（ライマンの法則に反しない場合）なら true。
fn is_valid_repeat(label: &str, last_fallback: &str) -> bool {
    if label == last_fallback {
        return true;
    }
    let mut chars = last_fallback.chars();
    if let Some(first) = chars.next() {
        if let Some(voiced) = rendaku_first(first) {
            if !last_fallback.chars().any(is_voiced_obstruent) {
                let rendaku: String = std::iter::once(voiced).chain(chars).collect();
                if label == rendaku {
                    return true;
                }
            }
        }
    }
    false
}

fn is_valid_kana_prediction(entry: &SourceEntry, label: &str, direct: &str) -> bool {
    if label == direct {
        return true;
    }
    if entry.compound_len == 1 {
        match entry.cp {
            0x306F if label == "ワ" => return true,
            0x3078 if label == "エ" => return true,
            0x3046 if label == "ー" => return true,
            _ => {}
        }
    }
    false
}

/// 助数詞用小書きカタカナの判定。
///
/// ヵ (U+30F5) と ヶ (U+30F6) は拗音形成に使わないため、
/// カタカナパススルーの対象から除外してモデルに委ねる。
#[inline]
fn is_counter_small_kana(cp: u32) -> bool {
    matches!(cp, 0x30F5 | 0x30F6)
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
            | 0x30EE // ヮ
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
        let config = PredictorConfig::new("dummy.mbm").with_numeric_confidence_threshold(0.4);

        assert_eq!(config.model_path(), Path::new("dummy.mbm"));
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
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/dummy.mbm")
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

    // ============================================================
    // get_source_segments / format_source_segmented / format_segmented
    // ============================================================

    #[test]
    fn get_source_segments_no_boundary() {
        let config = PredictorConfig::new(dummy_model_path());
        let predictor = Predictor::load(config).unwrap();

        // ASCII バイパスのみ → 境界スペースなし → 全体が 1 セグメント
        let result = predictor.predict("abc").unwrap();
        assert_eq!(result.get_source_segments(), vec!["abc"]);
    }

    #[test]
    fn get_source_segments_empty() {
        let config = PredictorConfig::new(dummy_model_path());
        let predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("").unwrap();
        assert!(result.get_source_segments().is_empty());
    }

    #[test]
    fn format_source_segmented_no_boundary() {
        let config = PredictorConfig::new(dummy_model_path());
        let predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("abc").unwrap();
        assert_eq!(result.format_source_segmented(), "abc");
    }

    #[test]
    fn format_segmented_ascii() {
        let config = PredictorConfig::new(dummy_model_path());
        let predictor = Predictor::load(config).unwrap();

        // ASCII バイパス: 各文字が 1:1 でかなに対応
        let result = predictor.predict("abc").unwrap();
        assert_eq!(result.format_segmented(), "a/b/c");
    }

    #[test]
    fn format_segmented_symbol() {
        let config = PredictorConfig::new(dummy_model_path());
        let predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("、").unwrap();
        assert_eq!(result.format_segmented(), "、");
    }

    // ============================================================
    // kana_to_source_char / source_to_kana_char
    // ============================================================

    #[test]
    fn kana_to_source_char_ascii() {
        let config = PredictorConfig::new(dummy_model_path());
        let predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("abc").unwrap();
        // a→0, b→1, c→2 のコードポイントインデックス
        assert_eq!(result.kana_to_source_char(), vec![0, 1, 2]);
    }

    #[test]
    fn source_to_kana_char_ascii() {
        let config = PredictorConfig::new(dummy_model_path());
        let predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("abc").unwrap();
        // 原文 a(0)→かな 0, b(1)→1, c(2)→2
        assert_eq!(
            result.source_to_kana_char(),
            vec![vec![0], vec![1], vec![2]]
        );
    }

    #[test]
    fn kana_to_source_char_and_source_to_kana_char_roundtrip() {
        let config = PredictorConfig::new(dummy_model_path());
        let predictor = Predictor::load(config).unwrap();

        // 記号 (マルチバイト) でも整合性が取れること
        let result = predictor.predict("a、b").unwrap();
        let k2s = result.kana_to_source_char();
        let s2k = result.source_to_kana_char();

        // k2s の長さ = かな文字数 (コードポイント数)
        assert_eq!(k2s.len(), result.kana_text().chars().count());
        // s2k の長さ = 原文文字数
        assert_eq!(s2k.len(), result.source_text().chars().count());

        // 往復整合性: s2k[s][j] = i ならば k2s[i] = s
        for (s, kana_positions) in s2k.iter().enumerate() {
            for &kana_char_idx in kana_positions {
                assert_eq!(k2s[kana_char_idx], s);
            }
        }
    }
}
