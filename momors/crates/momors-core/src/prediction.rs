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

use crate::bracket::{self, Outer, Role};
use crate::char_type::CharType;
use crate::feature::FeatureKey;
use crate::featurize::{compute_source_features, to_source_seq, SourceEntry};
use crate::model::MomoModel;
use crate::numeric::{
    convert_japanese_numeric, is_digit_label, is_kun_counter_signal, NumericFallback,
};
use crate::weight_model::WeightModel;
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

/// 人名辞書読みフォールバックを発動させる自信度の上限。
/// 人名スパン内の読みがこれ未満の自信度のとき、辞書の固定読みで置換する。
/// Python 版 `_LR_LOW_THRESHOLD` と対応。
const NAME_READING_CONF_THRESHOLD: f32 = 0.5;

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
    pub(crate) kanji_dict_path: Option<PathBuf>,
    pub(crate) numeric_confidence_threshold: f32,
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

    /// 点字のコードポイント位置 → 原文のコードポイント位置 のマッピング。
    ///
    /// `kana_to_braille` には `JapaneseResult::kana_to_braille()` を渡す。
    /// `braille_char_count` には点字テキストのコードポイント数を渡す。
    ///
    /// 各点字位置はそのセルを生成したかな文字を経由して原文位置に帰属する。
    /// 開始フラグ（⠼ ⠰ ⠠）はそのフラグを発動させた原文文字のインデックスに帰属し、
    /// 終了フラグ（⠤ など）とクラス遷移スペースは直前の原文文字に帰属する
    /// （`JapaneseTranslator` が `brl_pos` を遷移スペースの直後・開始フラグの前で記録するため）。
    /// 複合音（キャ → ⠈⠡）が 2 セル占める場合、両セルとも複合音の原文位置を指す。
    pub fn braille_char_to_source(
        &self,
        kana_to_braille: &[usize],
        braille_char_count: usize,
    ) -> Vec<usize> {
        let kana_to_src = self.kana_to_source_char();
        let n_kana = kana_to_braille.len();
        let mut result = vec![0usize; braille_char_count];

        for i in 0..n_kana {
            let start = kana_to_braille[i];
            let end = if i + 1 < n_kana {
                kana_to_braille[i + 1]
            } else {
                braille_char_count
            };
            let src = kana_to_src.get(i).copied().unwrap_or(0);
            for p in start..end {
                if p < braille_char_count {
                    result[p] = src;
                }
            }
        }

        result
    }

    /// 原文のコードポイント位置 → 点字のコードポイント位置のリスト。
    ///
    /// `kana_to_braille` には `JapaneseResult::kana_to_braille()` を渡す。
    /// 複合音（キャ など）は 2 つのかな文字が同一の点字位置を指すため、
    /// 重複を除いた昇順リストを返す。
    pub fn source_to_braille_char(&self, kana_to_braille: &[usize]) -> Vec<Vec<usize>> {
        self.source_to_kana_char()
            .into_iter()
            .map(|kana_indices| {
                let mut braille: Vec<usize> = kana_indices
                    .iter()
                    .filter_map(|&ki| kana_to_braille.get(ki).copied())
                    .collect();
                braille.dedup();
                braille
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
/// 重みの持ち方は型パラメータ `M` で切り替える。既定の [`MomoModel`]
/// （`.mbm`、int8 量子化）を使う場合は `Predictor` とだけ書けば良い。
/// 量子化前の float32 (`.mbmf`) を使いたい場合は [`crate::FloatPredictor`]
/// （`Predictor<FloatMomoModel>` の別名）を使う。両者は同じ `predict()` API を
/// 持つため、量子化前後でテキスト推論の結果を直接比較できる。
///
/// [`load`]: Predictor::load
/// [`predict`]: Predictor::predict
#[derive(Debug)]
pub struct Predictor<M: WeightModel = MomoModel> {
    config: PredictorConfig,
    model: M,
    /// ソート済み漢字辞書。binary_search でルックアップする。
    kanji_dict: Vec<(char, Vec<String>)>,
}

impl<M: WeightModel> Predictor<M> {
    /// 設定からモデルを読み込んで予測器を構築する。
    ///
    /// 単一漢字辞書は `kanji_dict_path` の明示指定があればそれを使い、
    /// なければモデルファイルに同梱された辞書（学習時と同一）を使う。
    pub fn load(config: PredictorConfig) -> Result<Self> {
        let mut model = M::load(config.model_path())?;
        let kanji_dict = if let Some(ref path) = config.kanji_dict_path {
            load_kanji_dict(path)?
        } else {
            model.take_kanji_dict()
        };
        Ok(Self {
            config,
            model,
            kanji_dict,
        })
    }

    /// バイト列からモデルを読み込んで予測器を構築する (WASM / インメモリ用)。
    ///
    /// デフォルト設定 (numeric_confidence_threshold=0.5) を使用する。
    /// 単一漢字辞書はモデルファイルに同梱されたものを使う。
    pub fn from_model_bytes(bytes: &[u8]) -> Result<Self> {
        let mut model = M::load_from_bytes(bytes)?;
        let config = PredictorConfig {
            model_path: PathBuf::from("<memory>"),
            numeric_confidence_threshold: 0.5,
            kanji_dict_path: None,
        };
        let kanji_dict = model.take_kanji_dict();
        Ok(Self {
            config,
            model,
            kanji_dict,
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
        // --- 正規化 ---
        // 康煕部首・CJK互換漢字等の畳み込みに加え、欧文リガチャ等の1→N展開を
        // 行う。Python 版 predict() の normalize_input() 呼び出しと対応。
        // パイプラインは正規化後テキスト基準で動かし、最後に原文基準へ戻す。
        let normalized = crate::normalize::normalize_input(text);
        let mut result = self.predict_with_brackets(&normalized.text)?;
        if let Some(byte_map) = &normalized.byte_map {
            remap_result_to_source(&mut result, text, byte_map);
        }
        Ok(result)
    }

    /// 括弧類を除去して推論し、結果へルールで書き戻す。
    ///
    /// 括弧は本文の分かち書きをほぼ変えない「オーバーレイ」なので、bypass として
    /// モデルに流すと (1) 特徴量ウィンドウを分断し (2) 周囲のマスあけを境界モデル
    /// 任せにしてしまう。そこで [`strip_brackets`] で本文から抜いて素の文として
    /// 推論し、[`reinsert_brackets`] で開き=後続に密着 / 閉じ=前接に密着させて戻す。
    /// マスあけは括弧種の外エッジ規則（[`crate::bracket::Outer`]）で決める。
    ///
    /// 入出力の座標系は引数 `text`（正規化後テキスト）基準。括弧が無ければ
    /// [`predict_normalized`] をそのまま呼ぶ（オーバーヘッドゼロ）。
    ///
    /// [`predict_normalized`]: Predictor::predict_normalized
    fn predict_with_brackets(&self, text: &str) -> Result<PredictionResult> {
        let stripped = strip_brackets(text);
        if stripped.brackets.is_empty() {
            return self.predict_normalized(text);
        }
        let mut result = self.predict_normalized(&stripped.text)?;
        reinsert_brackets(&mut result, text, &stripped);
        Ok(result)
    }

    /// 正規化済みテキストに対する推論本体。
    ///
    /// 結果の source_text とインデックスは正規化後テキスト基準。
    fn predict_normalized(&self, text: &str) -> Result<PredictionResult> {
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

        // 人名辞書マッチ (B/I フラグ + ユニット別固定読み)。辞書なしモデルでは
        // 全て O で、NameFlag* 特徴量は発火しない。
        let (name_flags, name_readings) =
            crate::name_dict::compute_name_matches(&source_seq, self.model.name_dict());
        let all_feat_keys = compute_source_features(&source_seq, &name_flags);
        let all_feat_ids: Vec<Vec<u32>> = all_feat_keys
            .iter()
            .map(|keys| lookup_feature_ids(keys, &self.model))
            .collect();

        let n_cls = self.model.n_classes() as usize;
        let mut read_scratch = self.model.new_scratch();
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
            // スコア計算 + argmax（漢字辞書制約付き）
            // ============================================================
            let (best_cls, best_score) =
                self.read_argmax(entry, &all_feat_ids[i], &mut read_scratch, &mut scores);

            let conf = sigmoid(best_score);
            let label = self.model.read_class(best_cls as u32).unwrap_or("");

            // ============================================================
            // JAPANESE_NUMERIC ルールベース変換
            //
            // 助数詞コヒーレンス: 低自信度でも、直後が訓読み助数詞（日→カ・つ→ツ・
            // 人→リ 等）ならば数字化せずモデルの読み（ミッ 等）を信用する。
            // 助数詞の読み自体が曖昧性を解決する（人→リ＝ふたり は信用、
            // 人→ニン＝さんにん は数字化）。漢数字専用（アラビアは別ロジック）。
            // ============================================================
            let next_is_kun_counter = if entry.ctype == CharType::JapaneseNumeric
                && conf < self.config.numeric_confidence_threshold
                && i + 1 < n
            {
                let (next_cls, _) = self.read_argmax(
                    &source_seq[i + 1],
                    &all_feat_ids[i + 1],
                    &mut read_scratch,
                    &mut scores,
                );
                let next_label = self.model.read_class(next_cls as u32).unwrap_or("");
                let next_char = char::from_u32(source_seq[i + 1].cp)
                    .map(|c| c.to_string())
                    .unwrap_or_default();
                is_kun_counter_signal(&next_char, next_label)
            } else {
                false
            };

            if entry.ctype == CharType::JapaneseNumeric
                && conf < self.config.numeric_confidence_threshold
                && !next_is_kun_counter
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
            // NUMERIC (アラビア数字) の採否（モデルを信用する）
            //
            //   - モデルが数字を返した: 原文と一致→採用 / 不一致→原文に戻す（数の崩壊防止）
            //   - モデルがかなを返した: 直後が既知の訓読み助数詞シグナル（日→カ・タチ、
            //                          つ→ツ、人→リ）→採用（みっか・はつか等） /
            //                          それ以外→原文に戻す
            //
            // 助数詞シグナルが立つのは直後が具体的な訓読みの助数詞のときだけなので、
            // 数字が隣接する場合（多桁数の内部、例「120日」の 0）や、直後に何もない
            // 孤立した数字（例「3:5」の 5）は自然に revert される。
            // CONTINUE/SKIP は後段の救済に委ねる。
            // ============================================================
            if entry.ctype == CharType::Numeric && label != LABEL_CONTINUE && label != LABEL_SKIP {
                let src_ch = char::from_u32(entry.cp)
                    .map(|c| c.to_string())
                    .unwrap_or_default();
                let revert = if is_digit_label(label) {
                    label != src_ch
                } else {
                    let next_is_kun_counter = if i + 1 < n {
                        let (next_cls, _) = self.read_argmax(
                            &source_seq[i + 1],
                            &all_feat_ids[i + 1],
                            &mut read_scratch,
                            &mut scores,
                        );
                        let next_label = self.model.read_class(next_cls as u32).unwrap_or("");
                        let next_char = char::from_u32(source_seq[i + 1].cp)
                            .map(|c| c.to_string())
                            .unwrap_or_default();
                        is_kun_counter_signal(&next_char, next_label)
                    } else {
                        false
                    };
                    !next_is_kun_counter
                };
                if revert {
                    let has_split = sigmoid(self.compute_boundary_score(&all_feat_ids[i])) >= 0.5;
                    self.emit_char_passthrough(entry, conf, &mut result);
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
                // revert しない場合は下の通常ラベル出力へ流す（label をそのまま emit）。
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
            } else if let Some(reading) = name_readings[i]
                .as_deref()
                .filter(|r| conf < NAME_READING_CONF_THRESHOLD && *r != label)
            {
                // 人名辞書読みフォールバック:
                // 人名スパン内 かつ モデルが低自信度のときだけ、辞書の固定読みで
                // 置換する（Python 版 DICT_NAME と対応）。自信のある予測は尊重する
                // ので、人名と同表層の別語（森林・関係など）を壊さない。
                reading.to_string()
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

    /// 1文字分の読みスコアを計算し、（漢字辞書制約付き）argmax のクラスとスコアを返す。
    /// `scratch` / `scores` はスクラッチバッファ（呼び出しごとに上書きされる）。
    fn read_argmax(
        &self,
        entry: &SourceEntry,
        feat_ids: &[u32],
        scratch: &mut M::Scratch,
        scores: &mut [f32],
    ) -> (usize, f32) {
        self.model.compute_read_scores(feat_ids, scratch, scores);

        // 漢字辞書制約付き argmax
        if entry.ctype == CharType::Kanji && !self.kanji_dict.is_empty() {
            if let Some(ch) = char::from_u32(entry.cp) {
                if let Some(readings) = lookup_kanji_dict(&self.kanji_dict, ch) {
                    // 辞書の読みが1つもモデルのクラスに存在しない場合は None。
                    // その場合は制約なし argmax にフォールバックする。
                    if let Some(result) =
                        constrained_argmax(scores, self.model.read_classes(), readings)
                    {
                        return result;
                    }
                }
            }
        }
        unconstrained_argmax(scores)
    }

    /// 境界モデルの生スコア (sigmoid 前) を計算する。
    fn compute_boundary_score(&self, feat_ids: &[u32]) -> f32 {
        self.model.compute_boundary_score(feat_ids)
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

/// 正規化テキスト基準の予測結果を原文（正規化前）基準へ戻す。
///
/// `normalize_input()` は欧文リガチャ等で 1→N 展開を行うため、パイプラインが
/// 参照するバイト位置は原文とずれ得る。source_text を原文へ差し替え、
/// kana_to_src_index / src_to_kana_index を原文のバイト位置へ変換する。
/// 展開で同じ原文文字に複数の正規化後文字が対応する場合、その仮名位置は
/// すべて原文の1文字（の先頭バイト位置）に集約される。
///
/// `byte_map` は正規化後テキストの各バイト位置 → 原文バイト位置の対応
/// （[`crate::normalize::NormalizedText::byte_map`]）。
fn remap_result_to_source(result: &mut PredictionResult, source_text: &str, byte_map: &[usize]) {
    for v in &mut result.kana_to_src_index {
        *v = byte_map[*v];
    }
    let mut src_to_kana: Vec<Vec<usize>> = vec![Vec::new(); source_text.len()];
    for (norm_byte, kana_positions) in std::mem::take(&mut result.src_to_kana_index)
        .into_iter()
        .enumerate()
    {
        if !kana_positions.is_empty() {
            src_to_kana[byte_map[norm_byte]].extend(kana_positions);
        }
    }
    result.src_to_kana_index = src_to_kana;
    result.source_text = source_text.to_string();
}

// ============================================================
// 括弧の除去 / 再挿入
// ============================================================

/// 除去した括弧1つ分の記録。
struct BracketMark {
    /// 括弧文字。
    ch: char,
    role: Role,
    outer: Outer,
    /// 除去前テキスト（norm 基準）での **UTF-8 バイト位置**。
    norm_byte: usize,
}

/// [`strip_brackets`] の結果。
struct Stripped {
    /// 括弧を除いたテキスト。
    text: String,
    /// stripped の各バイト → 由来する norm 文字の先頭バイト位置。
    /// `kana_to_src_index`（= stripped の文字先頭バイト）を norm 座標へ戻すのに使う。
    strip_to_norm: Vec<usize>,
    /// 除去した括弧（norm のバイト昇順）。
    brackets: Vec<BracketMark>,
    /// 除去前テキスト（norm）の総バイト数。
    norm_len: usize,
}

/// テキストから括弧類を除去し、位置と分類を記録する。
///
/// `norm_text` は正規化後テキスト。括弧は [`bracket::lookup`] で判定する。
fn strip_brackets(norm_text: &str) -> Stripped {
    let mut text = String::with_capacity(norm_text.len());
    let mut strip_to_norm = Vec::with_capacity(norm_text.len());
    let mut brackets = Vec::new();
    for (b, c) in norm_text.char_indices() {
        if let Some((role, outer)) = bracket::lookup(c) {
            brackets.push(BracketMark {
                ch: c,
                role,
                outer,
                norm_byte: b,
            });
        } else {
            text.push(c);
            for _ in 0..c.len_utf8() {
                strip_to_norm.push(b);
            }
        }
    }
    Stripped {
        text,
        strip_to_norm,
        brackets,
        norm_len: norm_text.len(),
    }
}

/// 外エッジ規則と現状のスペース有無から、その境界にスペースを置くか決める。
fn outer_wants_space(outer: Outer, had_space: bool) -> bool {
    match outer {
        Outer::Attach => false,
        Outer::Space => true,
        Outer::Defer => had_space,
    }
}

/// 除去した括弧を推論結果へ書き戻す。
///
/// `result` は stripped テキストに対する推論結果（座標系は stripped）。
/// 本関数は結果を **norm 座標＋括弧入り** に作り替える:
///
/// - 開き括弧は「直後に食いつく内容ユニット」の直前に、閉じ括弧は「直前に食いつく
///   内容ユニット」の直後に挿入する。
/// - 括弧まわりのマスあけは外エッジ規則で決める。閉じ→開きが隣接する境界は
///   必ずあける（点字表記法のカッコ連続規則に合わせる）。
/// - 括弧なしの境界（内容ユニット同士）は境界モデルの判定をそのまま使う。
fn reinsert_brackets(result: &mut PredictionResult, norm_text: &str, stripped: &Stripped) {
    // --- 1. かな→原文インデックスを stripped 座標から norm 座標へ写す ---
    for v in &mut result.kana_to_src_index {
        *v = stripped.strip_to_norm[*v];
    }

    // --- 2. かな列をセル（1かな文字＝1セル）へ分解 ---
    struct Cell {
        ch: char,
        src: usize,
        conf: f32,
        space: bool,
    }
    let mut cells: Vec<Cell> = Vec::with_capacity(result.kana_text.len());
    for (kb, ch) in result.kana_text.char_indices() {
        cells.push(Cell {
            ch,
            src: result.kana_to_src_index[kb],
            conf: result.confidences[kb],
            space: ch == ' ',
        });
    }
    // 内容セル（非スペース）の index。src は左→右で非減少。
    let content: Vec<usize> = cells
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.space)
        .map(|(i, _)| i)
        .collect();

    // --- 3. 括弧を内容セルの前(open)／後(close)へ割り付ける ---
    use std::collections::HashMap;
    let mut prefix: HashMap<usize, Vec<&BracketMark>> = HashMap::new();
    let mut suffix: HashMap<usize, Vec<&BracketMark>> = HashMap::new();
    // 内容セルが片側に無い退化ケース（文頭の閉じ・文末の開き）。
    let mut leading: Vec<&BracketMark> = Vec::new();
    let mut trailing: Vec<&BracketMark> = Vec::new();
    for bm in &stripped.brackets {
        match bm.role {
            Role::Open => match content.iter().find(|&&ci| cells[ci].src > bm.norm_byte) {
                Some(&ci) => prefix.entry(ci).or_default().push(bm),
                None => trailing.push(bm),
            },
            Role::Close => match content
                .iter()
                .rev()
                .find(|&&ci| cells[ci].src < bm.norm_byte)
            {
                Some(&ci) => suffix.entry(ci).or_default().push(bm),
                None => leading.push(bm),
            },
        }
    }

    // --- 4. 再構築 ---
    let mut kana = String::new();
    let mut k2s: Vec<usize> = Vec::new();
    let mut conf: Vec<f32> = Vec::new();

    let push_text = |kana: &mut String,
                     k2s: &mut Vec<usize>,
                     conf: &mut Vec<f32>,
                     s: &str,
                     src: usize,
                     c: f32| {
        kana.push_str(s);
        for _ in 0..s.len() {
            k2s.push(src);
            conf.push(c);
        }
    };
    let push_bracket =
        |kana: &mut String, k2s: &mut Vec<usize>, conf: &mut Vec<f32>, bm: &BracketMark| {
            let mut buf = [0u8; 4];
            push_text(
                kana,
                k2s,
                conf,
                bm.ch.encode_utf8(&mut buf),
                bm.norm_byte,
                1.0,
            );
        };

    // 内容セルが無い場合は括弧を norm 順に並べるだけ。
    if content.is_empty() {
        for bm in &stripped.brackets {
            push_bracket(&mut kana, &mut k2s, &mut conf, bm);
        }
    } else {
        // 文頭側の退化した閉じ括弧。
        for bm in &leading {
            push_bracket(&mut kana, &mut k2s, &mut conf, bm);
        }

        for j in 0..content.len() {
            let ci = content[j];

            // このセルに前置する開き括弧。
            if let Some(opens) = prefix.get(&ci) {
                for bm in opens {
                    push_bracket(&mut kana, &mut k2s, &mut conf, bm);
                }
            }
            // 内容セル本体。
            let mut buf = [0u8; 4];
            push_text(
                &mut kana,
                &mut k2s,
                &mut conf,
                cells[ci].ch.encode_utf8(&mut buf),
                cells[ci].src,
                cells[ci].conf,
            );
            // このセルに後置する閉じ括弧。
            if let Some(closes) = suffix.get(&ci) {
                for bm in closes {
                    push_bracket(&mut kana, &mut k2s, &mut conf, bm);
                }
            }

            // 次の内容セルとの境界のマスあけ。
            if j + 1 < content.len() {
                let next_ci = content[j + 1];
                let had_space = cells[ci + 1..next_ci].iter().any(|c| c.space);
                // この境界に面する括弧の外エッジ:
                //   左＝直前セルの後置閉じの最後 / 右＝次セルの前置開きの最初
                let close_edge = suffix.get(&ci).and_then(|v| v.last()).map(|bm| bm.outer);
                let open_edge = prefix
                    .get(&next_ci)
                    .and_then(|v| v.first())
                    .map(|bm| bm.outer);
                let want_space = match (close_edge, open_edge) {
                    // 閉じ→開きの隣接は必ずあける。
                    (Some(_), Some(_)) => true,
                    (Some(o), None) | (None, Some(o)) => outer_wants_space(o, had_space),
                    // 括弧の無い境界はモデル判定のまま。
                    (None, None) => had_space,
                };
                if want_space {
                    push_text(&mut kana, &mut k2s, &mut conf, " ", cells[ci].src, 1.0);
                }
            }
        }

        // 文末側の退化した開き括弧。
        for bm in &trailing {
            push_bracket(&mut kana, &mut k2s, &mut conf, bm);
        }
    }

    // --- 5. 結果を norm 座標で確定し、逆引きを作り直す ---
    result.kana_text = kana;
    result.kana_to_src_index = k2s;
    result.confidences = conf;
    result.source_text = norm_text.to_string();
    let mut src_to_kana: Vec<Vec<usize>> = vec![Vec::new(); stripped.norm_len];
    for (j, &src) in result.kana_to_src_index.iter().enumerate() {
        if src < stripped.norm_len {
            src_to_kana[src].push(j);
        }
    }
    result.src_to_kana_index = src_to_kana;
}

/// スコア配列の argmax を返す。
///
/// `n_classes >= 1` は loader (`MomoModel` 読み込み時) が保証する不変条件。
fn unconstrained_argmax(scores: &[f32]) -> (usize, f32) {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &s)| (i, s))
        .expect("n_classes >= 1")
}

/// 辞書の読み候補 `readings` のいずれかと一致するクラスの中で、
/// 最もスコアが高いものを返す。一致するクラスが1つも無ければ `None`
/// （呼び出し側で `unconstrained_argmax` にフォールバックする）。
fn constrained_argmax(
    scores: &[f32],
    class_labels: &[String],
    readings: &[String],
) -> Option<(usize, f32)> {
    let mut best: Option<(usize, f32)> = None;
    for (cls, lbl) in class_labels.iter().enumerate() {
        if readings.iter().any(|r| r == lbl) {
            let s = scores[cls];
            if best.is_none_or(|(_, best_s)| s > best_s) {
                best = Some((cls, s));
            }
        }
    }
    best
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
fn lookup_feature_ids<M: WeightModel>(keys: &[FeatureKey], model: &M) -> Vec<u32> {
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

    // ============================================================
    // strip_brackets / reinsert_brackets
    // ============================================================

    /// stripped 上の推論結果を手で組み立てるヘルパ。
    /// `kana` の各文字が `srcs[i]`（stripped バイト位置）に対応するとみなす。
    fn make_result(stripped_text: &str, kana: &str, srcs: &[usize]) -> PredictionResult {
        let chars: Vec<char> = kana.chars().collect();
        assert_eq!(
            chars.len(),
            srcs.len(),
            "kana 文字数と srcs 長が一致すること"
        );
        let mut kana_to_src = Vec::new();
        for (ch, &s) in chars.iter().zip(srcs) {
            for _ in 0..ch.len_utf8() {
                kana_to_src.push(s);
            }
        }
        PredictionResult {
            source_text: stripped_text.to_string(),
            kana_text: kana.to_string(),
            confidences: vec![0.9; kana_to_src.len()],
            kana_to_src_index: kana_to_src,
            src_to_kana_index: vec![Vec::new(); stripped_text.len()],
        }
    }

    #[test]
    fn strip_brackets_records_positions() {
        // a「b」c : a=0 「=1(3B) b=4 」=5(3B) c=8
        let s = strip_brackets("a「b」c");
        assert_eq!(s.text, "abc");
        assert_eq!(s.strip_to_norm, vec![0, 4, 8]);
        assert_eq!(s.brackets.len(), 2);
        assert_eq!(s.brackets[0].ch, '「');
        assert_eq!(s.brackets[0].norm_byte, 1);
        assert_eq!(s.brackets[1].ch, '」');
        assert_eq!(s.brackets[1].norm_byte, 5);
        assert_eq!(s.norm_len, "a「b」c".len());
    }

    #[test]
    fn strip_brackets_none_when_absent() {
        let s = strip_brackets("あいう");
        assert!(s.brackets.is_empty());
        assert_eq!(s.text, "あいう");
    }

    #[test]
    fn reinsert_quote_attaches_inner_no_extra_space() {
        // norm "a「b」c" / モデルは空白なしで "abc" と分割 → 括弧が b に食いつく
        let stripped = strip_brackets("a「b」c");
        let mut r = make_result(&stripped.text, "abc", &[0, 1, 2]);
        reinsert_brackets(&mut r, "a「b」c", &stripped);
        assert_eq!(r.kana_text, "a「b」c");
        assert_eq!(r.source_text, "a「b」c");
        // インデックス長の整合
        assert_eq!(r.kana_to_src_index.len(), r.kana_text.len());
        assert_eq!(r.confidences.len(), r.kana_text.len());
        assert_eq!(r.src_to_kana_index.len(), "a「b」c".len());
    }

    #[test]
    fn reinsert_quote_defer_keeps_model_spaces() {
        // 「」= Defer: モデルが "a b c" と空けたら外側の空きは温存される
        let stripped = strip_brackets("a「b」c");
        let mut r = make_result(&stripped.text, "a b c", &[0, 0, 1, 1, 2]);
        reinsert_brackets(&mut r, "a「b」c", &stripped);
        // a␣「b」␣c : 開きの左・閉じの右の空きはモデル判定のまま
        assert_eq!(r.kana_text, "a 「b」 c");
    }

    #[test]
    fn reinsert_paren_attach_removes_seam_space() {
        // オケ（オーケストラ）型の縮約: （）= Attach → 継ぎ目の空きを詰める
        // norm "a（b）c" : a=0 （=1(3B) b=4 ）=5(3B) c=8
        let stripped = strip_brackets("a（b）c");
        assert_eq!(stripped.text, "abc");
        // モデルは "a b c" と3分割したと仮定
        let mut r = make_result(&stripped.text, "a b c", &[0, 0, 1, 1, 2]);
        reinsert_brackets(&mut r, "a（b）c", &stripped);
        // 外エッジ Attach で両側の境界スペースが消える
        assert_eq!(r.kana_text, "a（b）c");
    }

    #[test]
    fn reinsert_close_open_adjacent_forces_space() {
        // ）（ の隣接は必ずあける（Attach 同士でも）
        // norm "a）（b" : a=0 ）=1(3B) （=4(3B) b=7
        let stripped = strip_brackets("a）（b");
        assert_eq!(stripped.text, "ab");
        let mut r = make_result(&stripped.text, "ab", &[0, 1]);
        reinsert_brackets(&mut r, "a）（b", &stripped);
        assert_eq!(r.kana_text, "a）（b".replace("）（", "） （"));
    }

    #[test]
    fn reinsert_indices_roundtrip() {
        let stripped = strip_brackets("a「b」c");
        let mut r = make_result(&stripped.text, "abc", &[0, 1, 2]);
        reinsert_brackets(&mut r, "a「b」c", &stripped);
        // s2k[s] に j が含まれるなら kana_to_src[j] == s
        for (s, kana_positions) in r.src_to_kana_index.iter().enumerate() {
            for &j in kana_positions {
                assert_eq!(r.kana_to_src_index[j], s);
            }
        }
    }

    #[test]
    fn config_builder_chains() {
        let config = PredictorConfig::new("fixture.mbm").with_numeric_confidence_threshold(0.4);

        assert_eq!(config.model_path(), Path::new("fixture.mbm"));
        assert!((config.numeric_confidence_threshold() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_basics() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn constrained_argmax_picks_best_matching_class() {
        let scores = vec![1.0, 5.0, 3.0];
        let labels = vec!["カ".to_string(), "キ".to_string(), "ク".to_string()];
        // 「キ」と「ク」が辞書候補 → スコアが高い「キ」(index 1) を選ぶ
        let readings = vec!["キ".to_string(), "ク".to_string()];
        assert_eq!(
            constrained_argmax(&scores, &labels, &readings),
            Some((1, 5.0))
        );
    }

    #[test]
    fn constrained_argmax_returns_none_when_no_reading_matches() {
        let scores = vec![1.0, 5.0, 3.0];
        let labels = vec!["カ".to_string(), "キ".to_string(), "ク".to_string()];
        // 辞書の読みがモデルのどのクラスラベルとも一致しない
        let readings = vec!["ケ".to_string(), "コ".to_string()];
        assert_eq!(constrained_argmax(&scores, &labels, &readings), None);
    }

    #[test]
    fn read_argmax_falls_back_to_unconstrained_when_dict_readings_absent_from_model() {
        // fixture.mbm はクラス [カ, キ, ク] のみを持つ。
        // 辞書の読みがどれとも一致しない場合、read_argmax は
        // best_s=NEG_INFINITY のまま class 0 を返してはならず、
        // unconstrained_argmax と同じ結果にフォールバックすること。
        let dict_path = std::env::temp_dir().join(format!(
            "momors_test_kanji_dict_no_match_{}.tsv",
            std::process::id()
        ));
        // "漢" (U+6F22) の読みを、モデルに存在しない読みだけにする
        std::fs::write(&dict_path, "漢\tケ\tコ\n").unwrap();

        let config = PredictorConfig::new(fixture_model_path()).with_kanji_dict_path(&dict_path);
        let predictor: Predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("漢").unwrap();
        std::fs::remove_file(&dict_path).ok();
        // フォールバックが機能していれば、confidence は NEG_INFINITY 由来の
        // sigmoid(-inf)=0.0 ではなく、モデルの素の argmax に基づく値になる。
        assert!(result.confidences[0] > 0.0);
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

    // --- predict() の動作確認テスト (テスト用モデルで) ---

    fn fixture_model_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/fixture.mbm")
    }

    /// `fixture_model_path()` の量子化前 (`.mbmf`) 版。
    /// `gen_fixture_mbmf.py` が `int8_val * scale` を機械的に導出しているため、
    /// `fixture.mbm` と厳密に同じ重みを持つ。
    fn fixture_model_path_float() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/fixture.mbmf")
    }

    #[test]
    fn predict_empty_string() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).expect("fixture.mbm が読めること");

        let result = predictor.predict("").unwrap();
        assert_eq!(result.source_text(), "");
        assert_eq!(result.kana_text(), "");
        assert!(result.confidences().is_empty());
    }

    #[test]
    fn predict_bypass_symbol() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("、").unwrap();
        assert_eq!(result.kana_text(), "、");
        assert!(result.confidences().iter().all(|&c| (c - 1.0).abs() < 1e-6));
        assert_eq!(result.source_to_kana().len(), "、".len());
    }

    #[test]
    fn predict_bypass_alpha() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("abc").unwrap();
        assert_eq!(result.kana_text(), "abc");
    }

    #[test]
    fn predict_mixed_bypass() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("a、b").unwrap();
        assert_eq!(result.kana_text(), "a、b");
    }

    #[test]
    fn predict_keeps_original_source_text_for_compat_ideographs() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        // ⺟ (U+2E9F, CJK部首補助) は内部で「母」(U+6BCD) に畳み込んで推論するが、
        // 結果の source_text とインデックスは原文（入力そのまま）基準で返す。
        let result = predictor.predict("⺟").unwrap();
        assert_eq!(result.source_text(), "⺟");
        assert_eq!(result.source_to_kana().len(), "⺟".len());
    }

    #[test]
    fn predict_expands_latin_ligature() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        // ﬃ (U+FB03) は "ffi" に展開され ALPHA としてバイパスされる。
        // source_text とインデックスは原文基準。
        let src = "o\u{FB03}ce";
        let result = predictor.predict(src).unwrap();
        assert_eq!(result.source_text(), src);
        assert_eq!(
            result.kana_text().replace(' ', ""),
            "office",
            "リガチャが展開されて仮名（バイパス）に現れること"
        );
        assert_eq!(result.source_to_kana().len(), src.len());

        // ﬃ（原文バイト1〜3）に展開後の f/f/i すべてが対応する
        let ligature_kana: Vec<char> = result.source_to_kana()[1]
            .iter()
            .map(|&kb| result.kana_text()[kb..].chars().next().unwrap())
            .filter(|&c| c != ' ')
            .collect();
        assert_eq!(ligature_kana, vec!['f', 'f', 'i']);

        // kana_to_src_index はすべて原文の有効な文字境界を指す
        for &src_byte in result.kana_to_source() {
            assert!(src.is_char_boundary(src_byte));
        }
    }

    #[test]
    fn predict_source_to_kana_indices() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("a").unwrap();
        assert_eq!(result.kana_to_source(), &[0usize]);
        assert_eq!(result.source_to_kana()[0], vec![0usize]);
    }

    #[test]
    fn predict_does_not_crash_on_complex_input() {
        // fixture.mbm では意味のあるカナは出ないが、ロジックが落ちないことを確認
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        // 小書き仮名、々、漢数字、句読点 - 全ロジックパスを通す
        let inputs = ["きょう", "学校々", "三百二十一円。", "abc、漢字!"];
        for text in &inputs {
            let _ = predictor.predict(text).expect("crash しないこと");
        }
    }

    #[test]
    fn quantized_and_float_predictors_agree_on_fixture() {
        // fixture.mbm (量子化) と fixture.mbmf (量子化前) は
        // gen_fixture_mbmf.py が int8_val * scale を機械的に導出しているため
        // 厳密に同じ重みを持つ。よって predict() の結果も完全一致するはず
        // （WeightModel トレイトによる一般化が量子化パスの挙動を変えていないことの
        // クロスチェック）。
        let config = PredictorConfig::new(fixture_model_path());
        let quantized: Predictor = Predictor::load(config).unwrap();

        let float_config = PredictorConfig::new(fixture_model_path_float());
        let float_predictor: crate::FloatPredictor =
            crate::FloatPredictor::load(float_config).unwrap();

        assert_eq!(quantized.n_classes(), float_predictor.n_classes());
        assert_eq!(quantized.n_features(), float_predictor.n_features());

        let inputs = ["きょう", "学校々", "三百二十一円。", "abc、漢字!", "漢字"];
        for text in &inputs {
            let q = quantized.predict(text).unwrap();
            let f = float_predictor.predict(text).unwrap();
            assert_eq!(q.kana_text(), f.kana_text(), "text={text}");
        }
    }

    // ============================================================
    // get_source_segments / format_source_segmented / format_segmented
    // ============================================================

    #[test]
    fn get_source_segments_no_boundary() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        // ASCII バイパスのみ → 境界スペースなし → 全体が 1 セグメント
        let result = predictor.predict("abc").unwrap();
        assert_eq!(result.get_source_segments(), vec!["abc"]);
    }

    #[test]
    fn get_source_segments_empty() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("").unwrap();
        assert!(result.get_source_segments().is_empty());
    }

    #[test]
    fn format_source_segmented_no_boundary() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("abc").unwrap();
        assert_eq!(result.format_source_segmented(), "abc");
    }

    #[test]
    fn format_segmented_ascii() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        // ASCII バイパス: 各文字が 1:1 でかなに対応
        let result = predictor.predict("abc").unwrap();
        assert_eq!(result.format_segmented(), "a/b/c");
    }

    #[test]
    fn format_segmented_symbol() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("、").unwrap();
        assert_eq!(result.format_segmented(), "、");
    }

    // ============================================================
    // kana_to_source_char / source_to_kana_char
    // ============================================================

    #[test]
    fn kana_to_source_char_ascii() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("abc").unwrap();
        // a→0, b→1, c→2 のコードポイントインデックス
        assert_eq!(result.kana_to_source_char(), vec![0, 1, 2]);
    }

    #[test]
    fn source_to_kana_char_ascii() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("abc").unwrap();
        // 原文 a(0)→かな 0, b(1)→1, c(2)→2
        assert_eq!(
            result.source_to_kana_char(),
            vec![vec![0], vec![1], vec![2]]
        );
    }

    #[test]
    fn kana_to_source_char_and_source_to_kana_char_roundtrip() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

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

    // ============================================================
    // source_to_braille_char
    // ============================================================

    #[test]
    fn source_to_braille_char_one_to_one() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        // ASCII バイパス: a→kana[0], b→kana[1], c→kana[2]
        // kana_to_braille が 1:1 のとき、source_to_braille も 1:1 になる
        let result = predictor.predict("abc").unwrap();
        let kana_to_braille = vec![0usize, 1, 2];
        let s2b = result.source_to_braille_char(&kana_to_braille);

        assert_eq!(s2b.len(), 3);
        assert_eq!(s2b[0], vec![0]);
        assert_eq!(s2b[1], vec![1]);
        assert_eq!(s2b[2], vec![2]);
    }

    #[test]
    fn source_to_braille_char_dedup_compound() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        // ASCII バイパス: a→kana[0], b→kana[1]
        // kana[0] と kana[1] が同じ点字位置 (複合音を模倣) → dedup されること
        let result = predictor.predict("ab").unwrap();
        let kana_to_braille = vec![0usize, 0]; // 両方とも点字位置 0
        let s2b = result.source_to_braille_char(&kana_to_braille);

        assert_eq!(s2b.len(), 2);
        assert_eq!(s2b[0], vec![0]);
        assert_eq!(s2b[1], vec![0]);
    }

    #[test]
    fn source_to_braille_char_empty() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("").unwrap();
        let s2b = result.source_to_braille_char(&[]);
        assert!(s2b.is_empty());
    }

    // ============================================================
    // braille_char_to_source
    // ============================================================

    #[test]
    fn braille_char_to_source_empty() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        let result = predictor.predict("").unwrap();
        assert!(result.braille_char_to_source(&[], 0).is_empty());
    }

    #[test]
    fn braille_char_to_source_one_to_one() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        // ASCII バイパス: a→kana[0]→braille[0], b→kana[1]→braille[1], c→kana[2]→braille[2]
        let result = predictor.predict("abc").unwrap();
        let b2s = result.braille_char_to_source(&[0, 1, 2], 3);
        assert_eq!(b2s, vec![0, 1, 2]);
    }

    #[test]
    fn braille_char_to_source_with_flag_cells() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        // フラグセルを模倣: kana[0](a→src0) が braille [0,2) を占有（フラグ込み）
        //                   kana[1](b→src1) が braille [2,4) を占有
        let result = predictor.predict("ab").unwrap();
        let b2s = result.braille_char_to_source(&[0, 2], 4);
        assert_eq!(b2s, vec![0, 0, 1, 1]);
    }

    #[test]
    fn braille_char_to_source_compound() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        // 複合音を模倣: kana[0](a→src0) と kana[1](b→src1) が同じ braille pos 0 を指す。
        // kana[0] の範囲 [0,0) は空になり、kana[1] の範囲 [0,2) が両セルを引き継ぐ。
        // (実際の キャ では kana[0]=キ と kana[1]=ャ が同一 src を指すが、
        //  ASCII proxy ではsrc[0] と src[1] が異なるため kana[1] の src が採用される)
        let result = predictor.predict("abc").unwrap();
        let b2s = result.braille_char_to_source(&[0, 0, 2], 3);
        // pos 0,1 → kana[1](b) → src 1
        // pos 2   → kana[2](c) → src 2
        assert_eq!(b2s, vec![1, 1, 2]);
    }

    #[test]
    fn braille_char_to_source_roundtrip_consistency() {
        let config = PredictorConfig::new(fixture_model_path());
        let predictor: Predictor = Predictor::load(config).unwrap();

        // source_to_braille_char と braille_char_to_source の整合性:
        // s2b[s] に bi が含まれるなら b2s[bi] == s でなければならない
        let result = predictor.predict("abc").unwrap();
        let kana_to_braille = vec![0usize, 1, 2];
        let s2b = result.source_to_braille_char(&kana_to_braille);
        let b2s = result.braille_char_to_source(&kana_to_braille, 3);

        for (s, braille_positions) in s2b.iter().enumerate() {
            for &bi in braille_positions {
                assert_eq!(b2s[bi], s, "b2s[{bi}] should be {s}");
            }
        }
    }
}
