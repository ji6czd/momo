#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "crf_features.h"
#include "utils.h"

// crfsuite C++ API の前方宣言（実装は predictor.cpp 内でのみ crfsuite.hpp をインクルード）
namespace CRFSuite {
class Tagger;
}

namespace momo {

// ---------------------------------------------------------------------------
// DecisionSource — 各文字の読み決定根拠タグ
// ---------------------------------------------------------------------------
enum class DecisionSource {
  CRF,               ///< CRFの予測をそのまま採用
  CRF_LOW,           ///< 自信度は低いがCRFを採用
  FALLBACK_KANJI,    ///< KANJI低自信度フォールバック辞書
  FALLBACK_NUMERIC,  ///< JAPANESE_NUMERICルールベース変換
  FALLBACK_ORPHAN,   ///< 文字消失バグ救済ロジック
  FALLBACK_REPEAT,   ///< 々の繰り返し処理
  DICT,              ///< カスタム辞書による強制置換
  BYPASS,            ///< ASCIIバイパス
};

// ---------------------------------------------------------------------------
// PredictorConfig — Predictor の設定
// ---------------------------------------------------------------------------
struct PredictorConfig {
  std::string path_read;                      ///< 読みモデル（.crfsuite）のパス
  std::string path_boundary;                  ///< 境界モデル（.crfsuite）のパス（空文字列なら分かち書きなし）
  std::string path_single_kanji_dict;         ///< 単一漢字辞書TSVのパス（空文字列なら使用しない）
  std::string path_custom_dict;               ///< カスタム辞書TSVのパス（空文字列なら使用しない）
  float confidence_threshold = 0.3f;          ///< KANJIフォールバック発動閾値
  float numeric_confidence_threshold = 0.8f;  ///< JAPANESE_NUMERICルールベース発動閾値
};

// ---------------------------------------------------------------------------
// PredictionResult — 推論結果
// ---------------------------------------------------------------------------
struct PredictionResult {
  std::string source_text;                          ///< 入力テキスト（UTF-8）
  std::string kana_text;                            ///< 変換後テキスト（UTF-8）
  std::vector<float> confidences;                   ///< かな各文字の自信度
  std::vector<int> kana_to_src_index;               ///< かな位置 → 原文UTF-32インデックス
  std::vector<std::vector<int>> src_to_kana_index;  ///< 原文UTF-32インデックス → かな位置リスト
  std::vector<DecisionSource> decision_sources;     ///< かな各文字の決定根拠
};

// ---------------------------------------------------------------------------
// カスタム辞書型
// ---------------------------------------------------------------------------

/// 表層形（UTF-32） → 各文字の読みラベル列（UTF-8）
using CustomDict = std::unordered_map<std::u32string, std::vector<std::string>>;

/// 先頭文字（char32_t） → [(表層形, 読みラベル列), ...] (長い順)
using DictIndex = std::unordered_map<char32_t, std::vector<std::pair<std::u32string, std::vector<std::string>>>>;

// ---------------------------------------------------------------------------
// Predictor クラス
// ---------------------------------------------------------------------------
class Predictor {
 public:
  /// コンストラクタ。モデルと辞書をロードする。
  /// ロードに失敗した場合は std::runtime_error を投げる。
  explicit Predictor(const PredictorConfig& config);

  ~Predictor();

  // コピー禁止（crfsuite ハンドルの所有権管理のため）
  Predictor(const Predictor&) = delete;
  Predictor& operator=(const Predictor&) = delete;

  /// メインの推論エントリポイント。UTF-8 テキストを受け取り結果を返す。
  PredictionResult predict(const std::string& utf8_text);

 private:
  PredictorConfig _config;

  // CRFSuite C++ API タガー（不完全型のためunique_ptrで保持）
  std::unique_ptr<CRFSuite::Tagger> _tagger_read;
  std::unique_ptr<CRFSuite::Tagger> _tagger_boundary;
  bool _has_boundary = false;

  // フォールバック辞書: char32_t → {on読み（UTF-8）, kun読み（UTF-8）}
  struct FallbackEntry {
    std::string on;
    std::string kun;
  };
  std::unordered_map<char32_t, FallbackEntry> _single_kanji_dict;

  // カスタム辞書インデックス
  DictIndex _dict_index;

  // --- 初期化ヘルパー ---
  void _load_single_kanji_dict(const std::string& path);
  void _load_custom_dict(const std::string& path);

  // --- 推論パイプライン ---
  struct PreprocessResult {
    std::vector<SourceUnit> source_seq;
    std::unordered_set<int> bypass_indices;
    std::unordered_map<int, std::string> ascii_overrides;
    std::unordered_map<int, std::string> dict_overrides;
  };

  PreprocessResult _preprocess(const std::u32string& text) const;

  void _run_inference(const std::vector<SourceUnit>& source_seq, std::vector<std::string>& raw_labels,
                      std::vector<float>& raw_confidences, std::vector<std::string>& boundary_labels);

  struct RefineResult {
    std::vector<std::string> labels;
    std::vector<float> confidences;
    std::vector<bool> has_splits;
    std::vector<DecisionSource> decision_sources;
  };

  RefineResult _refine(const std::vector<SourceUnit>& source_seq, const std::vector<std::string>& raw_labels,
                       const std::vector<float>& raw_confidences, const std::vector<std::string>& boundary_labels,
                       const std::unordered_set<int>& bypass_indices,
                       const std::unordered_map<int, std::string>& ascii_overrides,
                       const std::unordered_map<int, std::string>& dict_overrides) const;

  PredictionResult _assemble(const std::string& utf8_text, const std::u32string& text,
                             const std::vector<SourceUnit>& source_seq, const RefineResult& refined,
                             const std::unordered_set<int>& bypass_indices) const;

  // --- フォールバックロジック ---
  std::string _kurai_fallback(char32_t ch, char32_t left_char, CharType left_ctype, CharType right_ctype) const;

  // --- カスタム辞書ヘルパー ---
  std::pair<int, std::vector<std::string>> _find_longest_match(const std::vector<SourceUnit>& source_seq,
                                                               int pos) const;
};

}  // namespace momo
