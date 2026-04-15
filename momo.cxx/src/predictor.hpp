#pragma once
#include <string>
#include <vector>

#include "model.hpp"

// ============================================================
// 予測結果
// Python側の PredictionResult と対応
// ============================================================

struct PredictionResult {
  std::string source_text;                          // 入力テキスト（UTF-8）
  std::string kana_text;                            // 出力かな（UTF-8）
  std::vector<float> confidences;                   // 各かな文字の自信度
  std::vector<int> kana_to_src_index;               // かな位置 → 原文コードポイント位置
  std::vector<std::vector<int>> src_to_kana_index;  // 原文コードポイント位置 → かなバイト位置リスト
};

// ============================================================
// 予測設定
// Python側の PredictorConfig と対応
// ============================================================

class PredictorConfig {
 public:
  explicit PredictorConfig(std::string model_path) : model_path_(std::move(model_path)) {}

  const std::string& model_path() const { return model_path_; }

  float confidence_threshold() const { return confidence_threshold_; }
  void set_confidence_threshold(float v) { confidence_threshold_ = v; }

  float numeric_confidence_threshold() const { return numeric_confidence_threshold_; }
  void set_numeric_confidence_threshold(float v) { numeric_confidence_threshold_ = v; }

 private:
  std::string model_path_;                     // モデルファイルのパス（.mbm）
  float confidence_threshold_ = 0.5f;          // KANJIフォールバックを発動させる自信度の上限
  float numeric_confidence_threshold_ = 0.5f;  // JAPANESE_NUMERICルールベース変換を発動させる自信度の上限
};

// ============================================================
// 予測器
// ============================================================

class Predictor {
 public:
  explicit Predictor(PredictorConfig config);

  const PredictorConfig& config() const { return config_; }
  PredictorConfig& config() { return config_; }

  // モデルを読み込む。失敗時は std::runtime_error を投げる。
  void load();

  PredictionResult predict(const std::string& text) const;

 private:
  MomoModel model_;
  PredictorConfig config_;

  // 特徴量ID列 → 各クラスの生整数スコアに加算（CSC形式でアクセス）
  // scale・intercept の適用は呼び出し側で行う
  void compute_read_scores(const std::vector<uint32_t>& feat_ids, std::vector<int32_t>& scores) const;

  // 特徴量ID列 → 境界スコア（sigmoid変換前）
  float compute_boundary_score(const std::vector<uint32_t>& feat_ids) const;

  static float sigmoid(float x);

  float read_confidence(const std::vector<float>& scores, int class_id) const;
};
