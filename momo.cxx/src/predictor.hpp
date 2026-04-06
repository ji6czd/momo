#pragma once
#include <string>
#include <vector>
#include "model.hpp"

// ============================================================
// 予測結果
// Python側の PredictionResult と対応
// ============================================================

struct PredictionResult {
    std::string              source_text;       // 入力テキスト（UTF-8）
    std::string              kana_text;         // 出力かな（UTF-8）
    std::vector<float>       confidences;       // 各かな文字の自信度
    std::vector<int>         kana_to_src_index; // かな位置 → 原文コードポイント位置
};

// ============================================================
// 予測器
// ============================================================

class Predictor {
public:
    explicit Predictor(MomoModel model);

    PredictionResult predict(const std::string& text) const;

    float confidence_threshold         = 0.3f;
    float numeric_confidence_threshold = 0.5f;

private:
    MomoModel model_;

    // 特徴量ID列 → 各クラスのスコアに加算（CSC形式でアクセス）
    void compute_read_scores(
        const std::vector<uint32_t>& feat_ids,
        std::vector<float>&          scores) const;

    // 特徴量ID列 → 境界スコア（sigmoid変換前）
    float compute_boundary_score(
        const std::vector<uint32_t>& feat_ids) const;

    static float sigmoid(float x);

    float read_confidence(const std::vector<float>& scores, int class_id) const;
};
