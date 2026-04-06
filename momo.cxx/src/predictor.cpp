#include "predictor.hpp"
#include "momo_features.hpp"
#include "utf8.hpp"
#include <cmath>
#include <algorithm>

// ============================================================
// bypass（素通し）扱いにする文字種
// ============================================================

static bool is_bypass(CharType ct) {
    switch (ct) {
        case CharType::ALPHA:
        case CharType::SYMBOL:
        case CharType::SYMBOL_CLOSE:
        case CharType::SYMBOL_OPEN:
        case CharType::SYMBOL_STOP:
        case CharType::SYMBOL_PAUSE:
            return true;
        default:
            return false;
    }
}

// ============================================================
// Predictor
// ============================================================

Predictor::Predictor(MomoModel model)
    : model_(std::move(model))
{}

float Predictor::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float Predictor::read_confidence(
    const std::vector<float>& scores, int class_id) const
{
    return sigmoid(scores[class_id]);
}

// 特徴量キーをモデルの語彙マップで引いて feature_id のリストを作る
static std::vector<uint32_t> lookup_feature_ids(
    const std::vector<FeatureKey>& keys,
    const VocabMap&                vocab)
{
    std::vector<uint32_t> ids;
    ids.reserve(keys.size());
    for (const auto& k : keys) {
        auto it = vocab.find(k);
        if (it != vocab.end()) {
            ids.push_back(it->second);
        }
    }
    return ids;
}

void Predictor::compute_read_scores(
    const std::vector<uint32_t>& feat_ids,
    std::vector<float>&          scores) const
{
    // CSC形式: feature_id 列の非ゼロ要素を直接アクセスして加算
    // 1文字あたりの feat_ids は十数個、各列の非ゼロ要素だけ触れば済む
    const float scale = model_.read_scale;

    for (uint32_t feat_id : feat_ids) {
        if (feat_id >= model_.n_features) continue;

        const uint32_t col_start = model_.csc_colptr[feat_id];
        const uint32_t col_end   = model_.csc_colptr[feat_id + 1];

        for (uint32_t j = col_start; j < col_end; ++j) {
            const uint32_t cls = model_.csc_rowind[j];
            scores[cls] += static_cast<float>(model_.csc_data[j]) * scale;
        }
    }
}

float Predictor::compute_boundary_score(
    const std::vector<uint32_t>& feat_ids) const
{
    float score = model_.boundary_intercept[1];
    const float scale = model_.boundary_scale;
    for (uint32_t feat_id : feat_ids) {
        if (feat_id < model_.n_features) {
            score += static_cast<float>(model_.boundary_data[feat_id]) * scale;
        }
    }
    return score;
}

// ============================================================
// 推論本体
// ============================================================

PredictionResult Predictor::predict(const std::string& text) const {
    PredictionResult result;
    result.source_text = text;

    if (text.empty()) return result;

    const std::vector<char32_t> text32 = utf8_to_utf32(text);
    if (text32.empty()) return result;

    const std::vector<SourceEntry> source_seq = to_source_seq(text32);
    const int n = static_cast<int>(source_seq.size());

    const auto all_feat_keys = compute_source_features(source_seq);

    std::vector<std::vector<uint32_t>> all_feat_ids(n);
    for (int i = 0; i < n; ++i) {
        all_feat_ids[i] = lookup_feature_ids(all_feat_keys[i], model_.vocab);
    }

    const uint32_t n_cls = model_.n_classes;
    std::vector<float> scores(n_cls);

    for (int i = 0; i < n; ++i) {
        const auto& entry = source_seq[i];

        if (is_bypass(entry.ctype)) {
            result.kana_text += char32_to_utf8(entry.cp);
            result.kana_to_src_index.push_back(static_cast<int>(entry.orig_idx));
            result.confidences.push_back(1.0f);
            continue;
        }

        // 読みスコア計算（intercept で初期化してから CSC アクセス）
        for (uint32_t cls = 0; cls < n_cls; ++cls) {
            scores[cls] = model_.intercept_read[cls];
        }
        compute_read_scores(all_feat_ids[i], scores);

        // argmax
        const int best_cls = static_cast<int>(
            std::max_element(scores.begin(), scores.end()) - scores.begin());

        const float conf = read_confidence(scores, best_cls);
        const std::string& label = model_.read_classes[best_cls];

        if (label == "---" || label == "_") continue;

        // 境界判定
        const bool has_split = sigmoid(compute_boundary_score(all_feat_ids[i])) >= 0.5f;

        // かな出力
        for (const char c : label) result.kana_text += c;
        for (std::size_t j = 0; j < label.size(); ++j) {
            result.kana_to_src_index.push_back(static_cast<int>(entry.orig_idx));
            result.confidences.push_back(conf);
        }

        if (has_split) {
            result.kana_text += ' ';
            result.kana_to_src_index.push_back(static_cast<int>(entry.orig_idx));
            result.confidences.push_back(conf);
        }
    }

    return result;
}
