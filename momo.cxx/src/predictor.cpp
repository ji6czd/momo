#include "predictor.hpp"

#include <algorithm>
#include <cmath>

#include "momo_features.hpp"
#include "utf8.hpp"
#ifdef MOMO_TRACE
#include <chrono>
#include <cstdio>
#define TRACE_NOW() std::chrono::high_resolution_clock::now()
#define TRACE_US(a, b) std::chrono::duration_cast<std::chrono::nanoseconds>((b) - (a)).count()
#else
#define TRACE_NOW() 0
#define TRACE_US(a, b) 0
#endif

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

Predictor::Predictor(MomoModel model) : model_(std::move(model)) {}

float Predictor::sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float Predictor::read_confidence(const std::vector<float>& scores, int class_id) const {
  return sigmoid(scores[class_id]);
}

// 特徴量キーをモデルの語彙テーブルで引いて feature_id のリストを作る
static std::vector<uint32_t> lookup_feature_ids(const std::vector<FeatureKey>& keys, const VocabVec& vocab) {
  std::vector<uint32_t> ids;
  ids.reserve(keys.size());
  for (const auto& k : keys) {
    const uint32_t id = vocab_find(vocab, k);
    if (id != UINT32_MAX) {
      ids.push_back(id);
    }
  }
  return ids;
}

void Predictor::compute_read_scores(const std::vector<uint32_t>& feat_ids, std::vector<int32_t>& scores) const {
  // CSC形式: feature_id 列の非ゼロ要素を直接アクセスして加算
  // int8_t → int32_t への暗黙昇格のみ。scale は呼び出し側で一括適用する。
  for (uint32_t feat_id : feat_ids) {
    if (feat_id >= model_.n_features) continue;

    const uint32_t col_start = model_.csc_colptr[feat_id];
    const uint32_t col_end = model_.csc_colptr[feat_id + 1];

    for (uint32_t j = col_start; j < col_end; ++j) {
      const uint32_t cls = model_.csc_rowind[j];
      scores[cls] += model_.csc_data[j];  // int8_t → int32_t 昇格のみ
    }
  }
}

float Predictor::compute_boundary_score(const std::vector<uint32_t>& feat_ids) const {
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

#ifdef MOMO_TRACE
  long long tr_utf8 = 0, tr_seq = 0, tr_feat_key = 0, tr_vocab = 0;
  long long tr_fill = 0, tr_read_scores = 0, tr_scale = 0, tr_boundary = 0;
  auto _t0 = TRACE_NOW();
#endif

  const std::vector<char32_t> text32 = utf8_to_utf32(text);
  if (text32.empty()) return result;
#ifdef MOMO_TRACE
  {
    auto _t1 = TRACE_NOW();
    tr_utf8 = TRACE_US(_t0, _t1);
    _t0 = _t1;
  }
#endif

  const std::vector<SourceEntry> source_seq = to_source_seq(text32);
  const int n = static_cast<int>(source_seq.size());
#ifdef MOMO_TRACE
  {
    auto _t1 = TRACE_NOW();
    tr_seq = TRACE_US(_t0, _t1);
    _t0 = _t1;
  }
#endif

  const auto all_feat_keys = compute_source_features(source_seq);
#ifdef MOMO_TRACE
  {
    auto _t1 = TRACE_NOW();
    tr_feat_key = TRACE_US(_t0, _t1);
    _t0 = _t1;
  }
#endif

  std::vector<std::vector<uint32_t>> all_feat_ids(n);
  for (int i = 0; i < n; ++i) {
    all_feat_ids[i] = lookup_feature_ids(all_feat_keys[i], model_.vocab);
  }
#ifdef MOMO_TRACE
  {
    auto _t1 = TRACE_NOW();
    tr_vocab = TRACE_US(_t0, _t1);
  }
#endif

  const uint32_t n_cls = model_.n_classes;
  std::vector<int32_t> int_scores(n_cls);
  std::vector<float> scores(n_cls);

  for (int i = 0; i < n; ++i) {
    const auto& entry = source_seq[i];

    if (is_bypass(entry.ctype)) {
      result.kana_text += char32_to_utf8(entry.cp);
      result.kana_to_src_index.push_back(static_cast<int>(entry.orig_idx));
      result.confidences.push_back(1.0f);
      continue;
    }

    // 整数スコアを 0 で初期化して CSC アクセスで累積
#ifdef MOMO_TRACE
    _t0 = TRACE_NOW();
#endif
    std::fill(int_scores.begin(), int_scores.end(), 0);
#ifdef MOMO_TRACE
    {
      auto _t1 = TRACE_NOW();
      tr_fill += TRACE_US(_t0, _t1);
      _t0 = _t1;
    }
#endif
    compute_read_scores(all_feat_ids[i], int_scores);
#ifdef MOMO_TRACE
    {
      auto _t1 = TRACE_NOW();
      tr_read_scores += TRACE_US(_t0, _t1);
      _t0 = _t1;
    }
#endif

    // scale と intercept を一括適用して float スコアに変換
    const float scale = model_.read_scale;
    for (uint32_t cls = 0; cls < n_cls; ++cls) {
      scores[cls] = model_.intercept_read[cls] + int_scores[cls] * scale;
    }

    // argmax
    const int best_cls = static_cast<int>(std::max_element(scores.begin(), scores.end()) - scores.begin());

    const float conf = read_confidence(scores, best_cls);
    const std::string& label = model_.read_classes[best_cls];
#ifdef MOMO_TRACE
    {
      auto _t1 = TRACE_NOW();
      tr_scale += TRACE_US(_t0, _t1);
      _t0 = _t1;
    }
#endif

    if (label == "---" || label == "_") continue;

    // 境界判定
    const bool has_split = sigmoid(compute_boundary_score(all_feat_ids[i])) >= 0.5f;
#ifdef MOMO_TRACE
    {
      auto _t1 = TRACE_NOW();
      tr_boundary += TRACE_US(_t0, _t1);
      _t0 = _t1;
    }
#endif

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

#ifdef MOMO_TRACE
  const long long tr_total =
      tr_utf8 + tr_seq + tr_feat_key + tr_vocab + tr_fill + tr_read_scores + tr_scale + tr_boundary;
  std::fprintf(stderr,
               "[trace] predict n=%d  total=%lld ns\n"
               "  utf8_to_utf32      %6lld ns\n"
               "  to_source_seq      %6lld ns\n"
               "  compute_feat_keys  %6lld ns\n"
               "  vocab lookup       %6lld ns\n"
               "  fill int_scores    %6lld ns\n"
               "  compute_read_scores%6lld ns\n"
               "  scale+argmax+conf  %6lld ns\n"
               "  boundary           %6lld ns\n",
               n, tr_total, tr_utf8, tr_seq, tr_feat_key, tr_vocab, tr_fill, tr_read_scores, tr_scale, tr_boundary);
#endif

  return result;
}
