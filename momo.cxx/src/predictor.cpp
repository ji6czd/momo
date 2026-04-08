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

// 小書き仮名セット（Python の _SMALL_KANA と対応）
static const char32_t SMALL_KANA_LIST[] = {
  // 小書きひらがな
  U'\u3041', U'\u3043', U'\u3045', U'\u3047', U'\u3049',  // ぁぃぅぇぉ
  U'\u3063',                                               // っ
  U'\u3083', U'\u3085', U'\u3087',                         // ゃゅょ
  // 小書きカタカナ
  U'\u30A1', U'\u30A3', U'\u30A5', U'\u30A7', U'\u30A9',  // ァィゥェォ
  U'\u30C3',                                               // ッ
  U'\u30E3', U'\u30E5', U'\u30E7',                         // ャュョ
};

static bool is_small_kana(char32_t cp) {
  for (auto c : SMALL_KANA_LIST) {
    if (cp == c) return true;
  }
  return false;
}

// 小書きひらがな → 対応するカタカナ（+0x60）、カタカナはそのまま
// Python: chr(ord(char) + 0x60) if "ぁ" <= char <= "ん" else char
static char32_t small_kana_to_kana(char32_t cp) {
  if (cp >= U'\u3041' && cp <= U'\u3093') return cp + 0x60;
  return cp;
}

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

  // 救済ロジック用の親追跡（Python の parent_idx 相当）
  int parent_src_idx = -1;            // source_seq 上の親インデックス（-1 = なし）
  bool parent_is_bypass = false;      // 親が bypass 文字か
  bool parent_has_small_kana = false; // 親ラベルに小書き仮名を追記済みか
  size_t parent_kana_char_end = 0;    // 親ラベル出力後の kana_to_src_index サイズ
  size_t parent_kana_byte_end = 0;    // 親ラベル出力後の kana_text バイトサイズ

  for (int i = 0; i < n; ++i) {
    const auto& entry = source_seq[i];

    if (is_bypass(entry.ctype)) {
      const std::string ch_utf8 = char32_to_utf8(entry.cp);
      result.kana_text += ch_utf8;
      result.kana_to_src_index.push_back(static_cast<int>(entry.orig_idx));
      result.confidences.push_back(1.0f);
      parent_src_idx = i;
      parent_is_bypass = true;
      parent_has_small_kana = false;
      parent_kana_char_end = result.kana_to_src_index.size();
      parent_kana_byte_end = result.kana_text.size();
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

    // --------------------------------------------------------
    // 救済ロジック（Python の _refine_predictions と対応）
    // --------------------------------------------------------

    if (label == "_") {  // LABEL_CONTINUE
      // 1. 孤立 CONTINUE 救済：有効な親がない場合は文字そのままを出力
      //    Python: label == LABEL_CONTINUE and (parent_idx == -1 or parent_idx in bypass_indices)
      if (parent_src_idx < 0 || parent_is_bypass) {
        const std::string ch_utf8 = char32_to_utf8(entry.cp);
        result.kana_text += ch_utf8;
        for (size_t j = 0; j < ch_utf8.size(); ++j) {
          result.kana_to_src_index.push_back(static_cast<int>(entry.orig_idx));
          result.confidences.push_back(conf);
        }
        parent_src_idx = i;
        parent_is_bypass = false;
        parent_has_small_kana = false;
        parent_kana_char_end = result.kana_to_src_index.size();
        parent_kana_byte_end = result.kana_text.size();
#ifdef MOMO_TRACE
        _t0 = TRACE_NOW();
#endif
        const bool has_split_rescue = sigmoid(compute_boundary_score(all_feat_ids[i])) >= 0.5f;
#ifdef MOMO_TRACE
        { auto _t1 = TRACE_NOW(); tr_boundary += TRACE_US(_t0, _t1); }
#endif
        if (has_split_rescue) {
          result.kana_text += ' ';
          result.kana_to_src_index.push_back(static_cast<int>(entry.orig_idx));
          result.confidences.push_back(conf);
        }
        continue;
      }

      // 2. NUMERIC + CONTINUE 救済：文字そのままを出力
      //    Python: if clean_label in (LABEL_SKIP, LABEL_CONTINUE) and ctype == CharType.NUMERIC
      if (entry.ctype == CharType::NUMERIC) {
        const std::string ch_utf8 = char32_to_utf8(entry.cp);
        result.kana_text += ch_utf8;
        for (size_t j = 0; j < ch_utf8.size(); ++j) {
          result.kana_to_src_index.push_back(static_cast<int>(entry.orig_idx));
          result.confidences.push_back(conf);
        }
        continue;
      }

      // 3. 小書き仮名 + CONTINUE 救済：親ラベルに追記
      //    Python: elif clean_label == LABEL_CONTINUE and char in _SMALL_KANA
      //            if parent_idx >= 0 and _SMALL_KANA.isdisjoint(refined_labels[parent_idx])
      if (is_small_kana(entry.cp) && !parent_has_small_kana) {
        const char32_t kana_char = small_kana_to_kana(entry.cp);
        const std::string kana_utf8 = char32_to_utf8(kana_char);
        result.kana_text.insert(parent_kana_byte_end, kana_utf8);
        result.kana_to_src_index.insert(
          result.kana_to_src_index.begin() + static_cast<std::ptrdiff_t>(parent_kana_char_end),
          static_cast<int>(source_seq[parent_src_idx].orig_idx));
        result.confidences.insert(
          result.confidences.begin() + static_cast<std::ptrdiff_t>(parent_kana_char_end),
          conf);
        parent_kana_char_end++;
        parent_kana_byte_end += kana_utf8.size();
        parent_has_small_kana = true;
      }
      // それ以外の CONTINUE はスキップ（処理済みとして親情報は更新しない）
      continue;
    }

    if (label == "---") {  // LABEL_SKIP
      // NUMERIC + SKIP 救済：文字そのままを出力
      //    Python: if clean_label in (LABEL_SKIP, LABEL_CONTINUE) and ctype == CharType.NUMERIC
      if (entry.ctype == CharType::NUMERIC) {
        const std::string ch_utf8 = char32_to_utf8(entry.cp);
        result.kana_text += ch_utf8;
        for (size_t j = 0; j < ch_utf8.size(); ++j) {
          result.kana_to_src_index.push_back(static_cast<int>(entry.orig_idx));
          result.confidences.push_back(conf);
        }
      }
      continue;
    }

    // --------------------------------------------------------
    // 通常ラベル出力
    // --------------------------------------------------------

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

    // 親情報を更新（境界スペース前）
    parent_src_idx = i;
    parent_is_bypass = false;
    parent_has_small_kana = false;
    parent_kana_char_end = result.kana_to_src_index.size();
    parent_kana_byte_end = result.kana_text.size();

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
