#pragma once
#include <cstdint>
#include <vector>

#include "char_type.hpp"
#include "model.hpp"

// ============================================================
// ソース文字エントリ
// Python側の SourceEntry = (char, orig_idx, ctype) と対応
// ============================================================

struct SourceEntry {
  char32_t cp;        // コードポイント
  uint32_t orig_idx;  // 原文中のバイト位置
  CharType ctype;
};

// ============================================================
// 特徴量計算
// ============================================================

// SourceEntry 列から各文字の FeatureKey 列を計算する。
// 戻り値: features[i] = i番目の文字に対応する FeatureKey のリスト
std::vector<std::vector<FeatureKey>> compute_source_features(const std::vector<SourceEntry>& source_seq);

// テキスト（char32_t列）を SourceEntry 列に変換する。
// 位取り文字（十百千万億兆）の JAPANESE_NUMERIC 昇格もここで行う。
// Python側の get_units() + _preprocess_text() の推論時パスと対応。
std::vector<SourceEntry> to_source_seq(const std::vector<char32_t>& text);
