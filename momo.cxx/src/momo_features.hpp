#pragma once
#include <cstdint>
#include <vector>

#include "char_type.hpp"
#include "model.hpp"

// ============================================================
// ソース文字エントリ
// Python側の SourceEntry = (char, orig_idx, ctype) と対応
// 複合ユニット（拗音・漢字語）は cp2/cp3 に追加コードポイントを格納。
// ============================================================

struct SourceEntry {
  char32_t cp;              // 先頭コードポイント（後方互換・文脈特徴量に使用）
  char32_t cp2 = 0;         // 2文字目（複合ユニットのみ）
  char32_t cp3 = 0;         // 3文字目（複合ユニットのみ）
  uint32_t orig_idx;        // 原文中の文字インデックス（先頭文字）
  CharType ctype;
  uint8_t compound_len = 1; // ユニットのコードポイント数（通常1、複合は2か3）
};

// ============================================================
// 特徴量計算
// ============================================================

// SourceEntry 列から各文字の FeatureKey 列を計算する。
// 戻り値: features[i] = i番目の文字に対応する FeatureKey のリスト
std::vector<std::vector<FeatureKey>> compute_source_features(const std::vector<SourceEntry>& source_seq);

// テキスト（char32_t列）を SourceEntry 列に変換する。
// 拗音（ひらがな/カタカナ＋小書き仮名）は複合ユニットとして1エントリにまとめる。
// 位取り文字（十百千万億兆）の JAPANESE_NUMERIC 昇格もここで行う。
// Python側の get_units() + _preprocess_text() の推論時パスと対応。
std::vector<SourceEntry> to_source_seq(const std::vector<char32_t>& text);
