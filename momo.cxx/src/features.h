#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "utils.h"

namespace momo {

// ---------------------------------------------------------------------------
// 型定義
// ---------------------------------------------------------------------------

/// ソース文字系列の1要素。
/// Python の Tuple[str, int, str] = (文字列, 原文開始位置, 文字種) に対応。
struct SourceUnit {
    std::u32string text;     ///< 文字列（1文字または複数文字ブロック）
    int            orig_idx; ///< 原文中の開始バイト位置（UTF-32インデックス）
    CharType       ctype;    ///< 文字種
};

/// CRF特徴量。pycrfsuite ネイティブの { "feature_name=value": 1.0 } 形式。
using FeatureDict = std::unordered_map<std::string, float>;

// ---------------------------------------------------------------------------
// 定数
// ---------------------------------------------------------------------------

constexpr const char* MORA_SPLIT     = "+S";
constexpr const char* LABEL_CONTINUE = "---";
constexpr const char* LABEL_SKIP     = "_";

// ---------------------------------------------------------------------------
// 関数宣言
// ---------------------------------------------------------------------------

/// テキストを音節ユニットに分解し、SourceUnit のリストを返す。
///
/// ブロック化ルール（優先順位順）:
///   1. [...]  ブラケットブロック
///   2. かな＋小書きかなの連続
///   3. ASCII印字文字（0x21-0x7E）＋スペース/タブの連続
///   4. 空白の連続
///   5. それ以外の1文字
///
/// JAPANESE_NUMERIC の位取り文字（十百千万億兆）は左右の文脈で昇格する。
std::vector<SourceUnit> get_units(const std::u32string& text);

/// ソース文字系列全体に対して、各文字の文脈特徴量を一括計算する。
std::vector<FeatureDict> compute_source_features(const std::vector<SourceUnit>& source_seq);

} // namespace momo
