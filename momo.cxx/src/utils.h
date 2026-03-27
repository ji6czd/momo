#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace momo {

// ---------------------------------------------------------------------------
// CharType
// ---------------------------------------------------------------------------

enum class CharType {
  SPACE,
  NUM,
  SYMBOL,
  HIRAGANA,
  KATAKANA,
  KANJI,
  JAPANESE_NUMERIC,
  ALPHA,
  OTHER,
};

const char* char_type_to_str(CharType ct);

// ---------------------------------------------------------------------------
// UTF-8 <-> char32_t 変換
// ---------------------------------------------------------------------------

/// UTF-8 文字列を char32_t 列に変換する。
/// 不正なバイト列はスキップする。
std::u32string utf8_to_utf32(const std::string& utf8);

/// char32_t 列を UTF-8 文字列に変換する。
std::string utf32_to_utf8(const std::u32string& utf32);

/// char32_t 1文字を UTF-8 文字列に変換する。
std::string char32_to_utf8(char32_t cp);

// ---------------------------------------------------------------------------
// 文字種判定
// ---------------------------------------------------------------------------

/// 1文字を受け取り、基本カテゴリを返す。
/// SPACE / NUM / SYMBOL の判定は行わない（get_char_type() が担当）。
CharType get_basic_char_category(char32_t cp);

/// 文字種を判定する。
/// 位取り文字（十百千万億兆）の JAPANESE_NUMERIC への昇格は
/// get_units() の文脈判定で行われるため、この関数では KANJI を返す。
CharType get_char_type(char32_t cp);

// ---------------------------------------------------------------------------
// カナ変換・母音判定
// ---------------------------------------------------------------------------

/// ひらがなならカタカナに変換して返す。それ以外はそのまま返す。
char32_t convert_to_katakana(char32_t cp);

/// char（カナ1文字）が vowel（母音）を含むか判定する。
/// ひらがな・カタカナ両対応。引数は同じ種類の文字であること。
/// ン・ッ など母音を持たない文字は false を返す。
/// 引数がカナ以外、またはひらがな/カタカナ混在の場合は std::invalid_argument を投げる。
bool has_vowel(char32_t char_cp, char32_t vowel_cp);

}  // namespace momo
