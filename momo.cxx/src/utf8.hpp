#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "char_type.hpp"

// ============================================================
// UTF-8 ↔ char32_t 変換、および文字種判定
// ============================================================

// UTF-8 文字列を char32_t 列に変換する。
// 不正なバイト列は U+FFFD (REPLACEMENT CHARACTER) に置き換える。
std::vector<char32_t> utf8_to_utf32(const std::string& utf8);

// char32_t 列を UTF-8 文字列に変換する。
std::string utf32_to_utf8(const std::vector<char32_t>& utf32);

// char32_t 1文字を UTF-8 文字列に変換する。
std::string char32_to_utf8(char32_t cp);

// 文字種判定
// Python側の get_char_type() と対応する（utils.py 参照）
// 位取り文字（十百千万億兆）の JAPANESE_NUMERIC への昇格は
// get_units() 相当の文脈判定で行われるため、この関数では KANJI を返す。
CharType get_char_type(char32_t cp);
