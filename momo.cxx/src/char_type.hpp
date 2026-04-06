#pragma once
#include <cstdint>

// ============================================================
// CharType — 文字種列挙型
//
// bit7-4 : カテゴリ
//   0x0_ : 空白系
//   0x1_ : 文字系（ラテン・数字）
//   0x2_ : 予約
//   0x3_ : 記号系
//   0x4_ : 日本語文字系
//   0xF_ : その他・不明
//
// Python側の CharType.value と対応する（utils.py 参照）
// ============================================================

enum class CharType : uint8_t {
    SPACE            = 0x00,

    ALPHA            = 0x10,
    NUMERIC          = 0x11,

    RESERVED         = 0x20,  // 将来拡張用

    SYMBOL           = 0x30,
    SYMBOL_CLOSE     = 0x31,
    SYMBOL_OPEN      = 0x32,
    SYMBOL_STOP      = 0x33,
    SYMBOL_PAUSE     = 0x34,

    HIRAGANA         = 0x40,
    KATAKANA         = 0x41,
    KANJI            = 0x42,
    JAPANESE_NUMERIC = 0x43,

    OTHER            = 0xFF,
};

// カテゴリ取得（上位ニブル）
inline uint8_t chartype_category(CharType ct) {
    return static_cast<uint8_t>(ct) & 0xF0;
}

inline bool is_japanese(CharType ct) {
    return chartype_category(ct) == 0x40;
}

inline bool is_symbol(CharType ct) {
    return chartype_category(ct) == 0x30;
}

inline bool is_latin(CharType ct) {
    return chartype_category(ct) == 0x10;
}
