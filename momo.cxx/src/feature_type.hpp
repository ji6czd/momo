#pragma once
#include <cstdint>

#include "char_type.hpp"

// ============================================================
// FeatureType — 特徴量種別列挙型
//
// bit7-6 : ペイロード種別
//   0b00 (0x0_) = なし
//   0b01 (0x4_-0x7_) = CharType × N
//   0b10 (0x8_-0xB_) = char32_t × N
//   0b11 (0xC_) = uint8 × 1
//
// bit5-4 : 個数
//   0b00 = 0個（またはuint8×1固定）
//   0b01 = 1個
//   0b10 = 2個
//   0b11 = 3個
//
// Python側の exporter.py FT クラスと対応する
// ============================================================

enum class FeatureType : uint8_t {
  // 0b00'00'xxxx  ペイロードなし
  BIAS = 0x00,
  KANJI_POS_FIRST = 0x01,

  // 0b01'01'xxxx  CharType×1
  TYPE_SELF = 0x50,   // type=T
  TYPE_PREV1 = 0x51,  // -1:type=T
  TYPE_PREV2 = 0x52,  // -2:type=T
  TYPE_NEXT1 = 0x53,  // +1:type=T
  TYPE_NEXT2 = 0x54,  // +2:type=T
  TYPE_PREV3 = 0x55,  // -3:type=T
  TYPE_NEXT3 = 0x56,  // +3:type=T

  // 0b01'10'xxxx  CharType×2
  TYPE_TRANSITION = 0x60,  // type_transition=T->T  (前1→対象)

  // 0b01'11'xxxx  CharType×3
  TYPE_TRI_PREV2_PREV1_SELF = 0x70,   // type_tri=T-T-T  (前2-前1-対象)
  TYPE_TRI_PREV1_SELF_NEXT1 = 0x71,   // type_tri=T-T-T  (前1-対象-後1)
  TYPE_TRI_SELF_NEXT1_NEXT2 = 0x72,   // type_tri=T-T-T  (対象-後1-後2)
  TYPE_TRI_PREV3_PREV2_PREV1 = 0x73,  // type_tri=T-T-T  (前3-前2-前1)
  TYPE_TRI_NEXT1_NEXT2_NEXT3 = 0x74,  // type_tri=T-T-T  (後1-後2-後3)

  // 0b10'01'xxxx  char32_t×1
  CHAR_SELF = 0x90,   // char=X
  CHAR_PREV1 = 0x91,  // -1:char=X
  CHAR_PREV2 = 0x92,  // -2:char=X
  CHAR_NEXT1 = 0x93,  // +1:char=X
  CHAR_NEXT2 = 0x94,  // +2:char=X
  CHAR_PREV3 = 0x95,  // -3:char=X
  CHAR_NEXT3 = 0x96,  // +3:char=X

  // 0b10'10'xxxx  char32_t×2
  BIGRAM_PREV1_SELF = 0xA0,     // -1:bi=XY
  BIGRAM_PREV2_PREV1 = 0xA1,    // -2:-1:bi=XY
  BIGRAM_SELF_NEXT1 = 0xA2,     // +1:bi=XY
  BIGRAM_NEXT1_NEXT2 = 0xA3,    // +1:+2:bi=XY
  BIGRAM_PREV3_PREV2 = 0xA4,    // -3:-2:bi=XY
  BIGRAM_NEXT2_NEXT3 = 0xA5,    // +2:+3:bi=XY
  CHAR_SELF_COMPOUND_2 = 0xA6,  // char32_t×2: 2文字複合ユニットの char_s

  // 0b10'11'xxxx  char32_t×3
  TRIGRAM_PREV2_PREV1_SELF = 0xB0,   // tri=XYZ  (前2-前1-対象)
  TRIGRAM_PREV1_SELF_NEXT1 = 0xB1,   // tri=XYZ  (前1-対象-後1)
  TRIGRAM_SELF_NEXT1_NEXT2 = 0xB2,   // tri=XYZ  (対象-後1-後2)
  TRIGRAM_PREV3_PREV2_PREV1 = 0xB3,  // tri=XYZ  (前3-前2-前1)
  TRIGRAM_NEXT1_NEXT2_NEXT3 = 0xB4,  // tri=XYZ  (後1-後2-後3)
  CHAR_SELF_COMPOUND_3 = 0xB5,       // char32_t×3: 3文字複合ユニットの char_s

  // 0b11'00'xxxx  uint8×1
  KANJI_RUN_LEN = 0xC0,                  // kanji_run_len=N
  JAPANESE_NUMERIC_RUN_LEN = 0xC1,       // japanese_numeric_run_len=N
  PREV_JAPANESE_NUMERIC_RUN_LEN = 0xC2,  // -1:japanese_numeric_run_len=N
};

// ペイロードの CharType 個数
inline int feature_chartype_count(FeatureType ft) {
  const uint8_t v = static_cast<uint8_t>(ft);
  if ((v & 0xC0) != 0x40) return 0;
  return (v >> 4) & 0x03;
}

// ペイロードの char32_t 個数
inline int feature_char32_count(FeatureType ft) {
  const uint8_t v = static_cast<uint8_t>(ft);
  if ((v & 0xC0) != 0x80) return 0;
  return (v >> 4) & 0x03;
}

// ペイロードが uint8×1 かどうか
inline bool feature_is_uint8_payload(FeatureType ft) { return (static_cast<uint8_t>(ft) & 0xC0) == 0xC0; }
