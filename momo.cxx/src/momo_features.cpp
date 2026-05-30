#include "momo_features.hpp"

#include <array>

#include "utf8.hpp"

// ============================================================
// 小書き仮名判定（拗音複合ユニット検出に使用）
// Python側の _SMALL_KANA と対応
// ============================================================

static bool is_small_kana_cp(char32_t cp) {
  static const char32_t list[] = {
      U'ぁ', U'ぃ', U'ぅ', U'ぇ', U'ぉ',  // ぁぃぅぇぉ
      U'っ',                                               // っ
      U'ゃ', U'ゅ', U'ょ',                         // ゃゅょ
      U'ゎ',                                               // ゎ
      U'ァ', U'ィ', U'ゥ', U'ェ', U'ォ',  // ァィゥェォ
      U'ッ',                                               // ッ
      U'ャ', U'ュ', U'ョ',                         // ャュョ
      U'ヮ',                                               // ヮ
  };
  for (char32_t c : list) {
    if (cp == c) return true;
  }
  return false;
}

// ひらがな・カタカナ（小書き含む）かどうか
static bool is_base_kana(char32_t cp) {
  return (cp >= U'ぁ' && cp <= U'ん') ||  // ぁ〜ん
         (cp >= U'ァ' && cp <= U'ヶ');    // ァ〜ヶ
}

// ============================================================
// 位取り文字の判定
// ============================================================

static bool is_kurai_char(char32_t cp) {
  static const char32_t kurai[] = {
      U'十', U'百', U'千', U'万', U'億', U'兆',
  };
  for (char32_t c : kurai) {
    if (cp == c) return true;
  }
  return false;
}

// ============================================================
// テキスト → SourceEntry 列
// ============================================================

std::vector<SourceEntry> to_source_seq(
    const std::vector<char32_t>& text,
    const std::unordered_set<std::string>* compound_units) {
  const int n = static_cast<int>(text.size());

  std::vector<CharType> ctypes(n);
  for (int i = 0; i < n; ++i) {
    ctypes[i] = get_char_type(text[i]);
  }

  // 位取り文字の JAPANESE_NUMERIC 昇格（左→右パス）
  for (int i = 1; i < n; ++i) {
    if (is_kurai_char(text[i]) && ctypes[i - 1] == CharType::JAPANESE_NUMERIC) {
      ctypes[i] = CharType::JAPANESE_NUMERIC;
    }
  }
  // 右→左パス
  for (int i = n - 2; i >= 0; --i) {
    if (is_kurai_char(text[i]) && ctypes[i + 1] == CharType::JAPANESE_NUMERIC) {
      ctypes[i] = CharType::JAPANESE_NUMERIC;
    }
  }

  std::vector<SourceEntry> seq;
  seq.reserve(n);

  for (int i = 0; i < n;) {
    // --- 拗音複合ユニット検出（アルゴリズム） ---
    // ひらがな/カタカナ基底文字 + 小書き仮名
    if (is_base_kana(text[i]) && i + 1 < n && is_small_kana_cp(text[i + 1])) {
      SourceEntry e;
      e.cp = text[i];
      e.cp2 = text[i + 1];
      e.orig_idx = static_cast<uint32_t>(i);
      e.ctype = ctypes[i];
      e.compound_len = 2;
      seq.push_back(e);
      i += 2;
      continue;
    }

    // --- 漢字複合ユニット検出（辞書ベース）---
    if (compound_units && !compound_units->empty()) {
      bool found = false;
      // 最長一致（最大3文字まで）
      for (int len = 3; len >= 2; --len) {
        if (i + len > n) continue;
        // UTF-8 文字列に変換して辞書引き
        std::string candidate;
        for (int k = 0; k < len; ++k) candidate += char32_to_utf8(text[i + k]);
        if (compound_units->count(candidate)) {
          SourceEntry e;
          e.cp = text[i];
          e.cp2 = (len >= 2) ? text[i + 1] : 0;
          e.cp3 = (len >= 3) ? text[i + 2] : 0;
          e.orig_idx = static_cast<uint32_t>(i);
          e.ctype = ctypes[i];
          e.compound_len = static_cast<uint8_t>(len);
          seq.push_back(e);
          i += len;
          found = true;
          break;
        }
      }
      if (found) continue;
    }

    // --- 単一文字 ---
    SourceEntry e;
    e.cp = text[i];
    e.orig_idx = static_cast<uint32_t>(i);
    e.ctype = ctypes[i];
    e.compound_len = 1;
    seq.push_back(e);
    ++i;
  }

  return seq;
}

// ============================================================
// 特徴量計算
// ============================================================

static int kanji_run_length(const std::vector<SourceEntry>& seq, int i) {
  const int n = static_cast<int>(seq.size());
  int run = 1;
  for (int j = i + 1; j < n && seq[j].ctype == CharType::KANJI; ++j) ++run;
  for (int j = i - 1; j >= 0 && seq[j].ctype == CharType::KANJI; --j) ++run;
  return run;
}

static int numeric_run_length(const std::vector<SourceEntry>& seq, int i) {
  const int n = static_cast<int>(seq.size());
  int run = 1;
  for (int j = i + 1; j < n && seq[j].ctype == CharType::JAPANESE_NUMERIC; ++j) ++run;
  for (int j = i - 1; j >= 0 && seq[j].ctype == CharType::JAPANESE_NUMERIC; --j) ++run;
  return run;
}

static uint8_t clamp_run(int run) { return static_cast<uint8_t>(run <= 4 ? run : 5); }

static FeatureKey make_bias() { return {FeatureType::BIAS}; }
static FeatureKey make_kanji_pos_first() { return {FeatureType::KANJI_POS_FIRST}; }

static FeatureKey make_type(FeatureType ft, CharType ct) {
  FeatureKey k;
  k.type = ft;
  k.ct[0] = ct;
  return k;
}

static FeatureKey make_type_transition(CharType prev, CharType cur) {
  FeatureKey k;
  k.type = FeatureType::TYPE_TRANSITION;
  k.ct[0] = prev;
  k.ct[1] = cur;
  return k;
}

static FeatureKey make_type_tri(FeatureType ft, CharType a, CharType b, CharType c) {
  FeatureKey k;
  k.type = ft;
  k.ct[0] = a;
  k.ct[1] = b;
  k.ct[2] = c;
  return k;
}

static FeatureKey make_char(FeatureType ft, char32_t cp) {
  FeatureKey k;
  k.type = ft;
  k.cp[0] = cp;
  return k;
}

static FeatureKey make_bigram(FeatureType ft, char32_t a, char32_t b) {
  FeatureKey k;
  k.type = ft;
  k.cp[0] = a;
  k.cp[1] = b;
  return k;
}

static FeatureKey make_trigram(FeatureType ft, char32_t a, char32_t b, char32_t c) {
  FeatureKey k;
  k.type = ft;
  k.cp[0] = a;
  k.cp[1] = b;
  k.cp[2] = c;
  return k;
}

static FeatureKey make_run(FeatureType ft, uint8_t run) {
  FeatureKey k;
  k.type = ft;
  k.u8val = run;
  return k;
}

// 複合ユニットの CHAR_SELF 特徴量を生成する。
// 単一文字は CHAR_SELF（0x90）、2文字は CHAR_SELF_COMPOUND_2（0xA6）、
// 3文字は CHAR_SELF_COMPOUND_3（0xB5）。
static FeatureKey make_char_self_compound(const SourceEntry& e) {
  if (e.compound_len == 2) {
    return make_bigram(FeatureType::CHAR_SELF_COMPOUND_2, e.cp, e.cp2);
  }
  if (e.compound_len == 3) {
    return make_trigram(FeatureType::CHAR_SELF_COMPOUND_3, e.cp, e.cp2, e.cp3);
  }
  return make_char(FeatureType::CHAR_SELF, e.cp);
}

std::vector<std::vector<FeatureKey>> compute_source_features(const std::vector<SourceEntry>& seq) {
  const int n = static_cast<int>(seq.size());
  std::vector<std::vector<FeatureKey>> result(n);

  for (int i = 0; i < n; ++i) {
    // 文脈特徴量は各エントリの先頭コードポイント（cp）を使用。
    // 複合ユニットでも cp は先頭1文字なので bigram/trigram のサイズは変わらない。
    // CHAR_SELF のみ複合ユニット専用の特徴量型を使用する。
    const char32_t c = seq[i].cp;
    const CharType ctype = seq[i].ctype;

    const char32_t prev_c = (i > 0) ? seq[i - 1].cp : 0;
    const CharType prev_ctype = (i > 0) ? seq[i - 1].ctype : CharType::OTHER;
    const char32_t prev2_c = (i > 1) ? seq[i - 2].cp : 0;
    const CharType prev2_ctype = (i > 1) ? seq[i - 2].ctype : CharType::OTHER;
    const char32_t prev3_c = (i > 2) ? seq[i - 3].cp : 0;
    const CharType prev3_ctype = (i > 2) ? seq[i - 3].ctype : CharType::OTHER;
    const char32_t next_c = (i < n - 1) ? seq[i + 1].cp : 0;
    const CharType next_ctype = (i < n - 1) ? seq[i + 1].ctype : CharType::OTHER;
    const char32_t next2_c = (i < n - 2) ? seq[i + 2].cp : 0;
    const CharType next2_ctype = (i < n - 2) ? seq[i + 2].ctype : CharType::OTHER;
    const char32_t next3_c = (i < n - 3) ? seq[i + 3].cp : 0;
    const CharType next3_ctype = (i < n - 3) ? seq[i + 3].ctype : CharType::OTHER;

    auto& feats = result[i];

    // bias, char_s（複合ユニット対応）, type_s
    feats.push_back(make_bias());
    feats.push_back(make_char_self_compound(seq[i]));
    feats.push_back(make_type(FeatureType::TYPE_SELF, ctype));

    if (i > 0) {
      feats.push_back(make_char(FeatureType::CHAR_PREV1, prev_c));
      feats.push_back(make_type(FeatureType::TYPE_PREV1, prev_ctype));
      feats.push_back(make_bigram(FeatureType::BIGRAM_PREV1_SELF, prev_c, c));
      feats.push_back(make_type_transition(prev_ctype, ctype));

      if (i > 1) {
        feats.push_back(make_char(FeatureType::CHAR_PREV2, prev2_c));
        feats.push_back(make_type(FeatureType::TYPE_PREV2, prev2_ctype));
        feats.push_back(make_bigram(FeatureType::BIGRAM_PREV2_PREV1, prev2_c, prev_c));
        // trigram: 前2-前1-対象
        feats.push_back(make_trigram(FeatureType::TRIGRAM_PREV2_PREV1_SELF, prev2_c, prev_c, c));
        feats.push_back(make_type_tri(FeatureType::TYPE_TRI_PREV2_PREV1_SELF, prev2_ctype, prev_ctype, ctype));

        if (i > 2) {
          feats.push_back(make_char(FeatureType::CHAR_PREV3, prev3_c));
          feats.push_back(make_type(FeatureType::TYPE_PREV3, prev3_ctype));
          feats.push_back(make_bigram(FeatureType::BIGRAM_PREV3_PREV2, prev3_c, prev2_c));
          // trigram: 前3-前2-前1
          feats.push_back(make_trigram(FeatureType::TRIGRAM_PREV3_PREV2_PREV1, prev3_c, prev2_c, prev_c));
          feats.push_back(make_type_tri(FeatureType::TYPE_TRI_PREV3_PREV2_PREV1, prev3_ctype, prev2_ctype, prev_ctype));
        }
      }
    }

    if (i < n - 1) {
      feats.push_back(make_char(FeatureType::CHAR_NEXT1, next_c));
      feats.push_back(make_type(FeatureType::TYPE_NEXT1, next_ctype));
      feats.push_back(make_bigram(FeatureType::BIGRAM_SELF_NEXT1, c, next_c));

      if (i < n - 2) {
        feats.push_back(make_char(FeatureType::CHAR_NEXT2, next2_c));
        feats.push_back(make_type(FeatureType::TYPE_NEXT2, next2_ctype));
        feats.push_back(make_bigram(FeatureType::BIGRAM_NEXT1_NEXT2, next_c, next2_c));
        // trigram: 対象-後1-後2
        feats.push_back(make_trigram(FeatureType::TRIGRAM_SELF_NEXT1_NEXT2, c, next_c, next2_c));
        feats.push_back(make_type_tri(FeatureType::TYPE_TRI_SELF_NEXT1_NEXT2, ctype, next_ctype, next2_ctype));

        if (i < n - 3) {
          feats.push_back(make_char(FeatureType::CHAR_NEXT3, next3_c));
          feats.push_back(make_type(FeatureType::TYPE_NEXT3, next3_ctype));
          feats.push_back(make_bigram(FeatureType::BIGRAM_NEXT2_NEXT3, next2_c, next3_c));
          // trigram: 後1-後2-後3
          feats.push_back(make_trigram(FeatureType::TRIGRAM_NEXT1_NEXT2_NEXT3, next_c, next2_c, next3_c));
          feats.push_back(make_type_tri(FeatureType::TYPE_TRI_NEXT1_NEXT2_NEXT3, next_ctype, next2_ctype, next3_ctype));
        }
      }
    }

    // trigram: 前1-対象-後1
    if (i > 0 && i < n - 1) {
      feats.push_back(make_trigram(FeatureType::TRIGRAM_PREV1_SELF_NEXT1, prev_c, c, next_c));
      feats.push_back(make_type_tri(FeatureType::TYPE_TRI_PREV1_SELF_NEXT1, prev_ctype, ctype, next_ctype));
    }

    // 漢字連続長
    if (ctype == CharType::KANJI) {
      const int run = kanji_run_length(seq, i);
      feats.push_back(make_run(FeatureType::KANJI_RUN_LEN, clamp_run(run)));

      if (i == 0 || seq[i - 1].ctype != CharType::KANJI) {
        feats.push_back(make_kanji_pos_first());
      }

      if (i > 0 && seq[i - 1].ctype == CharType::JAPANESE_NUMERIC) {
        int num_run = 0;
        for (int j = i - 1; j >= 0 && seq[j].ctype == CharType::JAPANESE_NUMERIC; --j) {
          ++num_run;
        }
        feats.push_back(make_run(FeatureType::PREV_JAPANESE_NUMERIC_RUN_LEN, clamp_run(num_run)));
      }
    }

    // 漢数字連続長
    if (ctype == CharType::JAPANESE_NUMERIC) {
      const int run = numeric_run_length(seq, i);
      feats.push_back(make_run(FeatureType::JAPANESE_NUMERIC_RUN_LEN, clamp_run(run)));
    }
  }

  return result;
}
