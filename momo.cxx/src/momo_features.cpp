#include "momo_features.hpp"
#include "utf8.hpp"
#include <array>

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

std::vector<SourceEntry> to_source_seq(const std::vector<char32_t>& text) {
    const int n = static_cast<int>(text.size());

    std::vector<CharType> ctypes(n);
    for (int i = 0; i < n; ++i) {
        ctypes[i] = get_char_type(text[i]);
    }

    // 位取り文字の JAPANESE_NUMERIC 昇格（左→右パス）
    for (int i = 1; i < n; ++i) {
        if (is_kurai_char(text[i])
            && ctypes[i - 1] == CharType::JAPANESE_NUMERIC) {
            ctypes[i] = CharType::JAPANESE_NUMERIC;
        }
    }
    // 右→左パス
    for (int i = n - 2; i >= 0; --i) {
        if (is_kurai_char(text[i])
            && ctypes[i + 1] == CharType::JAPANESE_NUMERIC) {
            ctypes[i] = CharType::JAPANESE_NUMERIC;
        }
    }

    std::vector<SourceEntry> seq;
    seq.reserve(n);
    for (int i = 0; i < n; ++i) {
        seq.push_back({text[i], static_cast<uint32_t>(i), ctypes[i]});
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

static uint8_t clamp_run(int run) {
    return static_cast<uint8_t>(run <= 4 ? run : 5);
}

static FeatureKey make_bias()            { return {FeatureType::BIAS}; }
static FeatureKey make_bos()             { return {FeatureType::BOS}; }
static FeatureKey make_eos()             { return {FeatureType::EOS}; }
static FeatureKey make_kanji_pos_first() { return {FeatureType::KANJI_POS_FIRST}; }

static FeatureKey make_type(FeatureType ft, CharType ct) {
    FeatureKey k; k.type = ft; k.ct[0] = ct; return k;
}

static FeatureKey make_type_transition(CharType prev, CharType cur) {
    FeatureKey k;
    k.type  = FeatureType::TYPE_TRANSITION;
    k.ct[0] = prev; k.ct[1] = cur;
    return k;
}

static FeatureKey make_type_tri(FeatureType ft, CharType a, CharType b, CharType c) {
    FeatureKey k;
    k.type  = ft;
    k.ct[0] = a; k.ct[1] = b; k.ct[2] = c;
    return k;
}

static FeatureKey make_char(FeatureType ft, char32_t cp) {
    FeatureKey k; k.type = ft; k.cp[0] = cp; return k;
}

static FeatureKey make_bigram(FeatureType ft, char32_t a, char32_t b) {
    FeatureKey k; k.type = ft; k.cp[0] = a; k.cp[1] = b; return k;
}

static FeatureKey make_trigram(FeatureType ft, char32_t a, char32_t b, char32_t c) {
    FeatureKey k; k.type = ft; k.cp[0] = a; k.cp[1] = b; k.cp[2] = c; return k;
}

static FeatureKey make_run(FeatureType ft, uint8_t run) {
    FeatureKey k; k.type = ft; k.u8val = run; return k;
}

std::vector<std::vector<FeatureKey>>
compute_source_features(const std::vector<SourceEntry>& seq) {
    const int n = static_cast<int>(seq.size());
    std::vector<std::vector<FeatureKey>> result(n);

    for (int i = 0; i < n; ++i) {
        const char32_t c     = seq[i].cp;
        const CharType ctype = seq[i].ctype;

        const char32_t  prev_c      = (i > 0)   ? seq[i-1].cp    : 0;
        const CharType  prev_ctype  = (i > 0)   ? seq[i-1].ctype : CharType::OTHER;
        const char32_t  prev2_c     = (i > 1)   ? seq[i-2].cp    : 0;
        const CharType  prev2_ctype = (i > 1)   ? seq[i-2].ctype : CharType::OTHER;
        const char32_t  next_c      = (i < n-1) ? seq[i+1].cp    : 0;
        const CharType  next_ctype  = (i < n-1) ? seq[i+1].ctype : CharType::OTHER;
        const char32_t  next2_c     = (i < n-2) ? seq[i+2].cp    : 0;
        const CharType  next2_ctype = (i < n-2) ? seq[i+2].ctype : CharType::OTHER;

        auto& feats = result[i];

        // bias, char=X, type=T
        feats.push_back(make_bias());
        feats.push_back(make_char(FeatureType::CHAR_SELF, c));
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
                feats.push_back(make_trigram(FeatureType::TRIGRAM_PREV2_PREV1_SELF,
                                             prev2_c, prev_c, c));
                feats.push_back(make_type_tri(FeatureType::TYPE_TRI_PREV2_PREV1_SELF,
                                              prev2_ctype, prev_ctype, ctype));
            }
        } else {
            feats.push_back(make_bos());
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
                feats.push_back(make_trigram(FeatureType::TRIGRAM_SELF_NEXT1_NEXT2,
                                             c, next_c, next2_c));
                feats.push_back(make_type_tri(FeatureType::TYPE_TRI_SELF_NEXT1_NEXT2,
                                              ctype, next_ctype, next2_ctype));
            }
        } else {
            feats.push_back(make_eos());
        }

        // trigram: 前1-対象-後1
        if (i > 0 && i < n - 1) {
            feats.push_back(make_trigram(FeatureType::TRIGRAM_PREV1_SELF_NEXT1,
                                         prev_c, c, next_c));
            feats.push_back(make_type_tri(FeatureType::TYPE_TRI_PREV1_SELF_NEXT1,
                                          prev_ctype, ctype, next_ctype));
        }

        // 漢字連続長
        if (ctype == CharType::KANJI) {
            const int run = kanji_run_length(seq, i);
            feats.push_back(make_run(FeatureType::KANJI_RUN_LEN, clamp_run(run)));

            if (i == 0 || seq[i-1].ctype != CharType::KANJI) {
                feats.push_back(make_kanji_pos_first());
            }

            if (i > 0 && seq[i-1].ctype == CharType::JAPANESE_NUMERIC) {
                int num_run = 0;
                for (int j = i - 1; j >= 0 && seq[j].ctype == CharType::JAPANESE_NUMERIC; --j) {
                    ++num_run;
                }
                feats.push_back(make_run(FeatureType::PREV_JAPANESE_NUMERIC_RUN_LEN,
                                         clamp_run(num_run)));
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
