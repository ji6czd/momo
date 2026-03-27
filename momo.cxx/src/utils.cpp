#include "utils.h"

#include <stdexcept>

namespace momo {

// ---------------------------------------------------------------------------
// CharType → 文字列
// ---------------------------------------------------------------------------

const char* char_type_to_str(CharType ct) {
    switch (ct) {
        case CharType::SPACE:            return "CharType.SPACE";
        case CharType::NUM:              return "CharType.NUM";
        case CharType::SYMBOL:           return "CharType.SYMBOL";
        case CharType::HIRAGANA:         return "CharType.HIRAGANA";
        case CharType::KATAKANA:         return "CharType.KATAKANA";
        case CharType::KANJI:            return "CharType.KANJI";
        case CharType::JAPANESE_NUMERIC: return "CharType.JAPANESE_NUMERIC";
        case CharType::ALPHA:            return "CharType.ALPHA";
        case CharType::OTHER:            return "CharType.OTHER";
    }
    return "CharType.OTHER";
}

// ---------------------------------------------------------------------------
// UTF-8 <-> char32_t 変換
// ---------------------------------------------------------------------------

std::u32string utf8_to_utf32(const std::string& utf8) {
    std::u32string result;
    size_t i = 0;
    while (i < utf8.size()) {
        unsigned char b0 = static_cast<unsigned char>(utf8[i]);
        char32_t cp;
        size_t extra;

        if (b0 < 0x80) {
            cp    = b0;
            extra = 0;
        } else if ((b0 & 0xE0) == 0xC0) {
            cp    = b0 & 0x1F;
            extra = 1;
        } else if ((b0 & 0xF0) == 0xE0) {
            cp    = b0 & 0x0F;
            extra = 2;
        } else if ((b0 & 0xF8) == 0xF0) {
            cp    = b0 & 0x07;
            extra = 3;
        } else {
            // 不正バイトはスキップ
            ++i;
            continue;
        }

        ++i;
        bool valid = true;
        for (size_t j = 0; j < extra; ++j) {
            if (i >= utf8.size() || (static_cast<unsigned char>(utf8[i]) & 0xC0) != 0x80) {
                valid = false;
                break;
            }
            cp = (cp << 6) | (static_cast<unsigned char>(utf8[i]) & 0x3F);
            ++i;
        }
        if (valid) {
            result += cp;
        }
    }
    return result;
}

std::string char32_to_utf8(char32_t cp) {
    std::string result;
    if (cp < 0x80) {
        result += static_cast<char>(cp);
    } else if (cp < 0x800) {
        result += static_cast<char>(0xC0 | (cp >> 6));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        result += static_cast<char>(0xE0 | (cp >> 12));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        result += static_cast<char>(0xF0 | (cp >> 18));
        result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return result;
}

std::string utf32_to_utf8(const std::u32string& utf32) {
    std::string result;
    for (char32_t cp : utf32) {
        result += char32_to_utf8(cp);
    }
    return result;
}

// ---------------------------------------------------------------------------
// 文字種判定
// ---------------------------------------------------------------------------

// 常に JAPANESE_NUMERIC となる漢数字のコードポイント集合
static bool is_japanese_numeric_char(char32_t cp) {
    // 〇一二三四五六七八九
    static const char32_t table[] = {
        U'〇', U'一', U'二', U'三', U'四',
        U'五', U'六', U'七', U'八', U'九',
    };
    for (char32_t c : table) {
        if (cp == c) return true;
    }
    return false;
}

// Unicode の句読点・記号カテゴリに相当する範囲チェック。
// Python の unicodedata.category(c).startswith('P'/'S') の代替。
// 日本語テキスト処理で実際に現れる記号・約物の主要範囲を押さえる。
static bool is_symbol_or_punctuation(char32_t cp) {
    // ASCII 記号（!～/、:～@、[～`、{～~）
    if (cp >= 0x0021 && cp <= 0x002F) return true;
    if (cp >= 0x003A && cp <= 0x0040) return true;
    if (cp >= 0x005B && cp <= 0x0060) return true;
    if (cp >= 0x007B && cp <= 0x007E) return true;
    // 一般句読点・記号（…、—、"" 等）
    if (cp >= 0x2000 && cp <= 0x206F) return true;
    // 通貨記号
    if (cp >= 0x20A0 && cp <= 0x20CF) return true;
    // 矢印・数学記号・その他技術記号
    if (cp >= 0x2190 && cp <= 0x23FF) return true;
    // 囲み英数字・罫線など
    if (cp >= 0x2460 && cp <= 0x27BF) return true;
    // CJK記号・句読点（。、「」…）
    if (cp >= 0x3000 && cp <= 0x303F) return true;
    // 全角記号
    if (cp >= 0xFF01 && cp <= 0xFF0F) return true;
    if (cp >= 0xFF1A && cp <= 0xFF20) return true;
    if (cp >= 0xFF3B && cp <= 0xFF40) return true;
    if (cp >= 0xFF5B && cp <= 0xFF65) return true;
    // 半角カタカナ句読点
    if (cp >= 0xFF61 && cp <= 0xFF65) return true;
    return false;
}

CharType get_basic_char_category(char32_t cp) {
    if (cp >= 0x3040 && cp <= 0x309F) return CharType::HIRAGANA;
    if (cp >= 0x30A0 && cp <= 0x30FF) return CharType::KATAKANA;
    if (is_japanese_numeric_char(cp))  return CharType::JAPANESE_NUMERIC;
    if ((cp >= 0x4E00 && cp <= 0x9FFF) ||
        (cp >= 0x3400 && cp <= 0x4DBF)) return CharType::KANJI;
    if ((cp >= 0x0041 && cp <= 0x005A) ||
        (cp >= 0x0061 && cp <= 0x007A) ||
        (cp >= 0xFF21 && cp <= 0xFF3A) ||
        (cp >= 0xFF41 && cp <= 0xFF5A)) return CharType::ALPHA;
    return CharType::OTHER;
}

CharType get_char_type(char32_t cp) {
    if (cp == 0)                         return CharType::SPACE;
    // 空白文字（スペース・タブ・改行等）
    if (cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r' ||
        cp == 0x3000 /* 全角スペース */) return CharType::SPACE;
    // ASCII数字
    if (cp >= U'0' && cp <= U'9')        return CharType::NUM;
    // 記号・句読点
    if (is_symbol_or_punctuation(cp))    return CharType::SYMBOL;

    return get_basic_char_category(cp);
}

// ---------------------------------------------------------------------------
// カナ変換・母音判定
// ---------------------------------------------------------------------------

char32_t convert_to_katakana(char32_t cp) {
    if (get_basic_char_category(cp) == CharType::HIRAGANA) {
        return cp + 0x60;
    }
    return cp;
}

bool has_vowel(char32_t char_cp, char32_t vowel_cp) {
    CharType char_type  = get_basic_char_category(char_cp);
    CharType vowel_type = get_basic_char_category(vowel_cp);

    if (char_type != CharType::HIRAGANA && char_type != CharType::KATAKANA) {
        throw std::invalid_argument("char はひらがなまたはカタカナでなければなりません");
    }
    if (vowel_type != CharType::HIRAGANA && vowel_type != CharType::KATAKANA) {
        throw std::invalid_argument("vowel はひらがなまたはカタカナでなければなりません");
    }
    if (char_type != vowel_type) {
        throw std::invalid_argument("char と vowel は同じ文字種でなければなりません");
    }

    char32_t char_kata  = convert_to_katakana(char_cp);
    char32_t vowel_kata = convert_to_katakana(vowel_cp);

    static const std::unordered_map<char32_t, char32_t> vowel_map = {
        {U'ア', U'ア'}, {U'イ', U'イ'}, {U'ウ', U'ウ'}, {U'エ', U'エ'}, {U'オ', U'オ'},
        {U'ァ', U'ア'}, {U'ィ', U'イ'}, {U'ゥ', U'ウ'}, {U'ェ', U'エ'}, {U'ォ', U'オ'},
        {U'カ', U'ア'}, {U'キ', U'イ'}, {U'ク', U'ウ'}, {U'ケ', U'エ'}, {U'コ', U'オ'},
        {U'ガ', U'ア'}, {U'ギ', U'イ'}, {U'グ', U'ウ'}, {U'ゲ', U'エ'}, {U'ゴ', U'オ'},
        {U'サ', U'ア'}, {U'シ', U'イ'}, {U'ス', U'ウ'}, {U'セ', U'エ'}, {U'ソ', U'オ'},
        {U'ザ', U'ア'}, {U'ジ', U'イ'}, {U'ズ', U'ウ'}, {U'ゼ', U'エ'}, {U'ゾ', U'オ'},
        {U'タ', U'ア'}, {U'チ', U'イ'}, {U'ツ', U'ウ'}, {U'テ', U'エ'}, {U'ト', U'オ'},
        {U'ダ', U'ア'}, {U'ヂ', U'イ'}, {U'ヅ', U'ウ'}, {U'デ', U'エ'}, {U'ド', U'オ'},
        {U'ナ', U'ア'}, {U'ニ', U'イ'}, {U'ヌ', U'ウ'}, {U'ネ', U'エ'}, {U'ノ', U'オ'},
        {U'ハ', U'ア'}, {U'ヒ', U'イ'}, {U'フ', U'ウ'}, {U'ヘ', U'エ'}, {U'ホ', U'オ'},
        {U'バ', U'ア'}, {U'ビ', U'イ'}, {U'ブ', U'ウ'}, {U'ベ', U'エ'}, {U'ボ', U'オ'},
        {U'パ', U'ア'}, {U'ピ', U'イ'}, {U'プ', U'ウ'}, {U'ペ', U'エ'}, {U'ポ', U'オ'},
        {U'マ', U'ア'}, {U'ミ', U'イ'}, {U'ム', U'ウ'}, {U'メ', U'エ'}, {U'モ', U'オ'},
        {U'ヤ', U'ア'}, {U'ユ', U'ウ'}, {U'ヨ', U'オ'},
        {U'ャ', U'ア'}, {U'ュ', U'ウ'}, {U'ョ', U'オ'},
        {U'ラ', U'ア'}, {U'リ', U'イ'}, {U'ル', U'ウ'}, {U'レ', U'エ'}, {U'ロ', U'オ'},
        {U'ワ', U'ア'}, {U'ヲ', U'オ'},
        {U'ヴ', U'ウ'},
    };

    auto it = vowel_map.find(char_kata);
    if (it == vowel_map.end()) return false;  // ン・ッ など母音なし
    return it->second == vowel_kata;
}
} // namespace momo
