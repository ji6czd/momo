#include "utf8.hpp"
#include <cstdint>

// ============================================================
// UTF-8 → char32_t
// ============================================================

// 1文字分デコードして cp に書き込み、p を次の文字の先頭に進める。
// 不正なバイト列の場合は U+FFFD を返して1バイト消費する。
static void decode_one(const uint8_t*& p, const uint8_t* end, char32_t& cp) {
    const uint8_t b = *p;

    if (b < 0x80) {
        // 1バイト: 0xxxxxxx
        cp = b;
        ++p;
        return;
    }

    if ((b & 0xE0) == 0xC0) {
        // 2バイト: 110xxxxx 10xxxxxx
        if (end - p < 2 || (p[1] & 0xC0) != 0x80) {
            cp = 0xFFFD; ++p; return;
        }
        cp = ((char32_t)(b & 0x1F) << 6) | (p[1] & 0x3F);
        if (cp < 0x80) { cp = 0xFFFD; ++p; return; }  // 過長エンコード
        p += 2;
        return;
    }

    if ((b & 0xF0) == 0xE0) {
        // 3バイト: 1110xxxx 10xxxxxx 10xxxxxx
        if (end - p < 3 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80) {
            cp = 0xFFFD; ++p; return;
        }
        cp = ((char32_t)(b & 0x0F) << 12)
           | ((char32_t)(p[1] & 0x3F) << 6)
           | (p[2] & 0x3F);
        if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) {
            cp = 0xFFFD; ++p; return;  // 過長エンコード or サロゲート
        }
        p += 3;
        return;
    }

    if ((b & 0xF8) == 0xF0) {
        // 4バイト: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        if (end - p < 4 || (p[1] & 0xC0) != 0x80
                        || (p[2] & 0xC0) != 0x80
                        || (p[3] & 0xC0) != 0x80) {
            cp = 0xFFFD; ++p; return;
        }
        cp = ((char32_t)(b & 0x07) << 18)
           | ((char32_t)(p[1] & 0x3F) << 12)
           | ((char32_t)(p[2] & 0x3F) << 6)
           | (p[3] & 0x3F);
        if (cp < 0x10000 || cp > 0x10FFFF) {
            cp = 0xFFFD; ++p; return;  // 過長エンコード or 範囲外
        }
        p += 4;
        return;
    }

    // 不正バイト
    cp = 0xFFFD;
    ++p;
}

std::vector<char32_t> utf8_to_utf32(const std::string& utf8) {
    std::vector<char32_t> result;
    const uint8_t* p   = reinterpret_cast<const uint8_t*>(utf8.data());
    const uint8_t* end = p + utf8.size();

    while (p < end) {
        char32_t cp;
        decode_one(p, end, cp);
        result.push_back(cp);
    }
    return result;
}

// ============================================================
// char32_t → UTF-8
// ============================================================

std::string char32_to_utf8(char32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp <= 0x10FFFF) {
        s += static_cast<char>(0xF0 | (cp >> 18));
        s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    // 範囲外は無視
    return s;
}

std::string utf32_to_utf8(const std::vector<char32_t>& utf32) {
    std::string result;
    for (char32_t cp : utf32) {
        result += char32_to_utf8(cp);
    }
    return result;
}

// ============================================================
// 文字種判定
// Python側の get_char_type() / get_basic_char_category() と対応
// ============================================================

// 漢数字（常に JAPANESE_NUMERIC）
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

// 記号サブカテゴリ
static CharType symbol_subtype(char32_t cp) {
    // 閉じ括弧類
    static const char32_t close_syms[] = {
        U'」', U'』', U')', U'）', U'】', U'〕', U'}', U'｝',
        U'〉', U'》', U'"', U'\'',
    };
    for (char32_t c : close_syms) { if (cp == c) return CharType::SYMBOL_CLOSE; }

    // 開き括弧類
    static const char32_t open_syms[] = {
        U'「', U'『', U'(', U'（', U'【', U'〔', U'{', U'｛',
        U'〈', U'《',
    };
    for (char32_t c : open_syms) { if (cp == c) return CharType::SYMBOL_OPEN; }

    // 文末記号類
    static const char32_t stop_syms[] = {
        U'。', U'！', U'？', U'.', U'!', U'?',
    };
    for (char32_t c : stop_syms) { if (cp == c) return CharType::SYMBOL_STOP; }

    // 読点・中点類
    static const char32_t pause_syms[] = {
        U'、', U'・', U',',
    };
    for (char32_t c : pause_syms) { if (cp == c) return CharType::SYMBOL_PAUSE; }

    return CharType::SYMBOL;
}

CharType get_char_type(char32_t cp) {
    // 空白
    if (cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r'
        || cp == U'　') {
        return CharType::SPACE;
    }

    // ASCII数字
    if (cp >= U'0' && cp <= U'9') return CharType::NUMERIC;

    // 全角数字
    if (cp >= U'０' && cp <= U'９') return CharType::NUMERIC;

    // ひらがな (U+3041–U+309F)
    if (cp >= 0x3041 && cp <= 0x309F) return CharType::HIRAGANA;

    // カタカナ (U+30A0–U+30FF)
    if (cp >= 0x30A0 && cp <= 0x30FF) return CharType::KATAKANA;

    // 漢数字
    if (is_japanese_numeric_char(cp)) return CharType::JAPANESE_NUMERIC;

    // CJK統合漢字 (U+4E00–U+9FFF) および拡張A (U+3400–U+4DBF)
    if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF)) {
        return CharType::KANJI;
    }

    // ラテン文字（ASCII大文字・小文字）
    if ((cp >= U'A' && cp <= U'Z') || (cp >= U'a' && cp <= U'z')) {
        return CharType::ALPHA;
    }

    // 全角ラテン文字 (U+FF21–U+FF3A, U+FF41–U+FF5A)
    if ((cp >= 0xFF21 && cp <= 0xFF3A) || (cp >= 0xFF41 && cp <= 0xFF5A)) {
        return CharType::ALPHA;
    }

    // 記号類（Unicode の P* / S* カテゴリに相当する主要範囲）
    // ASCII 記号
    if ((cp >= 0x21 && cp <= 0x2F)       // !"#$%&'()*+,-./
     || (cp >= 0x3A && cp <= 0x40)       // :;<=>?@
     || (cp >= 0x5B && cp <= 0x60)       // [\]^_`
     || (cp >= 0x7B && cp <= 0x7E)) {    // {|}~
        return symbol_subtype(cp);
    }

    // CJK記号・句読点 (U+3000–U+303F)
    if (cp >= 0x3000 && cp <= 0x303F) return symbol_subtype(cp);

    // 全角記号 (U+FF01–U+FF0F, U+FF1A–U+FF20, U+FF3B–U+FF40, U+FF5B–U+FF65)
    if ((cp >= 0xFF01 && cp <= 0xFF0F)
     || (cp >= 0xFF1A && cp <= 0xFF20)
     || (cp >= 0xFF3B && cp <= 0xFF40)
     || (cp >= 0xFF5B && cp <= 0xFF65)) {
        return symbol_subtype(cp);
    }

    return CharType::OTHER;
}
