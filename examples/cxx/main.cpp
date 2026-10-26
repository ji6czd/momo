#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "momo.h"

// FFI の文字列はすべて UTF-16。出力は Windows ではコンソールを UTF-16
// モードにして wcout へ、それ以外では UTF-8 に変換して cout へ流す。
#ifdef _WIN32
void set_console() {
  // 標準出力をUTF-16に設定
  _setmode(_fileno(stdout), _O_U16TEXT);
}

void put(const std::u16string& s) {
  std::wcout << reinterpret_cast<const wchar_t*>(s.c_str());
}
#else
void set_console() {}

void put(const std::u16string& s) {
  std::string utf8;
  for (size_t i = 0; i < s.size(); ++i) {
    char32_t cp = s[i];
    // サロゲートペアを 1 コードポイントに戻す
    if (cp >= 0xD800 && cp < 0xDC00 && i + 1 < s.size() && s[i + 1] >= 0xDC00 &&
        s[i + 1] < 0xE000) {
      cp = 0x10000 + ((cp - 0xD800) << 10) + (s[i + 1] - 0xDC00);
      ++i;
    }
    if (cp < 0x80) {
      utf8 += static_cast<char>(cp);
    } else if (cp < 0x800) {
      utf8 += static_cast<char>(0xC0 | (cp >> 6));
      utf8 += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      utf8 += static_cast<char>(0xE0 | (cp >> 12));
      utf8 += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      utf8 += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      utf8 += static_cast<char>(0xF0 | (cp >> 18));
      utf8 += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      utf8 += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      utf8 += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }
  std::cout << utf8;
}
#endif

void put_int(int n) {
  std::u16string s;
  for (char c : std::to_string(n)) {
    s += static_cast<char16_t>(c);
  }
  put(s);
}

void put_endl() { put(u"\n"); }

const uint16_t* model_path = (const uint16_t*)u"model.mbm";
const uint16_t* src_text = (const uint16_t*)u"日本語のテキストを入力します。";

int main() {
  set_console();
  PredictorHandle* predictor = momo_predictor_new_w(model_path, nullptr);
  if (!predictor) {
    std::cerr << "Failed to create predictor." << std::endl;
    return 1;
  }
  // 組み込みテーブルの点訳器（日本語は既定テーブル、英語行は UEB）
  BrailleTranslatorHandle* translator = momo_braille_translator_from_embedded();
  if (!translator) {
    std::cerr << "Failed to create translator." << std::endl;
    momo_predictor_free(predictor);
    return 1;
  }
  put(u"Predictor created successfully.");
  put_endl();

  // 一時バッファサイズ
  constexpr int len = 1024;
  std::vector<uint16_t> kana_buffer(len);
  std::vector<uint16_t> braille_buffer(len);

  BrailleResultHandle* result = momo_braille_translator_translate_w(translator, src_text, predictor);
  if (result != nullptr) {
    // 読み（かな）を取得
    int kana_len = momo_braille_result_reading_text_w(result, kana_buffer.data(), static_cast<int>(kana_buffer.size()));
    if (kana_len > static_cast<int>(kana_buffer.size())) {
      kana_buffer.resize(kana_len);
      momo_braille_result_reading_text_w(result, kana_buffer.data(), kana_len);
    }

    // 点字テキストを取得
    int braille_len = momo_braille_result_braille_text_w(result, braille_buffer.data(), static_cast<int>(braille_buffer.size()));
    if (braille_len > static_cast<int>(braille_buffer.size())) {
      braille_buffer.resize(braille_len);
      momo_braille_result_braille_text_w(result, braille_buffer.data(), braille_len);
    }

    put(reinterpret_cast<const char16_t*>(kana_buffer.data()));
    put_endl();
    put(reinterpret_cast<const char16_t*>(braille_buffer.data()));
    put_endl();
    // インデックス表示準備
    int brl_count = momo_braille_result_braille_char_count(result);
    int src_count = momo_braille_result_source_char_count(result);

    // ソース文字ごとの点字位置を表示
    std::vector<int32_t> s2b_row(src_count + 1);
    std::vector<int32_t> s2b_col(brl_count);
    momo_braille_result_source_to_braille(result, s2b_row.data(), s2b_col.data());

    put(u"--- src → braille ---");
    put_endl();
    for (int i = 0; i < src_count; ++i) {
      // ソース文字（BMP なので uint16_t 1 つ = 1 コードポイント）
      int start = s2b_row[i];
      int end = s2b_row[i + 1];

      put(std::u16string(1, static_cast<char16_t>(src_text[i])));
      put(u": ");
      if (start == end) {
        put(u"(なし)");
      } else {
        // 対応する点字セルを列挙
        for (int j = start; j < end; ++j) {
          put(std::u16string(1, static_cast<char16_t>(braille_buffer[s2b_col[j]])));
        }
        // インデックス範囲
        put(u"  [");
        put_int(s2b_col[start]);
        if (end - start > 1) {
          put(u"..");
          put_int(s2b_col[end - 1]);
        }
        put(u"]");
      }
      put_endl();
    }

    // 点字文字ごとのソース文字位置を表示
    std::vector<int32_t> b2s(brl_count);
    momo_braille_result_braille_to_source(result, b2s.data());
    put(u"--- braille → src ---");
    put_endl();
    for (int i = 0; i < brl_count; ++i) {
      put(std::u16string(1, static_cast<char16_t>(braille_buffer[i])));
      put(u": ");
      put_int(b2s[i]);
      put_endl();
    }
  }

  momo_braille_result_free(result);
  momo_braille_translator_free(translator);
  momo_predictor_free(predictor);
  return 0;
}
