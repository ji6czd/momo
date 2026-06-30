#include <fcntl.h>
#include <io.h>

#include <iostream>
#include <vector>

#include "momo.h"

void set_console() {
  // 標準出力をUTF-16に設定
  _setmode(_fileno(stdout), _O_U16TEXT);
  // 標準エラー出力をUTF-16に設定
  _setmode(_fileno(stderr), _O_U16TEXT);
}

const uint16_t* model_path = (const uint16_t*)u"model.mbm";
const uint16_t* src_text = (const uint16_t*)u"日本語のテキストを入力します。";

int main() {
  set_console();
  PredictorHandle* predictor = momo_predictor_new_w(model_path);
  if (!predictor) {
    std::wcerr << L"Failed to create predictor." << std::endl;
    return 1;
  }
  std::wcout << L"Predictor created successfully." << std::endl;

  // 一時バッファサイズ
  constexpr int len = 1024;
  std::vector<uint16_t> kana_buffer(len);
  std::vector<uint16_t> braille_buffer(len);

  PredictionHandle* result = momo_predict_w(predictor, src_text);
  if (result != nullptr) {
    // かなテキストを取得
    int kana_len = momo_prediction_kana_w(result, kana_buffer.data(), static_cast<int>(kana_buffer.size()));
    if (kana_len > static_cast<int>(kana_buffer.size())) {
      kana_buffer.resize(kana_len);
      momo_prediction_kana_w(result, kana_buffer.data(), kana_len);
    }

    // 点字テキストを取得
    int braille_len = momo_prediction_braille_w(result, braille_buffer.data(), static_cast<int>(braille_buffer.size()));
    if (braille_len > static_cast<int>(braille_buffer.size())) {
      braille_buffer.resize(braille_len);
      momo_prediction_braille_w(result, braille_buffer.data(), braille_len);
    }

    std::wcout << reinterpret_cast<const wchar_t*>(kana_buffer.data()) << std::endl;
    std::wcout << reinterpret_cast<const wchar_t*>(braille_buffer.data()) << std::endl;
    // インデックス表示準備
    int brl_count = momo_prediction_braille_char_count(result);
    int src_count = momo_prediction_src_char_count(result);

    // ソース文字ごとの点字位置を表示
    std::vector<int32_t> s2b_row(src_count + 1);
    std::vector<int32_t> s2b_col(brl_count);
    momo_prediction_src_to_braille(result, s2b_row.data(), s2b_col.data());

    std::wcout << L"--- src → braille ---" << std::endl;
    for (int i = 0; i < src_count; ++i) {
      // ソース文字（BMP なので uint16_t 1 つ = 1 コードポイント）
      wchar_t src_ch[2] = {static_cast<wchar_t>(src_text[i]), L'\0'};

      int start = s2b_row[i];
      int end = s2b_row[i + 1];

      std::wcout << src_ch << L": ";
      if (start == end) {
        std::wcout << L"(なし)";
      } else {
        // 対応する点字セルを列挙
        for (int j = start; j < end; ++j) {
          wchar_t brl_ch[2] = {static_cast<wchar_t>(braille_buffer[s2b_col[j]]), L'\0'};
          std::wcout << brl_ch;
        }
        // インデックス範囲
        std::wcout << L"  [" << s2b_col[start];
        if (end - start > 1) {
          std::wcout << L".." << s2b_col[end - 1];
        }
        std::wcout << L"]";
      }
      std::wcout << std::endl;
    }

    // 点字文字ごとのソース文字位置を表示
    std::vector<int32_t> b2s(brl_count);
    momo_prediction_braille_to_src(result, b2s.data());
    std::wcout << L"--- braille → src ---" << std::endl;
    for (int i = 0; i < brl_count; ++i) {
      wchar_t brl_ch[2] = {static_cast<wchar_t>(braille_buffer[i]), L'\0'};
      std::wcout << brl_ch << L": " << b2s[i] << std::endl;
    }
  }

  momo_prediction_free(result);
  momo_predictor_free(predictor);
  return 0;
}
