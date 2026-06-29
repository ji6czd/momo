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
  auto result = momo_predict_w(predictor, reinterpret_cast<const uint16_t*>(u"日本語のテキストを入力します。"));
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
    // ソース文字ごとの点字位置を表示
  }
  momo_prediction_free(result);
  momo_predictor_free(predictor);
  return 0;
}
