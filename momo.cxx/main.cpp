#include <iostream>
#include <string>
#include <chrono>

#include "CLI/CLI.hpp"
#include "predictor.hpp"

namespace momo {

/**
 * @param c 判定する文字
 * @return cがUTF-8の先頭バイトであればtrue、そうでなければfalse
 * UTF-8の先頭バイトかどうかを判定する関数
 * UTF-8の先頭バイトは、0xxxxxxx（ASCII）または110xxxxx、1110xxxx、11110xxxのいずれかの形式を持ちます。
 * つまり、先頭バイトは0x00～0x7F、0xC0～0xDF、0xE0～0xEF、0xF0～0xF7の範囲にあります。
 * 続きバイトは0x80～0xBFの範囲にあるため、先頭バイトは0x80以上で0xC0未満の値を持ちません。
 */
bool is_utf8_firstbyte(unsigned char c) { return (c & 0xC0) != 0x80; }

/**
 * @brief モデルを読み込み、標準入力からテキストを受け取って予測を行う関数
 * @param config 予測設定（model_path を含む）
 * @param compute_confidence 自信度スコアを計算するかどうかのフラグ
 * @return 終了コード（0は成功、1はエラー）
 */
int predict(const PredictorConfig& config, bool compute_confidence, bool profile) {
  Predictor predictor(config);
  try {
    predictor.load();
  } catch (const std::exception& e) {
    std::cerr << "モデル読み込みエラー: " << e.what() << "\n";
    return 1;
  }

  // 標準入力から1行ずつ処理
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      std::cout << "\n";
      continue;
    }
    try {
      auto start = std::chrono::high_resolution_clock::now();
      const PredictionResult result = predictor.predict(line);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      if (profile) std::cout << "予測時間: " << elapsed.count() << "秒\n";
      std::cout << result.kana_text << "\n";
      if (compute_confidence) {
        for (size_t i = 0; i < result.confidences.size(); ++i) {
          if (is_utf8_firstbyte(static_cast<unsigned char>(result.kana_text[i]))) {
            std::cout << result.confidences[i] << " ";
          }
        }
        std::cout << "\n";
      }
    } catch (const std::exception& e) {
      std::cerr << "予測エラー: " << e.what() << "\n";
    }
  }

  return 0;
}

}  // namespace momo

int main(int argc, char* argv[]) {
  CLI::App app{"Momo - Japanese Braille Predictor"};
  std::string model_path = "basic_data.mbm";
  float confidence_threshold = 0.3f;
  float numeric_confidence_threshold = 0.5f;
  bool compute_confidence = false;
  bool profile = false;
  app.add_option("--model", model_path, "Path to model file (.mbm)")->default_val("basic_data.mbm");
  app.add_flag("--confidence", compute_confidence, "Compute confidence scores (marginal probabilities)");
  app.add_flag("--profile", profile, "Enable profiling (not implemented yet)")->default_val(false);
  app.add_option("--confidence-threshold", confidence_threshold, "KANJI fallback threshold (default: 0.3)");
  app.add_option("--numeric-confidence-threshold", numeric_confidence_threshold,
                 "JAPANESE_NUMERIC rule-based conversion threshold (default: 0.5)");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  PredictorConfig config(model_path);
  config.set_confidence_threshold(confidence_threshold);
  config.set_numeric_confidence_threshold(numeric_confidence_threshold);

  return momo::predict(config, compute_confidence, profile);
}
