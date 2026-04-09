#include <iostream>
#include <string>

#include "CLI/CLI.hpp"
#include "loader.hpp"
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
 * @param model_file_prefix モデルファイルのパスプレフィックス（例: "basic_data" なら "basic_data.mbm" を読み込む）
 * @param compute_confidence 自信度スコアを計算するかどうかのフラグ
 * @return 終了コード（0は成功、1はエラー）
 */
int predict(const std::string& model_file_prefix, bool compute_confidence) {
  // モデル読み込み
  MomoModel model;
  try {
    model = load_model(model_file_prefix + ".mbm");
  } catch (const std::exception& e) {
    std::cerr << "モデル読み込みエラー: " << e.what() << "\n";
    return 1;
  }

  std::cerr << "モデル読み込み完了: " << model.n_classes << " クラス, " << model.n_features << " 特徴量\n";

  // 予測器を構築
  Predictor predictor(std::move(model));

  // 標準入力から1行ずつ処理
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      std::cout << "\n";
      continue;
    }
    try {
      const PredictionResult result = predictor.predict(line);
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
  std::string model_file_prefix;
  app.add_option("--model", model_file_prefix, "Path prefix for model files")->default_val("basic_data");
  bool compute_confidence = false;
  app.add_flag("--confidence", compute_confidence, "Compute confidence scores (marginal probabilities)");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  return momo::predict(model_file_prefix, compute_confidence);
  return 0;
}
