#include <iostream>
#include <string>

#include "CLI/CLI.hpp"
#include "loader.hpp"
#include "predictor.hpp"

namespace momo {

int predict(const std::string& model_file_prefix, bool compute_confidence = false) {
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
