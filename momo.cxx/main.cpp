#include <iostream>
#include <string>

#include "CLI/CLI.hpp"
#include "predictor.h"
#include "trainer.h"
#include "utils.h"

namespace momo {

int predict(const std::string& model_file_prefix, bool compute_confidence = false) {
  PredictorConfig cfg;
  cfg.path_read       = model_file_prefix + "_read.crfsuite";
  cfg.path_boundary   = model_file_prefix + "_boundary.crfsuite";
  cfg.compute_confidence = compute_confidence;
  Predictor prd(cfg);

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    PredictionResult res = prd.predict(line);
    std::cout << res.kana_text << std::endl;
    if (compute_confidence) {
      for (auto c : res.confidences) {
        std::cout << c << ", ";
      }
      std::cout << std::endl;
    }
  }
  return 0;
}

int train(const std::string& tsv_file, bool transitions = false) {
  Trainer trainer;
  trainer.train(tsv_file, transitions);
  return 0;
}

}  // namespace momo

int main(int argc, char* argv[]) {
  CLI::App app{"Momo - Japanese Braille Predictor"};
  app.require_subcommand(1);

  // --- predict サブコマンド ---
  CLI::App* predict_cmd = app.add_subcommand("predict", "Run the predictor on standard input");
  std::string model_file_prefix;
  predict_cmd->add_option("--model", model_file_prefix, "Path prefix for model files")
      ->default_val("basic_data");
  bool compute_confidence = false;
  predict_cmd->add_flag("--confidence", compute_confidence,
                        "Compute confidence scores (marginal probabilities)");

  // --- train サブコマンド ---
  CLI::App* train_cmd = app.add_subcommand("train", "Train a new model from TSV data");
  std::string tsv_file;
  train_cmd->add_option("--tsv", tsv_file, "Path to training data in TSV format")
      ->default_val("basic_data.tsv");
  bool transitions = false;
  train_cmd->add_flag("--transitions", transitions,
                      "Use transition features (slower inference, potentially higher accuracy)");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  if (predict_cmd->parsed()) {
    return momo::predict(model_file_prefix, compute_confidence);
  } else if (train_cmd->parsed()) {
    return momo::train(tsv_file, transitions);
  }

  return 0;
}
