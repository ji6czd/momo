#include <iostream>
#include <string>

#include "func.h"
#include "predictor.h"
#include "utils.h"
namespace momo {
int test() {
  PredictorConfig cfg;
  cfg.path_read = "basic_data_read.crfsuite";
  cfg.path_boundary = "basic_data_boundary.crfsuite";
  Predictor prd(cfg);
  std::string input = "漢字の読みを予測します。";
  PredictionResult res = prd.predict(input);
  std::cout << res.kana_text << std::endl;
  return 0;
}
}  // namespace momo
int main(int argc, char* argv[]) {
  momo::test();
  return 0;
}
