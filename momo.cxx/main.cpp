#include <iostream>
#include <string>

#include "predictor.h"
#include "utils.h"
namespace momo {
int test() {
  PredictorConfig cfg;
  cfg.path_read = "basic_data_read.crfsuite";
  cfg.path_boundary = "basic_data_boundary.crfsuite";
  Predictor prd(cfg);

  // 標準入力から読み取って推論する
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    PredictionResult res = prd.predict(line);
    std::cout << res.kana_text << std::endl;
    for (auto c : res.confidences) {
      std::cout << c << ", ";
    }
    std::cout << std::endl;
  }
  return 0;
}
}  // namespace momo
int main(int argc, char* argv[]) {
  momo::test();
  return 0;
}
