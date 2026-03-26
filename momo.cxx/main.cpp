#include <iostream>
#include <string>
#include <string>

#include "func.h"
#include "utils.h"
#include "predictor.h"
namespace momo {
int test() {
    PredictorConfig cfg;
    cfg.path_read = "basic_data_read.crfsuite";
    cfg.path_boundary = "basic_data_boundary.crfsuite";
    Predictor prd(cfg);
    return 0;
}
}
int main(int argc, char* argv[]) {
  std::cout << "Hello, World!" << std::endl;
    momo::test();
    return 0;
}
