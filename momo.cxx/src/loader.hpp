#pragma once
#include <string>

#include "model.hpp"

// .mbm ファイルを読み込んで MomoModel を返す。
// 失敗した場合は std::runtime_error を投げる。
MomoModel load_model(const std::string& path);
