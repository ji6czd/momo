/*
 * crfsuite_impl.cpp
 *
 * crfsuite.hpp はクラス実装をヘッダーに含むため、
 * 複数の翻訳単位からインクルードすると多重定義エラーになる。
 * このファイルだけが crfsuite.hpp をインクルードし、
 * 他のファイル（predictor.cpp, trainer.cpp）は
 * crfsuite_api.hpp（宣言のみ）を使う。
 */
#include <crfsuite.hpp>
