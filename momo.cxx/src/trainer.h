#pragma once

#include <string>

namespace momo {

/**
 * CRF学習器。
 * TSVファイルを読み込み、読みモデルと境界モデルを学習して
 * ZIPファイルに出力する。
 *
 * TSVフォーマット:
 *   各行: 文字\t読みラベル\t文字種\tBIOSEタグ\t原文インデックス
 *   空行はシーケンスの区切り
 *   '#' で始まる行はコメント
 *
 * 出力ファイル:
 *   {output_prefix}_read.crfsuite     - 読みモデル
 *   {output_prefix}_boundary.crfsuite - 境界モデル
 *   {output_prefix}.zip               - 上記2ファイル + version_info.json
 */
class Trainer {
 public:
  Trainer();

  /**
   * 学習を実行する。
   * @param tsv_path  学習データTSVファイルのパス
   * @param transitions  遷移特徴量を使用するかどうか
   *                     false の場合、O(T*L) Viterbi fast path が有効になる
   */
  void train(const std::string& tsv_path, bool transitions = false);
};

}  // namespace momo
