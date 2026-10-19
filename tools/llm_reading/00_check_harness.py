"""評価の仕組みの確認: momors_py で回帰セットを推論し、eval/run_regression.sh と同じ
通過数になるかを見る（同じモデルなら一致するはず）。

    uv run python 00_check_harness.py [--model dataset/basic_data_7.mbm]

あわせて、basic_raw.tsv 全行の正解展開（parse_gold）が通ることと、本番モデルの予測を
スロットに割り付けたときに、どのスロットにも入らないかなが出ないことを確かめる。
"""

import argparse
import os

import momors_py

from _common import (
    DATASET,
    RAW_PATH,
    align_prediction,
    parse_gold,
    read_raw_lines,
    read_regression,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=os.path.join(DATASET, "basic_data_7.mbm"))
    args = ap.parse_args()
    predictor = momors_py.Predictor(args.model)

    cases = read_regression()
    failed = [(i, e, predictor.predict(i).kana) for i, e in cases]
    failed = [(i, e, g) for i, e, g in failed if g != e]
    print(f"回帰セット: {len(cases) - len(failed)}/{len(cases)} 通過")
    for inp, exp, got in failed:
        print(f"  ✗ {inp}\n    期待: {exp}\n    実際: {got}")

    lines = read_raw_lines(RAW_PATH)
    stray = 0
    for n, line in enumerate(lines, 1):
        g = parse_gold(line, n)
        p = align_prediction(g, predictor.predict(g.text))
        if p.stray:
            stray += 1
            if stray <= 5:
                print(f"  割り付け漏れ（行 {n}）: {p.stray!r} ← {g.text}")
    print(f"basic_raw.tsv {len(lines)} 行: 展開 OK、割り付け漏れのある行 {stray}")


if __name__ == "__main__":
    main()
