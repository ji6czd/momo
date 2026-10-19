"""A4: 学習データの中で、同じ漢字列の分け方が揺れていないかを一覧にする。

文ごとに連続する漢字スロット（漢字ラン）を取り出し、表層ごとに「分け方」（スロットごとの
読みの並び）を集計する。連結した読みが同じなのに分け方が違うものを揺れとして出す。
読みそのものが違うもの（今日＝キョー/コンニチ）は読み分けなので対象外。

規約の揺れは、w7 でも LLM でも「かなの各文字が原文のどこに当たるか」の誤りとして
学習されてしまうため、拡充の前に潰しておきたい。

    uv run python 04_split_audit.py [--raw dataset/basic_raw.tsv]

出力: work/split_audit.tsv（表層・連結読み・分け方ごとの件数・例の行番号）
"""

import argparse
import os
from collections import defaultdict

from _common import KANJI_TYPES, RAW_PATH, WORK, parse_gold, read_raw_numbered


def kanji_runs(slots):
    """連続する漢字スロットのまとまりを順に返す。"""
    run = []
    for s in slots:
        if s.ctype in KANJI_TYPES and (not run or run[-1].end == s.start):
            run.append(s)
            continue
        if run:
            yield run
        run = [s] if s.ctype in KANJI_TYPES else []
    if run:
        yield run


def pattern_str(pattern: tuple[str, ...]) -> str:
    return "/".join(r or "---" for r in pattern)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default=RAW_PATH)
    args = ap.parse_args()

    # (表層, 連結読み) → 分け方 → 出現した行番号
    seen: dict[tuple[str, str], dict[tuple[str, ...], list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    lines = read_raw_numbered(args.raw)
    for n, line in lines:
        for run in kanji_runs(parse_gold(line, n).slots):
            if len(run) < 2:
                continue
            surface = "".join(s.surface for s in run)
            pattern = tuple(s.gold for s in run)
            seen[(surface, "".join(pattern))][pattern].append(n)

    shaky = [(key, patterns) for key, patterns in seen.items() if len(patterns) > 1]
    shaky.sort(key=lambda kv: -sum(len(v) for v in kv[1].values()))

    out = os.path.join(WORK, "split_audit.tsv")
    os.makedirs(WORK, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("surface\treading\tpattern\tcount\texample_lines\n")
        for (surface, reading), patterns in shaky:
            for pattern, nums in sorted(patterns.items(), key=lambda kv: -len(kv[1])):
                examples = ",".join(str(x) for x in nums[:5])
                f.write(
                    f"{surface}\t{reading}\t{pattern_str(pattern)}\t{len(nums)}\t{examples}\n"
                )

    print(
        f"{len(lines)} 行を走査: 2文字以上の漢字列 {len(seen)} 種のうち、分け方が揺れているもの {len(shaky)} 種"
    )
    for (surface, reading), patterns in shaky[:20]:
        variants = "  ".join(
            f"{pattern_str(p)}×{len(v)}"
            for p, v in sorted(patterns.items(), key=lambda kv: -len(kv[1]))
        )
        print(f"  {surface}（{reading}）: {variants}")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
