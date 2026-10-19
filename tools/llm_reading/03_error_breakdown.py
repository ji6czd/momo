"""A3: w7（train 100%）の dev 上の漢字スロットの誤りを分類する。

- (a) 選び損ね : 正解の読みが単一文字辞書の候補にあるのに、別の候補を選んだ
- (b) 辞書にない: 正解の読みが候補にない（その漢字自体が辞書にない場合も含む）
- (c) まとまり  : 正解か予測の一方が継続（読みが空）。熟字訓のまとめ方・分け方の誤り
- (n) 漢数字    : 数字にするか読むか（一→1/イチ/ヒト）の判断。漢数字は辞書の対象外なので別枠
- (r) 々        : 直前の文字から規則で決まるので、辞書の内外では分けない

辞書は train だけから作ったもの（work/frac_100/single_character_dic.tsv）を使う。
(a) が多ければ読み分けの例文、(b) が多ければ新しい読みを持つ例文が拡充の軸になる。

    uv run python 03_error_breakdown.py [--model ...] [--dict ...]

出力: 件数のまとめ（標準出力）、work/error_breakdown.tsv（誤り1件1行）、
work/error_breakdown_b_by_char.tsv（(b) を漢字ごとに集計）
"""

import argparse
import os
from collections import Counter, defaultdict

import momors_py

from _common import KANJI_TYPES, WORK, load_dev, load_single_dict, predict_all

CATEGORIES = {
    "a": "(a) 選び損ね（正解が辞書の候補内）",
    "b": "(b) 辞書にない読み",
    "c": "(c) まとまり（熟字訓のまとめ方・分け方）",
    "n": "(n) 漢数字（数字にするか読むか）",
    "r": "(r) 々（直前の文字から規則で決まる）",
}


def classify(
    surface: str, ctype: str, gold: str, pred: str, dic: dict[str, set[str]]
) -> str:
    if ctype == "JAPANESE_NUMERIC":
        return "n"
    if surface == "々":
        return "r"
    if gold == "" or pred == "":
        return "c"
    return "a" if gold in dic.get(surface, ()) else "b"


def main() -> None:
    base = os.path.join(WORK, "frac_100")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=os.path.join(base, "basic_data_7.mbm"))
    ap.add_argument("--dict", default=os.path.join(base, "single_character_dic.tsv"))
    args = ap.parse_args()

    dic = load_single_dict(args.dict)
    pairs = predict_all(momors_py.Predictor(args.model), load_dev())

    errors = []
    n_kanji = 0
    for g, p in pairs:
        for s, r in zip(g.slots, p.readings):
            if s.ctype not in KANJI_TYPES:
                continue
            n_kanji += 1
            if s.gold != r:
                errors.append(
                    (classify(s.surface, s.ctype, s.gold, r, dic), g, p, s, r)
                )

    counts = Counter(cat for cat, *_ in errors)
    print(f"dev の漢字スロット {n_kanji} 個、誤り {len(errors)} 個")
    for cat, name in CATEGORIES.items():
        share = counts[cat] / len(errors) if errors else 0.0
        print(f"  {name}: {counts[cat]} ({share:.1%})")

    out = os.path.join(WORK, "error_breakdown.tsv")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("category\tchar\tgold\tpred\tdict_candidates\ttext\tprediction\n")
        for cat, g, p, s, r in sorted(errors, key=lambda e: (e[0], e[3].surface)):
            cands = "/".join(sorted(dic.get(s.surface, ())))
            f.write(
                f"{cat}\t{s.surface}\t{s.gold or '---'}\t{r or '---'}\t{cands}\t{g.text}\t{p.kana}\n"
            )

    by_char: dict[str, Counter] = defaultdict(Counter)
    for cat, _g, _p, s, _r in errors:
        if cat == "b":
            by_char[s.surface][s.gold] += 1
    out_b = os.path.join(WORK, "error_breakdown_b_by_char.tsv")
    with open(out_b, "w", encoding="utf-8", newline="\n") as f:
        f.write("char\tcount\tmissing_readings\tin_dict\n")
        for ch, c in sorted(by_char.items(), key=lambda kv: -sum(kv[1].values())):
            missing = " ".join(f"{r}×{n}" for r, n in c.most_common())
            f.write(
                f"{ch}\t{sum(c.values())}\t{missing}\t{'yes' if ch in dic else 'no'}\n"
            )

    print(f"\n→ {out}\n→ {out_b}")


if __name__ == "__main__":
    main()
