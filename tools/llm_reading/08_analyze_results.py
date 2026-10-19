"""07_gpu_eval.py の結果（results/）をモデル横並びで分析する（手元で実行）。

- しきい値ごとの成績: w7 の自信度 < t のスロットだけ LLM に置き換えた一致率（t=0.3〜0.8）と、
  LLM 単独。どこまで LLM に任せるのがよいか
- 組み合わせの上限: w7 と LLM のどちらかが正解しているスロットの割合（どう組み合わせても
  これ以上は上がらない）
- 文脈: dev / train の段落で、前の文を置いて良くなった・悪くなった漢字の数
- LLM の誤りの種類（dev 短文）: momo の表記の約束との違い（連濁・長音・促音・ノ）か、
  ありふれた読みへの偏り（辞書で頻度1位の読みを選んだ）か、それ以外（知識不足など）か。
  約束の違いが多ければファインチューンで直る見込みが大きい

    uv run python 08_analyze_results.py [--results work/gpu_results] [--dict work/gpu_kit/bundle/single_character_dic.tsv]

結果は gcloud compute scp --recurse <VM>:~/gpu_kit/results work/gpu_results で持ち帰る。
"""

import argparse
import csv
import json
import os
import unicodedata
from collections import Counter

from _common import WORK

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
ERROR_KINDS = [
    "連濁・半濁音だけ",
    "長音の書き方だけ",
    "促音だけ",
    "ノの有無だけ",
    "継続（まとめ方）",
    "頻度1位の読みへの偏り",
    "それ以外",
]


def load_short(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_dict_top(path: str) -> dict[str, str]:
    """単一文字辞書の、文字ごとの頻度1位の読み（--- と _ は除く）。"""
    top = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = [p for p in line.rstrip("\n").split("\t") if p not in ("---", "_")]
            if len(parts) >= 2:
                top[parts[0]] = parts[1]
    return top


def _strip_voicing(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def error_kind(char: str, gold: str, llm: str, top: dict[str, str]) -> str:
    """LLM の誤りを、momo の表記の約束との違いか、偏りか、それ以外かに分ける。"""
    if "---" in (gold, llm):
        return "継続（まとめ方）"
    if _strip_voicing(gold) == _strip_voicing(llm):
        return "連濁・半濁音だけ"
    if any(gold.replace("ー", v) == llm.replace("ー", v) for v in "アイウエオ"):
        return "長音の書き方だけ"
    if gold.rstrip("ッ") == llm.rstrip("ツチクキッ") or llm.rstrip("ッ") == gold.rstrip(
        "ツチクキッ"
    ):
        if "ッ" in gold + llm:
            return "促音だけ"
    if gold.rstrip("ノ") == llm.rstrip("ノ"):
        return "ノの有無だけ"
    if top.get(char) == llm:
        return "頻度1位の読みへの偏り"
    return "それ以外"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=os.path.join(WORK, "gpu_results"))
    ap.add_argument(
        "--dict",
        default=os.path.join(WORK, "gpu_kit", "bundle", "single_character_dic.tsv"),
        help="評価に使った単一文字辞書（頻度1位の読みの判定に使う）",
    )
    args = ap.parse_args()

    models = []
    for name in sorted(os.listdir(args.results)):
        path = os.path.join(args.results, name, "result.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                models.append((name, json.load(f)))
    if not models:
        raise SystemExit(f"結果がありません: {args.results}")

    print("== dev 短文: しきい値ごとの一致率（w7 の自信度 < t だけ LLM）")
    head = ["model", "w7", "LLM単独", *(f"t={t}" for t in THRESHOLDS), "最良", "上限"]
    print("\t".join(head))
    for name, r in models:
        s = r["short"]
        gated = {t: s[f"gated_{t}"] for t in THRESHOLDS}
        best_t = max(gated, key=gated.get)
        best = max(s["llm"], gated[best_t])
        best_label = "LLM単独" if s["llm"] >= gated[best_t] else f"t={best_t}"
        rows = load_short(os.path.join(args.results, name, "short.tsv"))
        oracle = sum(r["gold"] in (r["llm"], r["w7"]) for r in rows) / len(rows)
        cells = [s["w7"], s["llm"], *gated.values()]
        print(
            "\t".join(
                [
                    r["model"].split("/")[-1],
                    *(f"{c:.4f}" for c in cells),
                    f"{best:.4f}({best_label})",
                    f"{oracle:.4f}",
                ]
            )
        )

    print(
        "\n== 文脈: 前の文を置いて良くなった／悪くなった漢字（LLM 単独の一致率 なし→あり）"
    )
    print("model\tdev段落\ttrain段落")
    for _, r in models:
        cells = []
        for key in ("context", "context_train"):
            c = r.get(key)
            if not c:
                cells.append("-")
                continue
            cells.append(
                f"+{c['ctx_better']}/-{c['noctx_better']}"
                f"（{c['without']['llm']:.4f}→{c['with']['llm']:.4f}）"
            )
        print("\t".join([r["model"].split("/")[-1], *cells]))

    print("\n== 確認用の文（文脈あり / なし / w7）")
    for _, r in models:
        p = r["probe"]
        print(
            f"{r['model'].split('/')[-1]}\t{p['ctx']}/{p['n']}\t{p['noctx']}/{p['n']}\t{p['w7']}/{p['n']}"
        )

    top = load_dict_top(args.dict)
    print("\n== LLM の誤りの種類（dev 短文・LLM 単独）")
    print("\t".join(["model", "誤り", *ERROR_KINDS]))
    for name, r in models:
        rows = load_short(os.path.join(args.results, name, "short.tsv"))
        chars = [_slot_char(x) for x in rows]
        kinds = Counter(
            error_kind(ch, x["gold"], x["llm"], top)
            for ch, x in zip(chars, rows)
            if x["llm"] != x["gold"]
        )
        n = sum(kinds.values())
        cells = [f"{kinds[k]}({kinds[k] / n:.0%})" if n else "0" for k in ERROR_KINDS]
        print("\t".join([r["model"].split("/")[-1], str(n), *cells]))


def _slot_char(row: dict) -> str:
    """short.tsv の行の漢字（列 char があればそれ、なければ空）。"""
    return row.get("char", "")


if __name__ == "__main__":
    main()
