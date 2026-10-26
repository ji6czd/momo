"""フェーズ B の予備実験（手元の CPU）: 素の LLM に、文字ごとの候補から漢字の読みを選ばせる。

選び方の本体は llm_core.py（07_gpu_eval.py と共通）。ここは手元のデータ（momo_train で正解を展開し、
momors_py で w7 を推論する）を用意して、任意の文・確認用の文・dev の短い文で試す。

方式（--mode）: ruby（既定。「右側（みぎがわ）」の括弧書き）/ yomi（ひらがなの続け書き）/
interleave（1文字ずつ）。違いと実験の経緯は llm_core.py を参照。
文脈: 確認用の文の context 列を、原文の行の先頭に置き、その読み（w7）をふりがなの行の先頭に置く。

    uv run --group llm python 05_llm_select.py --text "上手に入るとは…"   # 任意の文を試す
    uv run --group llm python 05_llm_select.py --probe probe_context.tsv  # 文脈つき確認用の文
    uv run --group llm python 05_llm_select.py --dev 100 [--dev-offset 100]  # dev の短い文で w7 と比較
    uv run --group llm python 05_llm_select.py --model llm-jp/llm-jp-3-980m --dev 100

w7 は work/frac_100/（train だけで学習）を使う。出力は標準出力と work/llm_*.tsv。
"""

import argparse
import os
import time

import momors_py
import torch

from _common import (
    KANJI_TYPES,
    WORK,
    GoldSentence,
    align_prediction,
    load_dev,
    pick_fewshot,
    slots_for_text,
)
from llm_core import (
    LM,
    SELECTORS,
    THRESHOLDS,
    fmt_reading,
    fmt_sentence,
    gating_table,
    load_dict_ordered,
    ruby,
)

FRAC100 = os.path.join(WORK, "frac_100")


def w7_prediction(predictor, sent: GoldSentence):
    return align_prediction(sent, predictor.predict(sent.text))


def show(sent: GoldSentence, llm: list[str], w7: list[str], report) -> None:
    print(f"原文: {sent.text}")
    print(f"  LLM: {fmt_sentence(sent.slots, llm)}")
    print(f"  w7 : {fmt_sentence(sent.slots, w7)}")
    for run, table in report:
        if any(llm[j] != w7[j] for j in run):
            surface = "".join(sent.slots[j].surface for j in run)
            top = "  ".join(
                f"{'/'.join(map(fmt_reading, c))}:{lp:.2f}" for c, lp in table
            )
            print(f"    {surface}: [{top}]")


def run_probe(sel, w7, path: str, tag: str) -> None:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            pid, context, text, word, expected = line.rstrip("\n").split("\t")[:5]
            sent = GoldSentence("", text, slots_for_text(text))
            base = w7_prediction(w7, sent).readings
            ctx_ruby = ""
            if context:
                cs = GoldSentence("", context, slots_for_text(context))
                ctx_ruby = ruby(cs.slots, w7_prediction(w7, cs).readings)
            start = text.find(word)
            idx = [
                i
                for i, s in enumerate(sent.slots)
                if start <= s.start < start + len(word)
            ]
            got = {}
            for label, ctx, cr in (("ctx", context, ctx_ruby), ("noctx", "", "")):
                llm, _ = sel.select(text, sent.slots, base, ctx, cr)
                got[label] = "".join(llm[i] for i in idx)
            w7_word = "".join(base[i] for i in idx)
            rows.append(
                (pid, context, text, word, expected, got["ctx"], got["noctx"], w7_word)
            )

            def mark(r: str) -> str:
                return "○" if r == expected else "×"

            print(
                f"{pid}: {word}（期待 {expected}）  LLM文脈あり {got['ctx']}{mark(got['ctx'])}"
                f"  LLM文脈なし {got['noctx']}{mark(got['noctx'])}  w7 {w7_word}{mark(w7_word)}"
            )
    n = len(rows)
    for col, name in ((5, "LLM 文脈あり"), (6, "LLM 文脈なし"), (7, "w7")):
        print(f"{name}: {sum(r[col] == r[4] for r in rows)}/{n}")
    out = os.path.join(WORK, f"llm_probe_{tag}.tsv")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("id\tcontext\ttext\tword\texpected\tllm_ctx\tllm_noctx\tw7\n")
        f.writelines("\t".join(r) + "\n" for r in rows)
    print(f"→ {out}\n")


def run_dev(sel, w7, n: int, tag: str, offset: int = 0) -> None:
    dev = [g for g in load_dev() if len(g.text) <= 40][offset : offset + n]
    rows, detail = [], []  # rows: 漢字スロットの (正解, LLM, w7, w7 の自信度)
    t = time.time()
    for k, g in enumerate(dev, 1):
        pred = w7_prediction(w7, g)
        llm, _ = sel.select(g.text, g.slots, pred.readings)
        for s, a, b, conf in zip(g.slots, llm, pred.readings, pred.confidences):
            if s.ctype in KANJI_TYPES:
                rows.append((s.gold, a, b, conf))
                detail.append((s.surface, s.gold, a, b, conf, g.text))
        if k % 10 == 0:
            print(f"  {k}/{len(dev)} 文（{time.time() - t:.0f} 秒）", flush=True)

    tab = gating_table(rows)
    fixed = sum(a == g and b != g for g, a, b, _ in rows)
    broke = sum(b == g and a != g for g, a, b, _ in rows)
    print(f"\ndev 短文 {len(dev)} 文・漢字スロット {tab['slots']} 個（{tag}）")
    print(f"  LLM 一致率 {tab['llm']:.4f} / w7 一致率 {tab['w7']:.4f}")
    print(
        f"  w7 の誤りを LLM が直した {fixed} 個 / w7 が正しいのを LLM が崩した {broke} 個"
    )
    print("  w7 の自信度 < t のスロットだけ LLM に置き換えると:")
    for th in THRESHOLDS:
        n_low = sum(c < th for *_, c in rows)
        print(
            f"    t={th:.1f}: 一致率 {tab[f'gated_{th}']:.4f}（置換対象 {n_low}・"
            f"直した {tab[f'fixed_{th}']}・崩した {tab[f'broke_{th}']}）"
        )

    out = os.path.join(WORK, f"llm_dev_slots_{tag}.tsv")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("char\tgold\tllm\tw7\tw7_conf\ttext\n")
        for ch, gold, a, b, conf, text in detail:
            f.write(
                f"{ch}\t{fmt_reading(gold)}\t{fmt_reading(a)}\t{fmt_reading(b)}\t{conf:.3f}\t{text}\n"
            )
    print(f"→ {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="sbintuitions/sarashina2.2-0.5b")
    ap.add_argument("--mode", choices=list(SELECTORS), default="ruby")
    ap.add_argument("--shots", type=int, default=6)
    ap.add_argument("--text", action="append", default=[])
    ap.add_argument(
        "--probe", help="TSV: id, context, text, word, expected（word の読みの期待値）"
    )
    ap.add_argument(
        "--dev", type=int, default=0, help="dev の短い文（40字以下）を何文使うか"
    )
    ap.add_argument(
        "--dev-offset", type=int, default=0, help="dev の短い文の何文目から使うか"
    )
    ap.add_argument(
        "--calib", type=float, default=0.0, help="ruby のみ: PMI 較正の係数（0 で無効）"
    )
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)
    w7 = momors_py.Predictor(os.path.join(FRAC100, "basic_data_7.mbm"))
    dic = load_dict_ordered(os.path.join(FRAC100, "single_character_dic.tsv"))
    t0 = time.time()
    lm = LM(args.model)
    extra = {"calib": args.calib} if args.mode == "ruby" else {}
    sel = SELECTORS[args.mode](lm, pick_fewshot(args.shots), dic, **extra)
    tag = f"{args.model.split('/')[-1]}_{args.mode}" + (
        f"_c{args.calib:g}" if args.calib else ""
    )
    print(f"モデル読み込み・手本の前処理 {time.time() - t0:.0f} 秒（{tag}）")

    for text in args.text:
        sent = GoldSentence("", text, slots_for_text(text))
        base = w7_prediction(w7, sent).readings
        t = time.time()
        llm, report = sel.select(text, sent.slots, base)
        show(sent, llm, base, report)
        print(f"  （{time.time() - t:.1f} 秒）\n")

    if args.probe:
        run_probe(sel, w7, args.probe, tag)
    if args.dev:
        run_dev(sel, w7, args.dev, f"{tag}_o{args.dev_offset}", args.dev_offset)


if __name__ == "__main__":
    main()
