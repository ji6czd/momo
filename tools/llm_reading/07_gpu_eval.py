"""GPU キット: 複数のモデルで「候補から読みを選ばせる」評価を順に回し、結果を1つの表にまとめる。

クラウドの GPU で動かす想定（手元の CPU でも小さいモデルなら動く）。依存は torch と
transformers だけで、データは 06_export_bundle.py が書き出した bundle/ を読む。

測るもの（漢字スロットの一致率。w7 単独・LLM 単独・w7 の自信度が低いところだけ LLM）:
- short   dev の短い文: これまでの手元の実験と同じ条件（先頭 100 文が --dev-offset 0 に当たる）
- context dev の段落の2文目以降: 同じ段落の前の文（と w7 の読み）を前に置いた場合と置かない
          場合を比べる。文書の文脈が効くかどうかの本題
- context_train  同じものを train の段落から（dev の段落は少ないので数を補う）。w7 は学習済みなので
          w7 系の数字は参考にならない。見るのは LLM の文脈あり 対 なし
- probe   確認用の文: 文脈あり・なしで、注目する語の読み（期待値は注釈者が確認済み）

    uv run python 07_gpu_eval.py --smoke              # 動作確認（0.5B・各データ数件）
    uv run python 07_gpu_eval.py --preset l4          # 24GB 級の GPU: 8B まで
    uv run python 07_gpu_eval.py --preset a100        # 80GB 級の GPU: 33B まで全部
    uv run python 07_gpu_eval.py --models llm-jp/llm-jp-4-8b-base sbintuitions/sarashina2.2-3b

済んだモデルは飛ばす（results/<モデル>/result.json があれば。--force で回し直す）。
出力: results/<モデル>/{result.json,short.tsv,context.tsv,context_train.tsv,probe.tsv}、
      results/summary.tsv
"""

import argparse
import gc
import json
import os
import shutil
import time
import traceback

import torch

from llm_core import (
    LM,
    BSlot,
    RubySelector,
    fmt_reading,
    gating_table,
    load_dict_ordered,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SMALL = [
    "sbintuitions/sarashina2.2-0.5b",
    "sbintuitions/sarashina2.2-1b",
    "sbintuitions/sarashina2.2-3b",
    "llm-jp/llm-jp-3-980m",
    "llm-jp/llm-jp-3.1-1.8b",
    "llm-jp/llm-jp-3-3.7b",
]
PRESETS = {
    "l4": SMALL + ["sbintuitions/sarashina2-7b", "llm-jp/llm-jp-4-8b-base"],
    "a100": SMALL
    + [
        "sbintuitions/sarashina2-7b",
        "llm-jp/llm-jp-4-8b-base",
        "sbintuitions/sarashina2-13b",
        "llm-jp/llm-jp-3.1-13b",
        "llm-jp/llm-jp-4-32b-a3b-base",
        "llm-jp/llm-jp-4-33b-base",
    ],
}
GATE = 0.5  # 手元の実験で選んだしきい値（初見の 100 文でも最良だった）


class Rec:
    """bundle の1件。"""

    def __init__(self, d: dict):
        self.__dict__.update(d)
        self.slots = [BSlot(*s) for s in d["slots"]]


def load_jsonl(path: str) -> list[Rec]:
    with open(path, encoding="utf-8") as f:
        return [Rec(json.loads(line)) for line in f if line.strip()]


def kanji_rows(rec: Rec, llm: list[str]):
    """漢字スロットの (正解, LLM, w7, w7 の自信度)。"""
    return [
        (s.gold, a, w, c)
        for s, a, w, c in zip(rec.slots, llm, rec.w7, rec.conf)
        if s.ctype in ("KANJI", "JAPANESE_NUMERIC")
    ]


def write_tsv(path: str, header: list[str], rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        f.writelines("\t".join(map(str, r)) + "\n" for r in rows)


def evaluate(name: str, args, data: dict, out_dir: str) -> dict:
    t0 = time.time()
    lm = LM(name, args.device, args.dtype, args.batch)
    sel = RubySelector(lm, data["fewshot"], data["dic"])
    print(f"[{name}] 読み込み {time.time() - t0:.0f} 秒", flush=True)

    def progress(label: str, k: int, n: int, t: float) -> None:
        if k % 25 == 0 or k == n:
            print(f"[{name}] {label} {k}/{n}（{time.time() - t:.0f} 秒）", flush=True)

    # short: dev の短い文
    rows, detail, t = [], [], time.time()
    recs = data["short"][: args.n_short]
    for k, rec in enumerate(recs, 1):
        llm, _ = sel.select(rec.text, rec.slots, rec.w7)
        chars = [
            s.surface for s in rec.slots if s.ctype in ("KANJI", "JAPANESE_NUMERIC")
        ]
        for ch, (g, a, w, c) in zip(chars, kanji_rows(rec, llm)):
            rows.append((g, a, w, c))
            detail.append(
                (
                    rec.id,
                    ch,
                    fmt_reading(g),
                    fmt_reading(a),
                    fmt_reading(w),
                    c,
                    rec.text,
                )
            )
        progress("short", k, len(recs), t)
    short = gating_table(rows)
    write_tsv(
        os.path.join(out_dir, "short.tsv"),
        ["id", "char", "gold", "llm", "w7", "w7_conf", "text"],
        detail,
    )

    # context: 段落の前の文を置く／置かない（dev の段落と、数を補う train の段落）
    def eval_context(label: str, recs: list[Rec]) -> dict:
        with_rows, without_rows, detail, t = [], [], [], time.time()
        ctx_better = noctx_better = 0
        for k, rec in enumerate(recs, 1):
            with_ctx, _ = sel.select(
                rec.text, rec.slots, rec.w7, rec.context, rec.context_ruby
            )
            without, _ = sel.select(rec.text, rec.slots, rec.w7)
            for (g, a, w, c), (_, b, _, _) in zip(
                kanji_rows(rec, with_ctx), kanji_rows(rec, without)
            ):
                with_rows.append((g, a, w, c))
                without_rows.append((g, b, w, c))
                ctx_better += a == g and b != g
                noctx_better += b == g and a != g
                if a != b:
                    detail.append(
                        (
                            rec.id,
                            fmt_reading(g),
                            fmt_reading(a),
                            fmt_reading(b),
                            fmt_reading(w),
                            rec.text,
                            rec.context,
                        )
                    )
            progress(label, k, len(recs), t)
        write_tsv(
            os.path.join(out_dir, f"{label}.tsv"),
            ["id", "gold", "llm_ctx", "llm_noctx", "w7", "text", "context"],
            detail,
        )
        return {
            "items": len(recs),
            "with": gating_table(with_rows),
            "without": gating_table(without_rows),
            "ctx_better": ctx_better,
            "noctx_better": noctx_better,
        }

    context = eval_context("context", data["context"][: args.n_context])
    context_train = eval_context(
        "context_train", data["context_train"][: args.n_context_train]
    )

    # probe: 確認用の文
    detail, ok = [], {"ctx": 0, "noctx": 0, "w7": 0}
    for rec in data["probe"]:
        got = {}
        for label, ctx, ctx_ruby in (
            ("ctx", rec.context, rec.context_ruby),
            ("noctx", "", ""),
        ):
            llm, _ = sel.select(rec.text, rec.slots, rec.w7, ctx, ctx_ruby)
            got[label] = "".join(llm[i] for i in rec.word_idx)
        got["w7"] = "".join(rec.w7[i] for i in rec.word_idx)
        for key in ok:
            ok[key] += got[key] == rec.expected
        detail.append(
            (rec.id, rec.word, rec.expected, got["ctx"], got["noctx"], got["w7"])
        )
    write_tsv(
        os.path.join(out_dir, "probe.tsv"),
        ["id", "word", "expected", "llm_ctx", "llm_noctx", "w7"],
        detail,
    )

    result = {
        "model": name,
        "seconds": round(time.time() - t0),
        "device": args.device,
        "dtype": args.dtype,
        "short": short,
        "context": context,
        "context_train": context_train,
        "probe": {**ok, "n": len(data["probe"])},
    }
    del sel, lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


SUMMARY_COLS = [
    ("model", lambda r: r["model"]),
    ("minutes", lambda r: f"{r['seconds'] / 60:.0f}"),
    ("short_w7", lambda r: r["short"].get("w7")),
    ("short_llm", lambda r: r["short"].get("llm")),
    (f"short_gated{GATE}", lambda r: r["short"].get(f"gated_{GATE}")),
    ("ctx_w7", lambda r: r["context"]["without"].get("w7")),
    ("ctx_llm_without", lambda r: r["context"]["without"].get("llm")),
    ("ctx_llm_with", lambda r: r["context"]["with"].get("llm")),
    (
        f"ctx_gated{GATE}_without",
        lambda r: r["context"]["without"].get(f"gated_{GATE}"),
    ),
    (f"ctx_gated{GATE}_with", lambda r: r["context"]["with"].get(f"gated_{GATE}")),
    ("ctx_better", lambda r: r["context"]["ctx_better"]),
    ("noctx_better", lambda r: r["context"]["noctx_better"]),
    ("trainctx_llm_without", lambda r: r["context_train"]["without"].get("llm")),
    ("trainctx_llm_with", lambda r: r["context_train"]["with"].get("llm")),
    ("trainctx_ctx_better", lambda r: r["context_train"]["ctx_better"]),
    ("trainctx_noctx_better", lambda r: r["context_train"]["noctx_better"]),
    ("probe_ctx", lambda r: f"{r['probe']['ctx']}/{r['probe']['n']}"),
    ("probe_noctx", lambda r: f"{r['probe']['noctx']}/{r['probe']['n']}"),
    ("probe_w7", lambda r: f"{r['probe']['w7']}/{r['probe']['n']}"),
]


def fmt_cell(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def write_summary(results_dir: str, results: list[dict]) -> None:
    rows = [[fmt_cell(f(r)) for _, f in SUMMARY_COLS] for r in results]
    write_tsv(
        os.path.join(results_dir, "summary.tsv"), [c for c, _ in SUMMARY_COLS], rows
    )
    print("\n" + "\t".join(c for c, _ in SUMMARY_COLS))
    for row in rows:
        print("\t".join(row))


def purge_hf_cache(name: str) -> None:
    """そのモデルの Hugging Face のキャッシュを消す（ディスクが足りないとき）。"""
    root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    path = os.path.join(root, "models--" + name.replace("/", "--"))
    if os.path.isdir(path):
        shutil.rmtree(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", choices=list(PRESETS))
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument(
        "--smoke", action="store_true", help="動作確認（0.5B・各データ数件）"
    )
    ap.add_argument("--bundle", default=os.path.join(HERE, "bundle"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--dtype", default=None, help="既定: GPU は bfloat16、CPU は float32"
    )
    ap.add_argument("--batch", type=int, default=16, help="候補を一括採点する行数")
    ap.add_argument("--n-short", type=int, default=200)
    ap.add_argument("--n-context", type=int, default=200)
    ap.add_argument("--n-context-train", type=int, default=300)
    ap.add_argument("--force", action="store_true", help="済んだモデルも回し直す")
    ap.add_argument(
        "--purge-cache", action="store_true", help="モデルごとに HF のキャッシュを消す"
    )
    args = ap.parse_args()
    args.dtype = args.dtype or ("bfloat16" if args.device != "cpu" else "float32")

    models = list(args.models)
    if args.preset:
        models += PRESETS[args.preset]
    if args.smoke:
        models = models or ["sbintuitions/sarashina2.2-0.5b"]
        args.n_short, args.n_context, args.n_context_train = 3, 3, 3
        args.results = os.path.join(HERE, "results_smoke")
    if not models:
        ap.error("--preset / --models / --smoke のどれかを指定してください")

    data = {
        "fewshot": load_jsonl(os.path.join(args.bundle, "fewshot.jsonl")),
        "short": load_jsonl(os.path.join(args.bundle, "short.jsonl")),
        "context": load_jsonl(os.path.join(args.bundle, "context.jsonl")),
        "context_train": load_jsonl(os.path.join(args.bundle, "context_train.jsonl")),
        "probe": load_jsonl(os.path.join(args.bundle, "probe.jsonl")),
        "dic": load_dict_ordered(os.path.join(args.bundle, "single_character_dic.tsv")),
    }
    if args.smoke:
        data["probe"] = data["probe"][:2]
    print(
        f"device={args.device} dtype={args.dtype} short={min(args.n_short, len(data['short']))} "
        f"context={min(args.n_context, len(data['context']))} "
        f"context_train={min(args.n_context_train, len(data['context_train']))} "
        f"probe={len(data['probe'])}",
        flush=True,
    )

    os.makedirs(args.results, exist_ok=True)
    results = []
    for name in dict.fromkeys(models):
        out_dir = os.path.join(args.results, name.replace("/", "__"))
        result_path = os.path.join(out_dir, "result.json")
        if os.path.exists(result_path) and not args.force:
            print(f"[{name}] 済み（飛ばす）", flush=True)
            with open(result_path, encoding="utf-8") as f:
                results.append(json.load(f))
            continue
        os.makedirs(out_dir, exist_ok=True)
        try:
            result = evaluate(name, args, data, out_dir)
        except Exception:
            err = traceback.format_exc()
            print(f"[{name}] 失敗（次のモデルへ）\n{err}", flush=True)
            with open(os.path.join(out_dir, "error.txt"), "w", encoding="utf-8") as f:
                f.write(err)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        results.append(result)
        write_summary(args.results, results)
        if args.purge_cache:
            purge_hf_cache(name)

    write_summary(args.results, results)


if __name__ == "__main__":
    main()
