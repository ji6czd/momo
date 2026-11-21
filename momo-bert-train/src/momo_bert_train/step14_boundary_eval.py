"""dev の境界（マスあけ）を、train だけで学習した w7 と、読みと境界を同時学習した BERT で比べる。

    uv run step14_boundary_eval                      # 既定: work/w7_train/basic_data_7.mbm と work/finetuned/train_joint
                                                     #       （work/onnx/train_joint があれば Rust の経路も）
    uv run step14_boundary_eval --bert ""            # w7 だけ
    uv run step14_boundary_eval --threshold 0.4      # BERT 単体でマスをあける確率の閾値（既定 0.5）
    uv run step14_boundary_eval --onnx ""            # Rust の経路を省く
    uv run step14_boundary_eval --bert-f32           # Rust の経路で float32 版の ONNX を使う（既定は int8 版）

切れ目は step9_finetune の境界の学習と同じ（隣り合うスロットの間。行末と、原文の空白をはさむ切れ目は除く）。
- w7 は momo 本体の推論経路（英数字ランの保証・括弧・助数詞・カスタム辞書などの規則込み）が出した空白。
- BERT は境界の出力層の確率（PyTorch）で、規則は掛けていない。
- w7+BERT は momo 本体の推論経路に BERT を差し込んだもの（momors_py、export_onnx の出力を読む）。
  BERT が確率を付けた切れ目は BERT で、規則で決まる位置は規則で決まる。実際に配る形の数字。
w7 は train だけで学習したもの（作り方は README）。全データで学習した w7 は dev を学習済みで比較にならない。

出力: work/boundary_eval/<BERT の名前、w7 だけなら w7>/
  summary.json  それぞれの誤りの数（くっつけすぎ＝注釈はあけるのに続けた・分けすぎ）、F1、
                全部の切れ目が当たった行の割合
  by_type.tsv   切れ目の前後の文字種ごとの、切れ目の数とそれぞれの誤りの数
  diff.tsv      w7 と、BERT を使ったもの（w7+BERT があればそれ、無ければ BERT 単体）で判定が違う切れ目
                （どちらが注釈どおりか、BERT の確率、前後の文字つき）
  errors.tsv    どれかが外した切れ目すべて
"""

import argparse
import collections
import json
import time
from pathlib import Path

import momors_py
import torch

from .common import (
    DATA,
    DEFAULT_MODEL,
    WORK,
    Candidates,
    CharVocab,
    Labels,
    load_finetuned,
    load_jsonl,
    make_examples,
    predict_boundary,
)

CONTEXT = 10  # diff.tsv / errors.tsv に出す前後の文字数
NAMES = {"w7": "w7", "bert": "BERT", "w7bert": "w7+BERT"}


def momo_splits(predictor, rec) -> set[int]:
    """momo の推論経路が後ろにマスをあけたスロットの番号。

    境界の空白は、その直前の字（境界を問うた字）の原文位置に割り付けられている。"""
    owner = {p: i for i, s in enumerate(rec.slots) for p in range(s.start, s.end)}
    res = predictor.predict(rec.text)
    return {
        owner[src]
        for ch, src in zip(res.kana, res.kana_to_source)
        if ch == " " and src in owner
    }


def metrics(pairs: list[tuple[int, int]], lines_ok: int, n_lines: int) -> dict:
    """pairs は切れ目ごとの (注釈, 予測)。1 = マスあけ。"""
    c = collections.Counter(pairs)
    tp, fp, fn, tn = c[1, 1], c[0, 1], c[1, 0], c[0, 0]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    gaps = tp + fp + fn + tn
    return {
        "gaps": gaps,
        "spaces": tp + fn,
        "errors": fp + fn,
        "missed": fn,
        "extra": fp,
        "accuracy": (tp + tn) / gaps if gaps else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "lines_all_correct": lines_ok / n_lines if n_lines else 0.0,
    }


def summary_line(name: str, m: dict) -> str:
    return (
        f"{name:8s} 誤り {m['errors']:5d}/{m['gaps']}（くっつけすぎ {m['missed']}・分けすぎ {m['extra']}）"
        f" F1={m['f1']:.4f} 適合率={m['precision']:.4f} 再現率={m['recall']:.4f}"
        f" 全部当たった行 {m['lines_all_correct']:.1%}"
    )


def write_tsv(path: Path, header: list[str], rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        f.writelines("\t".join(map(str, r)) + "\n" for r in rows)


def mark(v: int) -> str:
    return "あけ" if v else "続け"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--w7",
        default=str(WORK / "w7_train" / "basic_data_7.mbm"),
        help="train だけで学習した w7（.mbm）",
    )
    ap.add_argument(
        "--bert",
        default="train_joint",
        help="work/finetuned/ 下の同時学習モデルの名前（空なら BERT 単体を省く）",
    )
    ap.add_argument(
        "--onnx",
        default=None,
        help="w7+BERT に使う export_onnx の出力（既定は work/onnx/<--bert>、空なら省く）",
    )
    ap.add_argument(
        "--bert-f32", action="store_true", help="w7+BERT で float32 版の ONNX を使う"
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    recs = load_jsonl(str(DATA / "dev_full.jsonl"))
    labels = Labels(recs)  # 読みの位置は使わない（make_example の引数を満たすだけ）
    cands = Candidates(recs, labels)
    vocab = CharVocab.from_model(DEFAULT_MODEL)
    exs = make_examples(recs, vocab, labels, cands)

    # 方式ごとの、行ごとの「後ろにマスをあけたスロットの番号」
    splits: dict[str, list[set[int]]] = {}
    t0 = time.perf_counter()
    w7 = momors_py.Predictor(args.w7)
    splits["w7"] = [momo_splits(w7, r) for r in recs]
    print(f"w7: {args.w7}（{time.perf_counter() - t0:.1f} 秒）", flush=True)

    bert_prob = None
    bert_dir = WORK / "finetuned" / args.bert if args.bert else None
    if bert_dir is not None and not (bert_dir / "model.pt").exists():
        print(f"{bert_dir} が無いので BERT 単体を省く")
        bert_dir = None
    if bert_dir is not None:
        names = json.loads((bert_dir / "labels.json").read_text(encoding="utf-8"))
        model = load_finetuned(bert_dir, len(names))
        if model.boundary_head is None:
            raise SystemExit(
                f"{bert_dir} は境界の出力層を持たない（--boundary で学習したものを渡す）"
            )
        t0 = time.perf_counter()
        bert_prob = predict_boundary(model, exs, vocab, args.device)
        splits["bert"] = [
            {si for si, p in zip(ex.bslot, probs) if p >= args.threshold}
            for ex, probs in zip(exs, bert_prob)
        ]
        print(
            f"BERT: {bert_dir}（{args.device}、{time.perf_counter() - t0:.1f} 秒）",
            flush=True,
        )

    onnx = args.onnx
    if onnx is None:
        onnx = str(WORK / "onnx" / args.bert) if args.bert else ""
        if onnx and not Path(onnx).is_dir():
            print(f"{onnx} が無いので w7+BERT を省く")
            onnx = ""
    if onnx:
        meta = json.loads((Path(onnx) / "meta.json").read_text(encoding="utf-8"))
        if not meta.get("boundary"):
            raise SystemExit(
                f"{onnx} は境界の出力を持たない（境界も学習したモデルを export_onnx で書き出す）"
            )
        t0 = time.perf_counter()
        rust = momors_py.Predictor(args.w7, bert=onnx, bert_f32=args.bert_f32)
        splits["w7bert"] = [momo_splits(rust, r) for r in recs]
        kind = "float32" if args.bert_f32 else "int8"
        print(
            f"w7+BERT: {onnx}（{kind}、{time.perf_counter() - t0:.1f} 秒）", flush=True
        )

    systems = list(splits)
    pairs = {k: [] for k in systems}
    lines_ok = dict.fromkeys(systems, 0)
    by_type: dict[tuple[str, str], collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    # diff.tsv で w7 と比べる相手
    other = "w7bert" if "w7bert" in splits else "bert" if "bert" in splits else None
    diff, errors = [], []
    for li, (r, ex) in enumerate(zip(recs, exs)):
        ok = dict.fromkeys(systems, True)
        probs = bert_prob[li] if bert_prob is not None else [None] * len(ex.bslot)
        for si, gold, p in zip(ex.bslot, ex.bgold, probs):
            a, b = r.slots[si], r.slots[si + 1]
            got = {k: int(si in splits[k][li]) for k in systems}
            key = (a.ctype, b.ctype)
            by_type[key]["gaps"] += 1
            by_type[key]["spaces"] += gold
            for k, v in got.items():
                pairs[k].append((gold, v))
                ok[k] &= v == gold
                by_type[key][k] += v != gold
            ctx = (
                r.text[max(0, a.end - CONTEXT) : a.end]
                + "｜"
                + r.text[b.start : b.start + CONTEXT]
            )
            prob = f"{p:.3f}" if p is not None else ""
            if any(v != gold for v in got.values()):
                errors.append(
                    [li, mark(gold), *(mark(got[k]) for k in systems), prob]
                    + [f"{a.ctype}|{b.ctype}", ctx]
                )
            if other is not None and got[other] != got["w7"]:
                diff.append(
                    [
                        li,
                        NAMES[other] if got[other] == gold else "w7",
                        mark(gold),
                        mark(got["w7"]),
                        mark(got[other]),
                        prob,
                        f"{a.ctype}|{b.ctype}",
                        ctx,
                    ]
                )
        for k in ok:
            lines_ok[k] += ok[k]

    name = args.bert or (Path(onnx).name if onnx else "w7")
    out = WORK / "boundary_eval" / (name if len(systems) > 1 else "w7")
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "w7": args.w7,
        "bert": str(bert_dir) if bert_dir is not None else None,
        "onnx": onnx or None,
        "onnx_f32": args.bert_f32 if onnx else None,
        "threshold": args.threshold,
        "lines": len(recs),
    }
    print()
    for k in systems:
        summary[k + "_metrics"] = m = metrics(pairs[k], lines_ok[k], len(recs))
        print(summary_line(NAMES[k], m))
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    type_rows = sorted(by_type.items(), key=lambda kv: -sum(kv[1][k] for k in systems))
    write_tsv(
        out / "by_type.tsv",
        ["前の文字種", "後の文字種", "切れ目", "注釈のマスあけ"]
        + [f"{NAMES[k]}の誤り" for k in systems],
        [
            [a, b, c["gaps"], c["spaces"], *(c[k] for k in systems)]
            for (a, b), c in type_rows
        ],
    )
    write_tsv(
        out / "errors.tsv",
        ["行", "注釈", *(NAMES[k] for k in systems), "BERTの確率"]
        + ["文字種", "前後（｜が切れ目）"],
        errors,
    )
    if other is not None:
        write_tsv(
            out / "diff.tsv",
            [
                "行",
                "注釈どおり",
                "注釈",
                "w7",
                NAMES[other],
                "BERTの確率",
                "文字種",
                "前後（｜が切れ目）",
            ],
            diff,
        )
        won = sum(1 for d in diff if d[1] != "w7")
        print(
            f"\nw7 と {NAMES[other]} で判定が違う切れ目 {len(diff)}:"
            f" {NAMES[other]} が注釈どおり {won}・w7 が注釈どおり {len(diff) - won}"
        )
    print(
        "\n誤りの多い文字種の組（前|後: 切れ目 "
        + " ".join(NAMES[k] for k in systems)
        + "）"
    )
    for (a, b), c in type_rows[:10]:
        print(f"  {a}|{b}: {c['gaps']} " + " ".join(str(c[k]) for k in systems))
    print(f"\n保存先: {out}")


if __name__ == "__main__":
    main()
