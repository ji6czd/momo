"""エンコーダ（BERT 系）を既存の学習データで微調整し、漢字スロットの読みを候補から選ばせる。

クラウドの GPU で動かす想定（手元の CPU では --smoke と、小さい割合・少ないエポックの予備実験）。
依存は torch と transformers だけで、データは 06_export_bundle.py が書き出した bundle/ を読む。

対象は ctype が KANJI / JAPANESE_NUMERIC のスロット。かな・記号は w7 の読みで固定する。
各スロットの読みを、その文字の辞書候補（学習に使う行だけから作る）に絞った softmax で選ぶ。
エポック数は固定（dev での選択はしない。dev は毎エポックの診断として出すだけ）。

測るもの（dev 全行、モデルは割合 × シード）:
- 漢字スロット一致率（w7 の learning_curve.tsv と同じ物差し）と、読み候補が2つ以上ある文字だけの一致率
- 誤りの内訳（辞書にない読み／辞書にあるが外した／頻度1位の読みへの偏り）と、頻度1位を選ぶだけの下限
- w7 の自信度がしきい値未満のスロットだけエンコーダに置き換えたときの一致率（gating_table）
- 確認用の文（probe）と読み分けの評価データ（eval_context）の、文脈あり・なし

    uv run python 10_encoder_train.py --smoke                    # 動作確認（極小モデル・数十行）
    uv run python 10_encoder_train.py --fracs 25 --epochs 3      # 手元の CPU での予備実験
    uv run python 10_encoder_train.py --fracs 25,50,75,100 --seeds 0,1,2   # 本番（GPU）
    uv run python 10_encoder_train.py --aggregate-only           # 集計だけやり直す

済んだ回は飛ばす（results_encoder/<モデル>/f<割合>_s<シード>/result.json があれば。--force で回し直す）。
出力: results_encoder/<モデル>/f<割合>_s<シード>/{result.json,dev_errors.tsv,eval_context.tsv,probe.tsv}、
      results_encoder/<モデル>/{summary.tsv,summary_agg.tsv}
"""

import argparse
import collections
import glob
import json
import os
import random
import statistics
import time

import torch

from encoder_core import (
    Candidates,
    CharVocab,
    Labels,
    build_model,
    load_jsonl,
    make_example,
    make_examples,
    predict,
    train,
    verify_records,
)
from llm_core import THRESHOLDS, gating_table

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "tohoku-nlp/bert-base-japanese-char-v3"
GATE = 0.5


def write_tsv(path: str, header: list[str], rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        f.writelines("\t".join(map(str, r)) + "\n" for r in rows)


def fmt_reading(r: str) -> str:
    return r if r else "---"


# ---------------------------------------------------------------- dev の漢字スロット


def eval_dev(model, dev, dev_exs, cands, labels, vocab, device) -> tuple[dict, list]:
    preds = predict(model, dev_exs, cands, vocab, device)
    rows, errors = [], []
    c = collections.Counter()
    for rec, ex, pr in zip(dev, dev_exs, preds):
        for si, (lab, _) in zip(ex.slot, pr):
            s = rec.slots[si]
            enc, w7, conf = labels.names[lab], rec.w7[si], rec.conf[si]
            rows.append((s.gold, enc, w7, conf))
            n_cands = cands.n_cands.get(s.surface, 0)
            gid = labels.id.get(s.gold, -1)
            in_cands = gid >= 0 and bool(cands.table[cands.row(s.surface), gid])
            top = labels.names[cands.top[s.surface]] if n_cands else None
            hit = enc == s.gold
            group = "unseen" if n_cands == 0 else "multi" if n_cands >= 2 else "single"
            c["n"] += 1
            c[f"{group}_n"] += 1
            c["hit"] += hit
            c[f"{group}_hit"] += hit
            c[f"{group}_w7_hit"] += w7 == s.gold
            c["top_hit"] += top == s.gold
            if not hit:
                c["err_not_in_cands" if not in_cands else "err_in_cands"] += 1
                c["err_bias"] += in_cands and enc == top
                errors.append(
                    (
                        rec.id,
                        s.surface,
                        fmt_reading(s.gold),
                        fmt_reading(enc),
                        fmt_reading(w7),
                        conf,
                        n_cands,
                        int(in_cands),
                        rec.text[max(0, s.start - 8) : s.end + 9],
                    )
                )

    def rate(a, b):
        return c[a] / c[b] if c[b] else 0.0

    res = {
        "slots": c["n"],
        "acc": rate("hit", "n"),
        "multi_n": c["multi_n"],
        "multi_acc": rate("multi_hit", "multi_n"),
        "multi_w7_acc": rate("multi_w7_hit", "multi_n"),
        "single_n": c["single_n"],
        "single_acc": rate("single_hit", "single_n"),
        "unseen_n": c["unseen_n"],
        "unseen_acc": rate("unseen_hit", "unseen_n"),
        "unseen_w7_acc": rate("unseen_w7_hit", "unseen_n"),
        "freq_top_acc": rate("top_hit", "n"),
        "errors": c["n"] - c["hit"],
        "err_not_in_cands": c["err_not_in_cands"],
        "err_in_cands": c["err_in_cands"],
        "err_bias": c["err_bias"],
        "gate": gating_table(rows),
    }
    return res, errors


# ---------------------------------------------------------------- 語の読み（probe・eval_context）


def word_readings(
    model, recs, cands, labels, vocab, device, use_context, window=0
) -> list[str]:
    """各行の語（word_idx のスロット）の読み。対象スロットはエンコーダ、それ以外は w7 のまま。"""
    exs = [
        make_example(
            r, vocab, labels, cands, r.context if use_context else "", window
        )
        for r in recs
    ]
    out = []
    for r, ex, pr in zip(recs, exs, predict(model, exs, cands, vocab, device)):
        chosen = {si: labels.names[lab] for si, (lab, _) in zip(ex.slot, pr)}
        out.append("".join(chosen.get(i, r.w7[i]) for i in r.word_idx))
    return out


def eval_probe(model, probe, cands, labels, vocab, device, window=0) -> tuple[dict, list]:
    noctx = word_readings(model, probe, cands, labels, vocab, device, False, window)
    ctx = word_readings(model, probe, cands, labels, vocab, device, True, window)
    w7 = ["".join(r.w7[i] for i in r.word_idx) for r in probe]
    res = {
        "n": len(probe),
        "ctx": sum(g == r.expected for g, r in zip(ctx, probe)),
        "noctx": sum(g == r.expected for g, r in zip(noctx, probe)),
        "w7": sum(g == r.expected for g, r in zip(w7, probe)),
    }
    rows = [
        (r.id, r.word, r.expected, c, n, w) for r, c, n, w in zip(probe, ctx, noctx, w7)
    ]
    return res, rows


def eval_words(model, recs, cands, labels, vocab, device, window=0) -> tuple[dict, list]:
    noctx = word_readings(model, recs, cands, labels, vocab, device, False, window)
    ctx = word_readings(model, recs, cands, labels, vocab, device, True, window)
    w7 = ["".join(r.w7[i] for i in r.word_idx) for r in recs]
    ok = lambda got: [g == r.expected for g, r in zip(got, recs)]  # noqa: E731
    ok_no, ok_ctx, ok_w7 = ok(noctx), ok(ctx), ok(w7)

    def group(idx: list[int]) -> dict:
        n = len(idx)
        return {
            "n": n,
            "noctx": sum(ok_no[i] for i in idx) / n if n else 0.0,
            "ctx": sum(ok_ctx[i] for i in idx) / n if n else 0.0,
            "w7": sum(ok_w7[i] for i in idx) / n if n else 0.0,
        }

    res = {}
    for split in ("dev", "train", "new"):
        res[split] = group([i for i, r in enumerate(recs) if r.split == split])
    res["dev_with_context"] = group(
        [i for i, r in enumerate(recs) if r.split == "dev" and r.context]
    )
    for kind in sorted({r.kind for r in recs}):
        res[f"dev_kind_{kind}"] = group(
            [i for i, r in enumerate(recs) if r.split == "dev" and r.kind == kind]
        )
    # 語ごとの全問正解（dev の行だけ。読みが2つ以上ある語は「多数派を選ぶだけ」では全問正解にならない）
    by_word = collections.defaultdict(list)
    for i, r in enumerate(recs):
        if r.split == "dev":
            by_word[r.word].append(i)
    multi = {
        w: ix for w, ix in by_word.items() if len({recs[i].expected for i in ix}) >= 2
    }
    for label, words in (("all", by_word), ("multi", multi)):
        res[f"dev_words_{label}"] = {
            "n": len(words),
            **{
                key: sum(all(oks[i] for i in ix) for ix in words.values())
                for key, oks in (("noctx", ok_no), ("ctx", ok_ctx), ("w7", ok_w7))
            },
        }
    rows = [
        (
            r.id,
            r.word,
            r.expected,
            r.split,
            r.kind,
            r.cue,
            n,
            c,
            w,
            int(a),
            int(b),
            int(d),
        )
        for r, n, c, w, a, b, d in zip(recs, noctx, ctx, w7, ok_no, ok_ctx, ok_w7)
    ]
    return res, rows


# ---------------------------------------------------------------- 実行と集計


def run_one(args, data, frac, seed, labels, vocab, cands, train_exs, dev_exs, out_dir):
    device = args.device
    random.seed(seed)
    torch.manual_seed(seed)
    t0 = time.time()
    model = build_model(args.model, len(labels), vocab)
    dev_n = sum(len(e.pos) for e in dev_exs)

    def diag(_epoch: int) -> str:
        preds = predict(model, dev_exs, cands, vocab, device)
        hit = sum(
            labels.names[lab] == r.slots[si].gold
            for r, ex, pr in zip(data["dev"], dev_exs, preds)
            for si, (lab, _) in zip(ex.slot, pr)
        )
        return f"dev={hit / dev_n:.4f}"

    train(
        model,
        train_exs,
        cands,
        vocab,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=seed,
        device=device,
        on_epoch=diag,
    )
    dev_res, errors = eval_dev(
        model, data["dev"], dev_exs, cands, labels, vocab, device
    )
    probe_res, probe_rows = eval_probe(
        model, data["probe"], cands, labels, vocab, device, args.window
    )
    words_res, words_rows = eval_words(
        model, data["eval_context"], cands, labels, vocab, device, args.window
    )
    result = {
        "model": args.model,
        "frac": frac,
        "seed": seed,
        "window": args.window,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "train_lines": len(train_exs),
        "seconds": round(time.time() - t0),
        "dev": dev_res,
        "probe": probe_res,
        "eval_context": words_res,
    }
    write_tsv(
        os.path.join(out_dir, "dev_errors.tsv"),
        [
            "id",
            "char",
            "gold",
            "enc",
            "w7",
            "w7_conf",
            "n_cands",
            "gold_in_cands",
            "context",
        ],
        errors,
    )
    write_tsv(
        os.path.join(out_dir, "probe.tsv"),
        ["id", "word", "expected", "enc_ctx", "enc_noctx", "w7"],
        probe_rows,
    )
    write_tsv(
        os.path.join(out_dir, "eval_context.tsv"),
        [
            "id",
            "word",
            "expected",
            "split",
            "kind",
            "cue",
            "enc_noctx",
            "enc_ctx",
            "w7",
            "ok_noctx",
            "ok_ctx",
            "ok_w7",
        ],
        words_rows,
    )
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


# 集計の列: (名前, 取り出し方)
RUN_COLS = [
    ("kanji_acc", lambda r: r["dev"]["acc"]),
    ("multi_acc", lambda r: r["dev"]["multi_acc"]),
    ("w7_multi_acc", lambda r: r["dev"]["multi_w7_acc"]),
    ("unseen_acc", lambda r: r["dev"]["unseen_acc"]),
    ("freq_top_acc", lambda r: r["dev"]["freq_top_acc"]),
    ("w7_acc", lambda r: r["dev"]["gate"].get("w7")),
    *[
        (f"gated_{t}", lambda r, t=t: r["dev"]["gate"].get(f"gated_{t}"))
        for t in THRESHOLDS
    ],
    ("err_not_in_cands", lambda r: r["dev"]["err_not_in_cands"]),
    ("err_in_cands", lambda r: r["dev"]["err_in_cands"]),
    ("err_bias", lambda r: r["dev"]["err_bias"]),
    ("evc_dev_noctx", lambda r: r["eval_context"]["dev"]["noctx"]),
    ("evc_dev_w7", lambda r: r["eval_context"]["dev"]["w7"]),
    ("evc_devctx_noctx", lambda r: r["eval_context"]["dev_with_context"]["noctx"]),
    ("evc_devctx_ctx", lambda r: r["eval_context"]["dev_with_context"]["ctx"]),
    ("evc_devctx_w7", lambda r: r["eval_context"]["dev_with_context"]["w7"]),
    ("evc_train_noctx", lambda r: r["eval_context"]["train"]["noctx"]),
    ("evc_words_noctx", lambda r: r["eval_context"]["dev_words_multi"]["noctx"]),
    ("evc_words_ctx", lambda r: r["eval_context"]["dev_words_multi"]["ctx"]),
    ("evc_words_w7", lambda r: r["eval_context"]["dev_words_multi"]["w7"]),
    ("evc_words_n", lambda r: r["eval_context"]["dev_words_multi"]["n"]),
    ("probe_noctx", lambda r: r["probe"]["noctx"]),
    ("probe_ctx", lambda r: r["probe"]["ctx"]),
    ("probe_w7", lambda r: r["probe"]["w7"]),
    ("probe_n", lambda r: r["probe"]["n"]),
    ("minutes", lambda r: r["seconds"] / 60),
]


def fmt_cell(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def write_summary(model_dir: str) -> None:
    results = []
    for path in sorted(glob.glob(os.path.join(model_dir, "f*_s*", "result.json"))):
        with open(path, encoding="utf-8") as f:
            results.append(json.load(f))
    if not results:
        return
    results.sort(key=lambda r: (r["frac"], r["seed"]))
    names = [n for n, _ in RUN_COLS]
    write_tsv(
        os.path.join(model_dir, "summary.tsv"),
        ["frac", "seed", *names],
        [
            [r["frac"], r["seed"], *(fmt_cell(f(r)) for _, f in RUN_COLS)]
            for r in results
        ],
    )
    agg = []
    for frac in sorted({r["frac"] for r in results}):
        rs = [r for r in results if r["frac"] == frac]
        cells = []
        for _, f in RUN_COLS:
            vals = [f(r) for r in rs]
            if any(v is None for v in vals):
                cells.append("")
            elif len(vals) > 1:
                cells.append(
                    f"{statistics.mean(vals):.4f}±{statistics.stdev(vals):.4f}"
                )
            else:
                cells.append(fmt_cell(vals[0]))
        agg.append([frac, len(rs), *cells])
    write_tsv(
        os.path.join(model_dir, "summary_agg.tsv"), ["frac", "seeds", *names], agg
    )
    print("\n" + "\t".join(["frac", "seeds", *names]))
    for row in agg:
        print("\t".join(map(str, row)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--fracs", default="25,50,75,100", help="train 先頭 N 行の割合（%%）"
    )
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--bundle", default=os.path.join(HERE, "bundle"))
    ap.add_argument("--results", default=os.path.join(HERE, "results_encoder"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit-train", type=int, help="学習に使う行の上限（予備実験用）")
    ap.add_argument("--limit-dev", type=int)
    ap.add_argument("--limit-eval", type=int)
    ap.add_argument(
        "--smoke", action="store_true", help="動作確認（極小モデル・数十行）"
    )
    ap.add_argument("--force", action="store_true", help="済んだ回も回し直す")
    ap.add_argument(
        "--window",
        type=int,
        default=0,
        help="スロットごとに前後 N 文字だけを見せる(0=行全体)。行全体の文脈を使っているかの切り分け用",
    )
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.model = "tiny"
        args.fracs, args.seeds, args.epochs = "100", "0", 1
        args.limit_train, args.limit_dev, args.limit_eval = 200, 40, 60
        args.results = os.path.join(HERE, "results_encoder_smoke")
    model_dir = os.path.join(
        args.results,
        args.model.replace("/", "__") + (f"__win{args.window}" if args.window else ""),
    )
    if args.aggregate_only:
        write_summary(model_dir)
        return

    data = {
        name: load_jsonl(os.path.join(args.bundle, f"{name}.jsonl"))
        for name in ("train_full", "dev_full", "probe", "eval_context")
    }
    data["train"], data["dev"] = data.pop("train_full"), data.pop("dev_full")
    if args.limit_train:
        data["train"] = data["train"][: args.limit_train]
    if args.limit_dev:
        data["dev"] = data["dev"][: args.limit_dev]
    if args.limit_eval:
        data["eval_context"] = data["eval_context"][: args.limit_eval]
        data["probe"] = data["probe"][:4]
    for recs in data.values():
        verify_records(recs)

    # 語彙は char-v3 のもの（tiny はこれを借りる）
    vocab = CharVocab.from_model(DEFAULT_MODEL if args.model == "tiny" else args.model)
    labels = Labels(data["train"])
    print(
        f"device={args.device} model={args.model} train={len(data['train'])} dev={len(data['dev'])} "
        f"labels={len(labels)} eval_context={len(data['eval_context'])} probe={len(data['probe'])}",
        flush=True,
    )

    os.makedirs(model_dir, exist_ok=True)
    for frac in [int(x) for x in args.fracs.split(",")]:
        sub = data["train"][: round(len(data["train"]) * frac / 100)]
        cands = Candidates(sub, labels)
        train_exs = make_examples(sub, vocab, labels, cands, args.window)
        dev_exs = make_examples(data["dev"], vocab, labels, cands, args.window)
        for seed in [int(x) for x in args.seeds.split(",")]:
            out_dir = os.path.join(model_dir, f"f{frac:03d}_s{seed}")
            if os.path.exists(os.path.join(out_dir, "result.json")) and not args.force:
                print(f"[{frac}% seed={seed}] 済み（飛ばす）", flush=True)
                continue
            os.makedirs(out_dir, exist_ok=True)
            print(f"[{frac}% seed={seed}] {len(sub)} 行", flush=True)
            r = run_one(
                args,
                data,
                frac,
                seed,
                labels,
                vocab,
                cands,
                train_exs,
                dev_exs,
                out_dir,
            )
            print(
                f"[{frac}% seed={seed}] dev={r['dev']['acc']:.4f} multi={r['dev']['multi_acc']:.4f} "
                f"gated{GATE}={r['dev']['gate'].get(f'gated_{GATE}', 0):.4f}（{r['seconds']} 秒）",
                flush=True,
            )
            write_summary(model_dir)
    write_summary(model_dir)


if __name__ == "__main__":
    main()
