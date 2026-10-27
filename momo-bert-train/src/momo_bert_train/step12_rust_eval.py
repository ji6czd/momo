"""Rust の推論経路（momors_py）で、w7 単独と w7＋BERT 読みモードを dev で比べる。

step9 までの一致率は BERT 単体（漢字スロットの分類だけ）の値。ここでは momo 本体の推論経路
（助数詞・漢数字・人名辞書・カスタム辞書・「々」の規則と境界モデルを含む）に BERT を差し込んだ
結果を、同じ物差し（dev の漢字スロット一致率）で w7 と並べる。

    uv run step12_rust_eval                                  # 既定: work/w7_train と work/onnx/train
    uv run step12_rust_eval --w7 path/to/basic_data_7.mbm --bert work/onnx/train --bert-f32

w7 は train だけで学習したもの（work/w7_train、作り方は README）を使う。全データで学習した w7 は
dev を学習済みで比較にならない。BERT 読みモードには feature `bert` 付きの momors_py が要る（README）。

出力: work/rust_eval/<BERT の名前>/
  changed.tsv   w7 と BERT で読みが変わった漢字スロット（どちらが正解か、前後の文字つき）
  errors.tsv    BERT が外した漢字スロット全部（w7 も同じ誤りのものを含む。前後の文字と行つき）
  context.tsv   eval_context.tsv（dev / new の行）の、語ごとの読みの当否
  summary.json  一致率と速度
"""

import argparse
import collections
import json
import time
from pathlib import Path

import momors_py

from .common import DATA, WORK, is_target, load_jsonl
from .export_data import ROOT, w7_readings

CONTEXT = 8  # changed.tsv に出す前後の文字数
EVAL_CONTEXT = ROOT / "dataset" / "eval_context.tsv"


def run(predictor, recs):
    """各行のスロットの読みと、かかった時間（秒）を返す。"""
    t0 = time.perf_counter()
    out = [w7_readings(predictor, r)[0] for r in recs]
    return out, time.perf_counter() - t0


def slot_accuracy(recs, readings) -> tuple[float, float, int]:
    """(漢字スロット一致率, 全スロットが当たった行の割合, 漢字スロット数)。"""
    hit = n = lines_ok = 0
    for r, rd in zip(recs, readings):
        ok = True
        for s, got in zip(r.slots, rd):
            if is_target(s):
                hit += got == s.gold
                n += 1
            ok &= got == s.gold or s.ctype not in ("KANJI", "JAPANESE_NUMERIC")
        lines_ok += ok
    return hit / n, lines_ok / len(recs), n


def word_reading(predictor, text: str, word: str, occ: int) -> str | None:
    """text の occ 番目の word に当たるかなを返す（空白は除く）。"""
    start = -1
    for _ in range(occ):
        start = text.find(word, start + 1)
        if start < 0:
            return None
    r = predictor.predict(text)
    span = range(start, start + len(word))
    return "".join(
        ch for ch, src in zip(r.kana, r.kana_to_source) if src in span and ch != " "
    )


def eval_context(predictors: dict, splits=("dev", "new")):
    """eval_context.tsv の行ごとの当否と、語ごとに全部の読みを当てたかを返す。"""
    rows = []
    with open(EVAL_CONTEXT, encoding="utf-8") as f:
        lines = [
            line.rstrip("\n") for line in f if line.strip() and not line.startswith("#")
        ]
    header = lines[0].split("\t")
    for line in lines[1:]:
        row = dict(zip(header, line.split("\t")))
        if row.get("split") not in splits:
            continue
        occ = int(row["occ"]) if row.get("occ") else 1
        got = {
            name: word_reading(p, row["text"], row["word"], occ)
            for name, p in predictors.items()
        }
        rows.append((row, got))
    by_word = collections.defaultdict(list)
    for row, got in rows:
        by_word[row["word"]].append((row, got))
    # 同じ語に 2 つ以上の読みがある語だけを「読み分け」として数える
    words = {
        w: rs for w, rs in by_word.items() if len({r["expected"] for r, _ in rs}) >= 2
    }
    summary = {}
    for name in predictors:
        summary[name] = {
            "rows": sum(got[name] == row["expected"] for row, got in rows)
            / max(len(rows), 1),
            "words_all_correct": sum(
                all(got[name] == row["expected"] for row, got in rs)
                for rs in words.values()
            ),
        }
    summary["n_rows"] = len(rows)
    summary["n_words"] = len(words)
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--w7", default=str(WORK / "w7_train" / "basic_data_7.mbm"))
    ap.add_argument(
        "--bert", default=str(WORK / "onnx" / "train"), help="export_onnx の出力"
    )
    ap.add_argument(
        "--bert-f32", action="store_true", help="float32 版を使う（既定は int8）"
    )
    ap.add_argument(
        "--threads", type=int, default=0, help="BERT のスレッド数（0=既定）"
    )
    args = ap.parse_args()

    recs = load_jsonl(str(DATA / "dev_full.jsonl"))
    w7 = momors_py.Predictor(args.w7)
    bert = momors_py.Predictor(
        args.w7, bert=args.bert, bert_f32=args.bert_f32, bert_threads=args.threads
    )
    if bert.bert_unknown_labels:
        print(f"BERT のラベルのうち w7 に無いもの: {bert.bert_unknown_labels[:10]}")

    n_chars = sum(len(r.text) for r in recs)
    rd_w7, sec_w7 = run(w7, recs)
    rd_bert, sec_bert = run(bert, recs)
    acc_w7, line_w7, n_slots = slot_accuracy(recs, rd_w7)
    acc_bert, line_bert, _ = slot_accuracy(recs, rd_bert)
    print(f"dev {len(recs)} 行 / {n_chars} 字 / 漢字スロット {n_slots}")
    print(
        f"  w7      漢字スロット {acc_w7:.4f}  行 {line_w7:.4f}  {sec_w7 * 1e6 / n_chars:7.1f} µs/字"
    )
    print(
        f"  w7+BERT 漢字スロット {acc_bert:.4f}  行 {line_bert:.4f}  {sec_bert * 1e6 / n_chars:7.1f} µs/字"
    )

    out = WORK / "rust_eval" / Path(args.bert).name
    out.mkdir(parents=True, exist_ok=True)
    changed = []
    errors = []
    gain = loss = 0
    for r, a, b in zip(recs, rd_w7, rd_bert):
        for s, x, y in zip(r.slots, a, b):
            if is_target(s) and y != s.gold:
                kind = (
                    "w7は正解"
                    if x == s.gold
                    else "w7も同じ誤り"
                    if x == y
                    else "w7は別の誤り"
                )
                ctx = r.text[max(0, s.start - CONTEXT) : s.end + CONTEXT]
                errors.append((kind, s.surface, s.gold, y, x, ctx, r.text))
            if not is_target(s) or x == y:
                continue
            verdict = "BERT" if y == s.gold else "w7" if x == s.gold else "両方誤り"
            gain += verdict == "BERT"
            loss += verdict == "w7"
            ctx = r.text[max(0, s.start - CONTEXT) : s.end + CONTEXT]
            changed.append((verdict, s.surface, s.gold, x, y, ctx))
    with open(out / "changed.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("正解側\t文字\t正解\tw7\tBERT\t前後\n")
        f.writelines("\t".join(map(str, c)) + "\n" for c in sorted(changed))
    print(
        f"  読みが変わったスロット {len(changed)}（BERT が正解 {gain} / w7 が正解 {loss}）"
    )
    with open(out / "errors.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("w7\t文字\t正解\tBERT\tw7の読み\t前後\t行\n")
        f.writelines("\t".join(map(str, e)) + "\n" for e in sorted(errors))
    kinds = collections.Counter(e[0] for e in errors)
    print(
        f"  BERT の誤り {len(errors)}（"
        + " / ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
        + "）"
    )

    rows, ctx_summary = eval_context({"w7": w7, "bert": bert})
    with open(out / "context.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("id\tword\texpected\tw7\tbert\tsplit\ttext\n")
        for row, got in rows:
            f.write(
                "\t".join(
                    [
                        row["id"],
                        row["word"],
                        row["expected"],
                        got["w7"] or "",
                        got["bert"] or "",
                        row["split"],
                        row["text"],
                    ]
                )
                + "\n"
            )
    print(
        f"  eval_context（dev/new {ctx_summary['n_rows']} 行・読み分け {ctx_summary['n_words']} 語）"
        f": 行 w7 {ctx_summary['w7']['rows']:.4f} / BERT {ctx_summary['bert']['rows']:.4f}、"
        f"全部の読みを当てた語 w7 {ctx_summary['w7']['words_all_correct']}"
        f" / BERT {ctx_summary['bert']['words_all_correct']}"
    )

    summary = {
        "w7": args.w7,
        "bert": args.bert,
        "bert_f32": args.bert_f32,
        "dev_lines": len(recs),
        "dev_chars": n_chars,
        "kanji_slots": n_slots,
        "acc_w7": acc_w7,
        "acc_bert": acc_bert,
        "line_w7": line_w7,
        "line_bert": line_bert,
        "us_per_char_w7": sec_w7 * 1e6 / n_chars,
        "us_per_char_bert": sec_bert * 1e6 / n_chars,
        "changed": len(changed),
        "changed_bert_right": gain,
        "changed_w7_right": loss,
        "eval_context": ctx_summary,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"保存先: {out}")


if __name__ == "__main__":
    main()
