"""GPU キット: クラウドへ持ち込むデータを書き出し、キットを zip にまとめる（手元で実行）。

クラウドの Linux では momo_py と momors_py（Windows 版の wheel しかない）を使わずに済むよう、
評価に要るものを手元で JSONL に書き出す。

- fewshot.jsonl  LLM に見せる手本（train の短い文、正解スロットつき）
- short.jsonl    dev の短い文（40字以下）: 正解スロット・w7 の読みと自信度
- context.jsonl  dev の長い行（段落）の2文目以降: 同じ段落の前の文を文脈とし、その読み（w7）も添える
- context_train.jsonl  同じものを train の段落から。basic_raw はほぼ1行1文で dev の段落は 43 行しか
                 ないので数を補う。w7 はこれらを学習済みなので w7 系の数字は参考にならないが、
                 LLM の「文脈あり 対 なし」の比較には使える（LLM はこのデータを見ていない）
- probe.jsonl    確認用の文（probe_context.tsv）: 文脈とその読み（w7）、注目する語のスロット、期待値
- single_character_dic.tsv  train だけから作った辞書
- train_full.jsonl   エンコーダの学習用: train の全行（スロットと正解）
- dev_full.jsonl     エンコーダの評価用: dev の全行（正解・w7 の読みと自信度）
- eval_context.jsonl 読み分けの評価データ dataset/eval_context.tsv の各行（w7 の読みつき）
- meta.json

キットの中身はこれら（bundle/）と、llm_core.py・07_gpu_eval.py・encoder_core.py・
10_encoder_train.py・gpu_kit/ の pyproject.toml と README.md。

    uv run --group llm python 06_export_bundle.py

出力: work/gpu_kit/（展開済み）と work/gpu_kit.zip。w7 は work/frac_100/（train だけで学習）。
"""

import importlib
import json
import os
import shutil
import time
import zipfile

import momors_py

from _common import (
    SCRIPT_DIR,
    TRAIN_RAW,
    WORK,
    GoldSentence,
    align_prediction,
    load_dev,
    parse_gold,
    pick_fewshot,
    read_raw_lines,
    slots_for_text,
)
from llm_core import ruby

FRAC100 = os.path.join(WORK, "frac_100")
KIT = os.path.join(WORK, "gpu_kit")
SENT_END = "。！？!?"
CLOSERS = "」』）)】〕"
MAX_CONTEXT = 300  # 文脈に置く前の文の上限（字）。超えたら古い文から落とす
TARGETS_PER_LINE = 3  # 1段落から取る対象の文の上限


def slot_rows(slots, offset: int = 0) -> list[list]:
    return [
        [s.surface, s.gold, s.ctype, s.start - offset, s.end - offset] for s in slots
    ]


def split_sentences(text: str) -> list[tuple[int, int]]:
    """文末記号（と直後の閉じ括弧）で区切った (開始, 終了) の並び。"""
    spans, start, i = [], 0, 0
    while i < len(text):
        if text[i] in SENT_END:
            i += 1
            while i < len(text) and text[i] in CLOSERS:
                i += 1
            spans.append((start, i))
            start = i
            continue
        i += 1
    if start < len(text):
        spans.append((start, len(text)))
    return [(a, b) for a, b in spans if text[a:b].strip()]


def export_short(w7, golds) -> list[dict]:
    out = []
    for n, g in enumerate(golds):
        p = align_prediction(g, w7.predict(g.text))
        out.append(
            {
                "id": f"short-{n}",
                "text": g.text,
                "slots": slot_rows(g.slots),
                "w7": p.readings,
                "conf": [round(c, 4) for c in p.confidences],
            }
        )
    return out


def export_context(w7, golds) -> list[dict]:
    """段落の2文目以降を対象に、同じ段落の前の文を文脈として添える。

    w7 は段落全体で推論する（実際の点訳と同じ）。文脈の読みは w7 のものを使う
    （本番では前の文はすでに処理済みで、正解は分からないため）。
    """
    out = []
    for n, g in enumerate(golds):
        p = align_prediction(g, w7.predict(g.text))
        spans = split_sentences(g.text)
        for k in range(1, min(len(spans), TARGETS_PER_LINE + 1)):
            s0, s1 = spans[k]
            idx = [i for i, s in enumerate(g.slots) if s0 <= s.start < s1]
            if s1 - s0 < 6 or not any(g.slots[i].ctype == "KANJI" for i in idx):
                continue
            c0 = s0
            for a, _ in reversed(spans[:k]):
                if s0 - a > MAX_CONTEXT:
                    break
                c0 = a
            ctx = [i for i, s in enumerate(g.slots) if c0 <= s.start < s0]
            out.append(
                {
                    "id": f"ctx-{n}-{k}",
                    "context": g.text[c0:s0],
                    "context_ruby": ruby(
                        [g.slots[i] for i in ctx], [p.readings[i] for i in ctx]
                    ),
                    "text": g.text[s0:s1],
                    "slots": slot_rows([g.slots[i] for i in idx], s0),
                    "w7": [p.readings[i] for i in idx],
                    "conf": [round(p.confidences[i], 4) for i in idx],
                }
            )
    return out


def find_nth(text: str, word: str, occ: int = 1) -> int:
    """text の中で word が occ 番目に現れる位置。"""
    pos = -1
    for _ in range(occ):
        pos = text.find(word, pos + 1)
        if pos < 0:
            raise ValueError(f"{word!r} が {occ} 回現れない: {text!r}")
    return pos


def export_word_rows(w7, items: list[dict]) -> list[dict]:
    """語の読みを問う行（id・context・text・word・expected・occ と、そのほかの列）に、
    スロットと w7 の読み・自信度、語に当たるスロットの番号（word_idx）を添える。"""
    out = []
    for it in items:
        text, context = it["text"], it["context"]
        sent = GoldSentence("", text, slots_for_text(text))
        p = align_prediction(sent, w7.predict(text))
        ctx_ruby = ""
        if context:
            cs = GoldSentence("", context, slots_for_text(context))
            ctx_ruby = ruby(
                cs.slots, align_prediction(cs, w7.predict(context)).readings
            )
        start = find_nth(text, it["word"], it.get("occ", 1))
        out.append(
            {
                **it,
                "context_ruby": ctx_ruby,
                "slots": slot_rows(sent.slots),
                "w7": p.readings,
                "conf": [round(c, 4) for c in p.confidences],
                "word_idx": [
                    i
                    for i, s in enumerate(sent.slots)
                    if start <= s.start < start + len(it["word"])
                ],
            }
        )
    return out


def export_probe(w7, path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            pid, context, text, word, expected = line.rstrip("\n").split("\t")[:5]
            items.append(
                {
                    "id": pid,
                    "context": context,
                    "text": text,
                    "word": word,
                    "expected": expected,
                }
            )
    return export_word_rows(w7, items)


def export_eval_context(w7) -> list[dict]:
    """dataset/eval_context.tsv の各行。"""
    rows = importlib.import_module("09_eval_context").load_rows()
    items = [
        {
            "id": r["id"],
            "context": r["context"],
            "text": r["text"],
            "word": r["word"],
            "occ": int(r["occ"] or 1),
            "expected": r["expected"],
            "kind": r["kind"],
            "cue": r["cue"],
            "split": r["split"],
        }
        for r in rows
    ]
    return export_word_rows(w7, items)


def export_full(golds, w7=None) -> list[dict]:
    """エンコーダ用に、行ごとのスロットと正解を書き出す（w7 を渡せば読みと自信度も添える）。"""
    out = []
    for n, g in enumerate(golds):
        row = {"id": n, "text": g.text, "slots": slot_rows(g.slots)}
        if w7 is not None:
            p = align_prediction(g, w7.predict(g.text))
            row["w7"] = p.readings
            row["conf"] = [round(c, 4) for c in p.confidences]
        out.append(row)
    return out


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main() -> None:
    w7 = momors_py.Predictor(os.path.join(FRAC100, "basic_data_7.mbm"))
    dev = load_dev()
    short_golds = [g for g in dev if len(g.text) <= 40]
    long_golds = [g for g in dev if len(g.text) > 40]
    train_paragraphs = [
        g
        for g in (
            parse_gold(line, n) for n, line in enumerate(read_raw_lines(TRAIN_RAW), 1)
        )
        if len(split_sentences(g.text)) >= 2
    ]

    if os.path.isdir(KIT):
        shutil.rmtree(KIT)
    bundle = os.path.join(KIT, "bundle")
    os.makedirs(bundle)

    fewshot = [{"text": g.text, "slots": slot_rows(g.slots)} for g in pick_fewshot(6)]
    short = export_short(w7, short_golds)
    context = export_context(w7, long_golds)
    context_train = export_context(w7, train_paragraphs)
    probe = export_probe(w7, os.path.join(SCRIPT_DIR, "probe_context.tsv"))
    write_jsonl(os.path.join(bundle, "fewshot.jsonl"), fewshot)
    write_jsonl(os.path.join(bundle, "short.jsonl"), short)
    write_jsonl(os.path.join(bundle, "context.jsonl"), context)
    write_jsonl(os.path.join(bundle, "context_train.jsonl"), context_train)
    write_jsonl(os.path.join(bundle, "probe.jsonl"), probe)
    train_full = export_full(
        parse_gold(line, n) for n, line in enumerate(read_raw_lines(TRAIN_RAW), 1)
    )
    dev_full = export_full(dev, w7)
    eval_context = export_eval_context(w7)
    write_jsonl(os.path.join(bundle, "train_full.jsonl"), train_full)
    write_jsonl(os.path.join(bundle, "dev_full.jsonl"), dev_full)
    write_jsonl(os.path.join(bundle, "eval_context.jsonl"), eval_context)
    shutil.copy(os.path.join(FRAC100, "single_character_dic.tsv"), bundle)
    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "w7": "work/frac_100/basic_data_7.mbm（train だけで学習）",
        "counts": {
            "fewshot": len(fewshot),
            "short": len(short),
            "context": len(context),
            "context_train": len(context_train),
            "probe": len(probe),
            "train_full": len(train_full),
            "dev_full": len(dev_full),
            "eval_context": len(eval_context),
        },
        "note": "probe の expected は注釈者が確認済み（2026-10-20）",
    }
    with open(os.path.join(bundle, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    for name in (
        "llm_core.py",
        "07_gpu_eval.py",
        "encoder_core.py",
        "10_encoder_train.py",
    ):
        shutil.copy(os.path.join(SCRIPT_DIR, name), KIT)
    for name in ("pyproject.toml", "README.md"):
        shutil.copy(os.path.join(SCRIPT_DIR, "gpu_kit", name), KIT)

    zip_path = os.path.join(WORK, "gpu_kit.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(KIT):
            for name in files:
                full = os.path.join(root, name)
                zf.write(full, os.path.relpath(full, WORK))

    print(json.dumps(meta["counts"], ensure_ascii=False))
    print(f"→ {KIT}\n→ {zip_path}（{os.path.getsize(zip_path) / 1024:.0f} KB）")


if __name__ == "__main__":
    main()
