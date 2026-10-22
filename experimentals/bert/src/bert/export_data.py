"""dataset/basic_raw.tsv から、実験とクラウドの微調整が使うデータを毎回作り直す。

原文のハッシュで train と dev に分ける（SHA-256 を 10 で割った余りが 0 なら dev）。
- 今ある文の振り分けは、何度作り直しても変わらない。データを足しても dev の中身は入れ替わらない
- 足した文は約 1 割が dev に入る
- 注釈（読みの列）を直しても振り分けは変わらない（原文だけで決める）
- 同じ原文の行は必ず同じ側に入る
規則は tools/llm_reading/01_split.py と同じ（過去の実験の dev とそろう）。

    uv run export_data                            # 学習・評価用の JSONL を作る
    uv run export_data --w7 path/to/basic_data_7.mbm   # dev に w7 の読みと自信度も添える

出力: work/data/
  train_full.jsonl, dev_full.jsonl  行ごとの原文とスロット（[文字, 読み, 文字種, 開始, 終了]）。
                                    --w7 を付けると dev に w7 の読み（w7）と自信度（conf）を添える
  train_raw.tsv, dev_raw.tsv        basic_raw.tsv 形式の分割。trainer に渡すと train だけの w7 を作れる
  meta.json

dev の w7 比較（step7_head）には、train だけで学習した w7 が要る。全データで学習した w7 を渡すと、
dev はその w7 の学習済みなので比較にならない。作り方は README を参照。
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .common import WORK
from .gold import GoldSentence, parse_gold, read_raw_lines

ROOT = Path(__file__).resolve().parents[4]
RAW_PATH = ROOT / "dataset" / "basic_raw.tsv"
DATA = WORK / "data"
MODULUS = 10


def is_dev(text: str, modulus: int = MODULUS) -> bool:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % modulus == 0


def slot_rows(g: GoldSentence) -> list[list]:
    return [[s.surface, s.gold, s.ctype, s.start, s.end] for s in g.slots]


def w7_readings(w7, g: GoldSentence) -> tuple[list[str], list[float]]:
    """w7 の予測を、正解と同じスロットに割り付ける（読みと自信度。マスあけは除く）。"""
    owner = {p: i for i, s in enumerate(g.slots) for p in range(s.start, s.end)}
    readings = [""] * len(g.slots)
    conf = [1.0] * len(g.slots)
    res = w7.predict(g.text)
    for ch, src, c in zip(res.kana, res.kana_to_source, res.confidences):
        si = owner.get(src)
        if si is None or (ch == " " and not g.slots[si].surface.isspace()):
            continue
        readings[si] += ch
        conf[si] = min(conf[si], c)
    return readings, conf


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def export(raw_path: Path = RAW_PATH, w7_path: str | None = None) -> dict[str, Any]:
    lines = read_raw_lines(str(raw_path))
    golds = [parse_gold(line, n) for n, line in enumerate(lines, 1)]
    dev_ix = {i for i, g in enumerate(golds) if is_dev(g.text)}

    w7 = None
    if w7_path:
        import momors_py

        w7 = momors_py.Predictor(w7_path)

    DATA.mkdir(parents=True, exist_ok=True)
    train = [(i, g) for i, g in enumerate(golds) if i not in dev_ix]
    dev = [(i, g) for i, g in enumerate(golds) if i in dev_ix]
    rows = {}
    for name, part in (("train", train), ("dev", dev)):
        out = []
        for n, (_, g) in enumerate(part):
            row = {"id": n, "text": g.text, "slots": slot_rows(g)}
            if name == "dev" and w7 is not None:
                row["w7"], row["conf"] = w7_readings(w7, g)
                row["conf"] = [round(c, 4) for c in row["conf"]]
            out.append(row)
        write_jsonl(DATA / f"{name}_full.jsonl", out)
        with open(DATA / f"{name}_raw.tsv", "w", encoding="utf-8", newline="\n") as f:
            f.writelines(lines[i] + "\n" for i, _ in part)
        rows[name] = len(out)
    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(raw_path.relative_to(ROOT))
        if raw_path.is_relative_to(ROOT)
        else str(raw_path),
        "modulus": MODULUS,
        "counts": rows,
        "w7": w7_path,
    }
    (DATA / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--w7", help="dev に添える w7 のモデル（.mbm。train だけで学習したもの）"
    )
    args = ap.parse_args()
    meta = export(w7_path=args.w7)
    c = meta["counts"]
    print(
        f"全 {c['train'] + c['dev']} 行 → train {c['train']} / dev {c['dev']}（{DATA}）"
    )


if __name__ == "__main__":
    main()
