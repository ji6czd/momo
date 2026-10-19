"""A1: basic_raw.tsv を文単位で train / dev に分ける（原文のハッシュで振り分ける）。

各行の原文（{…} と [...] の記法を除いたもの）の SHA-256 を 10 で割った余りが 0 なら dev。
- 今ある文の振り分けは、何度作り直しても変わらない。データを足しても dev の中身が入れ替わらない
- 後から足した文も同じ規則で約1割が dev に入るので、dev はデータとともに育つ
- 注釈（読みの列）を直しても振り分けは変わらない（原文の列だけで決める）
- 同じ原文の行は必ず同じ側に入る（train と dev に分かれて答えが漏れることがない）
- 将来 dev を減らすなら --modulus 20 にすれば、今の dev の部分集合に絞り込める

ある時点の dev を固定した物差しとして残すには --snapshot で原文のハッシュの一覧を書き出す
（splits/ は git 管理下）。dev は実験用のモデルからだけ外す。momo 本体は全データで学習する。

train は seed 固定で並べ替えて書く。先頭から N 行取れば入れ子の部分集合になる
（02_learning_curve.py の 25/50/75/100% はこれを使う）。

    uv run python 01_split.py [--modulus 10] [--snapshot 2026-10-20]

出力: work/split/train_raw.tsv, work/split/dev_raw.tsv、--snapshot なら splits/dev_<名前>.txt
"""

import argparse
import hashlib
import os
import random

from _common import (
    DEV_RAW,
    RAW_PATH,
    SCRIPT_DIR,
    SPLIT_DIR,
    TRAIN_RAW,
    parse_gold,
    read_raw_lines,
)

SNAPSHOT_DIR = os.path.join(SCRIPT_DIR, "splits")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_dev(text: str, modulus: int) -> bool:
    return int(text_hash(text), 16) % modulus == 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--modulus", type=int, default=10, help="dev にする割合の逆数（10 で約1割）"
    )
    ap.add_argument(
        "--seed", type=int, default=20261019, help="train の並べ替え（学習曲線用）"
    )
    ap.add_argument(
        "--snapshot", help="この名前で、今の dev の原文ハッシュ一覧を splits/ に残す"
    )
    args = ap.parse_args()

    lines = read_raw_lines(RAW_PATH)
    texts = [parse_gold(line, i).text for i, line in enumerate(lines, 1)]
    dev = [i for i, t in enumerate(texts) if is_dev(t, args.modulus)]
    dev_set = set(dev)
    train = [i for i in range(len(lines)) if i not in dev_set]
    random.Random(args.seed).shuffle(train)

    os.makedirs(SPLIT_DIR, exist_ok=True)
    for path, idx in ((TRAIN_RAW, train), (DEV_RAW, dev)):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(lines[i] + "\n" for i in idx))

    print(
        f"全 {len(lines)} 行 → train {len(train)} / dev {len(dev)}（原文ハッシュ mod {args.modulus} == 0）"
    )
    print(f"  {TRAIN_RAW}\n  {DEV_RAW}")

    if args.snapshot:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        path = os.path.join(SNAPSHOT_DIR, f"dev_{args.snapshot}.txt")
        hashes = sorted({text_hash(texts[i]) for i in dev})
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(
                f"# dev の原文 SHA-256（{len(hashes)} 文、mod {args.modulus}）。01_split.py --snapshot で作成\n"
            )
            f.writelines(h + "\n" for h in hashes)
        print(f"  スナップショット → {path}")


if __name__ == "__main__":
    main()
