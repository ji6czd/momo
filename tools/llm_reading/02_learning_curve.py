"""A2: 学習データの量を変えて w7 を学習し、dev での伸び方から律速を判定する。

各回は work/frac_XXX/ に train の先頭 N 行を basic_raw.tsv として置き、本番の
`task train` と同じ手順（createdata → makedic → train --window 7
--boundary-n-estimators 200）を momo-py の環境（uv run --project momo-py）で回す。
学習済みの回は飛ばす（--force で学び直す）。評価は momors_py で dev を推論する。

    uv run python 02_learning_curve.py [--fracs 25,50,75,100] [--jobs 16] [--force] [--eval-only]

出力: work/learning_curve.tsv（回ごとの指標）。本番モデル（dataset/basic_data_7.mbm）も
参考として並べるが、これは dev を含めて学習しているので値は高めに出る。
"""

import argparse
import os
import subprocess
import time

import momors_py

from _common import (
    DATASET,
    MOMO_PY,
    TRAIN_RAW,
    WORK,
    load_dev,
    predict_all,
    read_raw_lines,
    score,
)

COLUMNS = [
    "kanji_slot_acc",
    "slot_acc",
    "concat_reading_acc",
    "sentence_exact",
    "boundary_p",
    "boundary_r",
    "boundary_f1",
    "kanji_cont_gold",
    "kanji_cont_pred",
]


def frac_dir(frac: int) -> str:
    return os.path.join(WORK, f"frac_{frac:03d}")


def run(cmd: list[str]) -> None:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def train_fraction(frac: int, train_lines: list[str], jobs: int, force: bool) -> None:
    d = frac_dir(frac)
    raw = os.path.join(d, "basic_raw.tsv")
    data = os.path.join(d, "basic_data.tsv")  # createdata は *_raw.tsv → *_data.tsv
    mbm = os.path.join(d, "basic_data_7.mbm")
    if os.path.exists(mbm) and not force:
        print(f"[{frac}%] 学習済み: {mbm}")
        return

    os.makedirs(d, exist_ok=True)
    n = round(len(train_lines) * frac / 100)
    with open(raw, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(line + "\n" for line in train_lines[:n]))
    print(f"[{frac}%] {n} 行で学習", flush=True)

    uv = ["uv", "run", "--project", MOMO_PY]
    t0 = time.time()
    run([*uv, "trainer", "createdata", "--raw", raw])
    run([*uv, "makedic", data, os.path.join(d, "single_character_dic.tsv")])
    train_cmd = [
        *uv,
        "trainer",
        "train",
        "--tsv",
        data,
        "--window",
        "7",
        "--jobs",
        str(jobs),
        "--boundary-n-estimators",
        "200",
    ]
    name_dict = os.path.join(d, "person_name_dic.tsv")
    if os.path.exists(name_dict):
        train_cmd += ["--name-dict", name_dict]
    run(train_cmd)
    print(f"[{frac}%] 学習完了 {time.time() - t0:.0f} 秒", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fracs", default="25,50,75,100")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="学習済みの回も学び直す")
    ap.add_argument("--eval-only", action="store_true", help="学習せず評価だけ行う")
    args = ap.parse_args()
    fracs = [int(x) for x in args.fracs.split(",")]

    train_lines = read_raw_lines(TRAIN_RAW)
    if not args.eval_only:
        for frac in fracs:
            train_fraction(frac, train_lines, args.jobs, args.force)

    dev = load_dev()
    models = [
        (
            f"{frac}% ({round(len(train_lines) * frac / 100)}行)",
            os.path.join(frac_dir(frac), "basic_data_7.mbm"),
        )
        for frac in fracs
    ]
    models.append(
        ("本番 w7（dev を含む・参考）", os.path.join(DATASET, "basic_data_7.mbm"))
    )

    rows = []
    for label, mbm in models:
        if not os.path.exists(mbm):
            print(f"モデルなし（飛ばす）: {mbm}")
            continue
        m = score(predict_all(momors_py.Predictor(mbm), dev))
        rows.append((label, m))

    out = os.path.join(WORK, "learning_curve.tsv")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(["model", *COLUMNS]) + "\n")
        for label, m in rows:
            f.write("\t".join([label, *(f"{m[c]:.4f}" for c in COLUMNS)]) + "\n")

    print(f"\ndev {len(dev)} 文")
    print(f"{'model':<28}" + "".join(f"{c:>20}" for c in COLUMNS))
    for label, m in rows:
        print(f"{label:<28}" + "".join(f"{m[c]:>20.4f}" for c in COLUMNS))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
