"""保存した文字ベクトルで、読みの分類器だけを学習し、w7 と比べる（step6_extract の続き）。

分類器は、ベクトルから読みのラベルへの線形層（--hidden を付ければ1隠れ層）。微調整の実験と同じく、
その文字の辞書候補だけに絞った softmax で選ぶ。評価は dev の漢字スロット一致率で、同じスロットでの
w7 の読み（export_data --w7 で dev に添えたもの。無ければ w7 の欄は出さない）と並べる。

    uv run python -m bert.step7_head --vectors L1_w0
    uv run python -m bert.step7_head --vectors L4_w0 --hidden 512 --epochs 40

比べる数字（dev 923 行・漢字スロット一致率。memory: project-encoder-first-experiment）:
  w7 95.4% ／ 微調整（3シード平均）97.68%
"""

import argparse
import collections
import time

import numpy as np
import torch
from torch import nn

from .common import (
    DATA,
    WORK,
    Candidates,
    Labels,
    load_jsonl,
)

NEG = -1e9


def load_split(path):
    z = np.load(path)
    return z["x"], z["rec"], z["slot"], z["gold"]


def build_head(dim: int, n_labels: int, hidden: int) -> nn.Module:
    if hidden:
        return nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_labels),
        )
    return nn.Linear(dim, n_labels)


def masked(logits, table, rows):
    return logits.masked_fill(~table[rows], NEG)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vectors", default="L1_w0", help="work/vectors/ 下のタグ")
    ap.add_argument("--hidden", type=int, default=0, help="0 なら線形層だけ")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--save", help="最後のシードの分類器を work/heads/<名前>.pt に保存する")
    args = ap.parse_args()

    train = load_jsonl(str(DATA / "train_full.jsonl"))
    dev = load_jsonl(str(DATA / "dev_full.jsonl"))
    labels = Labels(train)
    cands = Candidates(train, labels)

    vec_dir = WORK / "vectors" / args.vectors
    xt, rt, st, gt = load_split(vec_dir / "train.npz")
    xd, rd, sd, gd = load_split(vec_dir / "dev.npz")
    xt, xd = torch.from_numpy(xt).float(), torch.from_numpy(xd).float()
    mean, std = xt.mean(0), xt.std(0).clamp_min(1e-6)
    xt, xd = (xt - mean) / std, (xd - mean) / std
    gt_t, gd_t = torch.from_numpy(gt), torch.from_numpy(gd)
    table = cands.table
    rows_t = torch.tensor(
        [cands.row(train[r].slots[s].surface) for r, s in zip(rt, st)]
    )
    rows_d = torch.tensor([cands.row(dev[r].slots[s].surface) for r, s in zip(rd, sd)])

    # 学習: 正解がラベル表にあるスロットだけ（train は全部ある）
    keep = gt_t >= 0
    xt, gt_t, rows_t = xt[keep], gt_t[keep], rows_t[keep]

    dev_gold = [dev[r].slots[s].gold for r, s in zip(rd, sd)]
    has_w7 = all(hasattr(r, "w7") for r in dev)
    w7_ok = (
        np.array([a == dev[r].w7[s] for a, r, s in zip(dev_gold, rd, sd)])
        if has_w7
        else np.ones(len(dev_gold), dtype=bool)
    )
    multi = np.array(
        [cands.n_cands.get(dev[r].slots[s].surface, 0) >= 2 for r, s in zip(rd, sd)]
    )
    print(
        f"train {len(xt)} スロット / dev {len(xd)} スロット（候補2つ以上 {multi.sum()}）"
        f" / ラベル {len(labels)} / 次元 {xt.shape[1]}"
    )
    if has_w7:
        print(f"w7 一致率: {w7_ok.mean():.4f}（候補2つ以上 {w7_ok[multi].mean():.4f}）")
    else:
        print("w7 の読みが無い（export_data --w7 で添える）。w7 との比較は出さない")

    accs, accs_multi, both_wrong, only_head = [], [], [], []
    for seed in (int(s) for s in args.seeds.split(",")):
        torch.manual_seed(seed)
        head = build_head(xt.shape[1], len(labels), args.hidden)
        opt = torch.optim.AdamW(
            head.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        steps = args.epochs * ((len(xt) + args.batch_size - 1) // args.batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=steps
        )
        t0 = time.time()
        for epoch in range(args.epochs):
            head.train()
            order = torch.randperm(
                len(xt), generator=torch.Generator().manual_seed(seed * 1000 + epoch)
            )
            for k in range(0, len(order), args.batch_size):
                i = order[k : k + args.batch_size]
                loss = nn.functional.cross_entropy(
                    masked(head(xt[i]), table, rows_t[i]), gt_t[i]
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        head.eval()
        if args.save and seed == int(args.seeds.split(",")[-1]):
            (WORK / "heads").mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state": head.state_dict(),
                    "mean": mean,
                    "std": std,
                    "hidden": args.hidden,
                    "layers": int(args.vectors.split("_")[0][1:]),
                    "labels": labels.names,
                },
                WORK / "heads" / f"{args.save}.pt",
            )
        with torch.no_grad():
            pred = masked(head(xd), table, rows_d).argmax(-1).numpy()
        ok = pred == gd
        accs.append(ok.mean())
        accs_multi.append(ok[multi].mean())
        both_wrong.append(int((~ok & ~w7_ok).sum()))
        only_head.append(int((~ok & w7_ok).sum()))
        vs_w7 = (
            f" w7と同じ誤り {both_wrong[-1]}・分類器だけの誤り {only_head[-1]}"
            f"・w7だけの誤り {int((~w7_ok & ok).sum())}"
            if has_w7
            else ""
        )
        print(
            f"  seed {seed}: 一致率 {ok.mean():.4f}（候補2つ以上 {ok[multi].mean():.4f}）"
            f"{vs_w7}（{time.time() - t0:.0f} 秒）"
        )

    m = lambda v: f"{np.mean(v):.4f}±{np.std(v):.4f}"  # noqa: E731
    print(
        f"\n分類器（{args.vectors}, hidden={args.hidden}）: {m(accs)}（候補2つ以上 {m(accs_multi)}）"
    )
    w7_text = f"w7: {w7_ok.mean():.4f}   " if has_w7 else ""
    print(f"{w7_text}微調整: 0.9768（旧データでの記録値）")


if __name__ == "__main__":
    main()
