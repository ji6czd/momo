"""本体を固定した BERT から、漢字スロットの文字ベクトルを取り出して保存する。

train_full と dev_full の全行を1行ずつ（行全体を文脈として）流し、対象スロット（漢字・漢数字）の
位置の隠れ状態を取る。保存したベクトルで読みの分類器だけを学習する（step7_head.py）ので、
BERT を回すのはここで1回きり。

    uv run python -m momo_bert_train.step6_extract                 # 最終層
    uv run python -m momo_bert_train.step6_extract --layers 4      # 最後の4層の平均
    uv run python -m momo_bert_train.step6_extract --window 3      # スロットごとに前後3文字だけ見せる
    uv run python -m momo_bert_train.step6_extract --limit 200     # 動作確認

出力: work/vectors/<タグ>/{train,dev}.npz
  x        (N, 768) float16  文字ベクトル
  rec      (N,) 行の番号
  slot     (N,) 行の中のスロット番号
  gold     (N,) 正解のラベル番号（ラベル表にない読みは -1）
"""

import argparse
import time

import numpy as np
import torch
from transformers import BertModel

from .common import (
    DATA,
    DEFAULT_MODEL,
    WORK,
    Candidates,
    CharVocab,
    Labels,
    amp_mode,
    load_jsonl,
    make_example,
    verify_records,
)


@torch.no_grad()
def extract(model, recs, vocab, labels, cands, *, layers, window, batch_size, device):
    exs = [make_example(r, vocab, labels, cands, window=window) for r in recs]
    # 窓つきはスロットごとの短い入力に展開する。(rec 番号, 入力) の並びにそろえる
    flat = []
    for ri, e in enumerate(exs):
        for p in e.parts if e.parts is not None else [e]:
            if p.pos:
                flat.append((ri, p))
    order = sorted(range(len(flat)), key=lambda i: len(flat[i][1].ids))
    dtype, _ = amp_mode(device)

    out_x, out_rec, out_slot, out_gold = [], [], [], []
    for k in range(0, len(order), batch_size):
        batch = [flat[i] for i in order[k : k + batch_size]]
        width = max(len(p.ids) for _, p in batch)
        ids = torch.full((len(batch), width), vocab.pad, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for b, (_, p) in enumerate(batch):
            ids[b, : len(p.ids)] = torch.tensor(p.ids)
            mask[b, : len(p.ids)] = 1
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
            hs = model(
                input_ids=ids.to(device),
                attention_mask=mask.to(device),
                output_hidden_states=layers > 1,
            )
        h = (
            torch.stack(hs.hidden_states[-layers:]).mean(0)
            if layers > 1
            else hs.last_hidden_state
        )
        for b, (ri, p) in enumerate(batch):
            out_x.append(h[b, p.pos].float().cpu())
            out_rec += [ri] * len(p.pos)
            out_slot += p.slot
            out_gold += p.gold
    return (
        torch.cat(out_x).numpy().astype(np.float16),
        np.array(out_rec),
        np.array(out_slot),
        np.array(out_gold),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layers", type=int, default=1, help="最後の何層を平均するか")
    ap.add_argument("--window", type=int, default=0, help="0 なら行全体")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--limit", type=int, default=0, help="各分割の先頭 N 行だけ（確認用）"
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    train = load_jsonl(str(DATA / "train_full.jsonl"))
    dev = load_jsonl(str(DATA / "dev_full.jsonl"))
    if args.limit:
        train, dev = train[: args.limit], dev[: args.limit]
    verify_records(train + dev)

    vocab = CharVocab.from_model(args.model)
    labels = Labels(train)
    cands = Candidates(train, labels)
    model = BertModel.from_pretrained(args.model, add_pooling_layer=False)
    model.to(args.device).eval()

    tag = f"L{args.layers}_w{args.window}" + (f"_n{args.limit}" if args.limit else "")
    out_dir = WORK / "vectors" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in (("train", train), ("dev", dev)):
        t0 = time.time()
        x, rec, slot, gold = extract(
            model,
            recs,
            vocab,
            labels,
            cands,
            layers=args.layers,
            window=args.window,
            batch_size=args.batch_size,
            device=args.device,
        )
        np.savez(out_dir / f"{name}.npz", x=x, rec=rec, slot=slot, gold=gold)
        print(
            f"{name}: {len(recs)} 行 → {len(x)} スロット（{time.time() - t0:.0f} 秒）"
        )
    print(f"保存先: {out_dir}")


if __name__ == "__main__":
    main()
