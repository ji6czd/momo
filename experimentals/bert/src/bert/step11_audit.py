"""学習データの注釈と、全データで学習したモデルの予測が食い違う箇所を一覧にする（注釈の点検用）。

step9_finetune --all-data で保存したモデル（既定 full）に、学習に使った全行（train + dev）を
通し、漢字スロットで「注釈の読み」と「モデルの最良の読み」が違うものを書き出す。
学習済みの文なので、8エポック学習しても合わせきれなかった箇所＝次のどれかである可能性が高い。

- 注釈の誤り（読みの打ち間違い、「ー」の抜け、継続と読みの区切りの付け間違い）
- 同じ文字列に別の読みが付いていて、文脈で読み分けられていない
- 正当な読み分けだが、データが少なくてモデルが覚えきれていない

機械では区別できないので、前後の文つきで出して人が判断する。注釈の確信が低い順に並べる。

    uv run step11_audit                    # full を使う
    uv run step11_audit --name full --top 3

出力: work/audit/<名前>/{mismatch.tsv, by_pair.tsv}
  mismatch.tsv  1行1スロット。注釈・予測・それぞれの確率・前後の文字
  by_pair.tsv   (文字, 注釈, 予測) ごとの件数。同じ食い違いが繰り返されていないかを見る
"""

import argparse
import collections
import json
import time

import torch
from transformers import BertConfig, BertModel

from .common import (
    BUNDLE,
    DEFAULT_MODEL,
    WORK,
    Candidates,
    CharVocab,
    Labels,
    ReadingModel,
    collate,
    load_jsonl,
    make_examples,
    masked_logits,
)

CONTEXT = 8  # 前後に出す文字数


def write_tsv(path, header, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        f.writelines("\t".join(map(str, r)) + "\n" for r in rows)


@torch.no_grad()
def score_all(model, exs, cands, vocab, batch_size=32):
    """各例の対象スロットごとの確率ベクトル（候補に絞った softmax）を返す。"""
    out = [None] * len(exs)
    order = sorted(range(len(exs)), key=lambda i: len(exs[i].ids))
    table = cands.table
    for k in range(0, len(order), batch_size):
        part = order[k : k + batch_size]
        ids, mask, pos_b, pos_t, rows, _ = collate(
            [exs[i] for i in part], vocab.pad, "cpu"
        )
        if not len(pos_t):
            for i in part:
                out[i] = torch.empty(0, table.shape[1])
            continue
        prob = masked_logits(model(ids, mask, pos_b, pos_t), table, rows).softmax(-1)
        j = 0
        for i in part:
            n = len(exs[i].pos)
            out[i] = prob[j : j + n]
            j += n
        if (k // batch_size) % 20 == 0:
            print(f"  {k + len(part)}/{len(exs)} 行", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--name",
        default="full",
        help="work/finetuned/ 下の名前（全データで学習したもの）",
    )
    ap.add_argument("--top", type=int, default=3, help="予測の上位候補を何個まで書くか")
    args = ap.parse_args()

    d = WORK / "finetuned" / args.name
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if not meta.get("all_data"):
        print(
            "注意: このモデルは dev を学習していない。dev の行の食い違いは未学習のため、点検には full を使う"
        )

    recs = load_jsonl(str(BUNDLE / "train_full.jsonl")) + load_jsonl(
        str(BUNDLE / "dev_full.jsonl")
    )
    labels = Labels(recs)
    assert labels.names == json.loads(
        (d / "labels.json").read_text(encoding="utf-8")
    ), "ラベル表が学習時と違う（データが変わった？）"
    cands = Candidates(recs, labels)

    cfg = BertConfig.from_json_file(str(d / "encoder.json"))
    model = ReadingModel(BertModel(cfg, add_pooling_layer=False), len(labels))
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    model.eval()
    vocab = CharVocab.from_model(meta.get("base_model") or DEFAULT_MODEL)

    exs = make_examples(recs, vocab, labels, cands)
    t0 = time.time()
    probs = score_all(model, exs, cands, vocab)
    print(f"推論 {len(recs)} 行（{time.time() - t0:.0f} 秒）")

    rows, pairs, n_slot = [], collections.Counter(), 0
    for r, ex, pr in zip(recs, exs, probs):
        for si, g, p in zip(ex.slot, ex.gold, pr):
            n_slot += 1
            s = r.slots[si]
            best = int(p.argmax())
            if best == g:
                continue
            top = p.topk(min(args.top, int((p > 0).sum())))
            alts = " ".join(
                f"{labels.names[i] or '---'}:{v:.2f}"
                for v, i in zip(top.values.tolist(), top.indices.tolist())
            )
            lo, hi = max(0, s.start - CONTEXT), s.end + CONTEXT
            ctx = f"{r.text[lo : s.start]}【{s.surface}】{r.text[s.end : hi]}"
            gp = float(p[g]) if g >= 0 else 0.0
            rows.append(
                (
                    r.id,
                    s.start,
                    s.surface,
                    s.gold or "---",
                    labels.names[best] or "---",
                    f"{gp:.3f}",
                    f"{float(p[best]):.3f}",
                    alts,
                    ctx,
                )
            )
            pairs[(s.surface, s.gold or "---", labels.names[best] or "---")] += 1
    rows.sort(key=lambda x: float(x[5]))

    out_dir = WORK / "audit" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(
        out_dir / "mismatch.tsv",
        ["id", "pos", "char", "gold", "pred", "p_gold", "p_pred", "top", "context"],
        rows,
    )
    write_tsv(
        out_dir / "by_pair.tsv",
        ["char", "gold", "pred", "count"],
        [(*k, v) for k, v in pairs.most_common()],
    )
    print(
        f"漢字スロット {n_slot} 個のうち、食い違い {len(rows)} 個（{len(rows) / n_slot:.2%}）"
    )
    print(f"出力: {out_dir}")


if __name__ == "__main__":
    main()
