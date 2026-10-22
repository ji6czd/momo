"""保存した分類器で、任意の文の漢字の読みを予測して表示する（step7_head --save の続き）。

    uv run python -m bert.step7_head --vectors L4_w0 --hidden 1024 --seeds 0 --save L4_h1024
    uv run python -m bert.step8_predict --head L4_h1024 "今日のテレビで、「今日の中東情勢」という番組が放送されていた。"

文を1行として丸ごと BERT に通す（step6 の行全体の入力と同じ）。対象は漢字・漢数字。各文字について、
辞書候補に絞った上位3つの読みと確率を出す。継続（その前の文字と一続きの読み）は `---`。
"""

import argparse
import unicodedata

import torch

from .common import (
    DATA,
    DEFAULT_MODEL,
    WORK,
    Candidates,
    CharVocab,
    Labels,
    Rec,
    load_jsonl,
)
from .step7_head import build_head, masked

from transformers import BertModel


def is_kanji(ch: str) -> bool:
    return ch == "々" or ch == "〇" or "CJK UNIFIED" in unicodedata.name(ch, "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--head", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("text", nargs="+")
    args = ap.parse_args()

    ck = torch.load(WORK / "heads" / f"{args.head}.pt")
    train = load_jsonl(str(DATA / "train_full.jsonl"))
    labels = Labels(train)
    assert labels.names == ck["labels"], "ラベル表が学習時と違う"
    cands = Candidates(train, labels)
    vocab = CharVocab.from_model(args.model)
    bert = BertModel.from_pretrained(args.model, add_pooling_layer=False).eval()
    head = build_head(ck["mean"].shape[0], len(labels), ck["hidden"])
    head.load_state_dict(ck["state"])
    head.eval()

    for text in args.text:
        ids = torch.tensor([[vocab.cls, *vocab.encode(text)[:254], vocab.sep]])
        with torch.no_grad():
            hs = bert(input_ids=ids, output_hidden_states=True)
            layers = ck["layers"]
            h = torch.stack(hs.hidden_states[-layers:]).mean(0)[0] if layers > 1 else hs.last_hidden_state[0]
            pos = [i for i, c in enumerate(text[:254]) if is_kanji(c)]
            x = (h[[p + 1 for p in pos]] - ck["mean"]) / ck["std"]
            rows = torch.tensor([cands.row(text[p]) for p in pos])
            prob = masked(head(x), cands.table, rows).softmax(-1)
        print(text)
        for p, pr in zip(pos, prob):
            top = pr.topk(min(args.top, int((pr > 0).sum())))
            alts = "  ".join(
                f"{labels.names[i] or '---'} {v:.2f}" for v, i in zip(top.values.tolist(), top.indices.tolist())
            )
            print(f"  {p:3d} {text[p]}  {alts}")


if __name__ == "__main__":
    main()
