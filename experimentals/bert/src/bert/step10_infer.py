"""step9_finetune で保存した微調整モデルを、手元の CPU で読み込んで読みを予測する。

    uv run python -m bert.step10_infer --name train "今日のテレビで、「今日の中東情勢」という番組が放送されていた。"

BERT 本体の重みは保存済みのモデルから読むので、学習済みの重みを取り直さない（語彙の取得だけ要る）。
文を1行として丸ごと通し、各漢字・漢数字の読みを辞書候補に絞った上位 N 個の確率つきで表示する。
継続（前の文字と一続きの読み）は `---`。
"""

import argparse
import json
import unicodedata

import torch
from transformers import BertConfig, BertModel

from .common import DEFAULT_MODEL, WORK, CharVocab, ReadingModel


def is_kanji(ch: str) -> bool:
    return ch in "々〇" or "CJK UNIFIED" in unicodedata.name(ch, "")


class FineTunedReader:
    def __init__(self, name: str, vocab_model: str = DEFAULT_MODEL):
        d = WORK / "finetuned" / name
        self.labels: list[str] = json.loads(
            (d / "labels.json").read_text(encoding="utf-8")
        )
        self.cands: dict[str, list[int]] = json.loads(
            (d / "cands.json").read_text(encoding="utf-8")
        )["chars"]
        cfg = BertConfig.from_json_file(str(d / "encoder.json"))
        self.model = ReadingModel(
            BertModel(cfg, add_pooling_layer=False), len(self.labels)
        )
        self.model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
        self.model.eval()
        self.vocab = CharVocab.from_model(vocab_model)
        self.max_chars = cfg.max_position_embeddings - 2

    @torch.no_grad()
    def predict(self, text: str, top: int = 3):
        """漢字ごとに (位置, 文字, [(読み, 確率), ...]) を返す。"""
        text = text[: self.max_chars]
        pos = [i for i, c in enumerate(text) if is_kanji(c)]
        if not pos:
            return []
        ids = torch.tensor([[self.vocab.cls, *self.vocab.encode(text), self.vocab.sep]])
        mask = torch.ones_like(ids)
        pos_t = torch.tensor([p + 1 for p in pos])
        logits = self.model(ids, mask, torch.zeros_like(pos_t), pos_t).float()
        out = []
        for p, lg in zip(pos, logits):
            allowed = self.cands.get(text[p]) or list(range(len(self.labels)))
            sel = torch.full_like(lg, -1e9)
            sel[allowed] = lg[allowed]
            v, i = sel.softmax(-1).topk(min(top, len(allowed)))
            alts = [
                (self.labels[k] or "---", s) for s, k in zip(v.tolist(), i.tolist())
            ]
            out.append((p, text[p], alts))
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="train", help="work/finetuned/ 下の名前")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("text", nargs="+")
    args = ap.parse_args()

    reader = FineTunedReader(args.name)
    for text in args.text:
        print(text)
        for p, ch, alts in reader.predict(text, args.top):
            print(f"  {p:3d} {ch}  " + "  ".join(f"{r} {s:.2f}" for r, s in alts))


if __name__ == "__main__":
    main()
