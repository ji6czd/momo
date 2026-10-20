import unicodedata

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel

MODEL_NAME = "tohoku-nlp/bert-base-japanese-char-v3"


def load_vocab() -> dict[str, int]:
    path = hf_hub_download(MODEL_NAME, "vocab.txt")
    with open(path, encoding="utf-8") as f:
        return {line.rstrip("\n"): i for i, line in enumerate(f)}


def split_clusters(text: str) -> list[str]:
    """基底文字と、それに続く結合文字（濁点など）をひとまとめにする。"""
    clusters: list[str] = []
    for c in text:
        if clusters and unicodedata.combining(c):
            clusters[-1] += c
        else:
            clusters.append(c)
    return clusters


def encode_with_alignment(text: str, vocab: dict[str, int]):
    """入力IDと、各トークンが元の何番目の文字に由来するかを返す。"""
    unk = vocab["[UNK]"]
    ids = [vocab["[CLS]"]]
    origin = [-1]  # -1: 元の文字に対応しない特殊トークン
    for i, cluster in enumerate(split_clusters(text)):
        for c in unicodedata.normalize("NFKC", cluster):
            if c.isspace():
                continue
            ids.append(vocab.get(c, unk))
            origin.append(i)
    ids.append(vocab["[SEP]"])
    origin.append(-1)
    return ids, origin


class CharEncoder:
    def __init__(self) -> None:
        self.vocab = load_vocab()
        self.model = AutoModel.from_pretrained(MODEL_NAME).eval()

    @torch.no_grad()
    def embed(self, text: str) -> list[tuple[str, torch.Tensor | None]]:
        """元の文字ごとに (文字, ベクトル) を返す。

        空白など落とされた文字は None。
        1文字が複数トークンに展開された場合（例: ㌔ -> キロ）は平均を取る。
        """
        ids, origin = encode_with_alignment(text, self.vocab)
        hidden = self.model(torch.tensor([ids])).last_hidden_state[0]
        clusters = split_clusters(text)
        result: list[tuple[str, torch.Tensor | None]] = []
        for i, cluster in enumerate(clusters):
            rows = [hidden[t] for t, o in enumerate(origin) if o == i]
            result.append((cluster, torch.stack(rows).mean(0) if rows else None))
        return result


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a, b, dim=0).item()


def vec_of(encoder: CharEncoder, text: str, char: str) -> list[torch.Tensor]:
    return [v for c, v in encoder.embed(text) if c == char and v is not None]


def main() -> None:
    enc = CharEncoder()

    print("== 1. 文字ごとのベクトル ==")
    text = "東北大学は仙台にあります。"
    for c, v in enc.embed(text):
        print(
            f"  {c}  shape={tuple(v.shape)}  先頭3次元={v[:3].tolist()}  L2ノルム={v.norm():.2f}"
        )
    print()

    print("== 2. 元の文字との対応（正規化・空白除去あり） ==")
    text = "ＡＢ C\n㌔ｶﾀｶﾅ"
    ids, origin = encode_with_alignment(text, enc.vocab)
    print("  元の文字:", split_clusters(text))
    print("  origin  :", origin)
    for c, v in enc.embed(text):
        print(f"  {c!r:6} -> {'(落とされた)' if v is None else 'ベクトルあり'}")
    print()

    print("== 3. 同じ文字でも文脈で変わるか（「行」） ==")
    s1 = "私は銀行に行く。"  # 銀行(こう) / 行く(い)
    s2 = "彼は銀行で働く。"  # 銀行(こう)
    s3 = "明日は京都に行く。"  # 行く(い)
    (g1, i1), (g2,), (i3,) = (vec_of(enc, s, "行") for s in (s1, s2, s3))
    print(f"  {s1} の銀行の行 vs {s2} の銀行の行 : {cos(g1, g2):.3f}  (同じ意味)")
    print(f"  {s1} の行くの行 vs {s3} の行くの行 : {cos(i1, i3):.3f}  (同じ意味)")
    print(f"  {s1} の銀行の行 vs 行くの行        : {cos(g1, i1):.3f}  (違う意味)")
    print(f"  {s1} の銀行の行 vs {s3} の行くの行 : {cos(g1, i3):.3f}  (違う意味)")


if __name__ == "__main__":
    main()
