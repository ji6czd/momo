import unicodedata

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "tohoku-nlp/bert-base-japanese-char-v3"

TEXTS = [
    "東北大学は仙台にあります。",
    "吾輩は猫である。名前はまだ無い。",
    "😀 𠮷野家 ABC abc",
    "ＡＢＣ１２３　全角と半角ｶﾀｶﾅ",
    "空白 と\n改行\tタブ",
    "今日は[MASK]がいい",
]


def load_vocab() -> dict[str, int]:
    path = hf_hub_download(MODEL_NAME, "vocab.txt")
    with open(path, encoding="utf-8") as f:
        return {line.rstrip("\n"): i for i, line in enumerate(f)}


def encode_naive(text: str, vocab: dict[str, int]) -> list[int]:
    """1文字ずつ語彙を引くだけ。前処理なし。"""
    unk = vocab["[UNK]"]
    return [vocab["[CLS]"], *(vocab.get(c, unk) for c in text), vocab["[SEP]"]]


def encode_normalized(text: str, vocab: dict[str, int]) -> list[int]:
    """NFKC正規化し、空白類を除去してから1文字ずつ引く。"""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if not c.isspace())
    return encode_naive(text, vocab)


def show(label: str, ids: list[int], tokenizer) -> None:
    print(f"  {label}:", tokenizer.convert_ids_to_tokens(ids))


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()
    vocab = load_vocab()
    print("vocab size:", len(vocab))
    print()

    for text in TEXTS:
        ref = tokenizer(text)["input_ids"]
        naive = encode_naive(text, vocab)
        norm = encode_normalized(text, vocab)

        print("text:", repr(text))
        show("tokenizer ", ref, tokenizer)
        print("  naive     一致:", naive == ref)
        if naive != ref:
            show("naive     ", naive, tokenizer)
        print("  normalized一致:", norm == ref)
        if norm != ref:
            show("normalized", norm, tokenizer)

        # IDが一致するなら、ベクトルも一致するはず
        if norm == ref:
            with torch.no_grad():
                a = model(torch.tensor([ref])).last_hidden_state
                b = model(torch.tensor([norm])).last_hidden_state
            print("  ベクトル最大差:", (a - b).abs().max().item())
        print()


if __name__ == "__main__":
    main()
