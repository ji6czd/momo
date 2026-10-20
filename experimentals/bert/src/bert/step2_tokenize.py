from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "tohoku-nlp/bert-base-japanese-char-v3"

TEXTS = [
    "東北大学は仙台にあります。",
    "吾輩は猫である。名前はまだ無い。",
    "😀 𠮷野家 ABC abc",
]


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    print("tokenizer:", type(tokenizer).__name__)
    print("model    :", type(model).__name__)
    print("vocab size:", tokenizer.vocab_size)
    print("special tokens:", tokenizer.special_tokens_map)
    print()

    for text in TEXTS:
        enc = tokenizer(text)
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        print("text  :", text)
        print("tokens:", tokens)
        print("ids   :", enc["input_ids"])
        print("count :", len(tokens))
        print()


if __name__ == "__main__":
    main()
