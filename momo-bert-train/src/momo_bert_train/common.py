"""実験の共通部分: データと出力の場所、encoder_core の取り込み。

データは export_data.py が dataset/basic_raw.tsv から作り直す work/data/（train_full.jsonl、
dev_full.jsonl）。クラウドのキット（make_kit.py）にも同じ場所に入るので、手元でもクラウドでも
同じパスで読める。
"""

from __future__ import annotations

from pathlib import Path

PKG_ROOT = (
    Path(__file__).resolve().parents[2]
)  # momo-bert-train（クラウドのキットでは展開先）
WORK = PKG_ROOT / "work"  # git 管理外
DATA = WORK / "data"

DEFAULT_MODEL = "tohoku-nlp/bert-base-japanese-char-v3"

from .encoder_core import (  # noqa: E402
    Candidates,
    CharVocab,
    Labels,
    ReadingModel,
    Rec,
    amp_mode,
    build_model,
    collate,
    is_target,
    load_jsonl,
    make_example,
    make_examples,
    masked_logits,
    predict,
    train,
    verify_records,
)

__all__ = [
    "Candidates",
    "CharVocab",
    "DATA",
    "DEFAULT_MODEL",
    "Labels",
    "ReadingModel",
    "Rec",
    "WORK",
    "amp_mode",
    "build_model",
    "collate",
    "is_target",
    "load_jsonl",
    "make_example",
    "make_examples",
    "masked_logits",
    "predict",
    "train",
    "verify_records",
]
