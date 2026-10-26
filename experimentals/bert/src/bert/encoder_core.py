"""エンコーダ（BERT 系）に、漢字スロットの読みを候補から選ばせる本体（torch と transformers だけで動く）。

tools/llm_reading/encoder_core.py（微調整の初回実験）を、experimentals/bert に引き取ったもの。
step6〜11 と、クラウドのキット（step9_finetune）が使う。momo_train・momors_py には依存しないので、
クラウドの Linux でもそのまま動く。データは export_data.py が書き出した JSONL。

課題の形:
- 対象は ctype が KANJI / JAPANESE_NUMERIC のスロット。かな・記号のスロットは予測しない。
- 1文字1トークンで行（段落）を丸ごと入れ、対象スロットの位置の隠れ状態から、読みのラベル
  （`カミ`、`ジョー`、継続 `""` など）を分類する。
- 分類は「その文字の辞書候補だけに絞った softmax」。辞書は学習に使う行だけから作る
  （Candidates）。学習時も推論時も同じ絞り込みをかける。辞書にない文字は、学習に出た全ラベルの
  argmax にフォールバックする。
"""

from __future__ import annotations

import collections
import json
import random
import time
import unicodedata
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from transformers import BertConfig, BertModel, get_linear_schedule_with_warmup

KANJI_TYPES = frozenset({"KANJI", "JAPANESE_NUMERIC"})
MAX_LEN = 256  # [CLS] と [SEP] を含むトークン数の上限（学習データの最長は 224 字）
NEG = -1e9  # 候補外のラベルの logit


# ---------------------------------------------------------------- データ


@dataclass
class Slot:
    surface: str
    gold: str  # 継続は ""
    ctype: str
    start: int
    end: int


class Rec(SimpleNamespace):
    """bundle の1件。slots は Slot の並びに直す。"""

    def __init__(self, d: dict):
        super().__init__(**d)
        self.slots = [Slot(*s) for s in d["slots"]]


def load_jsonl(path: str) -> list[Rec]:
    with open(path, encoding="utf-8") as f:
        return [Rec(json.loads(line)) for line in f if line.strip()]


def is_target(s: Slot) -> bool:
    return s.ctype in KANJI_TYPES


def verify_records(recs: list[Rec]) -> None:
    """スロットと原文の位置が合っていること（1文字1トークンへの分割でずれないこと）を検査する。"""
    for r in recs:
        for s in r.slots:
            if r.text[s.start : s.end] != s.surface:
                raise ValueError(
                    f"スロットと原文がずれている: {s} 原文={r.text[max(0, s.start - 3) : s.end + 3]!r}"
                )
            if is_target(s) and len(s.surface) != 1:
                raise ValueError(f"対象スロットが1文字でない: {s}")


# ---------------------------------------------------------------- 文字のトークン化


class CharVocab:
    """1文字1トークンの語彙。語彙にない文字は [UNK]（NFKC で1文字になって語彙にあればそれ）。"""

    def __init__(self, tokens: list[str]):
        self.index = {t: i for i, t in enumerate(tokens)}
        self.size = len(tokens)
        self.pad, self.unk = self.index["[PAD]"], self.index["[UNK]"]
        self.cls, self.sep = self.index["[CLS]"], self.index["[SEP]"]

    @classmethod
    def from_model(cls, model_name: str) -> "CharVocab":
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(model_name, "vocab.txt"), encoding="utf-8") as f:
            return cls([line.rstrip("\n") for line in f])

    def token_id(self, ch: str) -> int:
        i = self.index.get(ch)
        if i is not None:
            return i
        n = unicodedata.normalize("NFKC", ch)
        return self.index.get(n, self.unk) if len(n) == 1 else self.unk

    def encode(self, text: str) -> list[int]:
        return [self.token_id(c) for c in text]


# ---------------------------------------------------------------- ラベルと候補


class Labels:
    """読みのラベル表（頻度順）。全学習データから作り、どの割合の学習でも同じ表を使う。"""

    def __init__(self, train: list[Rec]):
        count = collections.Counter(
            s.gold for r in train for s in r.slots if is_target(s)
        )
        self.names = [g for g, _ in count.most_common()]
        self.id = {g: i for i, g in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)


class Candidates:
    """学習に使う行だけから作る、文字 → 読み候補の表（辞書）。

    table[row] は、その行の文字に許すラベルの bool ベクトル。辞書にない文字の行
    （fallback）は、学習に出たラベルすべてを許す。
    """

    def __init__(self, recs: list[Rec], labels: Labels):
        per_char: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        for r in recs:
            for s in r.slots:
                if is_target(s):
                    per_char[s.surface][labels.id[s.gold]] += 1
        self.chars = {c: i for i, c in enumerate(per_char)}
        self.fallback = len(self.chars)
        self.table = torch.zeros(len(self.chars) + 1, len(labels), dtype=torch.bool)
        self.top: dict[str, int] = {}  # 文字 → 頻度1位のラベル
        self.n_cands: dict[str, int] = {}
        for c, cnt in per_char.items():
            self.table[self.chars[c], list(cnt)] = True
            self.table[self.fallback, list(cnt)] = True
            self.top[c] = cnt.most_common(1)[0][0]
            self.n_cands[c] = len(cnt)

    def row(self, ch: str) -> int:
        return self.chars.get(ch, self.fallback)


# ---------------------------------------------------------------- 例の組み立て


@dataclass
class Example:
    ids: list[int]  # [CLS] 文脈 原文 [SEP]
    pos: list[int]  # 対象スロットのトークン位置
    slot: list[int]  # 対象スロットの rec.slots での番号
    rows: list[int]  # 候補表の行
    gold: list[int]  # 正解のラベル番号（表にない読みは -1）
    # 窓つき(window > 0)のときの、スロットごとの短い入力。predict / train はこちらを流す
    parts: list["Example"] | None = None


def make_example(
    rec: Rec,
    vocab: CharVocab,
    labels: Labels,
    cands: Candidates,
    context: str = "",
    window: int = 0,
) -> Example:
    """文脈（前の文）を付けるときは原文の前に連結し、位置をずらす。長すぎれば文脈を古い方から削る。

    window > 0 なら、スロットごとに前後 window 文字だけを見せる短い入力に切る（行全体の文脈を
    使っているかを確かめる実験用。w7 の window=7 は前後 3 文字）。"""
    room = MAX_LEN - 2 - len(rec.text)
    context = context[len(context) - room :] if 0 < room < len(context) else ""
    if room <= 0:
        context = ""
    off = 1 + len(context)
    ids = [vocab.cls, *vocab.encode(context + rec.text)][: MAX_LEN - 1] + [vocab.sep]
    ex = Example(ids, [], [], [], [])
    for i, s in enumerate(rec.slots):
        if is_target(s) and off + s.start < len(ids) - 1:
            ex.pos.append(off + s.start)
            ex.slot.append(i)
            ex.rows.append(cands.row(s.surface))
            ex.gold.append(labels.id.get(s.gold, -1))
    if not window:
        return ex
    chars = vocab.encode(context + rec.text)
    ex.parts = []
    for p, si, row, gold in zip(ex.pos, ex.slot, ex.rows, ex.gold):
        c = p - 1  # context + text の中での位置
        lo, hi = max(0, c - window), min(len(chars), c + window + 1)
        ids = [vocab.cls, *chars[lo:hi], vocab.sep]
        ex.parts.append(Example(ids, [1 + c - lo], [si], [row], [gold]))
    return ex


def make_examples(recs, vocab, labels, cands, window: int = 0) -> list[Example]:
    return [make_example(r, vocab, labels, cands, window=window) for r in recs]


# ---------------------------------------------------------------- モデル


class ReadingModel(nn.Module):
    def __init__(self, encoder: BertModel, n_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.config.hidden_size, n_labels)

    def forward(self, ids, mask, pos_b, pos_t):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.head(self.drop(h[pos_b, pos_t]))


def build_model(model_name: str, n_labels: int, vocab: CharVocab) -> ReadingModel:
    if model_name == "tiny":  # 動作確認用: 重みを落とさず、極小の乱数初期化で通す
        cfg = BertConfig(
            vocab_size=vocab.size,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=MAX_LEN,
        )
        enc = BertModel(cfg, add_pooling_layer=False)
    else:
        enc = BertModel.from_pretrained(model_name, add_pooling_layer=False)
    return ReadingModel(enc, n_labels)


def collate(exs: list[Example], pad: int, device: str):
    width = max(len(e.ids) for e in exs)
    ids = torch.full((len(exs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(exs), width), dtype=torch.long)
    pos_b, pos_t, rows, gold = [], [], [], []
    for b, e in enumerate(exs):
        ids[b, : len(e.ids)] = torch.tensor(e.ids)
        mask[b, : len(e.ids)] = 1
        pos_b += [b] * len(e.pos)
        pos_t += e.pos
        rows += e.rows
        gold += e.gold
    t = lambda x: torch.tensor(x, dtype=torch.long, device=device)  # noqa: E731
    return ids.to(device), mask.to(device), t(pos_b), t(pos_t), t(rows), t(gold)


def amp_mode(device: str):
    """(autocast の dtype, GradScaler)。bf16 が使えれば bf16、なければ fp16＋スケーラ、CPU は無し。"""
    if not device.startswith("cuda"):
        return None, None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler("cuda")


def masked_logits(logits, table, rows):
    return logits.float().masked_fill(~table[rows], NEG)


def train(
    model: ReadingModel,
    exs: list[Example],
    cands: Candidates,
    vocab: CharVocab,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: str,
    warmup: float = 0.06,
    weight_decay: float = 0.01,
    on_epoch=None,
    log=lambda s: print(s, flush=True),
) -> None:
    exs = [p for e in exs for p in (e.parts if e.parts is not None else [e])]
    exs = [e for e in exs if e.pos]
    table = cands.table.to(device)
    model.to(device).train()
    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {
            "params": [
                p for n, p in model.named_parameters() if not n.endswith(no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
            "weight_decay": 0.0,
        },
    ]
    opt = torch.optim.AdamW(params, lr=lr)
    steps = epochs * ((len(exs) + batch_size - 1) // batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(steps * warmup), steps)
    dtype, scaler = amp_mode(device)
    log(f"  学習 {len(exs)} 行・{steps} ステップ・精度={dtype or 'float32'}")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(exs)))
        random.Random(seed * 1000 + epoch).shuffle(order)
        total = n = 0
        for k in range(0, len(order), batch_size):
            ids, mask, pos_b, pos_t, rows, gold = collate(
                [exs[i] for i in order[k : k + batch_size]], vocab.pad, device
            )
            with torch.autocast(
                device_type="cuda", dtype=dtype, enabled=dtype is not None
            ):
                logits = model(ids, mask, pos_b, pos_t)
            loss = nn.functional.cross_entropy(masked_logits(logits, table, rows), gold)
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            total += loss.item() * len(gold)
            n += len(gold)
        msg = f"  epoch {epoch}/{epochs} loss={total / n:.4f}（{time.time() - t0:.0f} 秒）"
        if on_epoch:
            msg += " " + on_epoch(epoch)
        log(msg)


@torch.no_grad()
def predict(model, exs, cands, vocab, device, batch_size: int = 64):
    """各例の対象スロットごとの (予測ラベル番号, その確率)。入力と同じ順で返す（窓つきは部品ごとに流す）。"""
    flat = [p for e in exs for p in (e.parts if e.parts is not None else [e])]
    res = _predict_flat(model, flat, cands, vocab, device, batch_size)
    out, k = [], 0
    for e in exs:
        n = len(e.parts) if e.parts is not None else 1
        out.append([r for sub in res[k : k + n] for r in sub])
        k += n
    return out


@torch.no_grad()
def _predict_flat(
    model: ReadingModel,
    exs: list[Example],
    cands: Candidates,
    vocab: CharVocab,
    device: str,
    batch_size: int = 64,
) -> list[list[tuple[int, float]]]:
    """各例の対象スロットごとの (予測ラベル番号, その確率)。入力と同じ順で返す。"""
    model.to(device).eval()
    table = cands.table.to(device)
    dtype, _ = amp_mode(device)
    out: list = [None] * len(exs)
    order = sorted(range(len(exs)), key=lambda i: len(exs[i].ids))
    for k in range(0, len(order), batch_size):
        part = order[k : k + batch_size]
        ids, mask, pos_b, pos_t, rows, _ = collate(
            [exs[i] for i in part], vocab.pad, device
        )
        if not len(pos_t):
            for i in part:
                out[i] = []
            continue
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
            logits = model(ids, mask, pos_b, pos_t)
        prob = masked_logits(logits, table, rows).softmax(-1)
        best_p, best = prob.max(-1)
        best, best_p = best.tolist(), best_p.tolist()
        j = 0
        for i in part:
            n = len(exs[i].pos)
            out[i] = list(zip(best[j : j + n], best_p[j : j + n]))
            j += n
    return out
