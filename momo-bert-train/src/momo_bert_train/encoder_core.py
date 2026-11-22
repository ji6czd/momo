"""エンコーダ（BERT 系）に、漢字スロットの読みを候補から選ばせる本体（torch と transformers だけで動く）。

tools/llm_reading/encoder_core.py（微調整の初回実験）を、momo-bert-train に引き取ったもの。
step6〜11 と、クラウドのキット（step9_finetune）が使う。momo_train・momors_py には依存しないので、
クラウドの Linux でもそのまま動く。データは export_data.py が書き出した JSONL。

課題の形:
- 対象は ctype が KANJI / JAPANESE_NUMERIC のスロット。かな・記号のスロットは予測しない。
- 1文字1トークンで行（段落）を丸ごと入れ、対象スロットの位置の隠れ状態から、読みのラベル
  （`カミ`、`ジョー`、継続 `""` など）を分類する。
- 分類は「その文字の辞書候補だけに絞った softmax」。辞書は学習に使う行だけから作る
  （Candidates）。学習時も推論時も同じ絞り込みをかける。辞書にない文字は、学習に出た全ラベルの
  argmax にフォールバックする。

境界（マスあけ）の同時学習（ReadingModel(boundary=True)）:
- 境界の出力層（隠れ状態 → 1）を足し、BERT 本体を読みと共有する。BERT は 1 回通すだけで両方が出る。
- 切れ目は隣り合うスロットの間。その切れ目の直前の字（スロットの最後の字）の位置で、そこで
  マスをあけるか（注釈の +S）を当てる。momo の境界モデルがスロットの最後の字で問うのと同じ。
- 行末と、原文の空白をはさむ切れ目は使わない（どちらも momo が原文どおりに扱う）。
- 漢字の無い行も、境界の学習には使う。
- 断片（fragments）: 行の途中から途中までを切り出したものを、境界の学習にだけ足せる（読みの学習には
  使わない）。行全体だけで学習すると、文の続き方を手がかりにしすぎて、見出し・表の欄のような短い入力で
  マスあけを外す。切り出した中の切れ目の正解は元の行と同じ。
  切り出し方は 2 つ（cut）。既定の "unit" は、マスあけの単位（マスあけか原文の空白で区切ったひと続き）の
  頭から終わりまでを 2 単位以上。断片の終わりは必ず妥当な境界になり、行全体と同じ前提（入力の終わり＝
  行末）に立つ。"free" はスロットの切れ目ならどこでも切る（スクリーンリーダーが送る語の途中の切れ端に
  近い。2026-11-22 にこちらは優先しないと決め、比べるために残す）。
"""

from __future__ import annotations

import collections
import json
import os
import random
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
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
    space: bool = (
        False  # 注釈でこのスロットの後にマスあけがある（+S）。古いデータには無い
    )


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
    # 境界の切れ目: 切れ目の直前の字のトークン位置と、そこでマスをあけるか（1 / 0）。
    # 窓つきの部品には付けない
    bpos: list[int] = field(default_factory=list)
    bgold: list[int] = field(default_factory=list)
    bslot: list[int] = field(default_factory=list)  # 切れ目の直前のスロットの番号


def make_example(
    rec: Rec,
    vocab: CharVocab,
    labels: Labels,
    cands: Candidates,
    context: str = "",
    window: int = 0,
    last_gap: bool = False,
) -> Example:
    """文脈（前の文）を付けるときは原文の前に連結し、位置をずらす。長すぎれば文脈を古い方から削る。

    window > 0 なら、スロットごとに前後 window 文字だけを見せる短い入力に切る（行全体の文脈を
    使っているかを確かめる実験用。w7 の window=7 は前後 3 文字）。

    last_gap なら、断片（crop）の最後のスロットの後ろの切れ目も、元の行の注釈で境界の学習に使う
    （元の行でもそこが切れ目のとき。rec.tail_gap）。最後の字は [SEP] の直前に来るので、そこに学習の
    目標が無いと、その字の隠れ状態が [SEP] と同じに潰れてもかまわなくなる。"""
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
    for i, (a, b) in enumerate(zip(rec.slots, rec.slots[1:])):
        # 原文の空白をはさむ切れ目は、momo が原文どおりに残すので使わない
        if rec.text[a.end : b.start] or a.surface.isspace() or b.surface.isspace():
            continue
        if off + b.start >= len(ids) - 1:  # 次の字が切り詰めで入っていない
            break
        ex.bpos.append(off + a.end - 1)
        ex.bgold.append(int(a.space))
        ex.bslot.append(i)
    last = rec.slots[-1] if rec.slots else None
    if (
        last_gap
        and getattr(rec, "tail_gap", False)
        and off + last.end - 1 < len(ids) - 1
    ):
        ex.bpos.append(off + last.end - 1)
        ex.bgold.append(int(last.space))
        ex.bslot.append(len(rec.slots) - 1)
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


# ---------------------------------------------------------------- 断片（行の一部の切り出し）

# 断片の切り出し方。unit＝マスあけの単位の頭から終わりまで（2 単位以上）、free＝スロットの切れ目ならどこでも
FRAGMENT_CUTS = ("unit", "free")

# dev の断片の作り方（モデルどうしを同じ断片で比べるため固定。切り出し方は dev_fragments の引数）
DEV_FRAGMENTS = {
    "seed": "dev-fragments",
    "per_line": 1.0,
    "min_chars": 2,
    "max_chars": 20,
}


def crop(rec: Rec, i: int, j: int) -> Rec:
    """rec.slots[i] から rec.slots[j] まで（両端を含む）を 1 行にしたもの。位置は切り出した先頭から数え直す。

    tail_gap は、元の行で最後のスロットの後ろが切れ目か（行末でなく、空白をはさまない）。"""
    base = rec.slots[i].start
    nxt = rec.slots[j + 1] if j + 1 < len(rec.slots) else None
    tail_gap = (
        nxt is not None
        and not rec.text[rec.slots[j].end : nxt.start]
        and not rec.slots[j].surface.isspace()
        and not nxt.surface.isspace()
    )
    return Rec(
        {
            "text": rec.text[base : rec.slots[j].end],
            "slots": [
                [s.surface, s.gold, s.ctype, s.start - base, s.end - base, s.space]
                for s in rec.slots[i : j + 1]
            ],
            "tail_gap": tail_gap,
        }
    )


def units(rec: Rec) -> list[tuple[int, int]]:
    """マスあけの単位。スロット番号の (先頭, 最後)（両端を含む）の並び。

    注釈のマスあけ（+S）と原文の空白で区切ったひと続き。空白のスロットはどの単位にも入らない。"""
    out = []
    head = None
    for k, s in enumerate(rec.slots):
        if s.surface.isspace():
            if head is not None:
                out.append((head, k - 1))
                head = None
            continue
        if head is None:
            head = k
        nxt = rec.slots[k + 1] if k + 1 < len(rec.slots) else None
        if nxt is None or s.space or rec.text[s.end : nxt.start]:
            out.append((head, k))
            head = None
    return out


def fragments(
    recs: list[Rec],
    rng: random.Random,
    per_line: float,
    min_chars: int,
    max_chars: int,
    cut: str = "unit",
) -> list[Rec]:
    """行の途中から途中までを切り出した断片。1 行あたり平均 per_line 個。

    cut="unit": 先頭の単位（units）を選び、min_chars〜max_chars から一様に選んだ字数に届くまで単位を
    足す。少なくとも 2 単位（中にマスあけか原文の空白を 1 つ以上含む）。断片の終わりは必ず単位の終わり
    （元の行でもマスあけか行末）なので、最後の切れ目は学習に使わなくてよい（momo は行末の境界の
    スペースを落とす）。単位が 2 つ以下の行からは作れない（行全体と同じになる）。

    cut="free": 先頭のスロットを選び、同じく字数に届くまでスロットを足す（スロットの途中では切らない）。
    空白で始まる・終わる断片、切れ目の無い断片（1 スロットだけ）は作らない。

    どちらも、行全体と同じになる断片は作らない。"""
    if cut not in FRAGMENT_CUTS:
        raise ValueError(f"断片の切り出し方は {FRAGMENT_CUTS} のどれか: {cut!r}")
    out = []
    for r in recs:
        if cut == "unit":
            us = units(r)
            n = int(per_line) + (rng.random() < per_line % 1)
            for _ in range(n if len(us) > 2 else 0):
                u = rng.randrange(len(us) - 1)
                want = rng.randint(min_chars, max_chars)
                v = u + 1
                while (
                    v + 1 < len(us)
                    and r.slots[us[v][1]].end - r.slots[us[u][0]].start < want
                ):
                    v += 1
                if u == 0 and v == len(us) - 1:
                    continue
                out.append(crop(r, us[u][0], us[v][1]))
            continue
        starts = [k for k, s in enumerate(r.slots) if not s.surface.isspace()]
        n = int(per_line) + (rng.random() < per_line % 1)
        for _ in range(n if starts else 0):
            i = rng.choice(starts)
            want = rng.randint(min_chars, max_chars)
            j = i
            while j + 1 < len(r.slots) and r.slots[j].end - r.slots[i].start < want:
                j += 1
            while j > i and r.slots[j].surface.isspace():
                j -= 1
            if j == i or (i == 0 and j == len(r.slots) - 1):
                continue
            out.append(crop(r, i, j))
    return out


def dev_fragments(recs: list[Rec], cut: str = "unit") -> list[Rec]:
    """dev の断片（DEV_FRAGMENTS の決まった作り方。step9_finetune と step14_boundary_eval で同じもの）。

    cut="free" は 2026-11-22 までの dev の断片と同じもの（それまでの数字と比べられる）。"""
    p = DEV_FRAGMENTS
    return fragments(
        recs,
        random.Random(p["seed"]),
        p["per_line"],
        p["min_chars"],
        p["max_chars"],
        cut=cut,
    )


def boundary_examples(
    recs, vocab, labels, cands, last_gap: bool = False
) -> list[Example]:
    """境界だけの例（読みの位置を持たない）。断片を境界の学習にだけ使うときのもの。

    last_gap なら、断片の最後の切れ目も学習に使う（make_example）。"""
    out = []
    for r in recs:
        ex = make_example(r, vocab, labels, cands, last_gap=last_gap)
        if ex.bpos:
            out.append(
                Example(ex.ids, [], [], [], [], None, ex.bpos, ex.bgold, ex.bslot)
            )
    return out


# ---------------------------------------------------------------- モデル


class ReadingModel(nn.Module):
    """BERT 本体と読みの出力層。boundary=True なら、境界（その字の後でマスをあけるか）の出力層も持つ。

    境界の出力層の無いモデルは、state_dict が従来のものと同じ（古い model.pt をそのまま読める）。
    """

    def __init__(
        self,
        encoder: BertModel,
        n_labels: int,
        dropout: float = 0.1,
        boundary: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.config.hidden_size, n_labels)
        self.boundary_head = (
            nn.Linear(encoder.config.hidden_size, 1) if boundary else None
        )

    def encode(self, ids, mask):
        return self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state

    def forward(self, ids, mask, pos_b, pos_t):
        return self.head(self.drop(self.encode(ids, mask)[pos_b, pos_t]))

    def forward_joint(self, ids, mask, pos_b, pos_t, bpos_b, bpos_t):
        """読みの logit（対象スロットの位置）と、境界の logit（切れ目の位置）を、BERT 1 回で返す。"""
        h = self.encode(ids, mask)
        read = self.head(self.drop(h[pos_b, pos_t]))
        bnd = self.boundary_head(self.drop(h[bpos_b, bpos_t])).squeeze(-1)
        return read, bnd


def load_finetuned(d: Path, n_labels: int) -> ReadingModel:
    """step9_finetune が保存したモデル（d/encoder.json・d/model.pt）を読む。境界の出力層があればそれも。"""
    cfg = BertConfig.from_json_file(str(d / "encoder.json"))
    state = torch.load(d / "model.pt", map_location="cpu")
    model = ReadingModel(
        BertModel(cfg, add_pooling_layer=False),
        n_labels,
        boundary="boundary_head.weight" in state,
    )
    model.load_state_dict(state)
    return model.eval()


def build_model(
    model_name: str, n_labels: int, vocab: CharVocab, boundary: bool = False
) -> ReadingModel:
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
    return ReadingModel(enc, n_labels, boundary=boundary)


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


def collate_boundary(exs: list[Example], device: str):
    """境界の切れ目の (バッチ内の番号, トークン位置, 正解 0/1)。collate に渡したのと同じ並びの exs を渡す。"""
    pos_b, pos_t, gold = [], [], []
    for b, e in enumerate(exs):
        pos_b += [b] * len(e.bpos)
        pos_t += e.bpos
        gold += e.bgold
    t = lambda x: torch.tensor(x, dtype=torch.long, device=device)  # noqa: E731
    return t(pos_b), t(pos_t), torch.tensor(gold, dtype=torch.float32, device=device)


def amp_mode(device: str):
    """(autocast の dtype, GradScaler)。bf16 が使えれば bf16、なければ fp16＋スケーラ、CPU は無し。"""
    if not device.startswith("cuda"):
        return None, None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler("cuda")


def masked_logits(logits, table, rows):
    return logits.float().masked_fill(~table[rows], NEG)


def read_checkpoint(path: Path, log) -> dict | None:
    """途中保存を読む。壊れていれば（中身が切れているなど）、そう表示して None（最初から学習する）。"""
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        log(
            f"  {path.name} が壊れていて読めない（{type(e).__name__}: {e}）ので使わず、最初から学習する"
        )
        return None


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
    checkpoint: Path | None = None,
    run_key: dict | None = None,
    checkpoint_every: float = 300.0,
    boundary_weight: float = 1.0,
    extra: dict[int, list[Example]] | None = None,
) -> None:
    """checkpoint を渡すと、エポックの終わりと checkpoint_every 秒ごとに途中の状態を保存し、
    run_key が同じ保存があればその続きから学習する（クラウドの VM がプリエンプトされても、
    同じコマンドを打ち直せば続きから）。学習の結果を保存し終えたら、呼び出し側で消す。

    model が境界の出力層を持てば、読みと境界を同時に学習する。損失は
    読みの交差エントロピー + boundary_weight × 境界の二値交差エントロピー（どちらも位置の平均）。

    extra はエポック（1 から）ごとに exs へ足す例（断片など）。再開できるよう、呼び出し側で
    決まった作り方にしておく。"""
    joint = model.boundary_head is not None

    def usable(xs: list[Example]) -> list[Example]:
        xs = [p for e in xs for p in (e.parts if e.parts is not None else [e])]
        # 同時学習では、漢字の無い行も境界の学習に使う
        return [e for e in xs if e.pos or (joint and e.bpos)]

    exs = usable(exs)
    extra = {k: usable(v) for k, v in (extra or {}).items()}
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

    def epoch_exs(epoch: int) -> list[Example]:
        return exs + extra.get(epoch, [])

    steps = sum(
        (len(epoch_exs(e)) + batch_size - 1) // batch_size for e in range(1, epochs + 1)
    )
    sched = get_linear_schedule_with_warmup(opt, int(steps * warmup), steps)
    dtype, scaler = amp_mode(device)
    n_extra = sum(len(v) for v in extra.values())
    log(
        f"  学習 {len(exs)} 行"
        + (f"＋断片など 1 エポック平均 {n_extra / epochs:.0f} 個" if n_extra else "")
        + f"・{steps} ステップ・精度={dtype or 'float32'}"
        + (f"・読みと境界の同時学習（境界の重み {boundary_weight}）" if joint else "")
    )

    def save(
        epoch: int, k: int, total: float, n: int, btotal: float = 0.0, bn: int = 0
    ) -> None:
        state = {
            "run_key": run_key,
            "epoch": epoch,  # この epoch の k 番目の例から続ける
            "k": k,
            "total": total,
            "n": n,
            "btotal": btotal,
            "bn": bn,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }
        # 書いている途中で落ちても前の保存が残るように、別名に書いてから置き換える。
        # 中身をディスクへ書き出してから置き換えないと、VM が止まったときに置き換えだけが残り、
        # 中身の切れたファイルになることがある（プリエンプトで実際に起きた）
        tmp = checkpoint.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, checkpoint)
        # 置き換えそのものも書き出す（Linux。Windows には O_DIRECTORY が無い）
        if hasattr(os, "O_DIRECTORY"):
            fd = os.open(checkpoint.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    start_epoch, start_k, total, n, btotal, bn = 1, 0, 0.0, 0, 0.0, 0
    if checkpoint is not None and checkpoint.exists():
        state = read_checkpoint(checkpoint, log)
        if state is None:
            pass  # 壊れていた（read_checkpoint が表示した）
        elif state["run_key"] == run_key:
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            if scaler and state["scaler"]:
                scaler.load_state_dict(state["scaler"])
            torch.set_rng_state(state["rng_cpu"])
            if state["rng_cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["rng_cuda"])
            start_epoch, start_k = state["epoch"], state["k"]
            total, n = state["total"], state["n"]
            btotal, bn = state.get("btotal", 0.0), state.get("bn", 0)
            log(
                f"  {checkpoint.name} から再開（epoch {start_epoch} の {start_k} 例目から）"
            )
        else:
            log(f"  {checkpoint.name} は設定かデータが違うので使わず、最初から学習する")
        del state

    t0 = last_save = time.time()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        cur = epoch_exs(epoch)
        order = list(range(len(cur)))
        random.Random(seed * 1000 + epoch).shuffle(order)
        for k in range(start_k, len(order), batch_size):
            batch = [cur[i] for i in order[k : k + batch_size]]
            ids, mask, pos_b, pos_t, rows, gold = collate(batch, vocab.pad, device)
            with torch.autocast(
                device_type="cuda", dtype=dtype, enabled=dtype is not None
            ):
                if joint:
                    bpos_b, bpos_t, bgold = collate_boundary(batch, device)
                    logits, blogits = model.forward_joint(
                        ids, mask, pos_b, pos_t, bpos_b, bpos_t
                    )
                else:
                    logits = model(ids, mask, pos_b, pos_t)
            # 漢字の無い行だけのバッチ（同時学習のときだけ起こりうる）は、読みの損失を 0 にする
            loss = (
                nn.functional.cross_entropy(masked_logits(logits, table, rows), gold)
                if len(gold)
                else logits.float().sum() * 0.0
            )
            objective = loss
            if joint:
                bloss = (
                    nn.functional.binary_cross_entropy_with_logits(
                        blogits.float(), bgold
                    )
                    if len(bgold)
                    else blogits.float().sum() * 0.0
                )
                objective = loss + boundary_weight * bloss
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(objective).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                objective.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            total += loss.item() * len(gold)
            n += len(gold)
            if joint:
                btotal += bloss.item() * len(bgold)
                bn += len(bgold)
            if checkpoint is not None and time.time() - last_save >= checkpoint_every:
                save(epoch, k + batch_size, total, n, btotal, bn)
                last_save = time.time()
        msg = f"  epoch {epoch}/{epochs} loss={total / max(n, 1):.4f}"
        if joint:
            msg += f" 境界 loss={btotal / max(bn, 1):.4f}"
        msg += f"（{time.time() - t0:.0f} 秒）"
        if on_epoch:
            msg += " " + on_epoch(epoch)
        log(msg)
        start_k, total, n, btotal, bn = 0, 0.0, 0, 0.0, 0
        if checkpoint is not None:
            save(epoch + 1, 0, 0.0, 0)
            last_save = time.time()


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


@torch.no_grad()
def predict_boundary(
    model: ReadingModel,
    exs: list[Example],
    vocab: CharVocab,
    device: str,
    batch_size: int = 64,
) -> list[list[float]]:
    """各例の切れ目（Example.bpos）ごとの、マスをあける確率。入力と同じ順で返す（境界の出力層が要る）。"""
    model.to(device).eval()
    dtype, _ = amp_mode(device)
    out: list[list[float]] = [[] for _ in exs]
    order = sorted(range(len(exs)), key=lambda i: len(exs[i].ids))
    for k in range(0, len(order), batch_size):
        part = order[k : k + batch_size]
        batch = [exs[i] for i in part]
        bpos_b, bpos_t, _ = collate_boundary(batch, device)
        if not len(bpos_t):
            continue
        ids, mask, *_ = collate(batch, vocab.pad, device)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
            h = model.encode(ids, mask)
            logits = model.boundary_head(h[bpos_b, bpos_t]).squeeze(-1)
        prob = logits.float().sigmoid().tolist()
        j = 0
        for i in part:
            n = len(exs[i].bpos)
            out[i] = prob[j : j + n]
            j += n
    return out


@torch.no_grad()
def collapse_stats(
    model: ReadingModel,
    exs: list[Example],
    vocab: CharVocab,
    device: str,
    batch_size: int = 64,
    tol: float = 1e-2,
) -> dict:
    """最終層の隠れ状態が [SEP] と同じに潰れた字の割合（float32 で測る）。

    BERT には、目標の無い位置の隠れ状態を [SEP] と同じにしてしまう癖がある（事前学習のままでも
    行末の「。」などで起きる）。潰れた字どうしは出力層から区別できず、境界の出力層はそこで
    1 つの値（潰れた切れ目のうち注釈がマスあけの割合）しか出せない。

    返り値: all（全部の字）、last（[SEP] の直前の字）、inner（それ以外の字）の割合と、
    境界の出力層があれば sep_prob（[SEP] の位置のマスあけの確率の中央値）。"""
    model.to(device).eval()
    n_all = c_all = n_last = c_last = 0
    sep = []
    order = sorted(range(len(exs)), key=lambda i: len(exs[i].ids))
    for k in range(0, len(order), batch_size):
        batch = [exs[i] for i in order[k : k + batch_size]]
        ids, mask, *_ = collate(batch, vocab.pad, device)
        h = model.encode(ids, mask).float()
        for b, e in enumerate(batch):
            n = len(e.ids)
            if n < 3:
                continue
            same = (h[b, 1 : n - 1] - h[b, n - 1]).abs().amax(-1) < tol
            c_all += int(same.sum())
            n_all += n - 2
            c_last += int(same[-1])
            n_last += 1
            sep.append(h[b, n - 1])
    out = {
        "all": c_all / max(n_all, 1),
        "last": c_last / max(n_last, 1),
        "inner": (c_all - c_last) / max(n_all - n_last, 1),
    }
    if model.boundary_head is not None and sep:
        out["sep_prob"] = (
            model.boundary_head(torch.stack(sep)).squeeze(-1).sigmoid().median().item()
        )
    return out
