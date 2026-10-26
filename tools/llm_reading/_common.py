"""各スクリプトが共有するパス定義と、正解・予測の突き合わせ。

    from _common import *

- ROOT / DATASET / MOMO_TRAIN / WORK / SPLIT_DIR  パス
- parse_gold()       basic_raw.tsv の1行 → スロット列（momo_train の process_line_to_tsv を再利用）
- align_prediction() momors_py の予測を、正解と同じスロットに割り付ける
- score()            スロット一致率・マスあけ F1 などの指標

スロットは学習データの1行に当たる。漢字は1文字ずつ、拗音は「きょ」で1つ、[今日] の
「日」は読みが空の継続スロット。予測のかなは kana_to_source で原文位置へ戻してから
スロットに集めるので、読みが合っていても分け方（トー/キョー と トーキョー/---）が
違えば不一致になる。これを「対応の正しさ」として測る。
"""

from __future__ import annotations

import contextlib
import io
import os
import re
from dataclasses import dataclass, field

from momo_train.features import LABEL_CONTINUE, LABEL_SKIP, get_units
from momo_train.name_dict import parse_name_marks
from momo_train.trainer import process_line_to_tsv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATASET = os.path.join(ROOT, "dataset")
MOMO_TRAIN = os.path.join(ROOT, "momo-train")
WORK = os.path.join(SCRIPT_DIR, "work")
SPLIT_DIR = os.path.join(WORK, "split")

RAW_PATH = os.path.join(DATASET, "basic_raw.tsv")
REGRESSION_PATH = os.path.join(DATASET, "eval_regression.tsv")
TRAIN_RAW = os.path.join(SPLIT_DIR, "train_raw.tsv")
DEV_RAW = os.path.join(SPLIT_DIR, "dev_raw.tsv")

# 読みの正誤を主に見る文字種（漢字と漢数字）
KANJI_TYPES = frozenset({"KANJI", "JAPANESE_NUMERIC"})

_GROUP_RE = re.compile(r"\[(.*?)\]")  # get_units() と同じ [...] 記法


def read_raw_numbered(path: str) -> list[tuple[int, str]]:
    """basic_raw.tsv 形式のファイルから、空行と # 行を除いた (ファイルの行番号, 行) を返す。"""
    with open(path, encoding="utf-8") as f:
        return [
            (n, line.rstrip("\n"))
            for n, line in enumerate(f, 1)
            if line.strip() and not line.startswith("#")
        ]


def read_raw_lines(path: str) -> list[str]:
    """basic_raw.tsv 形式のファイルから、空行と # 行を除いた行を返す。"""
    return [line for _, line in read_raw_numbered(path)]


def read_regression() -> list[tuple[str, str]]:
    """eval_regression.tsv の (入力, 期待するかな出力) を返す。"""
    cases = []
    for line in read_raw_lines(REGRESSION_PATH):
        parts = line.split("\t")
        if len(parts) >= 2:
            cases.append((parts[0], parts[1]))
    return cases


def strip_group_marks(text: str) -> tuple[str, list[int]]:
    """[...] 記法の角括弧を除き、(除去後の文字列, 除去前位置 → 除去後位置) を返す。

    角括弧そのものの位置は -1 になる。
    """
    out: list[str] = []
    pos_map = [-1] * len(text)

    def keep(lo: int, hi: int) -> None:
        for j in range(lo, hi):
            pos_map[j] = len(out)
            out.append(text[j])

    tail = 0
    for m in _GROUP_RE.finditer(text):
        keep(tail, m.start())
        keep(m.start(1), m.end(1))
        tail = m.end()
    keep(tail, len(text))
    return "".join(out), pos_map


@dataclass
class Slot:
    surface: str  # 原文の文字（拗音は2文字）
    gold: str  # 正解の読み（継続は ""、SKIP は原文の文字そのもの）
    ctype: str
    start: int  # 推論入力（記法を除いた原文）での開始位置
    end: int
    space_after: bool  # 正解でこのスロットの後にマスあけがあるか


@dataclass
class GoldSentence:
    raw: str  # basic_raw.tsv の行そのもの
    text: str  # 推論に渡す原文（{…} と [...] の記法を除いたもの）
    slots: list[Slot]

    @property
    def space_before(self) -> set[int]:
        """マスあけの直後に来るスロットの番号。"""
        return {
            i + 1
            for i, s in enumerate(self.slots)
            if s.space_after and i + 1 < len(self.slots)
        }


def parse_gold(line: str, line_num: int = 0) -> GoldSentence:
    """basic_raw.tsv の1行を、学習データと同じ行（スロット）に展開する。"""
    raw_part = line.split("\t")[0]
    # process_line_to_tsv は注釈の不審点を print するので、ここでは黙らせる
    with contextlib.redirect_stdout(io.StringIO()):
        rows = process_line_to_tsv(line, line_num, None)
    clean, _ = parse_name_marks(raw_part)
    text, pos_map = strip_group_marks(clean)

    slots = []
    for row in rows:
        surface, label, ctype, orig = row.split("\t")[:4]
        space_after = "+S" in label
        label = label.replace("+S", "")
        if label == LABEL_CONTINUE:
            gold = ""
        elif label == LABEL_SKIP:
            gold = surface
        else:
            gold = label
        start = pos_map[int(orig)]
        slots.append(
            Slot(surface, gold, ctype, start, start + len(surface), space_after)
        )
    return GoldSentence(line, text, slots)


@dataclass
class PredSentence:
    kana: str  # 予測のかな（マスあけ込み）
    readings: list[str]  # スロットごとの予測の読み（マスあけを除く）
    space_before: set[int] = field(default_factory=set)
    stray: str = ""  # どのスロットにも割り付けられなかったかな
    # スロットごとの自信度（割り付いたかなの自信度の最小。かなが無いスロットは 1.0）
    confidences: list[float] = field(default_factory=list)


def align_prediction(gold: GoldSentence, result) -> PredSentence:
    """momors_py の PredictionResult を、正解と同じスロットに割り付ける。"""
    owner: dict[int, int] = {}
    for si, s in enumerate(gold.slots):
        for p in range(s.start, s.end):
            owner[p] = si

    pred = PredSentence(result.kana, [""] * len(gold.slots))
    pred.confidences = [1.0] * len(gold.slots)
    pending_space = False
    for ch, src, conf in zip(result.kana, result.kana_to_source, result.confidences):
        si = owner.get(src)
        is_content_space = si is not None and gold.slots[si].surface.isspace()
        if ch == " " and not is_content_space:
            pending_space = True
            continue
        if si is None:
            pred.stray += ch
            continue
        if pending_space and pred.readings[si] == "":
            pred.space_before.add(si)
        pending_space = False
        pred.readings[si] += ch
        pred.confidences[si] = min(pred.confidences[si], conf)
    return pred


def score(pairs: list[tuple[GoldSentence, PredSentence]]) -> dict[str, float]:
    """スロット一致率・連結読みの一致率・マスあけ P/R/F1・継続の使用率をまとめる。"""
    n_slot = ok_slot = 0
    n_kanji = ok_kanji = 0
    kanji_empty_gold = kanji_empty_pred = 0
    sent_concat_ok = sent_exact = 0
    tp = fp = fn = 0

    for g, p in pairs:
        all_ok = True
        for s, r in zip(g.slots, p.readings):
            hit = s.gold == r
            n_slot += 1
            ok_slot += hit
            all_ok &= hit
            if s.ctype in KANJI_TYPES:
                n_kanji += 1
                ok_kanji += hit
                kanji_empty_gold += s.gold == ""
                kanji_empty_pred += r == ""
        if "".join(s.gold for s in g.slots) == "".join(p.readings):
            sent_concat_ok += 1
        gb, pb = g.space_before, p.space_before
        tp += len(gb & pb)
        fp += len(pb - gb)
        fn += len(gb - pb)
        if all_ok and gb == pb and not p.stray:
            sent_exact += 1

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    n = len(pairs)
    return {
        "sentences": n,
        "kanji_slot_acc": ok_kanji / n_kanji if n_kanji else 0.0,
        "slot_acc": ok_slot / n_slot if n_slot else 0.0,
        "concat_reading_acc": sent_concat_ok / n if n else 0.0,
        "sentence_exact": sent_exact / n if n else 0.0,
        "boundary_p": prec,
        "boundary_r": rec,
        "boundary_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "kanji_cont_gold": kanji_empty_gold / n_kanji if n_kanji else 0.0,
        "kanji_cont_pred": kanji_empty_pred / n_kanji if n_kanji else 0.0,
    }


def load_dev() -> list[GoldSentence]:
    return [parse_gold(line, i) for i, line in enumerate(read_raw_lines(DEV_RAW), 1)]


def predict_all(
    predictor, golds: list[GoldSentence]
) -> list[tuple[GoldSentence, PredSentence]]:
    return [(g, align_prediction(g, predictor.predict(g.text))) for g in golds]


def load_single_dict(path: str) -> dict[str, set[str]]:
    """single_character_dic.tsv を 文字 → 読み候補の集合 で読む（--- と _ は除く）。"""
    dic: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                dic[parts[0]] = {
                    r for r in parts[1:] if r not in (LABEL_CONTINUE, LABEL_SKIP)
                }
    return dic


def slots_for_text(text: str) -> list[Slot]:
    """注釈のない原文を、学習データと同じ区切り（漢字は1文字ずつ）のスロットに分ける。"""
    slots = []
    for chars, idx, ctype in get_units(text):
        if chars.isspace():
            continue
        ctype = str(ctype)
        pieces = list(chars) if ctype in KANJI_TYPES and len(chars) > 1 else [chars]
        for i, piece in enumerate(pieces):
            start = idx + i
            slots.append(Slot(piece, "", ctype, start, start + len(piece), False))
    return slots


def pick_fewshot(k: int) -> list[GoldSentence]:
    """LLM に見せる手本を train の短い文から選ぶ（継続 --- と は→ワ と長音を含む文を優先）。"""
    golds = [parse_gold(line, n) for n, line in enumerate(read_raw_lines(TRAIN_RAW), 1)]
    short = [
        g
        for g in golds
        if 12 <= len(g.text) <= 32 and sum(s.ctype == "KANJI" for s in g.slots) >= 3
    ]
    picked = []
    for need in (
        lambda g: any(s.ctype == "KANJI" and s.gold == "" for s in g.slots),
        lambda g: any(s.surface == "は" and s.gold == "ワ" for s in g.slots),
        lambda g: any(s.gold.endswith("ー") and s.ctype == "KANJI" for s in g.slots),
    ):
        g = next((g for g in short if need(g) and g not in picked), None)
        if g:
            picked.append(g)
    for g in short:
        if len(picked) >= k:
            break
        if g not in picked:
            picked.append(g)
    return picked
