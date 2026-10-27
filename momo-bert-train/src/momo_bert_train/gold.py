"""basic_raw.tsv の1行を、学習データと同じ行（スロット）に展開する。

momo_train の process_line_to_tsv を使う。tools/llm_reading/_common.py の parse_gold と同じ規則で、
スロットは学習データの1行に当たる。漢字は1文字ずつ、拗音は「きょ」で1つ、[今日] の「日」は
読みが空の継続スロット。スロットの位置は推論に渡す原文（{…} と [...] の記法を除いたもの）の位置。
"""

from __future__ import annotations

import contextlib
import io
import re
from dataclasses import dataclass

from momo_train.features import LABEL_CONTINUE, LABEL_SKIP
from momo_train.name_dict import parse_name_marks
from momo_train.trainer import process_line_to_tsv

_GROUP_RE = re.compile(r"\[(.*?)\]")  # get_units() と同じ [...] 記法


def read_raw_lines(path: str) -> list[str]:
    """basic_raw.tsv 形式のファイルから、空行と # 行を除いた行を返す。"""
    with open(path, encoding="utf-8") as f:
        return [
            line.rstrip("\n") for line in f if line.strip() and not line.startswith("#")
        ]


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
    start: int  # 推論入力での開始位置
    end: int


@dataclass
class GoldSentence:
    raw: str  # basic_raw.tsv の行そのもの
    text: str  # 推論に渡す原文
    slots: list[Slot]


def parse_gold(line: str, line_num: int = 0) -> GoldSentence:
    raw_part = line.split("\t")[0]
    # process_line_to_tsv は注釈の不審点を print するので、ここでは黙らせる
    with contextlib.redirect_stdout(io.StringIO()):
        rows = process_line_to_tsv(line, line_num, None)
    clean, _ = parse_name_marks(raw_part)
    text, pos_map = strip_group_marks(clean)

    slots = []
    for row in rows:
        surface, label, ctype, orig = row.split("\t")[:4]
        label = label.replace("+S", "")
        if label == LABEL_CONTINUE:
            gold = ""
        elif label == LABEL_SKIP:
            gold = surface
        else:
            gold = label
        start = pos_map[int(orig)]
        slots.append(Slot(surface, gold, ctype, start, start + len(surface)))
    return GoldSentence(line, text, slots)
