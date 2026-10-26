"""人名辞書（gazetteer）関連。

学習データの原文には {佐藤} のように人名スパンをマークする。
姓と名は別スパンで囲む（例: {佐藤}{太郎}さん）。1スパン = 分かち書き1単位 = 辞書1エントリ。

マーク自体は特徴量に直接使わない（推論時の入力にはマークが存在しないため）。
マークから抽出した人名辞書に対する「辞書マッチ関数」(compute_name_flags) を
学習・推論の両方で同一に適用し、文字ごとの B/I フラグを特徴量に変換する。
"""

import re
from typing import Dict, Iterable, List, Optional, Tuple, Union

from .features import get_units, SourceEntry
from .utils import normalize_one_to_one

# 文字ごとの人名フラグ
NAME_FLAG_BEGIN = "B"  # 人名スパンの先頭
NAME_FLAG_INSIDE = "I"  # 人名スパンの継続
NAME_FLAG_OUT = "O"  # 人名スパン外

# デフォルトの人名辞書ファイル名（basic_data.tsv と同じディレクトリに置く）
NAME_DICT_FILENAME = "person_name_dic.tsv"

# 辞書エントリ: (表層形, ユニット別読み or None)
# 読みは get_units(表層形) のユニットに1:1で対応するカタカナのリスト。
# 人名の読みは文脈で変化しないため固定辞書として扱い、推論時の
# 低自信度フォールバックに使う。
NameEntry = Tuple[str, Optional[List[str]]]

_MARK_RE = re.compile(r"\{([^{}]*)\}")


def parse_name_marks(raw: str) -> Tuple[str, List[Tuple[int, int]]]:
    """原文から {…} 人名マークを取り除き、マーク除去後の文字列と
    人名スパン [(start, end), ...]（除去後座標・end は排他的）を返す。

    マークの対応が取れていない場合（閉じ忘れ・空スパンなど）は ValueError。
    """
    clean_parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    clean_len = 0
    pos = 0
    for m in _MARK_RE.finditer(raw):
        inner = m.group(1)
        if inner == "":
            raise ValueError(f"空の人名マーク {{}} があります: {raw!r}")
        before = raw[pos : m.start()]
        clean_parts.append(before)
        clean_len += len(before)
        spans.append((clean_len, clean_len + len(inner)))
        clean_parts.append(inner)
        clean_len += len(inner)
        pos = m.end()
    tail = raw[pos:]
    clean_parts.append(tail)
    clean = "".join(clean_parts)

    if "{" in clean or "}" in clean:
        raise ValueError(f"対応の取れていない人名マーク {{}} があります: {raw!r}")
    return clean, spans


def name_flag_for_unit(
    orig_idx: int, unit_len: int, spans: List[Tuple[int, int]]
) -> str:
    """ユニット（開始位置 orig_idx・文字数 unit_len）の人名フラグを返す。

    ユニットがスパン境界をまたぐ場合（拗音ユニットの途中でマークが
    切れているなど）はアノテーションエラーとして ValueError。
    """
    end = orig_idx + unit_len
    for start, stop in spans:
        if orig_idx >= start and end <= stop:
            return NAME_FLAG_BEGIN if orig_idx == start else NAME_FLAG_INSIDE
        # 部分的な重なりはエラー
        if orig_idx < stop and end > start:
            raise ValueError(
                f"人名マークが音節ユニットの途中で切れています "
                f"(ユニット位置 {orig_idx}, スパン {start}-{stop})"
            )
    return NAME_FLAG_OUT


def extract_names(raw: str) -> List[str]:
    """原文の {…} マークから人名表層形のリストを返す（重複はそのまま）。"""
    clean, spans = parse_name_marks(raw)
    return [clean[start:stop] for start, stop in spans]


def parse_name_dict_text(text: str) -> List[NameEntry]:
    """人名辞書テキストを解析して (表層形, ユニット別読み) リストを返す。

    フォーマット: 1行1エントリ「表層形[TAB]読み[TAB]出現回数」。
    読みはユニット別に '/' 区切り（例: ワタ/ナベ）。# 行は無視する。
    読み列が省略された行・数字のみの行（旧形式の出現回数）は読みなしとして扱う。
    表層形は推論時の入力正規化と揃えるため normalize_one_to_one を適用する。
    """
    entries: List[NameEntry] = []
    seen = set()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        surface = normalize_one_to_one(cols[0])
        if not surface or surface in seen:
            continue
        seen.add(surface)

        readings: Optional[List[str]] = None
        if len(cols) >= 2 and cols[1] and not cols[1].isdigit():
            readings = cols[1].split("/")
            # 読みは表層形のユニット数と1:1で対応していなければならない
            if len(readings) != len(get_units(surface)):
                print(
                    f"⚠️  人名辞書: {surface!r} の読み {cols[1]!r} がユニット数と"
                    "一致しないため読みを無視します"
                )
                readings = None
        entries.append((surface, readings))
    return entries


def load_name_dict(path: str) -> List[NameEntry]:
    """人名辞書ファイルを読み込む。フォーマットは parse_name_dict_text を参照。"""
    with open(path, encoding="utf-8") as f:
        return parse_name_dict_text(f.read())


# インデックス: 先頭文字 → [(ユニット列, ユニット別読み or None), ...]
NameIndex = Dict[str, List[Tuple[List[str], Optional[List[str]]]]]


def build_name_index(names: Iterable[Union[str, NameEntry]]) -> NameIndex:
    """人名エントリリストから照合用インデックスを構築する。

    キーは先頭ユニットの先頭1文字、値は (ユニット列, 読み) リスト
    （ユニット数の降順ソート済み）。照合は get_units() のユニット単位で行い、
    学習（TSV行）と推論（source_seq）の双方と表現が揃うようにする。
    表層形のみの文字列も受け付ける（読みなしエントリとして扱う）。
    """
    index: NameIndex = {}
    for item in names:
        surface, readings = (item, None) if isinstance(item, str) else item
        units = [u[0] for u in get_units(surface)]
        if not units:
            continue
        index.setdefault(units[0][0], []).append((units, readings))
    for key in index:
        index[key].sort(key=lambda x: len(x[0]), reverse=True)
    return index


def compute_name_matches(
    source_seq: List[SourceEntry],
    index: NameIndex,
) -> Tuple[List[str], List[Optional[str]]]:
    """ソース系列に人名辞書を最長一致で照合する。

    戻り値: (B/I/O フラグ列, ユニット別辞書読み列)。
    読みは辞書に読みが登録されているスパン内の位置のみ、それ以外は None。
    """
    n = len(source_seq)
    flags = [NAME_FLAG_OUT] * n
    readings_out: List[Optional[str]] = [None] * n
    i = 0
    while i < n:
        char = source_seq[i][0]
        candidates = index.get(char[0] if char else "")
        matched = 0
        matched_readings: Optional[List[str]] = None
        if candidates:
            for units, readings in candidates:
                length = len(units)
                if i + length > n:
                    continue
                if all(source_seq[i + j][0] == units[j] for j in range(length)):
                    matched = length
                    matched_readings = readings
                    break
        if matched:
            flags[i] = NAME_FLAG_BEGIN
            for j in range(1, matched):
                flags[i + j] = NAME_FLAG_INSIDE
            if matched_readings is not None:
                for j in range(matched):
                    readings_out[i + j] = matched_readings[j]
            i += matched
        else:
            i += 1
    return flags, readings_out


def compute_name_flags(
    source_seq: List[SourceEntry],
    index: NameIndex,
) -> List[str]:
    """ソース系列に人名辞書を最長一致で照合し、位置ごとの B/I/O フラグを返す。

    学習時（trainer）と推論時（predictor）の両方からこの照合を使うことで、
    特徴量の計算手順を完全に一致させる。
    """
    flags, _ = compute_name_matches(source_seq, index)
    return flags
