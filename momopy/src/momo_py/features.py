import re
from typing import List, Dict, Tuple

from .utils import (
    CharType, get_basic_char_category, get_char_type, has_vowel,
    _JAPANESE_NUMERIC_CHARS, _KURAI_CHARS,
)

# --- [共通定義・型定義] ---
FeatureDict = Dict[str, float]
# ソース文字系列 = [(char, orig_idx, ctype), ...]
SourceEntry = Tuple[str, int, str]

# --- [共通定数] ---
MORA_SPLIT     = "+S"
LABEL_CONTINUE = "---"
LABEL_SKIP     = "_"


def get_units(text: str) -> List[Tuple[str, int, str]]:
    """
    テキストを音節ユニットに分解し、(文字列, 原文開始位置, 文字種) のリストを返す。

    ひらがな+小書き（拗音など）は複数文字ブロックとしてまとめる。
    英数字連続は学習データ作成時の省略記法（"Hello" を1ブロックで書ける）を
    サポートするためにまとめる。推論時は _preprocess_text で1文字に展開し直す。
    JAPANESE_NUMERIC については以下の2段階で文字種を確定する:
      1. 漢数字（〇一二三…）は無条件で JAPANESE_NUMERIC。
      2. 位取り文字（十百千万億兆）は左右いずれかに JAPANESE_NUMERIC が
         隣接する場合のみ昇格させる。
         （「万全」の「万」は昇格しない。「三万円」の「万」は昇格する。）
    ブロック化は行わず、各要素は1文字（または既存の複数文字ブロック）のまま。
    """
    regex = r'\[(.*?)\]|([ぁ-んァ-ヶ][ぁぃぅぇぉゃゅょゎァィゥェォャュョヮヵヶ]+)|([!-~]+(?:[ \t]+[!-~]+)*)|(\s+)|(.)'
    raw_units: List[Tuple[str, int]] = []
    for m in re.finditer(regex, text):
        if (g := m.group(1)) is not None:
            raw_units.append((g, m.start(1)))
        elif (g := m.group(2)) is not None:
            raw_units.append((g, m.start(2)))
        elif (g := m.group(3)) is not None:
            raw_units.append((g, m.start(3)))
        elif (g := m.group(4)) is not None:
            raw_units.append((g, m.start(4)))
        else:
            raw_units.append((m.group(5) or "", m.start(5)))

    # --- 位取り文字の昇格判定 ---
    n = len(raw_units)
    is_numeric: List[bool] = [
        (len(val) == 1 and get_char_type(val) == CharType.JAPANESE_NUMERIC)
        for val, _ in raw_units
    ]

    # 左→右パス: 左隣が numeric なら位取り文字を昇格
    for i in range(1, n):
        val = raw_units[i][0]
        if len(val) == 1 and val in _KURAI_CHARS and is_numeric[i - 1]:
            is_numeric[i] = True

    # 右→左パス: 右隣が numeric なら位取り文字を昇格（「十一」の「十」など）
    for i in range(n - 2, -1, -1):
        val = raw_units[i][0]
        if len(val) == 1 and val in _KURAI_CHARS and is_numeric[i + 1]:
            is_numeric[i] = True

    # --- ctype を確定して返す（ブロック化しない） ---
    units: List[Tuple[str, int, str]] = []
    for i, (val, orig_idx) in enumerate(raw_units):
        if is_numeric[i]:
            ctype = CharType.JAPANESE_NUMERIC
        else:
            ctype = get_char_type(val[0]) if val else CharType.OTHER
        units.append((val, orig_idx, ctype))

    return units


def compute_source_features(source_seq: List[SourceEntry]) -> List[FeatureDict]:
    """
    ソース文字系列全体に対して、各文字の文脈特徴量を一括計算する。
    pycrfsuiteネイティブの { "feature_name=value": 1.0 } 形式で出力する。
    """
    result: List[FeatureDict] = []
    n = len(source_seq)

    for i, (char, _orig_idx, ctype) in enumerate(source_seq):
        prev2_char  = source_seq[i - 2][0] if i > 1 else ""
        prev2_ctype = source_seq[i - 2][2] if i > 1 else ""
        prev_char   = source_seq[i - 1][0] if i > 0 else ""
        prev_ctype  = source_seq[i - 1][2] if i > 0 else ""
        next_char   = source_seq[i + 1][0] if i < n - 1 else ""
        next_ctype  = source_seq[i + 1][2] if i < n - 1 else ""
        next2_char  = source_seq[i + 2][0] if i < n - 2 else ""
        next2_ctype = source_seq[i + 2][2] if i < n - 2 else ""

        features: FeatureDict = {
            'bias': 1.0,
            f'char={char}': 1.0,
            f'type={ctype}': 1.0,
        }

        if i > 0:
            features[f'-1:char={prev_char}'] = 1.0
            features[f'-1:type={prev_ctype}'] = 1.0
            features[f'-1:bi={prev_char}{char}'] = 1.0
            if i > 1:
                features[f'-2:char={prev2_char}'] = 1.0
                features[f'-2:type={prev2_ctype}'] = 1.0
                features[f'-2:-1:bi={prev2_char}{prev_char}'] = 1.0
        else:
            features['BOS'] = 1.0

        if i < n - 1:
            features[f'+1:char={next_char}'] = 1.0
            features[f'+1:type={next_ctype}'] = 1.0
            features[f'+1:bi={char}{next_char}'] = 1.0
            if i < n - 2:
                features[f'+2:char={next2_char}'] = 1.0
                features[f'+2:type={next2_ctype}'] = 1.0
                features[f'+1:+2:bi={next_char}{next2_char}'] = 1.0
        else:
            features['EOS'] = 1.0

        if i > 0:
            features[f'type_transition={prev_ctype}->{ctype}'] = 1.0

        if i > 0 and i < n - 1:
            features[f'tri={prev_char}{char}{next_char}'] = 1.0

        if i > 0 and i < n - 1:
            features[f'type_tri={prev_ctype}-{ctype}-{next_ctype}'] = 1.0

        if ctype == 'KANJI':
            run = 1
            j = i + 1
            while j < n and source_seq[j][2] == 'KANJI':
                run += 1; j += 1
            j = i - 1
            while j >= 0 and source_seq[j][2] == 'KANJI':
                run += 1; j -= 1

            run_key = str(run) if run <= 4 else '5+'
            features[f'kanji_run_len={run_key}'] = 1.0

            if i == 0 or source_seq[i - 1][2] != 'KANJI':
                features['kanji_pos_first'] = 1.0

            # 左隣の JAPANESE_NUMERIC 連続長を「日」などの隣接 KANJI に伝える
            if i > 0 and source_seq[i - 1][2] == 'JAPANESE_NUMERIC':
                num_run = 0
                j = i - 1
                while j >= 0 and source_seq[j][2] == 'JAPANESE_NUMERIC':
                    num_run += 1; j -= 1
                num_run_key = str(num_run) if num_run <= 4 else '5+'
                features[f'-1:japanese_numeric_run_len={num_run_key}'] = 1.0

        if ctype == 'JAPANESE_NUMERIC':
            run = 1
            j = i + 1
            while j < n and source_seq[j][2] == 'JAPANESE_NUMERIC':
                run += 1; j += 1
            j = i - 1
            while j >= 0 and source_seq[j][2] == 'JAPANESE_NUMERIC':
                run += 1; j -= 1

            run_key = str(run) if run <= 4 else '5+'
            features[f'japanese_numeric_run_len={run_key}'] = 1.0

        result.append(features)

    return result
