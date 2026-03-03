import unicodedata
import re
from typing import Union, List, Dict, Tuple

# --- [共通定義・型定義] ---
FeatureDict = Dict[str, Union[str, float, bool]]
# ソース文字系列 = [(char, orig_idx, ctype), ...]
SourceEntry = Tuple[str, int, str]

# --- [共通定数] ---
MORA_SPLIT = "+S"   # このラベルの後に分かち書きスペースを挿入する

def get_char_type(c: str) -> str:
    """文字種を判定。句読点・記号を独立したカテゴリ(SYMBOL)として扱う。"""
    if not c or c.isspace(): return 'SPACE'
    if c.isdigit() or ('0' <= c <= '9'): return 'NUM'
    
    # Unicodeカテゴリで句読点(P)や記号(S)を判定
    cat = unicodedata.category(c)
    if cat.startswith('P') or cat.startswith('S'):
        return 'SYMBOL'
        
    name = unicodedata.name(c, "")
    if "HIRAGANA" in name: return 'HIRAGANA'
    if "KATAKANA" in name: return 'KATAKANA'
    if "CJK UNIFIED IDEOGRAPH" in name: return 'KANJI'
    if "LATIN" in name: return 'ALPHA'
    
    return 'OTHER'

def get_units(text: str) -> List[Tuple[str, int]]:
    """
    テキストを音節ユニットに分解し、それぞれの原文開始位置を返す。
    ブラケット内の文字も、原文の正確なインデックス（0始まり）を保持する。
    英数字の連続（カンマやハイフン含む）は自動的に1つのブロックとしてまとめる。
    """
    regex = r'\[(.*?)\]|([ぁ-んァ-ヶ][ぁぃぅぇぉゃュょァィゥェォャュョ])|([a-zA-Z0-9\.\-,]+)|(\s+)|(.)'
    units: List[Tuple[str, int]] = []
    for m in re.finditer(regex, text):
        if (g := m.group(1)) is not None:
            units.append((g, m.start(1)))
        elif (g := m.group(2)) is not None:
            units.append((g, m.start(2)))
        elif (g := m.group(3)) is not None:
            units.append((g, m.start(3)))
        elif (g := m.group(4)) is not None:
            units.append((g, m.start(4)))
        else:
            units.append((m.group(5) or "", m.start(5)))
    return units

def compute_source_features(source_seq: List[SourceEntry]) -> List[FeatureDict]:
    """
    ソース文字系列全体に対して、各文字の文脈特徴量を一括計算する。
    1文字=1ラベルの設計に対応し、各文字に対応するFeatureDictのリストを返す。
    """
    result: List[FeatureDict] = []
    n = len(source_seq)

    for i, (char, _orig_idx, ctype) in enumerate(source_seq):
        prev_char  = source_seq[i - 1][0] if i > 0 else ""
        prev_ctype = source_seq[i - 1][2] if i > 0 else ""
        next_char  = source_seq[i + 1][0] if i < n - 1 else ""
        next_ctype = source_seq[i + 1][2] if i < n - 1 else ""

        features: FeatureDict = {
            'bias': 1.0,
            'char': char,
            'type': ctype,
        }

        # --- 前後1文字コンテキスト ---
        if i > 0:
            features['-1:char'] = prev_char
            features['-1:type'] = prev_ctype
            features['-1:bi']   = prev_char + char
            if i > 1:
                features['-2:char'] = source_seq[i - 2][0]
                features['-2:type'] = source_seq[i - 2][2]
                features['-2:-1:bi'] = source_seq[i - 2][0] + prev_char
        else:
            features['BOS'] = True

        if i < n - 1:
            features['+1:char'] = next_char
            features['+1:type'] = next_ctype
            features['+1:bi']   = char + next_char
            if i < n - 2:
                features['+2:char'] = source_seq[i + 2][0]
                features['+2:type'] = source_seq[i + 2][2]
                features['+1:+2:bi'] = next_char + source_seq[i + 2][0]
        else:
            features['EOS'] = True

        # --- 特徴量1: 文字種遷移パターン ---
        if i > 0:
            features['type_transition'] = prev_ctype + '->' + ctype

        # --- 特徴量2 & 3: 漢字連続長 / 漢字ラン先頭フラグ ---
        if ctype == 'KANJI':
            run = 1
            j = i + 1
            while j < n and source_seq[j][2] == 'KANJI':
                run += 1; j += 1
            j = i - 1
            while j >= 0 and source_seq[j][2] == 'KANJI':
                run += 1; j -= 1
            features['kanji_run_len']   = run
            features['kanji_pos_first'] = (i == 0 or source_seq[i - 1][2] != 'KANJI')

        result.append(features)

    return result