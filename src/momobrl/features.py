import unicodedata
import re
from typing import List, Dict, Tuple

# --- [共通定義・型定義] ---
# 🌟 pycrfsuiteネイティブ仕様に変更：値はすべて float（重み）になります
FeatureDict = Dict[str, float]
# ソース文字系列 = [(char, orig_idx, ctype), ...]
SourceEntry = Tuple[str, int, str]

# --- [共通定数] ---
MORA_SPLIT = "+S"   # このラベルの後に分かち書きスペースを挿入する

def get_basic_char_category(c: str) -> str:
    """1文字を受け取り、基本カテゴリ（かな/漢字/英字/その他）を返す。"""
    cp = ord(c)
    if 0x3040 <= cp <= 0x309F: return 'HIRAGANA'
    if 0x30A0 <= cp <= 0x30FF: return 'KATAKANA'
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF: return 'KANJI'
    # 半角英字・全角英字
    if 0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A or 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A:
        return 'ALPHA'
    return 'OTHER'

def get_char_type(c: str) -> str:
    """文字種を判定。極限まで高速化するために ord() による範囲判定を使用。"""
    if not c or c.isspace(): return 'SPACE'
    if c.isdigit() or ('0' <= c <= '9'): return 'NUM'
    
    cat = unicodedata.category(c)
    if cat.startswith('P') or cat.startswith('S'): return 'SYMBOL'

    return get_basic_char_category(c)

def get_units(text: str) -> List[Tuple[str, int]]:
    """
    テキストを音節ユニットに分解し、それぞれの原文開始位置を返す。
    ブラケット内の文字も、原文の正確なインデックス（0始まり）を保持する。
    英数字の連続（カンマやハイフン含む）は自動的に1つのブロックとしてまとめる。
    """
    regex = r'\[(.*?)\]|([ぁ-んァ-ヶ][ぁぃぅぇぉゃゅょァィゥェォャュョ])|([a-zA-Z0-9\.\-,]+)|(\s+)|(.)'
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
    pycrfsuiteネイティブの { "feature_name=value": 1.0 } 形式で出力する。
    """
    result: List[FeatureDict] = []
    n = len(source_seq)

    for i, (char, _orig_idx, ctype) in enumerate(source_seq):
        prev_char  = source_seq[i - 1][0] if i > 0 else ""
        prev_ctype = source_seq[i - 1][2] if i > 0 else ""
        next_char  = source_seq[i + 1][0] if i < n - 1 else ""
        next_ctype = source_seq[i + 1][2] if i < n - 1 else ""

        # 🌟 自身の文字と文字種（f文字列を使ってキー側に埋め込む）
        features: FeatureDict = {
            'bias': 1.0,
            f'char={char}': 1.0,
            f'type={ctype}': 1.0,
        }

        # --- 前後1文字コンテキスト ---
        if i > 0:
            features[f'-1:char={prev_char}'] = 1.0
            features[f'-1:type={prev_ctype}'] = 1.0
            features[f'-1:bi={prev_char}{char}'] = 1.0
            if i > 1:
                features[f'-2:char={source_seq[i - 2][0]}'] = 1.0
                features[f'-2:type={source_seq[i - 2][2]}'] = 1.0
                features[f'-2:-1:bi={source_seq[i - 2][0]}{prev_char}'] = 1.0
        else:
            features['BOS'] = 1.0

        if i < n - 1:
            features[f'+1:char={next_char}'] = 1.0
            features[f'+1:type={next_ctype}'] = 1.0
            features[f'+1:bi={char}{next_char}'] = 1.0
            if i < n - 2:
                features[f'+2:char={source_seq[i + 2][0]}'] = 1.0
                features[f'+2:type={source_seq[i + 2][2]}'] = 1.0
                features[f'+1:+2:bi={next_char}{source_seq[i + 2][0]}'] = 1.0
        else:
            features['EOS'] = 1.0

        # --- 特徴量1: 文字種遷移パターン ---
        if i > 0:
            features[f'type_transition={prev_ctype}->{ctype}'] = 1.0

        # --- 特徴量2 & 3: 漢字連続長 / 漢字ラン先頭フラグ ---
        if ctype == 'KANJI':
            run = 1
            j = i + 1
            while j < n and source_seq[j][2] == 'KANJI':
                run += 1; j += 1
            j = i - 1
            while j >= 0 and source_seq[j][2] == 'KANJI':
                run += 1; j -= 1
                
            # 数値（長さ）はそのまま重みとして渡す
            features['kanji_run_len'] = float(run)
            
            if i == 0 or source_seq[i - 1][2] != 'KANJI':
                features['kanji_pos_first'] = 1.0

        result.append(features)

    return result