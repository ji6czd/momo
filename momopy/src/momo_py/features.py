import unicodedata
import re
from enum import Enum
from typing import List, Dict, Tuple

# --- [共通定義・型定義] ---
# 🌟 pycrfsuiteネイティブ仕様に変更：値はすべて float（重み）になります
FeatureDict = Dict[str, float]
# ソース文字系列 = [(char, orig_idx, ctype), ...]
SourceEntry = Tuple[str, int, str]

class CharCategory(str, Enum):
    """文字カテゴリ列挙型（str継承で文字列互換性を維持）"""
    HIRAGANA = 'HIRAGANA'
    KATAKANA = 'KATAKANA'
    KANJI = 'KANJI'
    ALPHA = 'ALPHA'
    OTHER = 'OTHER'

class CharType(str, Enum):
    """文字種列挙型"""
    SPACE = 'SPACE'
    NUM = 'NUM'
    SYMBOL = 'SYMBOL'
    HIRAGANA = 'HIRAGANA'
    KATAKANA = 'KATAKANA'
    KANJI = 'KANJI'
    ALPHA = 'ALPHA'
    OTHER = 'OTHER'

# --- [共通定数] ---
MORA_SPLIT = "+S"   # このラベルの後に分かち書きスペースを挿入する
LABEL_CONTINUE = "---"
LABEL_SKIP = "_"

def get_basic_char_category(c: str) -> CharCategory:
    """1文字を受け取り、基本カテゴリ（かな/漢字/英字/その他）を返す。"""
    cp = ord(c)
    if 0x3040 <= cp <= 0x309F: return CharCategory.HIRAGANA
    if 0x30A0 <= cp <= 0x30FF: return CharCategory.KATAKANA
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF: return CharCategory.KANJI
    # 半角英字・全角英字
    if 0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A or 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A:
        return CharCategory.ALPHA
    return CharCategory.OTHER

def get_char_type(c: str) -> CharType:
    """文字種を判定。極限まで高速化するために ord() による範囲判定を使用。"""
    if not c or c.isspace(): return CharType.SPACE
    if c.isdigit() or ('0' <= c <= '9'): return CharType.NUM
    
    cat = unicodedata.category(c)
    if cat.startswith('P') or cat.startswith('S'): return CharType.SYMBOL

    # get_basic_char_categoryの結果を CharType に変換
    basic = get_basic_char_category(c)
    return CharType(basic.value)

def get_units(text: str) -> List[Tuple[str, int]]:
    """
    テキストを音節ユニットに分解し、それぞれの原文開始位置を返す。
    ブラケット内の文字も、原文の正確なインデックス（0始まり）を保持する。
    英数字の連続（カンマやハイフン含む）は自動的に1つのブロックとしてまとめる。
    """

    regex = r'\[(.*?)\]|([ぁ-んァ-ヶ][ぁぃぅぇぉゃゅょゎァィゥェォャュョヮヵヶ]+)|([!-~]+(?:[ \t]+[!-~]+)*)|(\s+)|(.)'
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

    変更点 (v2):
      1. kanji_run_len を離散化（数値重みをやめてカテゴリ特徴量に）
      2. 文字トリグラム（前後1文字を跨ぐ3文字連鎖）を追加
      3. 文字種トリグラムを追加
    """
    result: List[FeatureDict] = []
    n = len(source_seq)

    for i, (char, _orig_idx, ctype) in enumerate(source_seq):
        # --- 前後の文字・文字種を取得 ---
        prev2_char  = source_seq[i - 2][0] if i > 1 else ""
        prev2_ctype = source_seq[i - 2][2] if i > 1 else ""
        prev_char   = source_seq[i - 1][0] if i > 0 else ""
        prev_ctype  = source_seq[i - 1][2] if i > 0 else ""
        next_char   = source_seq[i + 1][0] if i < n - 1 else ""
        next_ctype  = source_seq[i + 1][2] if i < n - 1 else ""
        next2_char  = source_seq[i + 2][0] if i < n - 2 else ""
        next2_ctype = source_seq[i + 2][2] if i < n - 2 else ""

        # 🌟 自身の文字と文字種
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

        # --- 特徴量1: 文字種遷移パターン ---
        if i > 0:
            features[f'type_transition={prev_ctype}->{ctype}'] = 1.0

        # --- 🌟 特徴量2: 文字トリグラム（前後1文字を跨ぐ3文字連鎖）---
        # ひらがなの「は→ワ/ハ」「へ→エ/ヘ」など文脈依存の読みに効く
        # 長い会話文での助詞パターンにも有効
        if i > 0 and i < n - 1:
            features[f'tri={prev_char}{char}{next_char}'] = 1.0

        # --- 🌟 特徴量3: 文字種トリグラム ---
        # 「KANJI→HIRAGANA→KANJI」のような文字種の3連鎖パターンを捉える
        # 送り仮名の検出・熟語境界の推定に有効
        if i > 0 and i < n - 1:
            features[f'type_tri={prev_ctype}-{ctype}-{next_ctype}'] = 1.0

        # --- 🌟 特徴量4: 漢字連続長（離散化）---
        # 旧実装: features['kanji_run_len'] = float(run)
        #   → 数値重みはlbfgsの勾配計算を不安定にし、線形性を仮定してしまう
        # 新実装: カテゴリ特徴量として離散化
        #   → 各連続長が独立した重みを学習できる（非線形な関係を表現可能）
        #   → 5以上はすべて同じカテゴリ（長い熟語は長さの差が意味的に小さい）
        if ctype == 'KANJI':
            run = 1
            j = i + 1
            while j < n and source_seq[j][2] == 'KANJI':
                run += 1; j += 1
            j = i - 1
            while j >= 0 and source_seq[j][2] == 'KANJI':
                run += 1; j -= 1

            # 🌟 離散化: 5以上はすべて '5+' にまとめる
            run_key = str(run) if run <= 4 else '5+'
            features[f'kanji_run_len={run_key}'] = 1.0

            if i == 0 or source_seq[i - 1][2] != 'KANJI':
                features['kanji_pos_first'] = 1.0

        result.append(features)

    return result
