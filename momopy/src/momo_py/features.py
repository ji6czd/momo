import unicodedata
import re
from enum import Enum
from typing import List, Dict, Tuple

# --- [共通定義・型定義] ---
FeatureDict = Dict[str, float]
# ソース文字系列 = [(char, orig_idx, ctype), ...]
SourceEntry = Tuple[str, int, str]

class CharType(str, Enum):
    """文字種列挙型"""
    SPACE            = 'SPACE'
    NUM              = 'NUM'
    SYMBOL           = 'SYMBOL'
    HIRAGANA         = 'HIRAGANA'
    KATAKANA         = 'KATAKANA'
    KANJI            = 'KANJI'
    JAPANESE_NUMERIC = 'JAPANESE_NUMERIC'
    ALPHA            = 'ALPHA'
    OTHER            = 'OTHER'

# --- [共通定数] ---
MORA_SPLIT     = "+S"
LABEL_CONTINUE = "---"
LABEL_SKIP     = "_"

# 常に JAPANESE_NUMERIC となる漢数字
_JAPANESE_NUMERIC_CHARS = frozenset("〇一二三四五六七八九")

# 隣接文脈次第で JAPANESE_NUMERIC に昇格する位取り文字
_KURAI_CHARS = frozenset("十百千万億兆")


def get_basic_char_category(c: str) -> CharType:
    """
    1文字を受け取り、基本カテゴリ（かな/漢字/英字/その他）を返す。
    SPACE / NUM / SYMBOL の判定は行わない（get_char_type() が担当）。
    """
    cp = ord(c)
    if 0x3040 <= cp <= 0x309F: return CharType.HIRAGANA
    if 0x30A0 <= cp <= 0x30FF: return CharType.KATAKANA
    if c in _JAPANESE_NUMERIC_CHARS: return CharType.JAPANESE_NUMERIC
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF: return CharType.KANJI
    if (0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A
            or 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A):
        return CharType.ALPHA
    return CharType.OTHER

def get_char_type(c: str) -> CharType:
    """
    文字種を判定する。
    """
    if not c or c.isspace(): return CharType.SPACE
    if c.isdigit() or ('0' <= c <= '9'): return CharType.NUM

    cat = unicodedata.category(c)
    if cat.startswith('P') or cat.startswith('S'): return CharType.SYMBOL

    return get_basic_char_category(c)


def _has_vowel(char: str, vowel: str) -> bool:
    """
    char（カナ1文字）が vowel（母音）を含むかどうかを返す。
    ひらがな・カタカナ両対応。引数は同じ種類の文字であること。

    Args:
        char:  判定対象のカナ1文字（ひらがなまたはカタカナ）
        vowel: 母音1文字（charと同じ種類）

    Returns:
        char が vowel の母音を持つなら True

    Raises:
        ValueError: 引数の文字種が異なる場合、またはカナ以外の場合
    """

    char_type  = get_basic_char_category(char)
    vowel_type = get_basic_char_category(vowel)

    if char_type not in (CharType.HIRAGANA, CharType.KATAKANA):
        raise ValueError(f"char はひらがなまたはカタカナでなければなりません: {char!r}")
    if vowel_type not in (CharType.HIRAGANA, CharType.KATAKANA):
        raise ValueError(f"vowel はひらがなまたはカタカナでなければなりません: {vowel!r}")
    if char_type != vowel_type:
        raise ValueError(
            f"char と vowel は同じ文字種でなければなりません: "
            f"char={char!r}({char_type}), vowel={vowel!r}({vowel_type})"
        )

    # 比較のためカタカナに統一
    def to_kata(c: str) -> str:
        if get_basic_char_category(c) == CharType.HIRAGANA:
            return chr(ord(c) + 0x60)
        return c

    # カタカナの母音マップ: 各文字→その母音
    # 小書き文字（ァィゥェォ）も含む
    _VOWEL_MAP = {
        'ア': 'ア', 'イ': 'イ', 'ウ': 'ウ', 'エ': 'エ', 'オ': 'オ',
         'ァ': 'ア', 'ィ': 'イ', 'ゥ': 'ウ', 'ェ': 'エ', 'ォ': 'オ',
        'カ': 'ア', 'キ': 'イ', 'ク': 'ウ', 'ケ': 'エ', 'コ': 'オ',
        'ガ': 'ア', 'ギ': 'イ', 'グ': 'ウ', 'ゲ': 'エ', 'ゴ': 'オ',
        'サ': 'ア', 'シ': 'イ', 'ス': 'ウ', 'セ': 'エ', 'ソ': 'オ',
        'ザ': 'ア', 'ジ': 'イ', 'ズ': 'ウ', 'ゼ': 'エ', 'ゾ': 'オ',
        'タ': 'ア', 'チ': 'イ', 'ツ': 'ウ', 'テ': 'エ', 'ト': 'オ',
        'ダ': 'ア', 'ヂ': 'イ', 'ヅ': 'ウ', 'デ': 'エ', 'ド': 'オ',
        'ナ': 'ア', 'ニ': 'イ', 'ヌ': 'ウ', 'ネ': 'エ', 'ノ': 'オ',
        'ハ': 'ア', 'ヒ': 'イ', 'フ': 'ウ', 'ヘ': 'エ', 'ホ': 'オ',
        'バ': 'ア', 'ビ': 'イ', 'ブ': 'ウ', 'ベ': 'エ', 'ボ': 'オ',
        'パ': 'ア', 'ピ': 'イ', 'プ': 'ウ', 'ペ': 'エ', 'ポ': 'オ',
        'マ': 'ア', 'ミ': 'イ', 'ム': 'ウ', 'メ': 'エ', 'モ': 'オ',
        'ヤ': 'ア', 'ユ': 'ウ', 'ヨ': 'オ',
        'ャ': 'ア', 'ュ': 'ウ', 'ョ': 'オ',
        'ラ': 'ア', 'リ': 'イ', 'ル': 'ウ', 'レ': 'エ', 'ロ': 'オ',
        'ワ': 'ア', 'ヲ': 'オ',
        'ヴ': 'ウ',
    }

    char_kata  = to_kata(char)
    vowel_kata = to_kata(vowel)

    char_vowel = _VOWEL_MAP.get(char_kata)
    if char_vowel is None:
        # ン・ッ など母音を持たない文字
        return False

    return char_vowel == vowel_kata

def get_units(text: str) -> List[Tuple[str, int, str]]:
    """
    テキストを音節ユニットに分解し、(文字列, 原文開始位置, 文字種) のリストを返す。

    英数字連続・ひらがな+小書きなどは従来通り複数文字ブロックとしてまとめる。
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
