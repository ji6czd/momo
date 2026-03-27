import unicodedata
from enum import Enum
from typing import List


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

    位取り文字（十百千万億兆）の JAPANESE_NUMERIC への昇格は
    get_units() の文脈判定で行われるため、この関数では KANJI を返す。
    """
    if not c or c.isspace(): return CharType.SPACE
    if c.isdigit() or ('0' <= c <= '9'): return CharType.NUM

    cat = unicodedata.category(c)
    if cat.startswith('P') or cat.startswith('S'): return CharType.SYMBOL

    return get_basic_char_category(c)

def convert_to_katakana(c: str) -> str:
    """c（カナ1文字）がカタカナならそのまま、ひらがななら対応するカタカナを返す。
    それ以外の文字はそのまま返す。
    """

    if get_basic_char_category(c) == CharType.HIRAGANA:
        return chr(ord(c) + 0x60)
    return c

def has_vowel(char: str, vowel: str) -> bool:
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

    char_kata  = convert_to_katakana(char)
    vowel_kata = convert_to_katakana(vowel)

    char_vowel = _VOWEL_MAP.get(char_kata)
    if char_vowel is None:
        return False

    return char_vowel == vowel_kata

def split_on_unescaped_slash(s: str) -> List[str]:
    """バックスラッシュでエスケープされていない '/' で s を分割する。"""
    blocks: List[str] = []
    current: List[str] = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            current.append(s[i])
            current.append(s[i + 1])
            i += 2
        elif s[i] == '/':
            blocks.append(''.join(current))
            current = []
            i += 1
        else:
            current.append(s[i])
            i += 1
    blocks.append(''.join(current))
    return blocks
