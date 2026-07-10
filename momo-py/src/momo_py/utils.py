import unicodedata
from enum import Enum
from typing import Dict, List, Tuple


class CharType(str, Enum):
    """文字種列挙型"""

    SPACE = "SPACE"
    ALPHA = "ALPHA"
    NUMERIC = "NUM"
    SYMBOL = "SYMBOL"
    SYMBOL_CLOSE = "SYMBOL_CLOSE"  # 閉じ括弧類: 」』）】〕｝〉》 など
    SYMBOL_OPEN = "SYMBOL_OPEN"  # 開き括弧類: 「『（【〔｛〈《 など
    SYMBOL_STOP = "SYMBOL_STOP"  # 文末記号類: 。！？.!? など
    SYMBOL_PAUSE = "SYMBOL_PAUSE"  # 読点・中点類: 、・, など
    HIRAGANA = "HIRAGANA"
    KATAKANA = "KATAKANA"
    KANJI = "KANJI"
    JAPANESE_NUMERIC = "JAPANESE_NUMERIC"
    SKIP = "SKIP"  # NBSP・ZWJ など: 仮名に現れず原文位置のみ保持
    OTHER = "OTHER"  # 未定義文字: バイパスして momors-braille 側で処理

    def __str__(self) -> str:
        return self.value


# 常に JAPANESE_NUMERIC となる漢数字
JAPANESE_NUMERIC_CHARS = frozenset("〇一二三四五六七八九")

# ゼロ幅文字・フォーマット制御文字など、仮名・点字出力には現れるべきでないスキップ文字。
# 可視スペース系（NBSP 等）は isspace() == True のため SPACE に分類される（ここに含めない）。
_SKIP_CHARS = frozenset({
    "­",  # SOFT HYPHEN
    "​",  # ZERO WIDTH SPACE
    "‌",  # ZERO WIDTH NON-JOINER
    "‍",  # ZERO WIDTH JOINER
    "⁠",  # WORD JOINER
    "⁡",  # FUNCTION APPLICATION
    "⁢",  # INVISIBLE TIMES
    "⁣",  # INVISIBLE SEPARATOR
    "⁤",  # INVISIBLE PLUS
    "﻿",  # ZERO WIDTH NO-BREAK SPACE (BOM)
})

# 隣接文脈次第で JAPANESE_NUMERIC に昇格する位取り文字
KURAI_CHARS = frozenset("十百千万億兆")

# 記号サブカテゴリ
_CLOSE_SYMBOLS = frozenset("」』)）】〕}｝〉》\"'")
_OPEN_SYMBOLS = frozenset("「『(（【〔{｛〈《")
_STOP_SYMBOLS = frozenset("。！？.!?")
_PAUSE_SYMBOLS = frozenset("、・,")


# 漢字繰り返し符号（CJK Symbols and Punctuation ブロック内のため範囲判定では拾えない）
_KANJI_ITERATION_MARKS = frozenset("々〻")


def get_basic_char_category(c: str) -> CharType:
    """
    1文字を受け取り、基本カテゴリ（かな/漢字/英字/その他）を返す。
    SPACE / NUM / SYMBOL 系の判定は行わない（get_char_type() が担当）。
    """
    cp = ord(c)
    if 0x3040 <= cp <= 0x309F:
        return CharType.HIRAGANA
    if 0x30A0 <= cp <= 0x30FF:
        return CharType.KATAKANA
    if c in JAPANESE_NUMERIC_CHARS:
        return CharType.JAPANESE_NUMERIC
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
        return CharType.KANJI
    # 拡張B〜F・I (サロゲートペア必須。𠮟 U+20B9F 等の人名・地名用漢字を含む拡張Bが実用上重要)
    if 0x20000 <= cp <= 0x2EE5F:
        return CharType.KANJI
    # 拡張G・H (サロゲートペア必須)
    if 0x30000 <= cp <= 0x323AF:
        return CharType.KANJI
    if c in _KANJI_ITERATION_MARKS:
        return CharType.KANJI
    if (
        0x0041 <= cp <= 0x005A
        or 0x0061 <= cp <= 0x007A
        or 0xFF21 <= cp <= 0xFF3A
        or 0xFF41 <= cp <= 0xFF5A
    ):
        return CharType.ALPHA
    return CharType.OTHER


def get_char_type(c: str) -> CharType:
    """
    文字種を判定する。

    記号は内容に応じてサブカテゴリに分類する:
        SYMBOL_CLOSE  : 閉じ括弧類
        SYMBOL_OPEN   : 開き括弧類
        SYMBOL_STOP   : 文末記号類
        SYMBOL_PAUSE  : 読点・中点類
        SYMBOL        : その他の記号

    位取り文字（十百千万億兆）の JAPANESE_NUMERIC への昇格は
    get_units() の文脈判定で行われるため、この関数では KANJI を返す。
    """
    if c in _SKIP_CHARS:
        return CharType.SKIP
    if not c or c.isspace():
        return CharType.SPACE
    if c.isdigit() or ("0" <= c <= "9"):
        return CharType.NUMERIC

    cat = unicodedata.category(c)
    if cat.startswith("P") or cat.startswith("S"):
        if c in _CLOSE_SYMBOLS:
            return CharType.SYMBOL_CLOSE
        if c in _OPEN_SYMBOLS:
            return CharType.SYMBOL_OPEN
        if c in _STOP_SYMBOLS:
            return CharType.SYMBOL_STOP
        if c in _PAUSE_SYMBOLS:
            return CharType.SYMBOL_PAUSE
        return CharType.SYMBOL

    return get_basic_char_category(c)


def convert_to_katakana(c: str) -> str:
    """c（カナ1文字）がカタカナならそのまま、ひらがななら対応するカタカナを返す。
    それ以外の文字はそのまま返す。
    """
    if get_basic_char_category(c) == CharType.HIRAGANA:
        return chr(ord(c) + 0x60)
    return c


# ---------------------------------------------------------------
# 入力正規化
# ---------------------------------------------------------------
# 選択的NFKC正規化の対象コードポイント範囲。
# ブランケットNFKCは全角英数字の半角化など学習データ方針と衝突する変換を
# 含むため、対象カテゴリを明示的に列挙する。カテゴリを追加するときは
# ここに範囲を足し、Rust側テーブルを再生成する
# (momors/tools/gen_normalize_table.py)。

# 1→1変換のみのカテゴリ（原文の文字数・位置が変わらない）。
# 通常のCJK統合漢字と見た目が同じ・酷似する別符号点を畳み込む。
_COMPAT_IDEOGRAPH_RANGES: List[Tuple[int, int]] = [
    (0x2E80, 0x2EFF),  # CJK部首補助
    (0x2F00, 0x2FD5),  # 康煕部首
    (0xF900, 0xFAFF),  # CJK互換漢字
    (0x2F800, 0x2FA1F),  # CJK互換漢字補助
]

# 1→1変換のみのカテゴリ: 全角英数字→半角。
# 幅は読み・境界・点字のいずれにも情報を持たず、区別すると学習例が分断される
# だけのため半角へ畳む。全角記号（（！？ 等）は点字変換側で半角と扱いが
# 異なり得るため対象外（区別を保持する）。
_FULLWIDTH_ALNUM_RANGES: List[Tuple[int, int]] = [
    (0xFF10, 0xFF19),  # ０-９
    (0xFF21, 0xFF3A),  # Ａ-Ｚ
    (0xFF41, 0xFF5A),  # ａ-ｚ
]

# 1→1変換の全カテゴリ。Rust側テーブル（momors/tools/gen_normalize_table.py）は
# この変数を読んで生成される。
_ONE_TO_ONE_RANGES: List[Tuple[int, int]] = sorted(
    _COMPAT_IDEOGRAPH_RANGES + _FULLWIDTH_ALNUM_RANGES
)

# 1→N展開を含むカテゴリ（原文位置は normalize_input() のマップで追跡する）
_EXPANSION_RANGES: Dict[str, List[Tuple[int, int]]] = {
    # 欧文リガチャ: ﬃ→ffi 等。PDF抽出テキストやDTP由来のテキストに混入する
    "latin_ligatures": [(0xFB00, 0xFB06)],
}

_ALL_NORMALIZE_RANGES: List[Tuple[int, int]] = sorted(
    _ONE_TO_ONE_RANGES
    + [r for ranges in _EXPANSION_RANGES.values() for r in ranges]
)


def _in_ranges(cp: int, ranges: List[Tuple[int, int]]) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def normalize_one_to_one(text: str) -> str:
    """1→1変換のみの選択的NFKC正規化（互換漢字の畳み込み + 全角英数字の半角化）。

    文字数・原文位置が変わらないことが保証されるため、辞書表層形や学習データ
    生成など、インデックスマップを持てない箇所の正規化に使う。推論入力の
    正規化は normalize_input() を使うこと（1→N展開を含む）。
    """
    return "".join(
        unicodedata.normalize("NFKC", c)
        if _in_ranges(ord(c), _ONE_TO_ONE_RANGES)
        else c
        for c in text
    )


# 後方互換のためエイリアスを残す（旧名: 互換漢字のみだった頃の名称）
normalize_compat_ideographs = normalize_one_to_one


def normalize_input(text: str) -> Tuple[str, List[int]]:
    """推論入力テキストを選択的にNFKC正規化する。

    互換漢字系（1→1）に加え、_EXPANSION_RANGES のカテゴリ（欧文リガチャ等の
    1→N展開）も正規化する。

    Returns:
        (正規化後テキスト, 正規化後の各文字に対応する原文の文字インデックス)
        1→N展開された文字は、展開後の全文字が同じ原文インデックスを指す。
    """
    out_chars: List[str] = []
    out_map: List[int] = []
    for i, c in enumerate(text):
        if _in_ranges(ord(c), _ALL_NORMALIZE_RANGES):
            for nc in unicodedata.normalize("NFKC", c):
                out_chars.append(nc)
                out_map.append(i)
        else:
            out_chars.append(c)
            out_map.append(i)
    return "".join(out_chars), out_map


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
    char_type = get_basic_char_category(char)
    vowel_type = get_basic_char_category(vowel)

    if char_type not in (CharType.HIRAGANA, CharType.KATAKANA):
        raise ValueError(f"char はひらがなまたはカタカナでなければなりません: {char!r}")
    if vowel_type not in (CharType.HIRAGANA, CharType.KATAKANA):
        raise ValueError(
            f"vowel はひらがなまたはカタカナでなければなりません: {vowel!r}"
        )
    if char_type != vowel_type:
        raise ValueError(
            f"char と vowel は同じ文字種でなければなりません: "
            f"char={char!r}({char_type}), vowel={vowel!r}({vowel_type})"
        )

    _VOWEL_MAP = {
        "ア": "ア",
        "イ": "イ",
        "ウ": "ウ",
        "エ": "エ",
        "オ": "オ",
        "ァ": "ア",
        "ィ": "イ",
        "ゥ": "ウ",
        "ェ": "エ",
        "ォ": "オ",
        "カ": "ア",
        "キ": "イ",
        "ク": "ウ",
        "ケ": "エ",
        "コ": "オ",
        "ガ": "ア",
        "ギ": "イ",
        "グ": "ウ",
        "ゲ": "エ",
        "ゴ": "オ",
        "サ": "ア",
        "シ": "イ",
        "ス": "ウ",
        "セ": "エ",
        "ソ": "オ",
        "ザ": "ア",
        "ジ": "イ",
        "ズ": "ウ",
        "ゼ": "エ",
        "ゾ": "オ",
        "タ": "ア",
        "チ": "イ",
        "ツ": "ウ",
        "テ": "エ",
        "ト": "オ",
        "ダ": "ア",
        "ヂ": "イ",
        "ヅ": "ウ",
        "デ": "エ",
        "ド": "オ",
        "ナ": "ア",
        "ニ": "イ",
        "ヌ": "ウ",
        "ネ": "エ",
        "ノ": "オ",
        "ハ": "ア",
        "ヒ": "イ",
        "フ": "ウ",
        "ヘ": "エ",
        "ホ": "オ",
        "バ": "ア",
        "ビ": "イ",
        "ブ": "ウ",
        "ベ": "エ",
        "ボ": "オ",
        "パ": "ア",
        "ピ": "イ",
        "プ": "ウ",
        "ペ": "エ",
        "ポ": "オ",
        "マ": "ア",
        "ミ": "イ",
        "ム": "ウ",
        "メ": "エ",
        "モ": "オ",
        "ヤ": "ア",
        "ユ": "ウ",
        "ヨ": "オ",
        "ャ": "ア",
        "ュ": "ウ",
        "ョ": "オ",
        "ラ": "ア",
        "リ": "イ",
        "ル": "ウ",
        "レ": "エ",
        "ロ": "オ",
        "ワ": "ア",
        "ヲ": "オ",
        "ヴ": "ウ",
    }

    char_kata = convert_to_katakana(char)
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
        if s[i] == "\\" and i + 1 < len(s):
            current.append(s[i])
            current.append(s[i + 1])
            i += 2
        elif s[i] == "/":
            blocks.append("".join(current))
            current = []
            i += 1
        else:
            current.append(s[i])
            i += 1
    blocks.append("".join(current))
    return blocks
