import sys
from enum import Enum
from sudachipy import dictionary, tokenizer, Morpheme, MorphemeList

dic_obj = dictionary.Dictionary()
tokenizer_obj = dic_obj.create()

class CharCategory(str, Enum):
    """文字カテゴリ列挙型（str継承で文字列互換性を維持）"""
    HIRAGANA = 'HIRAGANA'
    KATAKANA = 'KATAKANA'
    KANJI = 'KANJI'
    ALPHA = 'ALPHA'
    OTHER = 'OTHER'

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

def lookup(c: str) -> MorphemeList:
    """テキストを受け取り、形態素解析の結果を返す。"""
    res = dic_obj.lookup(c)
    for m in res:
        print(m.reading_form())

def main():
    if len(sys.argv) < 2:
        print("Usage: python lookup.py <text>")
        return
    text = sys.argv[1]
    lookup(text)

if __name__ == "__main__":
    main()
