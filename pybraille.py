# -*- coding: utf-8 -*-
#
# This is a Python script to convert Japanese katakana text to Braille.

import unicodedata

en_braille_table = {
    '0': '⠚', '1': '⠁', '2': '⠃', '3': '⠉', '4': '⠙', '5': '⠑', '6': '⠋', '7': '⠛', '8': '⠓', '9': '⠊',
    ' ': '⠀', 'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚',
    'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞',
    'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵',}

numerical_sign = '⠼'

jp_braille_table = {
    ' ': '⠀', 'ア': '⠁', 'イ': '⠃', 'ウ': '⠉', 'エ': '⠋', 'オ': '⠊',
    'カ': '⠡', 'キ': '⠣', 'ク': '⠩', 'ケ': '⠫', 'コ': '⠪',
    'サ': '⠱', 'シ': '⠳', 'ス': '⠹', 'セ': '⠻', 'ソ': '⠺',
    'タ': '⠕', 'チ': '⠗', 'ツ': '⠝', 'テ': '⠟', 'ト': '⠞', 'ッ': '⠂',
    'ナ': '⠅', 'ニ': '⠇', 'ヌ': '⠍', 'ネ': '⠏', 'ノ': '⠎',
    'ハ': '⠥', 'ヒ': '⠧', 'フ': '⠭', 'ヘ': '⠯', 'ホ': '⠮',
    'マ': '⠵', 'ミ': '⠷', 'ム': '⠽', 'メ': '⠿', 'モ': '⠾',
    'ヤ': '⠌', 'ユ': '⠬', 'ヨ': '⠜',
    'ラ': '⠑', 'リ': '⠓', 'ル': '⠙', 'レ': '⠛', 'ロ': '⠚',
    'ワ': '⠄', 'ヲ': '⠔', 'ン': '⠴',
    'ガ': '⠐⠡', 'ギ': '⠐⠣', 'グ': '⠐⠩', 'ゲ': '⠐⠫', 'ゴ': '⠐⠪',
    'ザ': '⠐⠱', 'ジ': '⠐⠳', 'ズ': '⠐⠹', 'ゼ': '⠐⠻', 'ゾ': '⠐⠺',
    'ダ': '⠐⠕', 'ヂ': '⠐⠗', 'ヅ': '⠐⠝', 'デ': '⠐⠟', 'ド': '⠐⠞',
    'バ': '⠐⠥', 'ビ': '⠐⠧', 'ブ': '⠐⠭', 'ベ': '⠐⠯', 'ボ': '⠐⠮',
    'パ': '⠠⠥', 'ピ': '⠠⠧', 'プ': '⠠⠭', 'ペ': '⠠⠯', 'ポ': '⠠⠮',
    '。': '⠲⠀⠀', '、': '⠰⠀', '・': '⠐⠀', '！': '⠖⠀⠀', '？': '⠢⠀⠀',
    '「': '⠤', '」': '⠤', 'ー': '⠒',
    '０': '⠚', '１': '⠁', '２': '⠃', '３': '⠉', '４': '⠙', '５': '⠑', '６': '⠋', '７': '⠛', '８': '⠓', '９': '⠊',
    }

def to_braille(text):
    return ''.join([en_braille_table.get(c, ' ') for c in text.lower()])

def to_jp_braille(text):
    normalized_text = unicodedata.normalize('NFKC', text)
    braille_str = ""
    inDigitFlag = False
    inCapitalWordFlag = False
    for c in normalized_text:
        if c.isdigit():
            if not inDigitFlag:
                braille_str += numerical_sign
                inDigitFlag = True
            braille_str += en_braille_table.get(c, '⠀')
            continue
        else:
            inDigitFlag = False
        braille_str += jp_braille_table.get(c, '⠀')
    return braille_str

if __name__ == '__main__':
    print(to_braille('Hello World!.'))
    print(to_jp_braille("コンニチワ、セカイ！"))
    print(to_jp_braille("12345"))
    print(to_jp_braille("３ネン ２クミ"))
