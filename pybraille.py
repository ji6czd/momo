# -*- coding: utf-8 -*-
#
# This is a Python script to convert Japanese katakana text to Braille.

import unicodedata
import sys

en_braille_table = {
    '0': '⠚', '1': '⠁', '2': '⠃', '3': '⠉', '4': '⠙', '5': '⠑', '6': '⠋', '7': '⠛', '8': '⠓', '9': '⠊',
    '!': '⠖', '?': '⠢', '.': '⠲', ',': '⠄', ':': '⠒', ';': '⠆', '-': '⠤', '(': '⠦', ')': '⠴', '/': '⠌',
    ' ': '⠀', 'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚',
    'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞',
    'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵',}

numerical_sign = '⠼'
capital_sign = '⠠'
foreign_word_sign = '⠰'
space_sign = '⠀'

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
    'イェ': '⠈⠋','ウァ': '⠐⠖','ウィ': '⠢⠃','ウェ': '⠢⠋','ウォ': '⠢⠊',
    'キェ': '⠈⠫','キャ': '⠈⠡','キュ': '⠈⠩','キョ': '⠈⠪','クァ': '⠢⠡','クィ': '⠢⠣','クェ': '⠢⠫','クォ': '⠢⠪',
    'シェ': '⠈⠻','シャ': '⠈⠱','シュ': '⠈⠹','ショ': '⠈⠺','スィ': '⠈⠳',
    'チェ': '⠈⠟','チャ': '⠈⠕','チュ': '⠈⠝','チョ': '⠈⠞','ツァ': '⠢⠕','ツィ': '⠢⠗','ツェ': '⠢⠟','ツォ': '⠢⠞','ティ': '⠈⠗','テュ': '⠨⠝','トゥ': '⠢⠝',
    'ニェ': '⠈⠏','ニャ': '⠈⠅','ニュ': '⠈⠍','ニョ': '⠈⠎',
    'ヒャ': '⠈⠥','ヒュ': '⠈⠭','ヒョ': '⠈⠮','ファ': '⠢⠥','フィ': '⠢⠧','フェ': '⠢⠯','フォ': '⠢⠮','フュ': '⠨⠬','フョ': '⠨⠜',
    'ミャ': '⠈⠵','ミュ': '⠈⠽','ミョ': '⠈⠾',
    'リャ': '⠈⠑','リュ': '⠈⠙','リョ': '⠈⠚',
    'ヴァ': '⠲⠥','ヴィ': '⠲⠧','ヴェ': '⠲⠯','ヴォ': '⠲⠮','ヴュ': '⠸⠬','ヴョ': '⠸⠜',
    'ギャ': '⠘⠡','ギュ': '⠘⠩','ギョ': '⠘⠪','グァ': '⠲⠡','グィ': '⠲⠣','グェ': '⠲⠫','グォ': '⠲⠪',
    'ジェ': '⠘⠻','ジャ': '⠘⠱','ジュ': '⠘⠹','ジョ': '⠘⠺',
    'ヂェ': '⠘⠟','ヂャ': '⠘⠕','ヂュ': '⠘⠝','ヂョ': '⠘⠞','ディ': '⠘⠗','デュ': '⠸⠝','ドゥ': '⠲⠝',
    'ビャ': '⠘⠥','ビュ': '⠘⠭','ビョ': '⠘⠮','ピャ': '⠨⠥','ピュ': '⠨⠭','ピョ': '⠨⠮',
    '。': '⠲⠀⠀', '、': '⠰⠀', '・': '⠐⠀', '！': '⠖⠀⠀', '？': '⠢⠀⠀',
    '「': '⠤', '」': '⠤', 'ー': '⠒',
    '０': '⠚', '１': '⠁', '２': '⠃', '３': '⠉', '４': '⠙', '５': '⠑', '６': '⠋', '７': '⠛', '８': '⠓', '９': '⠊',
}

small_kana = 'ァィゥェォャュョ'

def to_braille(text: str) -> str:
    return ''.join([en_braille_table.get(c, ' ') for c in text.lower()])

def to_jp_braille(text: str) -> str:
    normalized_text = unicodedata.normalize('NFKC', text)
    braille_str = ""
    inDigitFlag = False # 数字中にTrue
    inCapitalWordFlag = False # 英文字列中 True
    inDoubleCapitalFlag = False # ２重大文字状態
    inForeignWordFlag = False # 単語が外国語であることを表すフラグ
    skip = False # 特殊音で次の文字をスキップ

    for index, c in enumerate(normalized_text):
        if skip:
            # 特殊音などで直前のループで処理済みの文字をスキップ
            skip = False
            continue
        # １文字ずつ処理
        if ord(c) < 0x7f:
            # ASCII文字
            if c.isalpha():
                inDigitFlag = False
                if inForeignWordFlag == False:
                    braille_str += foreign_word_sign
                    inForeignWordFlag = True
                if c.isupper():
                    if inCapitalWordFlag == False:
                        braille_str += capital_sign
                    inCapitalWordFlag = True
                    if (index+1 < len(normalized_text)
                        and normalized_text[index+1].isupper()
                        and inDoubleCapitalFlag == False):
                        braille_str += capital_sign
                        inDoubleCapitalFlag = True
                elif inCapitalWordFlag:
                    inCapitalWordFlag = False
                    inDoubleCapitalFlag == False
            elif c.isnumeric():
                inForeignWordFlag = False
                if inDigitFlag == False:
                    braille_str += numerical_sign
                inDigitFlag = True
            elif (c == '.'
            or c == ','):
                # 数字の小数点、カンマである可能性
                # 数字フラグはリセットしない
                inForeignWordFlag == False
            else:
                # その他の文字でも外国語フラグをリセット
                inForeignWordFlag = False
                inDigitFlag = False
            braille_str += en_braille_table.get(c.lower(), ' ')
        else:
            # 日本語文字
            if inForeignWordFlag:
                braille_str += space_sign
            inForeignWordFlag = False
            inCapitalWordFlag = False
            inDigitFlag = False
            if (index+1 < len(normalized_text)
            and small_kana.find(normalized_text[index+1]) != -1):
                # 特殊文字処理
                c = normalized_text[index] + normalized_text[index + 1]
                skip = True # 2文字処理したので次はスキップ
            braille_str += jp_braille_table.get(c, '⠀')
    return braille_str

if __name__ == '__main__':
    print(to_jp_braille('Hello World!.'))
    print(to_jp_braille("コンニチワ、セカイ！"))
    print(to_jp_braille("12345"))
    print(to_jp_braille("３ネン ２クミ"))
    print(to_jp_braille("ワタシワ ベンキョーガ キライナノデチュ。ヴァイオリンノ レンシューワ モット キライニャニェニョーー。"))
    print(to_jp_braille("NHK ラジオ ダイ１ノ シューハスーワ ５９４kHz デス。"))
    print(to_jp_braille("33min45sec"))
    print(to_jp_braille("エンシューリツワ、3.141592 グライデショー。"))