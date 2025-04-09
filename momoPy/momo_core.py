# -*- coding: utf-8 -*-
from typing import Optional

from sudachipy import tokenizer
from sudachipy import dictionary
from sudachipy import config
from sudachipy import MorphemeList, Morpheme
from . import braille_rules
from . import pybraille

mode = tokenizer.Tokenizer.SplitMode.B
tokenizer_obj = dictionary.Dictionary().create(mode)
braille_rules.load_braille_rules()

def is_english_alphanumeric(word: str) -> bool:
    # isalnum()で英数字かどうかを判定しつつ、isascii()でASCII文字のみかをチェック
    return word.isalnum() and all(ch.isascii() for ch in word)

def score_part_of_speech(morpheme: Morpheme, pos) -> Optional[int]:
    for index, m_pos in enumerate(morpheme.part_of_speech()):
        if (index <= len(morpheme.part_of_speech())
        and (m_pos == pos or pos == '*')):
            return index
    return None

def search_braille_rules(morpheme: Morpheme) -> Optional[int]:
    max_score = -1
    rule_index = -1
    for index, rule in enumerate(braille_rules.rules.rule):
        score = score_part_of_speech(morpheme, rule.current_pos.name)
        if score is not None and score > max_score:
            max_score = score
            rule_index = index
    if rule_index < 0:
        return None
    else:
        return rule_index

def search_next_rule(morpheme: Morpheme, rule) -> Optional[int]:
    max_score = -1
    next_index = -1
    for index, n_rule in enumerate(rule.next_pos):
        score = score_part_of_speech(morpheme, n_rule.name)
        if score is not None and score > max_score:
            max_score = score
            next_index = index
    if next_index < 0:
        return None
    return next_index

def is_space_required(current_morpheme: Morpheme, next_morpheme: Morpheme) -> bool:
    space_flag: bool = True # 与えられた2つの品詞間にスペースが必要かどうか
    rule_index = search_braille_rules(current_morpheme)
    if rule_index is not None:
        rule = braille_rules.rules.rule[rule_index]
        rule_index = search_next_rule(next_morpheme, rule)
        if rule_index is not None:
            space_flag = rule.next_pos[rule_index].before_space
        else:
            space_flag = False
    else:
        space_flag = False
    return space_flag

def is_kana_conversion_required(morphe: Morpheme) -> bool:
    if (morphe.part_of_speech()[0] == "補助記号"
        or morphe.part_of_speech()[0] == "空白"
        or morphe.part_of_speech()[1] == "数詞"
        or morphe.reading_form() == morphe.surface()
        or is_english_alphanumeric(morphe.surface()) == True
    ):
        return False
    return True

def convert_prolonged_sound_mark(morpheme: Morpheme) -> str:
    # 動詞以外の長音記号を変換する
    # カタカナ語は変換しない
    # 先頭が「ウ」の場合は変換しない
    ret = list(morpheme.reading_form())
    if morpheme.part_of_speech()[0] != "動詞":
        for index in range(len(ret)):
            if index == 0:
                continue
            if ret[index] == "ウ":
                ret[index] = 'ー'
    return "".join(ret)

def correct_counter_suffix_reading(morphemeNum: Morpheme, morphemeCounter: Morpheme) ->str:

    # 助数詞の読みを修正する
    reading = morphemeCounter.reading_form()
    if (score_part_of_speech(morphemeCounter, "助数詞可能") is not None
        or score_part_of_speech(morphemeCounter, "助数詞") is not None):
        reading = morphemeCounter.reading_form()
        number = morphemeNum.surface()
        # 助数詞を直前の数字に従って修正する
        if (morphemeCounter.surface() in "匹"):
            # 直前の数字が「1,6,8」の場合は、読みをパ行に変更する
            # 直前の数字が「3」の場合は、読みをバ行に変更する
            if number[len(number)-1] in "168":
                reading = chr(ord(reading[0])+2) + reading[1:]
            elif number[len(number)-1] == "3":
                reading = chr(ord(reading[0])+1) + reading[1:]
        elif (morphemeCounter.surface() == '本'):
            # 直前の数字が"24579"の場合は、読みをハ行に変更する
            # 直前の数字が"168"の場合は、読みをパ行に変更する
            # 直前の数字が"3"の場合は、読みをバ行に変更する
            if number[len(number)-1] in "24579":
                reading = chr(ord(reading[0])-2) + reading[1:]
            elif number[len(number)-1] in "3":
                reading = chr(ord(reading[0])-1) + reading[1:]
        elif (morphemeCounter.surface() == '版'):
            # 直前の数字が"24579"の場合は、読みをハ行に変更する
            # 直前の数字が"168"の場合は、読みをパ行に変更する
            # 直前の数字が"3"の場合は、読みをバ行に変更する
            if number[len(number)-1] in "24579":
                reading = chr(ord(reading[0])-1) + reading[1:]
            elif number[len(number)-1] in "168":
                reading = chr(ord(reading[0])+1) + reading[1:]
    else:
        # 助数詞でない場合は空白を返す
        reading = ''
    return reading

def segment_braille_rule(src_string: str):
    segmented_string: str = ""
    tokenized_list = tokenizer_obj.tokenize(src_string)
    for m_index, m in enumerate(tokenized_list):
        segmented_string += m.surface()
        if is_space_required(m, tokenized_list[m_index + 1]):
            segmented_string += " "
    return segmented_string

def convert_to_kana(src_string: str):
    kanaString: str = ""
    tokenized_list = tokenizer_obj.tokenize(src_string)
    for m_index, m in enumerate(tokenized_list):
        if m.part_of_speech()[0] == "助詞":
            # 助詞は、点字のルールで置換する
            if m.reading_form() == "ハ":
                kanaString += "ワ"
            elif m.reading_form() == "ヘ":
                kanaString += "エ"
            else:
                kanaString += m.reading_form()
        elif is_kana_conversion_required(m) == False:
            # その品詞は、そのまま表記を追加する
            kanaString += m.surface()
        else:
            # それ以外の品詞は、助数詞処理・長音処理を行って読みを追加する
            counter = correct_counter_suffix_reading(tokenized_list[m_index - 1], m)
            if counter != '':
                # 助数詞がある場合は、読みを追加する
                kanaString += counter
            else:
                # その他
                kanaString += convert_prolonged_sound_mark(m)
        # この品詞のルールを、ルールリストに基づいて処理する
        if m_index < len(tokenized_list) - 1:
            if is_space_required(m, tokenized_list[m_index + 1]):
                kanaString += " "
    return kanaString

def convert_to_braille(src: str):
    # 文字列を点字に変換する
    kana_str = convert_to_kana(src)
    return pybraille.to_jp_braille(kana_str)

if __name__ == '__main__':
    src = "原文を分割する機能も追加してみました。"
    print(src)
    print(segment_braille_rule(src))
    kana_str = convert_to_kana(src)
    print(kana_str)
    print(pybraille.to_jp_braille(kana_str))
