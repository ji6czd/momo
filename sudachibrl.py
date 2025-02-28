#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sudachipy import tokenizer
from sudachipy import dictionary
import braille_rules
import pybraille
import sys

mode = tokenizer.Tokenizer.SplitMode.B
tokenizer_obj = dictionary.Dictionary().create(mode)
braille_rules.LoadBrailleRules()

def searchBrailleRule(current_morpheme, next_morpheme):
    space_flag = True
    for rule in braille_rules.rules.rule:
        if current_morpheme.part_of_speech()[0] == rule.current_pos.name:
            for n_rule in rule.next_pos:
                if next_morpheme.part_of_speech()[0] == n_rule.name:
                    space_flag = n_rule.before_space
                    break
    return space_flag

def ProcessBrailleRules(tokenizedList):
    kanaString = ""
    for m_index, m in enumerate(tokenizedList):
        # 助詞を点字のルールに変換する
        if m.part_of_speech()[0] == "助詞":
            if m.reading_form() == "ハ":
                kanaString += "ワ"
            elif m.reading_form() == "ヘ":
                kanaString += "エ"
            else:
                kanaString += m.reading_form()
        else:
            kanaString += m.reading_form()
        # この品詞のルールを、ルールリストに基づいて処理する
        if m_index < len(tokenizedList) - 1:
            if searchBrailleRule(m, tokenizedList[m_index + 1]):
                kanaString += " "
    return kanaString

if __name__ == '__main__':
    print("Input Japanese sentence and hit enter key!")
    for line in sys.stdin:
        line = line.strip()
        list = tokenizer_obj.tokenize(line, mode)
        kana_string = ProcessBrailleRules(list)
        print(line)
        print(kana_string)
        print(pybraille.to_jp_braille(kana_string))
    print('end')
