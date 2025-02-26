#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sudachipy import tokenizer
from sudachipy import dictionary
from pybraille import *
import braillel_rules_pb2
import sys

if __name__ == '__main__':
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.B

    print("Input Japanese sentence and hit enter key!")
    for line in sys.stdin:
        line = line.strip()
        list = tokenizer_obj.tokenize(line, mode)
        kana_string = ' '.join([m.reading_form() for m in list])
        print(line)
        print(kana_string)
        print(to_jp_braille(kana_string))
    print('end')
