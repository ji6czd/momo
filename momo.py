#!/usr/bin/env python3

from momo_core import convert_to_kana
from momo_core import segment_braille_rule
import pybraille
import sys

if __name__ == '__main__':
    print("Input Japanese sentence and hit enter key!")
    line: str = ''
    for line in sys.stdin:
        line = line.strip()
        segmented_string = segment_braille_rule(line)
        kana_string: str = convert_to_kana(line)
        print(line)
        print(segmented_string)
        print(kana_string)
        print(pybraille.to_jp_braille(kana_string))
    print('end')
