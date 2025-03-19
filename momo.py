#!/usr/bin/env python3

from momo_core import convert_to_kana
import pybraille
import sys

if __name__ == '__main__':
    print("Input Japanese sentence and hit enter key!")
    line: str = ''
    for line in sys.stdin:
        line = line.strip()
        kana_string: str = convert_to_kana(line)
        print(line)
        print(kana_string)
        print(pybraille.to_jp_braille(kana_string))
    print('end')
