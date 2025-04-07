#!/usr/bin/env python3

def momo(source: str) -> str:
    """
    Convert a Japanese sentence to Braille.
    :param source: Japanese sentence
    :return: Braille representation of the sentence
    """
    from .momo_core import convert_to_kana
    from .momo_core import segment_braille_rule
    from . import pybraille

    kana_string: str = convert_to_kana(source)
    return pybraille.to_jp_braille(kana_string)

if __name__ == '__main__':
    import sys
    from .momo_core import convert_to_kana
    from .momo_core import segment_braille_rule
    from . import pybraille
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
