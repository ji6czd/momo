from .momobrl_core import convert_to_kana, segment_braille_rule, load_braille_rules
from .pybraille import to_jp_braille, to_braille


def main():
    import sys

    load_braille_rules()
    print("Input Japanese sentence and hit enter key!")
    line: str = ""
    for line in sys.stdin:
        line = line.strip()
        # segmented_string = segment_braille_rule(line)
        kana_string: str = convert_to_kana(line)
        print(line)
        # print(segmented_string)
        print(kana_string)
        print(to_jp_braille(kana_string))
    print("end")


if __name__ == "__main__":
    main()
