#!/usr/bin/env python3
def main():
    import sys
    from momoPy import (
        convert_to_kana,
        load_braille_rules,
        segment_braille_rule,
        to_jp_braille,
    )

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
