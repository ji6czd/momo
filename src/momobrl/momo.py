import momobrl
import sys

def main():
    momobrl.load_braille_rules()
    print("Input Japanese sentence and hit enter key!")
    line: str = ""
    for line in sys.stdin:
        line = line.strip()
        segmented_string = momobrl.segment_braille_rule(line)
        kana_string: str = momobrl.convert_to_kana(line)
        segmented_kana: str = momobrl.segment_to_kana(line)
        # print(line)
        print(segmented_string)
        # print(kana_string)
        print(segmented_kana)
        print(momobrl.to_jp_braille(kana_string))
    print("end")


if __name__ == "__main__":
    main()
