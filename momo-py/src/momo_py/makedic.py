import re
import sys
from collections import Counter, defaultdict
from typing import DefaultDict
from momo_py.features import CharType
from .utils import get_basic_char_category


def makedic(tsv_file: str, dic_file: str) -> None:
    dic: DefaultDict[str, Counter[str]] = defaultdict(Counter)

    try:
        with open(tsv_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue

                surface = parts[0]
                reading = re.sub(r"\+S", "", parts[1])
                category = get_basic_char_category(surface[0])
                # 漢字のみを対象とする
                if category == CharType.KANJI:
                    dic[surface][reading] += 1
    except FileNotFoundError:
        print(f"Error: The file '{tsv_file}' was not found.")
    except IOError as e:
        print(f"Error: An I/O error occurred: {e}")

    print(f"Total unique keys: {len(dic)}")
    # 頻度の多い順に並べる（最終的には手動編集で順序を管理する予定）
    try:
        with open(dic_file, "w", encoding="utf-8") as f:
            for kanji, counter in sorted(dic.items()):
                readings = [r for r, _ in counter.most_common()]
                f.write(kanji + "\t" + "\t".join(readings) + "\n")
    except IOError as e:
        print(f"Error: An I/O error occurred while writing to '{dic_file}': {e}")


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: makedic <input_tsv_file> <output_tsv_file>"
        )
        sys.exit(1)
    makedic(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
