import re
import sys
import json
import argparse
from collections import defaultdict
from typing import DefaultDict, Set
from momobrl.features import get_basic_char_category, CharType

def makedic(tsv_file: str, dic_file: str) -> None:
    dic: DefaultDict[str, Set[str]] = defaultdict(set)

    try:
        with open(tsv_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) < 4:
                    continue

                surface = parts[0]
                reading = re.sub(r'\+S', '', parts[1])
                category = get_basic_char_category(surface)
                # 漢字のみを対象とする
                if category == CharType.KANJI:
                    dic[surface].add(reading)  # 同一キー内の重複値を排除
    except FileNotFoundError:
        print(f"Error: The file '{tsv_file}' was not found.")
    except IOError as e:
         print(f"Error: An I/O error occurred: {e}")

    print(f"Total unique keys: {len(dic)}")
    # 出力用に Set を list に変換
    output = {k: sorted(list(v)) for k, v in sorted(dic.items())}
    try:
        with open(dic_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Error: An I/O error occurred while writing to '{dic_file}': {e}")

if __name__ == "__main__":
    makedic(sys.argv[1], sys.argv[2])
    