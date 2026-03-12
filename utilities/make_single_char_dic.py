import re
import sys
import json
from collections import defaultdict
from typing import DefaultDict, Set
from momobrl.features import get_basic_char_category, CharCategory

def main() -> None:
    dic: DefaultDict[str, Set[str]] = defaultdict(set)

    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) < 4:
            continue

        surface = parts[0]
        reading = re.sub(r'\+S', '', parts[1])
        category = get_basic_char_category(surface)
        # 漢字またはひらがな・カタカナを追加
        if category == CharCategory.KANJI:
            dic[surface].add(reading)  # 同一キー内の重複値を排除

    # 出力用に Set を list に変換
    output = {k: sorted(list(v)) for k, v in sorted(dic.items())}
    
    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()