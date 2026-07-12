import sys

def extract_words_from_tsv(tsv_file: str) -> list[tuple[str, str]]:
    words = []
    raw_buf = ""
    read_buf = ""

    try:
        with open(tsv_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 空行やヘッダー（#）はスキップ
                if not line or line.startswith('#'):
                    continue

                parts = line.split('\t')
                if len(parts) < 4:
                    continue

                char = parts[0]
                read = parts[1]
                tag = parts[3]

                # '---'（継続文字のプレースホルダ読み）は無視、'+S'（分かち書き記号）は表示用に削除
                clean_read = "" if read == "---" else read.replace("+S", "")

                # BIOESタグに基づく単語の組み立て
                if tag == 'B':
                    raw_buf = char
                    read_buf = clean_read
                elif tag == 'I':
                    raw_buf += char
                    read_buf += clean_read
                elif tag == 'E':
                    raw_buf += char
                    read_buf += clean_read
                    words.append((raw_buf, read_buf))
                    raw_buf = ""
                    read_buf = ""
                elif tag == 'S':
                    words.append((char, clean_read))

    except FileNotFoundError:
        print(f"❌ エラー: ファイル '{tsv_file}' が見つかりません。")
        sys.exit(1)

    return words

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python extract_dict.py <TSVファイル名>")
        sys.exit(1)
    
    tsv_path = sys.argv[1]
    extracted = extract_words_from_tsv(tsv_path)
    
    # 重複を排除しつつ、抽出した順番を保持する
    seen = set()
    unique_words = []
    for raw, read in extracted:
        if (raw, read) not in seen:
            unique_words.append((raw, read))
            seen.add((raw, read))
    
    print(f"📁 抽出完了: {len(unique_words)} 個のユニークな単語が見つかりました。\n")
    print("原文\t\t読み")
    print("-" * 30)
    for raw, read in unique_words:
        print(f"{raw}\t\t{read}")
