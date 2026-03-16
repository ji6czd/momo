from typing import Tuple
from sudachipy import tokenizer
from sudachipy import dictionary
from sudachipy import Morpheme

wordlist = []  # 単語を格納するリスト
def count_words_chars(text: str) -> Tuple[int, int]:
    """
    日本語テキストの単語数をカウントする関数。
    SudachiPyを使用して形態素解析を行い、単語数を計算します。
    """
    # テキストをタブ文字で分割して左側を得る
    text = text.split('\t')[0]
    # SudachiPyのトークナイザを初期化
    mode = tokenizer.Tokenizer.SplitMode.B
    tokenizer_obj = dictionary.Dictionary().create(mode)

    # テキストを形態素解析して単語数をカウント
    morphemes = tokenizer_obj.tokenize(text)
    word_count = len(morphemes)
    # 単語をリストに追加
    for morpheme in morphemes:
        wordlist.append(morpheme.surface())

    return word_count, len(text)

def main():
    """標準有力からファイルを読み込み、行毎に単語数を数えます。"""
    import sys

    if len(sys.argv) != 2:
        print("Usage: python wordcount.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            words = 0
            chars = 0
            line_number = 0
            for line in file:
                line = line.strip()
                if line:  # 空行を無視
                    w, c = count_words_chars(line)
                    print(f"L:{line_number}: {w} words and {c} characters")
                    words += w
                    chars += c
                line_number += 1
            # 単語リストから重複を排除する
            unique_words = set(wordlist)
            print(f"Unique words: {len(unique_words)} {chars} characters")
            # 単語リストをファイルに保存する
            with open('wordlist.txt', 'w', encoding='utf-8') as word_file:
                for word in unique_words:
                    word_file.write(word + '\n')
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except IOError as e:
        print(f"Error: An I/O error occurred: {e}")
if __name__ == "__main__":
    main()