import sys
import unicodedata

def get_char_type(c: str) -> str:
    if not c or c.isspace(): return 'SPACE'
    if c.isdigit() or ('0' <= c <= '9'): return 'NUM'
    name = unicodedata.name(c, "")
    if "HIRAGANA" in name: return 'HIRAGANA'
    if "KATAKANA" in name: return 'KATAKANA'
    if "CJK UNIFIED IDEOGRAPH" in name: return 'KANJI'
    return 'OTHER'

def is_suspicious(raw: str, read: str) -> bool:
    """原文と読みの対応をチェック。読みが '_' の場合はスキップ扱いとして許容する。"""
    if (raw == "" and read == " ") or read == "_": return False
    
    rt = get_char_type(raw[0]) if raw else 'NONE'
    yt = get_char_type(read[0]) if read else 'NONE'
    
    # 警告パターン
    if rt == 'OTHER' and yt in ['KANJI', 'HIRAGANA', 'KATAKANA']: return True
    if rt == 'KANJI' and yt == 'OTHER': return True
    if rt == 'NUM' and yt == 'OTHER': return True
    
    return False

def process_line(line: str, line_num: int) -> list[str]:
    line = line.strip()
    if not line or line.startswith('#'): return []
    if '\t' not in line:
        print(f"⚠️ {line_num}行目: タブがありません。")
        return []

    raw_full, read_full = line.split('\t')
    raw_blocks = raw_full.split('/')
    read_blocks = read_full.split('/')
    
    if len(raw_blocks) != len(read_blocks):
        print(f"\n❌ エラー(行{line_num}): 塊の数が合いません (左:{len(raw_blocks)} / 右:{len(read_blocks)})")
        return []

    # 妥当性チェック
    suspicious_found = False
    for i, (raw, read) in enumerate(zip(raw_blocks, read_blocks)):
        if is_suspicious(raw, read):
            if not suspicious_found:
                print(f"\n🔍 ズレの警告(行{line_num}): 以下のブロック付近で対応が怪しいです")
                suspicious_found = True
            print(f"  Block {i+1:3}: 原文[{raw:^5}] <-> 読み[{read:^5}] ⚠️ 文字種不一致")

    tsv_rows: list[str] = []
    current_orig_idx = 0
    for raw, read in zip(raw_blocks, read_blocks):
        # スペース処理
        if raw == "" and read == " ":
            tsv_rows.append(f"[SPACE]\t[SPACE]\tSPACE\tS\t{current_orig_idx}")
            continue
        
        block_len = len(raw)
        for i, char in enumerate(raw):
            ctype = get_char_type(char)
            # 読みが '_' の場合もそのままラベルとして保持
            r_val = read if i == 0 else "---"
            tag = "S" if block_len == 1 else ("B" if i == 0 else ("E" if i == block_len - 1 else "I"))
            
            tsv_rows.append(f"{char}\t{r_val}\t{ctype}\t{tag}\t{current_orig_idx}")
            current_orig_idx += 1
            
    return tsv_rows

def main(input_fn: str, output_fn: str) -> None:
    try:
        with open(input_fn, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"エラー: {input_fn} が見つかりません。")
        return

    all_tsv: list[str] = ["#原文\t読み\t文字種\tタグ\tOrigIdx"]
    success_count: int = 0
    
    for i, line in enumerate(lines, 1):
        rows = process_line(line, i)
        if rows:
            all_tsv.extend(rows)
            all_tsv.append("")
            success_count += 1

    with open(output_fn, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_tsv))
    
    print(f"\n✅ 処理完了: {success_count}/{len(lines)} 行 -> {output_fn}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_fn = sys.argv[1]
        # 以前のルール通り、'_'で区切った手前をベースに出力名を生成
        output_fn = sys.argv[1].rsplit('_', 1)[0] + "_data.tsv"
        main(input_fn, output_fn)
    else:
        print("使用法: python create_data.py <入力ファイル>")