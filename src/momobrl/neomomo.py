import sys
import os
import pickle
import argparse
import unicodedata
import re
import sklearn_crfsuite
from typing import Union, List, Dict

# --- [1. 共通定義] ---
FeatureDict = Dict[str, Union[str, float, bool]]

def get_char_type(c: str) -> str:
    if not c or c.isspace(): return 'SPACE'
    if c.isdigit() or ('0' <= c <= '9'): return 'NUM'
    name = unicodedata.name(c, "")
    if "HIRAGANA" in name: return 'HIRAGANA'
    if "KATAKANA" in name: return 'KATAKANA'
    if "CJK UNIFIED IDEOGRAPH" in name: return 'KANJI'
    return 'OTHER'

def is_suspicious(raw: str, read: str) -> bool:
    """
    データ作成のミスを検知するガード。
    """
    if (raw == "" and read == " ") or read == "_": return False
    clean_read = read.replace("+S", "")

    # 1. カタカナ -> カタカナの一致チェック（ハ・ヘ含む完全一致を要求）
    if all('KATAKANA' in unicodedata.name(c, "") for c in raw):
        if raw != clean_read:
            return True
        return False

    # 2. ひらがな -> カタカナの一致チェック
    # 「は/ワ」「へ/エ」のみ例外として認め、他は厳格にチェック
    PARTICLE_EXCEPTIONS = {"は": ["ハ", "ワ"], "へ": ["ヘ", "エ"]}
    if raw in PARTICLE_EXCEPTIONS:
        if clean_read not in PARTICLE_EXCEPTIONS[raw]:
            return True
        return False

    if all('HIRAGANA' in unicodedata.name(c, "") for c in raw):
        # ひらがなをカタカナに変換して比較
        expected_katakana = "".join([chr(ord(c) + 0x60) for c in raw])
        if expected_katakana != clean_read:
            return True
        return False

    # 3. 漢字や記号の基本的な型チェック
    rt = get_char_type(raw[0]) if raw else 'NONE'
    yt = get_char_type(clean_read[0]) if clean_read else 'NONE'
    if rt == 'OTHER' and yt in ['KANJI', 'HIRAGANA', 'KATAKANA']: return True
    if rt == 'KANJI' and yt == 'OTHER': return True
    
    return False

# --- [2. データ作成 (createdata) ロジック] ---

def process_line_to_tsv(line: str, line_num: int) -> List[str]:
    line = line.strip()
    if not line or line.startswith('#'): return []
    if '\t' not in line:
        print(f"\n❌ Error (Line {line_num}): タブ(Tab)が見つかりません。")
        sys.exit(1)

    raw_part, read_full = line.split('\t')
    read_blocks = read_full.split('/')
    
    regex = r'\[(.*?)\]|([ぁ-んァ-ヶ][ぁぃぅぇぉゃュょァィゥェォャュョ])|(\s+)|(.)'
    
    raw_units = []
    for m in re.finditer(regex, raw_part):
        bracket, combined, space, char = m.groups()
        if bracket is not None: raw_units.append(bracket)
        elif combined is not None: raw_units.append(combined)
        elif space is not None: raw_units.append(space)
        else: raw_units.append(char)

    tsv_rows = []
    raw_ptr = 0
    current_orig_pos = 0

    for r_label in read_blocks:
        if r_label == " ":
            if tsv_rows:
                parts = tsv_rows[-1].split('\t')
                if "+S" not in parts[1]:
                    parts[1] = parts[1] + "+S"
                    tsv_rows[-1] = "\t".join(parts)
            while raw_ptr < len(raw_units) and raw_units[raw_ptr].isspace():
                current_orig_pos += len(raw_units[raw_ptr])
                raw_ptr += 1
            continue

        while raw_ptr < len(raw_units) and raw_units[raw_ptr].isspace():
            current_orig_pos += len(raw_units[raw_ptr])
            raw_ptr += 1
            
        if raw_ptr >= len(raw_units):
            print(f"\n❌ Error (Line {line_num}): 読みラベルが多すぎます。")
            sys.exit(1)

        target_chars = raw_units[raw_ptr]
        if is_suspicious(target_chars, r_label):
            print(f"⚠️  Suspicious (Line {line_num}): '{target_chars}' に対して '{r_label}' が当てられています。")

        block_len = len(target_chars)
        for i, char in enumerate(target_chars):
            ctype = get_char_type(char)
            r_val = r_label if i == 0 else "---"
            tag = "S" if block_len == 1 else ("B" if i == 0 else ("E" if i == block_len - 1 else "I"))
            tsv_rows.append(f"{char}\t{r_val}\t{ctype}\t{tag}\t{current_orig_pos + i}")
            
        current_orig_pos += block_len
        raw_ptr += 1
            
    remaining = [u for u in raw_units[raw_ptr:] if not u.isspace()]
    if remaining:
        print(f"\n❌ Error (Line {line_num}): 原文が余っています ('{''.join(remaining)}')")
        sys.exit(1)

    return tsv_rows

# --- [3. 学習・予測ロジック (変更なし)] ---

def char2features(sentence: List[List[str]], i: int) -> FeatureDict:
    char, _, ctype = sentence[i][0], sentence[i][1], sentence[i][2]
    features = {'bias': 1.0, 'char': char, 'type': ctype}
    if i > 0:
        features.update({'-1:char': sentence[i-1][0], '-1:bi': sentence[i-1][0] + char})
        if i > 1: features['-2:char'] = sentence[i-2][0]
    else: features['BOS'] = True
    if i < len(sentence) - 1:
        features.update({'+1:char': sentence[i+1][0], '+1:bi': char + sentence[i+1][0]})
        if i < len(sentence) - 2: features['+2:char'] = sentence[i+2][0]
    else: features['EOS'] = True
    return features

def run_train(tsv_path: str) -> None:
    if not os.path.exists(tsv_path):
        print(f"❌ TSV未検出: {tsv_path}"); sys.exit(1)
    sentences, current = [], []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                if current: sentences.append(current); current = []
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 4: current.append(parts)
    if current: sentences.append(current)
    X = [[char2features(s, i) for i in range(len(s))] for s in sentences]
    y = [[s[i][1] for i in range(len(s))] for s in sentences]
    print(f"⚙️  学習開始...")
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100)
    crf.fit(X, y)
    model_path = tsv_path.rsplit('.', 1)[0] + ".model"
    with open(model_path, 'wb') as f: pickle.dump(crf, f)
    print(f"💾 モデル保存完了: {model_path}")

def run_predict(model_path: str) -> None:
    if not os.path.exists(model_path):
        print(f"❌ モデル未検出: {model_path}"); sys.exit(1)
    with open(model_path, 'rb') as f: model = pickle.load(f)
    print("🔮 予測モード (Ctrl+D / Ctrl+Z で終了)")
    for line in sys.stdin:
        text = line.strip()
        if not text: continue
        test_sentence = [[c, "", get_char_type(c)] for c in text]
        X_test = [char2features(test_sentence, i) for i in range(len(test_sentence))]
        y_pred = model.predict_single(X_test)
        result = "".join([p.replace("+S", " ") for p in y_pred if p not in ["_", "---"]])
        print(f"予測: {result.strip()}\n" + "-"*20)

def main():
    parser = argparse.ArgumentParser(prog="translate")
    subparsers = parser.add_subparsers(dest="command")
    cp = subparsers.add_parser("createdata"); cp.add_argument("--raw", required=True)
    tp = subparsers.add_parser("train"); tp.add_argument("--tsv", required=True)
    pp = subparsers.add_parser("predict"); pp.add_argument("--model", required=True)
    args = parser.parse_args()

    if args.command == "createdata":
        out = args.raw.rsplit('_', 1)[0] + "_data.tsv"
        with open(args.raw, 'r', encoding='utf-8') as f: lines = f.readlines()
        all_tsv = ["#原文\t読み\t文字種\tタグ\tOrigIdx"]
        success_count = 0
        for i, line in enumerate(lines, 1):
            rows = process_line_to_tsv(line, i)
            if rows: all_tsv.extend(rows); all_tsv.append(""); success_count += 1
        with open(out, 'w', encoding='utf-8') as f: f.write("\n".join(all_tsv))
        print(f"✅ TSV作成成功: {success_count}行処理完了。 -> {out}")
    elif args.command == "train": run_train(args.tsv)
    elif args.command == "predict": run_predict(args.model)

if __name__ == "__main__":
    main()