import sys
import os
import pickle
import argparse
import unicodedata
import re
import sklearn_crfsuite
from typing import Union, List, Dict, Tuple

# --- [1. 共通定義・型定義] ---
FeatureDict = Dict[str, Union[str, float, bool]]

def get_char_type(c: str) -> str:
    """文字種を判定。句読点・記号を独立したカテゴリ(SYMBOL)として扱う。"""
    if not c or c.isspace(): return 'SPACE'
    if c.isdigit() or ('0' <= c <= '9'): return 'NUM'
    
    # Unicodeカテゴリで句読点(P)や記号(S)を判定
    cat = unicodedata.category(c)
    if cat.startswith('P') or cat.startswith('S'):
        return 'SYMBOL'
        
    name = unicodedata.name(c, "")
    if "HIRAGANA" in name: return 'HIRAGANA'
    if "KATAKANA" in name: return 'KATAKANA'
    if "CJK UNIFIED IDEOGRAPH" in name: return 'KANJI'
    if "LATIN" in name: return 'ALPHA'
    
    return 'OTHER'

def get_units(text: str) -> List[Tuple[str, int]]:
    """
    テキストを音節ユニットに分解し、それぞれの原文開始位置を返す。
    ブラケット内の文字も、原文の正確なインデックス（0始まり）を保持する。
    英数字の連続（カンマやハイフン含む）は自動的に1つのブロックとしてまとめる。
    """
    # 🌟 グループ3で英数字の塊（iPhone, 120,000, Wi-Fiなど）を捉える
    regex = r'\[(.*?)\]|([ぁ-んァ-ヶ][ぁぃぅぇぉゃュょァィゥェォャュョ])|([a-zA-Z0-9\.\-,]+)|(\s+)|(.)'
    units = []
    for m in re.finditer(regex, text):
        if m.group(1) is not None:
            units.append((m.group(1), m.start(1)))
        elif m.group(2) is not None:
            units.append((m.group(2), m.start(2)))
        elif m.group(3) is not None:
            units.append((m.group(3), m.start(3)))
        elif m.group(4) is not None:
            units.append((m.group(4), m.start(4)))
        else:
            units.append((m.group(5), m.start(5)))
    return units

def is_suspicious(raw: str, read: str) -> bool:
    """
    ひらがな/カタカナの単純な対応ミスを検知。
    「は/ワ」「へ/エ」は点訳ルールとして許可。
    """
    if (raw == "" and read == " ") or read == "_": return False
    clean_read = read.replace("+S", "")
    
    # カタカナ -> カタカナの完全一致チェック
    if all('KATAKANA' in unicodedata.name(c, "") for c in raw):
        return raw != clean_read
        
    # ひらがな -> カタカナの一致チェック（例外あり）
    PARTICLE_EXCEPTIONS = {"は": ["ハ", "ワ"], "へ": ["ヘ", "エ"]}
    if raw in PARTICLE_EXCEPTIONS:
        return clean_read not in PARTICLE_EXCEPTIONS[raw]
        
    if all('HIRAGANA' in unicodedata.name(c, "") for c in raw):
        expected = "".join([chr(ord(c) + 0x60) for c in raw])
        return expected != clean_read
        
    return False

# --- [2. 特徴量抽出] ---

def char2features(sentence: List[List[Union[str, int]]], i: int) -> FeatureDict:
    """
    周辺コンテキスト（半径2文字/計5文字ウィンドウ）から特徴量を抽出。
    """
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

# --- [3. 学習・データ作成ロジック] ---

def process_line_to_tsv(line: str, line_num: int) -> List[str]:
    line = line.strip()
    parts = line.split('\t')
    
    if len(parts) < 2:
        print(f"\n❌ Error (Line {line_num}): タブが見つかりません。"); sys.exit(1)
    elif len(parts) > 2:
        print(f"\n❌ Error (Line {line_num}): タブが複数（{len(parts)-1}個）含まれています。")
        print(f"   -> 原文と読みを区切るためのタブは「1つだけ」にしてください。分かち書き等にタブが混ざっていないか確認してください。")
        print(parts[2])
        sys.exit(1)
        
    raw_part, read_full = parts[0], parts[1]
    
    # '//' の連続チェック
    if '//' in read_full:
        print(f"⚠️  Warning (Line {line_num}): 読み部分に連続した '/' が含まれています: '{read_full}'")
    
    read_blocks = read_full.split('/')
    raw_units_info = get_units(raw_part)
    
    tsv_rows, raw_ptr = [], 0
    
    # 0始まりでブロックをカウント
    for label_idx, r_label in enumerate(read_blocks):
        if r_label == " ":
            if tsv_rows:
                parts = tsv_rows[-1].split('\t')
                if "+S" not in parts[1]:
                    parts[1] += "+S"
                    tsv_rows[-1] = "\t".join(parts)
            while raw_ptr < len(raw_units_info) and raw_units_info[raw_ptr][0].isspace():
                raw_ptr += 1
            continue
            
        while raw_ptr < len(raw_units_info) and raw_units_info[raw_ptr][0].isspace():
            raw_ptr += 1
            
        if raw_ptr >= len(raw_units_info):
            print(f"\n❌ Error (Line {line_num}): 読みラベル過多。")
            print(f"   -> 読みインデックス [{label_idx}] のブロック '{r_label}' に対応する原文がありません！")
            sys.exit(1)
        
        target_chars, orig_idx = raw_units_info[raw_ptr]
        
        if is_suspicious(target_chars, r_label):
            print(f"⚠️  Suspicious (Line {line_num}): 読みインデックス [{label_idx}] '{target_chars}' -> '{r_label}' (原文インデックス: {orig_idx})")
        
        # 句読点を見つけたら、読みラベルに自動で「+S」を補う
        KUTOUTEN = ["。", "、", "？", "！", ".", ","]
        if target_chars in KUTOUTEN and "+S" not in r_label:
            r_label += "+S"
            
        block_len = len(target_chars)
        for i, char in enumerate(target_chars):
            ctype = get_char_type(char)
            r_val = r_label if i == 0 else "---"
            tag = "S" if block_len == 1 else ("B" if i == 0 else ("E" if i == block_len - 1 else "I"))
            # TSVにも正確な0オリジンのインデックスを記録
            tsv_rows.append(f"{char}\t{r_val}\t{ctype}\t{tag}\t{orig_idx + i}")
        raw_ptr += 1
    
    remaining = [u for u in raw_units_info[raw_ptr:] if not u[0].isspace()]
    if remaining:
        first_leftover_val, first_leftover_idx = remaining[0]
        print(f"\n❌ Error (Line {line_num}): 原文余り。")
        print(f"   -> 原文インデックス [{first_leftover_idx}] の '{first_leftover_val}' 以降に対応する読みラベルがありません！")
        sys.exit(1)
        
    return tsv_rows

# --- [4. 推論・インデックス同期] ---

def get_original_index(index_map: List[int], pos: int) -> int:
    """
    変換後テキストの指定位置から原文のインデックスを返す。
    """
    if 0 <= pos < len(index_map):
        return index_map[pos]
    return -1

def run_predict(model_path: str) -> None:
    if not os.path.exists(model_path):
        print(f"❌ モデル未検出: {model_path}"); sys.exit(1)
    with open(model_path, 'rb') as f: model = pickle.load(f)
    
    print("🔮 予測・同期モード (Ctrl+D で終了)")
    for line in sys.stdin:
        text = line.strip()
        if not text: continue
        
        units_info = get_units(text)
        test_data = []
        for val, idx in units_info:
            for i, c in enumerate(val):
                test_data.append([c, idx + i, get_char_type(c)])
        
        X_test = [char2features(test_data, i) for i in range(len(test_data))]
        y_pred = model.predict_single(X_test)
        
        translated, index_map = "", []
        for i, label in enumerate(y_pred):
            if label == "_": 
                continue
                
            orig_char = test_data[i][0]
            ctype = test_data[i][2]
            orig_idx = test_data[i][1]
            
            # 予測ラベルから +S を除去した純粋な読み
            clean_label = label.replace("+S", "")

            # 🌟 新規最適化：英数字の場合は常に原文を1文字ずつ出力
            if ctype in ['NUM', 'ALPHA']:
                translated += orig_char
                index_map.append(orig_idx)
            # 日本語の場合で、継続タグ（---）ではない場合のみ読みを出力
            elif clean_label != "---":
                for char in clean_label:
                    translated += char
                    index_map.append(orig_idx)
            
            # 🌟 分かち書きの処理 (+S が含まれていればスペースを足す)
            if "+S" in label:
                translated += " "
                index_map.append(orig_idx)

        print(f"予測: {translated}")
        if translated:
            test_pos = len(translated) // 2
            orig_pos = get_original_index(index_map, test_pos)
            print(f"同期検証: 変換後インデックス [{test_pos}] 『{translated[test_pos]}』 -> 原文インデックス [{orig_pos}] 付近")
        print("-" * 20)

# --- [5. メイン] ---

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
        success = 0
        for i, line in enumerate(lines, 1):
            if not line.strip() or line.startswith('#'): continue
            rows = process_line_to_tsv(line, i)
            if rows: all_tsv.extend(rows); all_tsv.append(""); success += 1
        with open(out, 'w', encoding='utf-8') as f: f.write("\n".join(all_tsv))
        print(f"✅ TSV作成完了 ({success}行): {out}")
    elif args.command == "train":
        sentences, current = [], []
        with open(args.tsv, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    if current: sentences.append(current); current = []
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 4: current.append(parts)
        if current: sentences.append(current)
        X = [[char2features(s, i) for i in range(len(s))] for s in sentences]
        y = [[s[i][1] for i in range(len(s))] for s in sentences]
        crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100)
        crf.fit(X, y)
        with open(args.tsv.rsplit('.', 1)[0] + ".model", 'wb') as f: pickle.dump(crf, f)
        print("💾 学習完了")
    elif args.command == "predict": run_predict(args.model)

if __name__ == "__main__":
    main()