import sys
import os
import pickle
import argparse
import unicodedata
import sklearn_crfsuite
from typing import Union, Optional, List, Dict
# --- [共通・データ定義] ---
FeatureDict = Dict[str, Union[str, float, bool]]
# [文[行[項目]]]
CorpusData = List[List[List[str]]]

# --- [1. データ作成 (createdata) 用] ---

def get_char_type(c: str) -> str:
    if not c or c.isspace(): return 'SPACE'
    if c.isdigit() or ('0' <= c <= '9'): return 'NUM'
    name = unicodedata.name(c, "")
    if "HIRAGANA" in name: return 'HIRAGANA'
    if "KATAKANA" in name: return 'KATAKANA'
    if "CJK UNIFIED IDEOGRAPH" in name: return 'KANJI'
    return 'OTHER'

def is_suspicious(raw: str, read: str) -> bool:
    """原文と読みの対応が怪しい場合にTrueを返す"""
    if (raw == "" and read == " ") or read == "_": return False
    rt = get_char_type(raw[0]) if raw else 'NONE'
    yt = get_char_type(read[0]) if read else 'NONE'
    
    # 基本的な不一致を検出
    if rt == 'OTHER' and yt in ['KANJI', 'HIRAGANA', 'KATAKANA']: return True
    if rt == 'KANJI' and yt == 'OTHER': return True
    if rt == 'NUM' and yt == 'OTHER': return True
    return False

def process_line_to_tsv(line: str, line_num: int) -> List[str]:
    line = line.strip()
    if not line or line.startswith('#'): return []
    if '\t' not in line: return []

    raw_full, read_full = line.split('\t')
    raw_blocks = raw_full.split('/')
    read_blocks = read_full.split('/')
    
    if len(raw_blocks) != len(read_blocks):
        print(f"❌ Error(Line {line_num}): Block count mismatch (Raw:{len(raw_blocks)} / Read:{len(read_blocks)})")
        return []

    tsv_rows = []
    current_orig_idx = 0
    
    for i, (raw, read) in enumerate(zip(raw_blocks, read_blocks)):
        # 警告の呼び出し
        if is_suspicious(raw, read):
            print(f"⚠️  Suspicious match (Line {line_num}, Block {i+1}): [{raw}] <-> [{read}]")

        # 分かち書き(//)の処理
        if raw == "" and read == " ":
            if tsv_rows:
                last_row_parts = tsv_rows[-1].split('\t')
                last_row_parts[1] = last_row_parts[1] + "+S"
                tsv_rows[-1] = "\t".join(last_row_parts)
            continue
        
        block_len = len(raw)
        for j, char in enumerate(raw):
            ctype = get_char_type(char)
            r_val = read if j == 0 else "---"
            tag = "S" if block_len == 1 else ("B" if j == 0 else ("E" if j == block_len - 1 else "I"))
            tsv_rows.append(f"{char}\t{r_val}\t{ctype}\t{tag}\t{current_orig_idx}")
            current_orig_idx += 1
            
    return tsv_rows

# --- [2. 学習・保存・予測用の関数群] ---

def save_pkl(model: sklearn_crfsuite.CRF, filename: str) -> bool:
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"💾 モデルを保存しました: {filename}")
        return True
    except Exception as e:
        print(f"❌ 保存失敗: {e}")
        return False

def load_pkl(filename: str) -> Optional[sklearn_crfsuite.CRF]:
    if not os.path.exists(filename):
        print(f"❌ モデルファイルが見つかりません: {filename}")
        return None
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"❌ 読み込み失敗: {e}")
        return None

def char2features(sentence: List[List[str]], i: int) -> FeatureDict:
    char = sentence[i][0]
    ctype = sentence[i][2]
    features = {'bias': 1.0, 'char': char, 'type': ctype}
    
    if i > 0:
        features.update({'-1:char': sentence[i-1][0], '-1:bi': sentence[i-1][0] + char})
        if i > 1: features['-2:char'] = sentence[i-2][0]
    else:
        features['BOS'] = True
        
    if i < len(sentence) - 1:
        features.update({'+1:char': sentence[i+1][0], '+1:bi': char + sentence[i+1][0]})
        if i < len(sentence) - 2: features['+2:char'] = sentence[i+2][0]
    else:
        features['EOS'] = True
        
    return features

def load_tsv(filename: str) -> CorpusData:
    sentences = []
    current_sentence = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                current_sentence.append(parts)
    if current_sentence:
        sentences.append(current_sentence)
    return sentences

# --- [3. サブコマンド実行ロジック] ---

def run_createdata(input_fn: str) -> None:
    output_fn = input_fn.rsplit('_', 1)[0] + "_data.tsv"
    print(f"📝 TSV作成モード: {input_fn} -> {output_fn}")
    try:
        with open(input_fn, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ エラー: {input_fn} が見つかりません。")
        return

    all_tsv = ["#原文\t読み\t文字種\tタグ\tOrigIdx"]
    success_count = 0
    for i, line in enumerate(lines, 1):
        rows = process_line_to_tsv(line, i)
        if rows:
            all_tsv.extend(rows)
            all_tsv.append("")
            success_count += 1

    with open(output_fn, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_tsv))
    print(f"✅ 完了: {success_count}/{len(lines)} 行を処理しました。")

def run_train(tsv_path: str) -> None:
    model_path = tsv_path.rsplit('.', 1)[0] + ".model"
    print(f"⚙️ 学習モード: {tsv_path} -> {model_path}")
    data = load_tsv(tsv_path)
    X = [[char2features(s, i) for i in range(len(s))] for s in data]
    y = [[s[i][1] for i in range(len(s))] for s in data]
    
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100)
    crf.fit(X, y)
    print("🤖 学習が完了しました！")
    save_pkl(crf, model_path)

def run_predict(model_path: str) -> None:
    model = load_pkl(model_path)
    if not model: return
    
    print(f"🔮 予測モード (Ctrl+D / Ctrl+Z で終了)")
    for line in sys.stdin:
        text = line.strip()
        if not text: continue
        
        # 内部の get_char_type を使用
        test_sentence = [[c, "", get_char_type(c)] for c in text]
        X_test = [char2features(test_sentence, i) for i in range(len(test_sentence))]
        y_pred = model.predict_single(X_test)
        
        result = ""
        for char, pred in zip(text, y_pred):
            if pred == "_": continue
            if pred == "---": continue
            
            if "+S" in pred:
                clean_read = pred.replace("+S", "")
                result += clean_read + " "
            else:
                result += pred
        
        print(f"原文: {text}")
        print(f"予測: {result.strip()}")
        print("-" * 20)

# --- [4. メイン・エントリポイント] ---

def main() -> None:
    parser = argparse.ArgumentParser(prog="translate", description="精密点訳CRFツール")
    subparsers = parser.add_subparsers(dest="command")

    cp = subparsers.add_parser("createdata", help="RAWテキストから学習用TSVを作成")
    cp.add_argument("--raw", required=True, help="入力RAWファイル")

    tp = subparsers.add_parser("train", help="TSVからモデルを学習")
    tp.add_argument("--tsv", required=True, help="学習用TSVファイル")

    pp = subparsers.add_parser("predict", help="モデルを使って予測")
    pp.add_argument("--model", required=True, help="モデルファイル")

    args = parser.parse_args()

    if args.command == "createdata": run_createdata(args.raw)
    elif args.command == "train": run_train(args.tsv)
    elif args.command == "predict": run_predict(args.model)
    else: parser.print_help()

if __name__ == "__main__":
    main()
