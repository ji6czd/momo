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

# --- [共通定数] ---
MORA_SPLIT = "+S"   # このラベルの後に分かち書きスペースを挿入する

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
        # 点字の長音変換と文字が一致したときはfalse
        if expected == 'ウ' and clean_read == 'ー':
            return False
        else:
            return expected != clean_read 
        
    return False

# --- [2. 特徴量抽出] ---
# ソース文字系列 = [(char, orig_idx, ctype), ...]
SourceEntry = Tuple[str, int, str]


def compute_source_features(source_seq: List[SourceEntry]) -> List[FeatureDict]:
    """
    ソース文字系列全体に対して、各文字の文脈特徴量を一括計算する。
    1文字=1ラベルの設計に対応し、各文字に対応するFeatureDictのリストを返す。

    特徴量一覧（char/type/バイグラムに加えて）:
    - type_transition  : 前文字->現文字 の文字種遷移パターン
    - kanji_run_len    : 現在位置を中心とした漢字連続数（両方向）
    - kanji_pos_first  : 漢字ランの先頭文字か否か
    """
    result: List[FeatureDict] = []
    n = len(source_seq)

    for i, (char, _orig_idx, ctype) in enumerate(source_seq):
        prev_char  = source_seq[i - 1][0] if i > 0 else ""
        prev_ctype = source_seq[i - 1][2] if i > 0 else ""
        next_char  = source_seq[i + 1][0] if i < n - 1 else ""
        next_ctype = source_seq[i + 1][2] if i < n - 1 else ""

        features: FeatureDict = {
            'bias': 1.0,
            'char': char,
            'type': ctype,
        }

        # --- 前後1文字コンテキスト ---
        if i > 0:
            features['-1:char'] = prev_char
            features['-1:type'] = prev_ctype
            features['-1:bi']   = prev_char + char
            if i > 1:
                features['-2:char'] = source_seq[i - 2][0]
                features['-2:type'] = source_seq[i - 2][2]
        else:
            features['BOS'] = True

        if i < n - 1:
            features['+1:char'] = next_char
            features['+1:type'] = next_ctype
            features['+1:bi']   = char + next_char
            if i < n - 2:
                features['+2:char'] = source_seq[i + 2][0]
                features['+2:type'] = source_seq[i + 2][2]
        else:
            features['EOS'] = True

        # --- 新特徴量1: 文字種遷移パターン ---
        if i > 0:
            features['type_transition'] = prev_ctype + '->' + ctype

        # --- 新特徴量2 & 3: 漢字連続長 / 漢字ラン先頭フラグ ---
        if ctype == 'KANJI':
            run = 1
            j = i + 1
            while j < n and source_seq[j][2] == 'KANJI':
                run += 1; j += 1
            j = i - 1
            while j >= 0 and source_seq[j][2] == 'KANJI':
                run += 1; j -= 1
            features['kanji_run_len']   = run
            features['kanji_pos_first'] = (i == 0 or source_seq[i - 1][2] != 'KANJI')

        result.append(features)

    return result


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

def predict_text(text: str, model) -> Tuple[str, List[int]]:
    """
    単一行のテキストから予測結果（読み）とインデックスマップを返す。
    run_predict() と predict_oneline() の両方から利用される共通処理。

    【1文字=1ラベル設計の推論フロー】
    1. get_units() でソース文字系列を構築
    2. compute_source_features() で文脈特徴量を計算
    3. CRFが各文字に対して1ラベルを予測
    4. ラベルからテキストを再構成（---は多文字ユニット継続、+Sは分かち書き）
    """
    units_info = get_units(text)

    # ソース文字系列を構築
    source_seq: List[SourceEntry] = []
    for val, idx in units_info:
        for i, c in enumerate(val):
            source_seq.append((c, idx + i, get_char_type(c)))

    if not source_seq:
        return "", []

    # 特徴量計算 → 予測
    src_features = compute_source_features(source_seq)
    y_pred = model.predict_single(src_features)

    # ラベル列からテキストを再構成
    translated: str       = ""
    index_map:  List[int] = []

    for i, (char, orig_idx, ctype) in enumerate(source_seq):
        label = y_pred[i]
        clean_label = label.replace(MORA_SPLIT, "")

        if clean_label == "_":
            pass  # スキップ文字（空白等）
        elif clean_label != "---":
            # 通常の読みラベル
            for ch in clean_label:
                translated += ch
                index_map.append(orig_idx)

        # 分かち書きスペース
        if MORA_SPLIT in label:
            translated += " "
            index_map.append(orig_idx)

    return translated, index_map


def load_model(modelpath: str) -> sklearn_crfsuite.CRF:
    """
    モデルをファイルから読み込む。
    """
    if not os.path.exists(modelpath):
        raise FileNotFoundError(f"❌ モデル未検出: {modelpath}")
    with open(modelpath, 'rb') as f:
        model = pickle.load(f)
    return model


def predict_oneline(text: str, model_path: str) -> str:
    """
    ウェブAPIで流用する関数。
    テキストを入力として、予測結果（読み）を出力。
    """
    model = load_model(model_path)
    translated, _ = predict_text(text, model)
    return translated

def run_predict(model_path: str) -> None:
    model = load_model(model_path)
    
    print("🔮 予測・同期モード (Ctrl+D で終了)")
    for line in sys.stdin:
        text = line.strip()
        if not text: continue
        
        translated, index_map = predict_text(text, model)

        print(f"予測: {translated}")
        if translated:
            test_pos = len(translated) // 2
            orig_pos = get_original_index(index_map, test_pos)
            print(f"同期検証: 変換後インデックス [{test_pos}] 『{translated[test_pos]}』 -> 原文インデックス [{orig_pos}] 付近")
        print("-" * 20)

# --- [5. メイン（デバッグ用ツール群）] ---

def get_labels(crf: sklearn_crfsuite.CRF, target_feature: str) -> List[Tuple[str, float]]:
    """
    指定された特徴量に対するラベルと重みのリストを返す
    """
    weights = []
    # ⚠️ 修正: dictのメソッドは items() です
    for (feature, label), weight in crf.state_features_.items():
        if feature == target_feature:
            weights.append((label, weight))

    # 点数の高い順（降順）にソート
    weights.sort(key=lambda x: x[1], reverse=True)
    return weights

def run_label_scanner(model_path: str) -> None:
    """
    標準入力から対話的に特徴量を受け取り、AIの脳内（配点表）を表示する
    """
    model = load_model(model_path)
    
    print("🧠 AI脳内スキャナー起動 (Ctrl+D で終了)")
    print("使い方: 見たい文字を1文字入力してください。（例: 上）")
    print("応用編: 特徴量名で直接検索も可能です。（例: -1:char:金, +1:char:手）")
    print("-" * 40)
    
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
            
        # 1文字だけ入力された場合は自動的に 'char:〇' に変換する親切設計
        if len(text) == 1:
            target_feature = f'char:{text}'
        else:
            target_feature = text
            
        weights = get_labels(model, target_feature)
        
        print(f"\n🔍 検索対象: '{target_feature}'")
        if not weights:
            print("  -> (この特徴量は学習データに存在しません)")
        else:
            for label, weight in weights:
                # 符号付き(±)で小数点以下3桁まで綺麗にフォーマット
                print(f"  ラベル: {label:6} | 点数: {weight:+.3f}")
        print("-" * 40)

def createdata(rawdata: str, tsvdata: str) -> None:
    """
    原データファイル(rawdata)を処理して、TSVファイル(tsvdata)に出力する。
    """
    with open(rawdata, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    all_tsv = ["#原文\t読み\t文字種\tタグ\tOrigIdx"]
    success = 0
    for i, line in enumerate(lines, 1):
        # '#'で始まる行はコメント行としてスキップする（TSVのヘッダー行もこれに含まれる）
        if not line.strip() or line.startswith('#'):
            continue
        rows = process_line_to_tsv(line, i)
        if rows:
            all_tsv.extend(rows)
            all_tsv.append("")
            success += 1
    with open(tsvdata, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_tsv))
    print(f"✅ TSV作成完了 ({success}行): {tsvdata}")

def train(tsvdata: str) -> None:
    """
    TSVファイルを読み込み、CRFモデルを学習して保存する。
    """
    sentences, current = [], []
    with open(tsvdata, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                if current:
                    sentences.append(current)
                current = []
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                current.append(parts)
    if current:
        sentences.append(current)
    X: List[List[FeatureDict]] = []
    y: List[List[str]] = []
    for sentence in sentences:
        # TSV列: [char, reading, ctype, tag, orig_idx]
        source_seq: List[SourceEntry] = [
            (p[0], int(p[4]) if len(p) > 4 and p[4].lstrip('-').isdigit() else idx,
             p[2])
            for idx, p in enumerate(sentence)
        ]
        labels = [p[1] for p in sentence]

        X.append(compute_source_features(source_seq))
        y.append(labels)

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,   # L1正則化: 低情報特徴量の重みをゼロに近づける
        c2=0.01,  # L2正則化: 重みを全体的に小さく抑える
        max_iterations=100,
        all_possible_transitions=False,  # あり得ない遷移パターンの計算をスキップ
        verbose=True
    )
    crf.fit(X, y)
    
    model_path = tsvdata.rsplit('.', 1)[0] + ".model"
    with open(model_path, 'wb') as f:
        pickle.dump(crf, f)
    print("💾 学習完了")

def main():
    parser = argparse.ArgumentParser(prog="translate")
    subparsers = parser.add_subparsers(dest="command")
    cp = subparsers.add_parser("createdata"); cp.add_argument("--raw", required=True)
    tp = subparsers.add_parser("train"); tp.add_argument("--tsv", required=True)
    pp = subparsers.add_parser("predict"); pp.add_argument("--model", required=True)
    lp = subparsers.add_parser("label"); lp.add_argument("--model", required=True)
    args = parser.parse_args()

    if args.command == "createdata":
        out = args.raw.rsplit('_', 1)[0] + "_data.tsv"
        createdata(args.raw, out)
    elif args.command == "train":
        train(args.tsv)
    elif args.command == "predict":
        run_predict(args.model)
    elif args.command == "label":
        run_label_scanner(args.model)  # 🌟 ここで対話型スキャナーを起動！

if __name__ == "__main__":
    main()