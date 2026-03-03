import sys
import os
import pickle
import argparse
import json

import sklearn_crfsuite
from typing import List, Tuple

try:
    from .features import get_units, get_char_type, compute_source_features, SourceEntry, MORA_SPLIT
    from .trainer import createdata, train
except ImportError:
    from features import get_units, get_char_type, compute_source_features, SourceEntry, MORA_SPLIT  # type: ignore
    from trainer import createdata, train  # type: ignore

# --- [4. 推論・インデックス同期] ---

def predict_text(text: str, model) -> Tuple[str, List[int], List[float]]:
    """
    単一行のテキストから予測結果（読み）とインデックスマップ、および各文字の予測確信度を返す。
    run_predict() から利用される共通処理。

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
        return "", [], []

    # 特徴量計算
    src_features = compute_source_features(source_seq)
    
    # 🌟 通常の予測に加えて、確率（自信度）の辞書リストも取得！
    y_pred = model.predict_single(src_features)
    y_marginals = model.predict_marginals_single(src_features)

    translated: str = ""
    index_map: List[int] = []
    confidences: List[float] = [] # 🌟 自信度を保存するリスト

    for i, (char, orig_idx, ctype) in enumerate(source_seq):
        label = y_pred[i]
        # 🌟 選ばれたラベルの確率を取得（0.0 〜 1.0）
        confidence = y_marginals[i].get(label, 0.0)
        
        clean_label = label.replace(MORA_SPLIT, "")

        if clean_label == "_":
            pass
        elif clean_label != "---":
            for ch in clean_label:
                translated += ch
                index_map.append(orig_idx)
                confidences.append(confidence) # 🌟 記録

        if MORA_SPLIT in label:
            translated += " "
            index_map.append(orig_idx)
            confidences.append(confidence) # 🌟 スペースにも直前の文字の自信度を記録

    return translated, index_map, confidences

def format_result_json(text: str, translated: str, index_map: List[int], confidences: List[float]) -> str:
    """
    予測結果を整形されたJSON文字列として返す。
    リスト（配列）部分は1行にまとめて視認性を高める。
    """
    text_safe = json.dumps(text, ensure_ascii=False)
    kana_safe = json.dumps(translated, ensure_ascii=False)
    
    # 配列だけを1行の文字列として作成（小数点以下3桁に丸める）
    conf_str = "[" + ", ".join([f"{c:.3f}" for c in confidences]) + "]"
    idx_str = "[" + ", ".join(map(str, index_map)) + "]"
    
    # 理想の形で組み上げる
    custom_json = f'{{\n  "text": {text_safe},\n  "kana": {kana_safe},\n  "index_map": {idx_str},\n  "confidences": {conf_str}\n}}'
    
    return custom_json

def load_model(modelpath: str) -> sklearn_crfsuite.CRF:
    """
    モデルをファイルから読み込む。
    """
    if not os.path.exists(modelpath):
        raise FileNotFoundError(f"❌ モデル未検出: {modelpath}")
    with open(modelpath, 'rb') as f:
        model = pickle.load(f)
    return model


def run_predict(model_path: str) -> None:
    model = load_model(model_path)
    # 標準入力からデータを読み込む。Ctrl+Cで終了。
    try:
        for line in sys.stdin:
            text = line.strip()
            if not text: continue
        
            translated, index_map, confidences = predict_text(text, model)
            print(format_result_json(text, translated, index_map, confidences))
    except KeyboardInterrupt:
        print("\n🛑 予測モード終了。お疲れ様でした！")

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