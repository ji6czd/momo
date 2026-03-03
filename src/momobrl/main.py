import sys
import argparse

import sklearn_crfsuite
from typing import List, Tuple

try:
    from .trainer import createdata, train
    from .pybraille import to_jp_braille, to_braille
    from .predictor import Predictor
    from .translator import Translator
except ImportError:
    from trainer import createdata, train  # type: ignore
    from pybraille import to_jp_braille, to_braille  # type: ignore
    from predictor import Predictor  # type: ignore
    from translator import Translator  # type: ignore

def run_translate():
    """
    Sudachiを使い、形態素解析ベースの点訳を行う。
    標準入力からテキストを読み込み、点字変換して標準出力に出力する対話型モード。
    Ctrl+Cで終了。
    """
    t = Translator()
    for line in sys.stdin:
        src = line.strip()
        if not src:
            continue
        kana = t.convert_to_kana(src)
        print(src)
        print(kana)

def run_predict(model_path: str) -> None:
    """標準入力からテキストを読み込み、予測結果をJSON形式で出力する対話型モード"""
    predictor = Predictor(model_path)
    # 標準入力からデータを読み込む。Ctrl+Cで終了。
    try:
        for line in sys.stdin:
            text = line.strip()
            if not text: continue

            result = predictor.predict(text)
            print(result.to_json())
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
    model = Predictor(model_path).model
    
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
    parser = argparse.ArgumentParser(prog="momo")
    subparsers = parser.add_subparsers(dest="command")
    pp = subparsers.add_parser("predict"); pp.add_argument("--model", required=True)
    tp = subparsers.add_parser("translate")
    lp = subparsers.add_parser("label"); lp.add_argument("--model", required=True)
    cp = subparsers.add_parser("createdata"); cp.add_argument("--raw", required=True)
    tp = subparsers.add_parser("train"); tp.add_argument("--tsv", required=True)
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "createdata":
        out = args.raw.rsplit('_', 1)[0] + "_data.tsv"
        createdata(args.raw, out)
    elif args.command == "train":
        train(args.tsv)
    elif args.command == "predict":
        run_predict(args.model)
    elif args.command == "translate":
        run_translate()
    elif args.command == "label":
        run_label_scanner(args.model)  # 🌟 ここで対話型スキャナーを起動！

if __name__ == "__main__":
    main()