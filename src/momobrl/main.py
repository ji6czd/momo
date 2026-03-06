import sys
import argparse
from typing import List, Tuple
from importlib.metadata import version

from .trainer import create_data, train
from .predictor import Predictor

from .pybraille import to_jp_braille, to_braille
from .translator import Translator

def run_translate(opt_segment: bool = False):
    """
    Sudachiを使い、形態素解析ベースの点訳を行う。
    標準入力からテキストを読み込み、点字変換して標準出力に出力する対話型モード。
    Ctrl+Cで終了。
    """
    t = Translator()
    try:
        for line in sys.stdin:
            src = line.strip()
            if not src:
                continue
            kana = t.segment_kana_string(src) if opt_segment else t.convert_to_kana(src)
            braille = t.convert_to_braille(src)
            print(src)
            print(kana)
            print(braille)
    except KeyboardInterrupt:
        print("\n🛑 翻訳モード終了。")

def run_predict(model_path: str) -> None:
    """標準入力からテキストを読み込み、予測結果をJSON形式で出力する対話型モード"""
    predictor = Predictor(model_path)
    try:
        for line in sys.stdin:
            text = line.strip()
            if not text: continue

            result = predictor.predict(text)
            print(result.to_json())
    except KeyboardInterrupt:
        print("\n🛑 予測モード終了。お疲れ様でした！")

# --- [デバッグ用ツール群（AI脳内スキャナー）] ---

def get_labels(tagger, target_feature: str) -> List[Tuple[str, float]]:
    """
    指定された特徴量に対するラベルと重みのリストを返す
    """
    weights = []
    
    # 🌟 本家 pycrfsuite の info().state_features を呼び出して辞書を取得
    model_info = tagger.info()
    
    for (feature, label), weight in model_info.state_features.items():
        if feature == target_feature:
            weights.append((label, weight))

    # 点数の高い順（降順）にソート
    weights.sort(key=lambda x: x[1], reverse=True)
    return weights

def run_label_scanner(model_path: str) -> None:
    """
    標準入力から対話的に特徴量を受け取り、AIの脳内（配点表）を表示する
    """
    # Predictorからtaggerインスタンスを取り出して使用
    predictor = Predictor(model_path)
    tagger = predictor.tagger
    
    print("🧠 AI脳内スキャナー起動 (Ctrl+D で終了)")
    print("使い方: 見たい文字を1文字入力してください。（例: 上）")
    # 🌟 検索例の表記も pycrfsuite ネイティブ形式（=）に更新！
    print("応用編: 特徴量名で直接検索も可能です。（例: -1:char=金, +1:char=手, type_transition=HIRAGANA->KANJI）")
    print("-" * 40)
    
    try:
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue
                
            # 1文字だけ入力された場合は自動的に 'char=〇' に変換
            if len(text) == 1:
                target_feature = f'char={text}'
            else:
                target_feature = text
                
            weights = get_labels(tagger, target_feature)
            
            print(f"\n🔍 検索対象: '{target_feature}'")
            if not weights:
                print("  -> (この特徴量は学習データに存在しません)")
            else:
                for label, weight in weights:
                    # 符号付き(±)で小数点以下3桁まで綺麗にフォーマット
                    print(f"  ラベル: {label:6} | 点数: {weight:+.3f}")
            print("-" * 40)
    except KeyboardInterrupt:
        print("\n🛑 スキャナー終了。")

def main():
    parser = argparse.ArgumentParser(prog="momo")
    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument("-v", "--version", action="version", version=f"Momo {version('momobrl')}")
    
    # コマンドの定義
    pp = subparsers.add_parser("predict")
    pp.add_argument("--model", required=True)
    
    tp = subparsers.add_parser("translate")
    tp.add_argument("-s", "--segment", action="store_true", help="ソーステキストの文字に対応するように仮名を分割して出力")
    
    lp = subparsers.add_parser("label")
    lp.add_argument("--model", required=True)
    
    cp = subparsers.add_parser("createdata")
    cp.add_argument("--raw", required=True)
    
    tp_train = subparsers.add_parser("train")
    tp_train.add_argument("--tsv", required=True)
    
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "createdata":
        out = args.raw.rsplit('_', 1)[0] + "_data.tsv"
        try:
            create_data(args.raw, out)
        except ValueError as e:
            print(f"\n❌ Error: {e}", file=sys.stderr)
            sys.exit(1)
            
    elif args.command == "train":
        train(args.tsv)
        
    elif args.command == "predict":
        run_predict(args.model)
        
    elif args.command == "translate":
        run_translate(args.segment)
        
    elif args.command == "label":
        run_label_scanner(args.model)

if __name__ == "__main__":
    main()