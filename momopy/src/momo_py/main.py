import sys
import argparse
import time
from importlib.metadata import version

from .trainer import create_data, train
from .predictor import Predictor, PredictorConfig

from .pybraille import to_jp_braille
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
            print(braille)
            print(kana)
    except KeyboardInterrupt:
        print("\n🛑 翻訳モード終了。")

def run_predict(config: PredictorConfig, show_trace: bool = False, show_profile: bool = False, segmented_output: bool = False, show_source: bool = True, show_kana: bool = True, show_braille: bool = True) -> None:
    """標準入力からテキストを読み込み、予測結果をJSON形式で出力する対話型モード。
    show_trace が True の場合、各文字の決定根拠をターミナルにも表示する。
    """
    predictor = Predictor(config)
    use_color = sys.stderr.isatty()
    try:
        for line in sys.stdin:
            text = line.strip()
            if not text: continue

            t1 = time.perf_counter() if show_profile else 0.0
            result = predictor.predict(text)
            print(f"予測時間: {(time.perf_counter() - t1)*1000:.2f} ms") if show_profile else None
            if show_trace:
                print(result.to_json())
            kana: str = result.format_segmented() if segmented_output else result.kana_text
            if show_source:
                print(result.source_text)
            if show_braille:
                print(to_jp_braille(result.kana_text))
            if show_kana:
                print(kana)

            if show_trace:
                print("─" * 48, file=sys.stderr)
                print(result.format_terminal(use_color=use_color), file=sys.stderr)
                print("─" * 48, file=sys.stderr)

    except KeyboardInterrupt:
        print("\n🛑 予測モード終了。お疲れ様でした！")

# --- [デバッグ用ツール群（AI脳内スキャナー）] ---

def run_label_scanner(config: PredictorConfig) -> None:
    """特徴量を入力してスコア上位のラベルを表示する"""
    predictor = Predictor(config)
    print("🧠 AI脳内スキャナー起動 (Ctrl+D で終了)")
    print("使い方: 見たい文字を1文字入力（例: 金）")
    print("応用編: 特徴量名で直接検索（例: char=金, -1:char=の）")
    print("-" * 40)

    vocab = predictor._vectorizer_read.vocabulary_

    try:
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue

            target_feature = f'char={text}' if len(text) == 1 else text

            if target_feature not in vocab:
                print(f"  -> (この特徴量は学習データに存在しません)")
                print("-" * 40)
                continue

            feat_idx = vocab[target_feature]
            col = predictor._coef_read_sparse.getcol(feat_idx).toarray().flatten()

            weights = sorted(
                zip(predictor._read_classes, col),
                key=lambda x: x[1],
                reverse=True
            )

            print(f"\n🔍 検索対象: '{target_feature}'")
            for label, weight in weights:
                if weight == 0.0:
                    break
                print(f"  ラベル: {label:6} | 点数: {weight:+.3f}")
            print("-" * 40)

    except KeyboardInterrupt:
        print("\n🛑 スキャナー終了。")

def main():
    parser = argparse.ArgumentParser(prog="momo")
    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument("-v", "--version", action="version", version=f"Momo {version('momo-py')}")
    
    # コマンドの定義
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--custom-dict", dest="custom_dict", default=None, help="カスタム辞書ファイルのパス")
    predict_parser.add_argument("--single-dict", dest="single_dict", help="単一漢字用辞書ファイルのパス")
    predict_parser.add_argument("--trace", action="store_true", help="各文字の決定根拠をターミナルに表示する")
    predict_parser.add_argument("--explain", type=int, default=8, metavar="N",
                                help="--trace時に表示する特徴量寄与度の上位件数（デフォルト: 8、0で無効）")
    predict_parser.add_argument("--profile", action="store_true", help="予測の実行時間を表示する")
    predict_parser.add_argument("--segment", action="store_true", help="予測結果を文字ごとに分割して出力する")
    predict_parser.add_argument("--no-source", action="store_true", help="予測結果のみ出力（ソーステキストを出力しない）")
    predict_parser.add_argument("--no-kana", action="store_true", help="予測結果の点字のみ出力（カナテキストを出力しない）")
    predict_parser.add_argument("--no-braille", action="store_true", help="予測結果のカナのみ出力（点字テキストを出力しない）")
    
    translate_parser = subparsers.add_parser("translate")
    translate_parser.add_argument("-s", "--segment", action="store_true", help="ソーステキストの文字に対応するように仮名を分割して出力")
    
    labelscan_parser = subparsers.add_parser("label")
    labelscan_parser.add_argument("--model", required=True)
    labelscan_parser.add_argument("--dict", dest="custom_dict", default=None, help="カスタム辞書ファイルのパス")    

    create_data_parser = subparsers.add_parser("createdata")
    create_data_parser.add_argument("--raw", required=True)
    
    trainer_parser = subparsers.add_parser("train")
    trainer_parser.add_argument("--tsv", required=True)
    trainer_parser.add_argument("--window", type=int, default=7, choices=[3, 5, 7],
                            help="特徴量ウィンドウサイズ（3, 5, 7、デフォルト: 7）")
    trainer_parser.add_argument("--dry-run", action="store_true", help="特徴量の抽出とモデルの初期化まで行い、学習はせずに終了する")
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
        train(tsvdata=args.tsv, window=args.window, dry_run=args.dry_run)
        
    elif args.command == "predict":
        # --trace なしのときは explain_top_n=0 にして計算を省略
        explain_top_n = args.explain if args.trace else 0
        config = PredictorConfig(
            model_path=args.model,
            custom_dict_path=args.custom_dict,
            single_kanji_dict_path=args.single_dict,
            explain_top_n=explain_top_n,
        )
        run_predict(config, show_trace=args.trace, show_profile=args.profile, segmented_output=args.segment, show_source=not args.no_source, show_kana=not args.no_kana, show_braille=not args.no_braille)
        
    elif args.command == "translate":
        run_translate(args.segment)
        
    elif args.command == "label":
        config = PredictorConfig(
            model_path=args.model,
            custom_dict_path=args.custom_dict,
        )
        run_label_scanner(config)

if __name__ == "__main__":
    main()
