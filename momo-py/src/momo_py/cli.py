import os
import sys
import argparse
import time
from importlib.metadata import version

from .trainer import create_data, train
from .predictor import Predictor, PredictorConfig


def run_predict(
    config: PredictorConfig,
    show_profile: bool = False,
    segmented_output: bool = False,
    segmented_source_output: bool = False,
    show_source: bool = True,
    show_kana: bool = True,
    show_braille: bool = True,
    create_tsv: bool = False,
) -> None:
    """標準入力からテキストを読み込み、予測結果を出力する対話型モード。

    各文字の決定根拠・特徴量寄与度を見る trace や、特徴量ごとのラベル重みを見る
    label 診断は Rust 版 `momo-inspect`（trace / label サブコマンド）へ移行した。
    """
    predictor = Predictor(config)
    try:
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue

            t1 = time.perf_counter() if show_profile else 0.0
            result = predictor.predict(text)
            print(
                f"予測時間: {(time.perf_counter() - t1) * 1000:.2f} ms"
            ) if show_profile else None

            if create_tsv:
                segmented_output = True

            kana: str = (
                result.format_segmented() if segmented_output else result.kana_text
            )
            src: str = (
                result.format_source_segmented()
                if segmented_source_output
                else result.source_text
            )

            if create_tsv:
                print(f"{src}\t{kana}")
            else:
                if show_source:
                    print(src)
                if show_kana:
                    print(kana)

    except KeyboardInterrupt:
        print("\n🛑 予測モード終了。お疲れ様でした！")


def main():
    parser = argparse.ArgumentParser(prog="momo")
    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument(
        "-v", "--version", action="version", version=f"Momo {version('momo-py')}"
    )

    # コマンドの定義
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", default=None)
    predict_parser.add_argument(
        "--custom-dict",
        dest="custom_dict",
        default=None,
        help="カスタム辞書ファイルのパス",
    )
    predict_parser.add_argument(
        "--single-dict",
        dest="single_dict",
        help="単一漢字用辞書ファイルのパス（省略時は実行ファイルと同じ場所またはパッケージ内蔵の辞書を使用）",
    )
    predict_parser.add_argument(
        "--name-dict",
        dest="name_dict",
        default=None,
        help="人名辞書ファイルのパス（人名B/Iフラグ特徴量。省略時はモデルまたは実行ファイルと同じ場所の person_name_dic.tsv を自動検出）",
    )
    predict_parser.add_argument(
        "--profile", action="store_true", help="予測の実行時間を表示する"
    )
    predict_parser.add_argument(
        "--segment", action="store_true", help="予測結果を文字ごとに分割して出力する"
    )
    predict_parser.add_argument(
        "--segment-source",
        action="store_true",
        help="原文を点字の分かち書き単位で分割して出力する",
    )
    predict_parser.add_argument(
        "--no-source",
        action="store_true",
        help="予測結果のみ出力（ソーステキストを出力しない）",
    )
    predict_parser.add_argument(
        "--no-kana",
        action="store_true",
        help="予測結果の点字のみ出力（カナテキストを出力しない）",
    )
    predict_parser.add_argument(
        "--no-braille",
        action="store_true",
        help="予測結果のカナのみ出力（点字テキストを出力しない）",
    )
    predict_parser.add_argument(
        "--window",
        type=int,
        default=7,
        choices=[4, 5, 7],
        help="特徴量ウィンドウサイズ（4, 5, 7、デフォルト: 7）",
    )
    predict_parser.add_argument(
        "--create-momo-tsv",
        action="store_true",
    )

    create_data_parser = subparsers.add_parser("createdata")
    create_data_parser.add_argument("--raw", required=True)
    create_data_parser.add_argument(
        "--name-dict",
        dest="name_dict",
        default=None,
        help="{…} マークから抽出した人名辞書の出力先（省略時は出力TSVと同じディレクトリの person_name_dic.tsv）",
    )

    trainer_parser = subparsers.add_parser("train")
    trainer_parser.add_argument("--tsv", required=True)
    trainer_parser.add_argument(
        "--model",
        help="学習済みモデルのファイルパス（デフォルトはtsvファイル名に基づく）",
    )
    trainer_parser.add_argument(
        "--window",
        type=int,
        default=7,
        choices=[4, 5, 7],
        help="特徴量ウィンドウサイズ（4, 5, 7、デフォルト: 7）",
    )
    trainer_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="特徴量の抽出とモデルの初期化まで行い、学習はせずに終了する",
    )
    trainer_parser.add_argument(
        "--sgd",
        action="store_true",
        help="読みモデルに SGDClassifier(loss=hinge) を使用する（メモリ節約、精度やや低下）",
    )
    trainer_parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="読みモデルの One-vs-Rest 学習を並列実行するワーカ数（デフォルト: 4）",
    )
    trainer_parser.add_argument(
        "--name-dict",
        dest="name_dict",
        default=None,
        help="人名辞書ファイルのパス（省略時はTSVと同じディレクトリの person_name_dic.tsv を自動検出）",
    )
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    if args.command == "createdata":
        out = args.raw.rsplit("_", 1)[0] + "_data.tsv"
        try:
            create_data(args.raw, out, name_dict_path=args.name_dict)
        except ValueError as e:
            print(f"\n❌ Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "train":
        train(
            tsvdata=args.tsv,
            model_file=args.model,
            window=args.window,
            dry_run=args.dry_run,
            use_svc=not args.sgd,
            n_jobs=args.jobs,
            name_dict=args.name_dict,
        )

    elif args.command == "predict":
        exe_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

        single_dict_path = args.single_dict
        if single_dict_path is None:
            candidate = os.path.join(exe_dir, "single_character_dic.tsv")
            if os.path.isfile(candidate):
                single_dict_path = candidate

        # 人名辞書: 明示指定がなければ実行ファイルと同じ場所を探す
        # （モデルファイルと同じディレクトリは Predictor 側が自動検出する）
        name_dict_path = args.name_dict
        if name_dict_path is None:
            candidate = os.path.join(exe_dir, "person_name_dic.tsv")
            if os.path.isfile(candidate):
                name_dict_path = candidate

        config = PredictorConfig(
            model_path=args.model,
            custom_dict_path=args.custom_dict,
            single_kanji_dict_path=single_dict_path,
            person_name_dict_path=name_dict_path,
            explain_top_n=0,  # 特徴量寄与度の表示は Rust 版 momo-inspect trace へ移行
            window=args.window,
        )
        run_predict(
            config,
            show_profile=args.profile,
            segmented_output=args.segment,
            segmented_source_output=args.segment_source,
            create_tsv=args.create_momo_tsv,
            show_source=not args.no_source,
            show_kana=not args.no_kana,
            show_braille=not args.no_braille,
        )


if __name__ == "__main__":
    main()
