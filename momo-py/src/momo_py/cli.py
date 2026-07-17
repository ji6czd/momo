import sys
import argparse
from importlib.metadata import version

from .trainer import create_data, train


def main():
    parser = argparse.ArgumentParser(prog="trainer", description="Momo Py: 学習ツール")
    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument(
        "-v", "--version", action="version", version=f"Momo {version('momo-py')}"
    )

    # momo-py は学習ツール。推論（predict）と診断（trace / label）は Rust 版
    # （`momo` コマンドと `momo-inspect`）へ移行した。ここには学習系の
    # createdata / train のみを置く。

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


if __name__ == "__main__":
    main()
