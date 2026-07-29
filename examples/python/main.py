"""MOMO (momors-py) を使った最小サンプル。

日本語テキストを「かな」「点字」に変換して表示する。

    uv run main.py
"""

import sys

from momors_py import BrailleTranslator, Predictor


def _use_utf8_stdio() -> None:
    """Windows でパイプ/リダイレクト時も日本語が化けないよう UTF-8 に統一する。"""
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def main() -> None:
    # Windows でパイプ/リダイレクト時も日本語が化けないよう UTF-8 に統一する。
    _use_utf8_stdio()

    # 同梱モデル（window=7）から予測器を作成する。初回はモデル読み込みに少し時間がかかる。
    predictor = Predictor.from_bundled(window=7)
    braille = BrailleTranslator()

    # 標準入力からテキストを取得する。空行または EOF（Ctrl+Z / パイプ終端）で終了する。
    while True:
        try:
            text = input()
        except EOFError:
            break
        text = text.strip()
        if not text:
            break

        # 日本語テキストを点字のルールで分かち書きされた仮名に変換する。
        result = predictor.predict(text)
        # 点訳には点字のルールで分かち書きされた仮名をわたす。
        cells = braille.translate(result.kana)

        print(f"原文      : {result.source}")
        print(f"かな      : {result.kana}")
        print(f"点字      : {cells.braille}")


if __name__ == "__main__":
    main()
