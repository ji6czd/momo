import gc
import signal
import threading
import sys
from types import FrameType
from importlib.metadata import version
from flask import Flask, request, abort
from markupsafe import escape
from momors_py import Predictor, PredictionResult, BrailleTranslator, BrailleResult

app = Flask(__name__)

version_info = f"Momo version {version('momors_py')}"

# セマフォで同時実行を1に制限する。
# Cloud Run の concurrency 設定に関係なく、コード側で強制する。
# これにより「現在のモデルを1つだけキャッシュ・切り替え時は解放」が安全に動作する。
_semaphore = threading.Semaphore(1)


# モデル名とウィンドーサイズの対応表
_models = {
    "small": 4,
    "medium": 5,
    "large": 7,
}
# 現在キャッシュしているモデル（同時に1つだけ保持）
_current_model: str = ""
_current_predictor: Predictor | None = None


def _get_predictor(model: str = "large") -> Predictor:
    global _current_model, _current_predictor
    if _current_predictor is None or _current_model != model:
        # 別モデルへの切り替え: 古いインスタンスを解放してからロード
        _current_predictor = None
        gc.collect()
        _current_predictor = Predictor.from_bundled(window=_models[model])
        _current_model = model
    return _current_predictor


# アプリ起動時に large モデルを先読みしておく（初回リクエストのコールドスタートを防ぐ）
_get_predictor("large")

# 点字表示に使うWebフォント。momo-editor と同じ SixBraille HLF (Horizontal Line Framed) を
# base64 で埋め込む（出典: fonts/sixbraille202002a/Webフォント関連/sixbraille41.woff2、Public Domain）。
_SIXBRAILLE_HLF_WOFF2_BASE64 = "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"

_braille_font_face = f"""<style>
@font-face {{
    font-family: 'SixBraille HLF';
    src: url(data:font/woff2;base64,{_SIXBRAILLE_HLF_WOFF2_BASE64}) format('woff2');
}}
</style>"""

top_page = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Momo - Japanese braille translator</title>
</head>
<body>
    <h1>自動点訳システムMomo</h1>
    <p>AI点訳だよ！まだまだお勉強中だけど、よろしくね！</p>
    <p>
    <form action="/predict" method="get">
        <label for="source">点訳したい文章を入力してね：</label><br>
        <input type="text" id="source" name="source" required><br>
        <input type="radio" id="small" name="model" value="small">
        <label for="small">Small (約12MB)</label><br>
        <input type="radio" id="medium" name="model" value="medium">
        <label for="medium">Medium (約24MB)</label><br>
        <input type="radio" id="large" name="model" value="large" checked>
        <label for="large">Large (約37MB)</label><br>
        <input type="submit" value="LR機械学習点訳">
    </form>
    </p>
    <p>{version_info}</p>
</body>
</html>
"""

predict_page = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Momo - 点訳結果</title>
    {braille_font_face}
</head>
<body>
    <h1>点訳結果</h1>
    <p>原文: {source}</p>
    <p>仮名: {result}</p>
    <p>点字: <span style="font-size: 28px; font-family: 'SixBraille HLF', monospace; letter-spacing: 2px;">{braille_result}</span></p>
    <p>{confidences_table}</p>
    <p>
    <form action="/predict" method="get">
        <label for="source">点訳したい文章を入力してね：</label><br>
        <input type="text" id="source" value="{source}" name="source" required><br>
        <input type="radio" id="small" name="model" value="small">
        <label for="small">Small (約8.7MB)</label><br>
        <input type="radio" id="medium" name="model" value="medium">
        <label for="medium">Medium (約14MB)</label><br>
        <input type="radio" id="large" name="model" value="large" checked>
        <label for="large">Large (約21MB)</label><br>
        <input type="submit" value="LR機械学習点訳">
    </form>
    </p>
    <p>{version_info}</p>
</body></html>
"""


def make_characters_table(res: PredictionResult) -> str:
    """
    点訳結果の詳細を表示する。
    """
    characters_table = "<table border='1'><tr><th>元の文字</th>"

    for i, idx in enumerate(res.kana_to_source):
        if res.kana[i] == " ":
            orig_char = " "
        else:
            orig_char = res.source[idx] if idx < len(res.source) else ""
        characters_table += f"<td>{orig_char}</td>"
    characters_table += "</tr><tr><th>文字</th>"
    # かな約結果を並べる
    for c in res.kana:
        characters_table += f"<td>{c}</td>"
    characters_table += "</tr><tr><th>確信度</th>"
    # 次に確信度を横に並べる
    for conf in res.confidences:
        characters_table += f"<td>{conf:.2f}</td>"
    characters_table += "</tr></table>"
    return characters_table


@app.route("/")
def hello() -> str:
    return top_page


@app.route("/predict", methods=["GET"])
def predict() -> str:
    source = request.args.get("source")
    if not source:
        abort(400, "source パラメータが必要です")

    model = request.args.get("model", "large")
    with _semaphore:
        prd = _get_predictor(model)
        res = prd.predict(source)

    translator = BrailleTranslator()
    braille = translator.translate(res.kana).braille

    return predict_page.format(
        braille_font_face=_braille_font_face,
        source=escape(source),
        result=res.kana,
        confidences_table=make_characters_table(res),
        braille_result=braille,
        version_info=version_info,
    )


def shutdown_handler(signal_int: int, frame: FrameType) -> None:
    # Safely exit program
    sys.exit(0)


if __name__ == "__main__":
    # handles Ctrl-C termination
    signal.signal(signal.SIGINT, shutdown_handler)

    app.run(host="localhost", port=8080, debug=True)
else:
    # handles Cloud Run container termination
    signal.signal(signal.SIGTERM, shutdown_handler)
