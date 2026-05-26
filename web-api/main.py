import signal
import sys
from types import FrameType
from importlib.metadata import version
from flask import Flask, request
from momo_py import PredictionResult, Predictor, PredictorConfig
from momo_py import pybraille

app = Flask(__name__)

version_info = f"Momo version {version('momo_py')}"

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
        <label for="small">Small (約8.7MB)</label><br>
        <input type="radio" id="medium" name="model" value="medium" checked>
        <label for="medium">Medium (約14MB)</label><br>
        <input type="radio" id="large" name="model" value="large">
        <label for="large">Large (約21MB)</label><br>
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
</head>
<body>
    <h1>点訳結果</h1>
    <p>原文: {source}</p>
    <p>仮名: {result}</p>
    <p>点字: <span style="font-size: 24px; font-family: monospace; letter-spacing: 2px;">{braille_result}</span></p>
    <p>{confidences_table}</p>
    <p>
    <form action="/predict" method="get">
        <label for="source">点訳したい文章を入力してね：</label><br>
        <input type="text" id="source" value="{source}" name="source" required><br>
        <input type="radio" id="small" name="model" value="small">
        <label for="small">Small (約5MB)</label><br>
        <input type="radio" id="medium" name="model" value="medium" checked>
        <label for="medium">Medium (約14MB)</label><br>
        <input type="radio" id="large" name="model" value="large">
        <label for="large">Large (約21MB)</label><br>
        <input type="submit" value="LR機械学習点訳">
    </form>
    </p>
    <p>Model version: {model_version}</p>
    <p>{version_info}</p>
</body></html>
"""

def make_characters_table(res: PredictionResult) -> str:
    """
    点訳結果の詳細を表示する。
    """
    characters_table = "<table border='1'><tr><th>元の文字</th>"

    for i, idx in enumerate(res.kana_to_src_index):
        if res.kana_text[i] == " ":
            orig_char = " "
        else:
            orig_char = res.source_text[idx] if idx < len(res.source_text) else ""
        characters_table += f"<td>{orig_char}</td>"
    characters_table += "</tr><tr><th>文字</th>"
    # かな約結果を並べる
    for c in res.kana_text:
        characters_table += f"<td>{c}</td>"
    characters_table += "</tr><tr><th>確信度</th>"
    # 次に確信度を横に並べる
    for conf in res.confidences:
        characters_table += f"<td>{conf:.2f}</td>"
    characters_table += "</tr><tr><th>ソース</th>"
    # 変換ソース
    for dec in res.decision_sources:
        characters_table += f"<td>{dec}</td>"
    characters_table += "</tr></table>"
    return characters_table


@app.route("/")
def hello() -> str:
    return top_page


@app.route("/predict", methods=["GET"])
def predict() -> str:
    source = request.args.get("source")
    model = request.args.get("model", "large")
    if model == "small":
        model_file = "./dataset/basic_data_4.zip"
    elif model == "medium":
        model_file = "./dataset/basic_data_5.zip"
    else:
        model_file = "./dataset/basic_data_7.zip"
    custom_dic_file = "./dataset/custom_dictionary.tsv"
    cfg = PredictorConfig(model_path=model_file, custom_dict_path=custom_dic_file)
    prd = Predictor(cfg)
    model_version = prd.get_version_info().get("trained_at", "不明")

    res = prd.predict(source)
    braille = pybraille.to_jp_braille(res.kana_text)

    return predict_page.format(
        source=source,
        result=res.kana_text,
        confidences_table=make_characters_table(res),
        braille_result=braille,
        model_version=model_version,
        version_info=version_info,
    )


@app.route("/translate", methods=["GET"])
def translate() -> str:
    source = request.args.get("source")
    t = Translator()
    kana = t.convert_to_kana(source)
    braille = pybraille.to_jp_braille(kana)
    return translate_page.format(
        source=source,
        result=kana,
        braille_result=braille,
        version_info=version_info,
        sudachi_version=version("sudachipy"),
    )


@app.route("/api/predict", methods=["GET"])
def api_predict() -> str:
    source = request.args.get("source")
    cfg = PredictorConfig(model_path=model_file, custom_dict_path=custom_dic_file)
    p = Predictor(cfg)
    return p.predict(source).to_json()


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
