import signal
import os
import time
import sys
import json
from types import FrameType
from flask import Flask, request
from momobrl import PredictionResult, Predictor, Translator

app = Flask(__name__)

top_page = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neomomo</title>
</head>
<body>
    <h1>Neomomo</h1>
    <p>AI点訳だよ！まだまだお勉強中だけど、よろしくね！</p>
    <p>
    <form action="/predict" method="get">
        <label for="source">点訳したい文章を入力してね：</label><br>
        <input type="text" id="source" name="source" required><br>
        <input type="submit" value="点訳する">
    </form>
    </p>
</body>
</html>
"""

predict_page = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neomomo - 点訳結果</title>
</head>
<body>
    <h1>点訳結果</h1>
    <p>元の文章: {source}</p>
    <p>点訳結果: {result}</p>
    {confidences_table}
    <p>
    <form action="/predict" method="get">
        <label for="source">点訳したい文章を入力してね：</label><br>
        <input type="text" id="source" value="{source}" name="source" required><br>
        <input type="submit" value="点訳する">
    </form>
    </p>
    <p>model version: {model_version}</p>
</body></html>
"""

model_file = "./dataset/training_data.zip"
def make_characters_table(res: PredictionResult) -> str:
    characters_table = "<table border='1'><tr><th>元の文字</th>"

    for i, idx in enumerate(res.kana_to_src_index):
        if res.kana_text[i] == ' ':
            orig_char = ' '
        else:
            orig_char = res.source_text[idx] if idx < len(res.source_text) else ''
        characters_table += f"<td>{orig_char}</td>"
    characters_table += "</tr><tr><th>文字</th>"
    # かな約結果を並べる
    for c in res.kana_text:
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
    source = request.args.get('source')
    p = Predictor(model_file)
    model_version = p.get_version_info().get('trained_at', '不明')

    res = p.predict(source)
    
    return predict_page.format(source=source, result=res.kana_text, confidences_table=make_characters_table(res), model_version=model_version)

def shutdown_handler(signal_int: int, frame: FrameType) -> None:
    # Safely exit program
    sys.exit(0)


if __name__ == "__main__":
    # Running application locally, outside of a Google Cloud Environment

    # handles Ctrl-C termination
    signal.signal(signal.SIGINT, shutdown_handler)

    app.run(host="localhost", port=8080, debug=True)
else:
    # handles Cloud Run container termination
    signal.signal(signal.SIGTERM, shutdown_handler)
