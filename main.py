import signal
import sys
from types import FrameType

from flask import Flask, request

from neomomo import predict_oneline

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
    <p>点訳したい文章: {source}</p>
    <p>点訳結果: {result}</p>
    <a href="/">トップページに戻る</a>
</body></html>
"""

@app.route("/")
def hello() -> str:
    return top_page

@app.route("/predict", methods=["GET"])
def predict() -> str:
    source = request.args.get('source')
    result = predict_oneline(source, "./dataset/training_data.model")
    return predict_page.format(source=source, result=result)

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
