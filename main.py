import signal
import sys
from types import FrameType

from flask import Flask

from translate import predict_oneline

app = Flask(__name__)

@app.route("/")
def hello() -> str:
    return "Hello, Google Cloud Run! Please enjoy!"

@app.route("/predict/<source>", methods=["GET"])
def predict(source: str) -> str:
    # return "Predicting..."
    return predict_oneline(source)

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
