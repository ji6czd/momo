"""momo-server: VSCode拡張のバックエンドサーバー

起動方法:
    uvicorn momo_py.server:app --host 127.0.0.1 --port 8765

または:
    momo-server
"""

from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .predictor import Predictor, PredictorConfig


# ---------------------------------------------------------------------------
# グローバルなPredictorインスタンス（起動時に1回だけ初期化）
# ---------------------------------------------------------------------------
predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """サーバー起動時にPredictorを初期化し、終了時にクリーンアップする。"""
    global predictor
    config = PredictorConfig(window=7)
    predictor = Predictor(config)
    print("Predictor initialized.")
    yield
    predictor = None
    print("Predictor released.")


app = FastAPI(
    title="momo-server",
    description="点字変換エンジン momo の HTTP API サーバー",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# スキーマ定義
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: Annotated[str, Field(description="変換するソーステキスト", min_length=1)]


class KanaResponse(BaseModel):
    kana_text: Annotated[str, Field(description="仮名に変換された文字列")]


# ---------------------------------------------------------------------------
# エンドポイント
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, str]:
    """サーバーの動作確認用エンドポイント。"""
    return {"status": "ok"}


@app.post("/predict/kana", response_model=KanaResponse)
def predict_kana(req: PredictRequest) -> KanaResponse:
    """ソーステキストを仮名文字列に変換して返す。

    - **text**: 変換したい日本語テキスト（1文字以上）
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor is not initialized.")
    result = predictor.predict(req.text)
    return KanaResponse(kana_text=result.kana_text)


@app.get("/predict/kana", response_model=KanaResponse)
def predict_kana_get(text: Annotated[str, Field(description="変換するソーステキスト", min_length=1)]) -> KanaResponse:
    """ソーステキストを仮名文字列に変換して返す（GETメソッド版）。

    - **text**: 変換したい日本語テキスト（クエリパラメータ）
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor is not initialized.")
    result = predictor.predict(text)
    return KanaResponse(kana_text=result.kana_text)


@app.post("/predict/predict")
def predict_json(req: PredictRequest) -> Response:
    """ソーステキストを、点字表記を含むJSON形式で返す。

    - **text**: 変換したい日本語テキスト（1文字以上）
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor is not initialized.")
    json_str = predictor.predict(req.text).to_json()
    return Response(content=json_str, media_type="application/json")


@app.get("/predict/predict")
def predict_json_get(text: Annotated[str, Field(description="変換するソーステキスト", min_length=1)]) -> Response:
    """ソーステキストを、点字表記を含むJSON形式で返す（GETメソッド版）。

    - **text**: 変換したい日本語テキスト（クエリパラメータ）
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor is not initialized.")
    json_str = predictor.predict(text).to_json()
    return Response(content=json_str, media_type="application/json")


# ---------------------------------------------------------------------------
# スクリプトエントリーポイント（momo-server コマンド用）
# ---------------------------------------------------------------------------
def main() -> None:
    uvicorn.run("momo_py.server:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
