"""momo-py: MOMO の学習ツール。

推論（かな変換）・点字変換・診断（trace / label）は Rust 実装（momors）へ移行済み。
このパッケージが担うのは学習系（createdata / train）と、Rust が読み込む .mbm /
.mbmf モデルのエクスポートのみ。

- 学習・データ生成: `momo_py.trainer`（`create_data` / `train`）
- モデルエクスポート: `momo_py.exporter`
- 学習結果のシリアライズ定義: `momo_py.bundle`（`LRModelBundle`）
"""

from .trainer import create_data, train

__all__ = [
    "create_data",
    "train",
]
