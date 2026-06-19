class PredictionResult:
    """予測結果。`Predictor.predict()` の戻り値。"""

    source: str
    """入力文字列（原文）"""
    kana: str
    """変換後のカナ文字列"""
    confidences: list[float]
    """各カナ文字の自信度（0.0〜1.0）"""
    segmented: str
    """分かち書きされたカナ（例: `ワガハイ/ワ/ネコ/デ/アル`）"""
    source_segmented: str
    """分かち書きされた原文（例: `吾輩/は/猫/で/ある`）"""
    kana_to_source: list[int]
    """かな文字インデックス → 原文文字インデックス"""
    source_to_kana: list[list[int]]
    """原文文字インデックス → かな文字インデックスのリスト"""

    def __repr__(self) -> str: ...


class Predictor:
    """日本語テキスト → カナ変換器。"""

    def __init__(
        self,
        model_path: str,
        *,
        kanji_dict: str | None = None,
        numeric_threshold: float = 0.5,
    ) -> None:
        """モデルファイルを読み込んで予測器を作成する。

        Args:
            model_path: `.mbm` モデルファイルのパス
            kanji_dict: 漢字辞書 TSV のパス（省略可）
            numeric_threshold: 数字ルールベース変換を発動させる自信度の上限
        """
        ...

    @classmethod
    def from_bundled(
        cls,
        window: int = 7,
        *,
        kanji_dict: str | None = None,
        numeric_threshold: float = 0.5,
    ) -> "Predictor":
        """同梱モデルから予測器を作成する。

        Args:
            window: コンテキストウィンドウサイズ（4, 5, 7）。デフォルトは 7。
            kanji_dict: 漢字辞書 TSV のパス（省略可）
            numeric_threshold: 数字ルールベース変換を発動させる自信度の上限
        """
        ...

    def predict(self, text: str) -> PredictionResult:
        """テキストをカナに変換する。"""
        ...

    def __repr__(self) -> str: ...


class BrailleResult:
    """点字変換結果。`BrailleConverter.convert()` の戻り値。"""

    braille: str
    """変換後の点字文字列"""
    kana_to_braille: list[int]
    """かな文字インデックス → 点字先頭セルインデックス"""

    def __repr__(self) -> str: ...


class BrailleConverter:
    """カナ文字列 → 日本語点字変換器。"""

    def __init__(self) -> None:
        """組み込みテーブルで変換器を作成する。"""
        ...

    def convert(self, kana: str) -> BrailleResult:
        """カナ文字列を点字に変換する。"""
        ...
