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
        single_char_dict: str | None = None,
        numeric_threshold: float = 0.5,
    ) -> None:
        """モデルファイルを読み込んで予測器を作成する。

        Args:
            model_path: `.mbm` モデルファイルのパス
            single_char_dict: 単一文字辞書 TSV のパス（省略可）
            numeric_threshold: 数字ルールベース変換を発動させる自信度の上限
        """
        ...

    @classmethod
    def from_bundled(
        cls,
        window: int = 7,
        *,
        single_char_dict: str | None = None,
        numeric_threshold: float = 0.5,
    ) -> "Predictor":
        """同梱モデルから予測器を作成する。

        Args:
            window: コンテキストウィンドウサイズ（4, 5, 7）。デフォルトは 7。
            single_char_dict: 単一文字辞書 TSV のパス（省略可）
            numeric_threshold: 数字ルールベース変換を発動させる自信度の上限
        """
        ...

    def predict(self, text: str) -> PredictionResult:
        """テキストをカナに変換する。"""
        ...

    def __repr__(self) -> str: ...


class BrailleResult:
    """点字変換結果。`BrailleTranslator.translate()` の戻り値。"""

    braille: str
    """変換後の点字文字列"""
    text_to_braille: list[int]
    """原文文字インデックス → 点字先頭セルインデックス"""

    def __repr__(self) -> str: ...


class BrailleTranslator:
    """点訳器。行の言語を判定して日本語（かな→点字）／英語（UEB）へ振り分ける。"""

    def __init__(self) -> None:
        """組み込みテーブルで変換器を作成する。"""
        ...

    def translate(self, text: str) -> BrailleResult:
        """1行を言語判定して点字に変換する（日本語=かな / 英語=UEB）。"""
        ...

    def translate_japanese(self, kana: str) -> BrailleResult:
        """言語判定せず、必ず日本語として点訳する（英字は外字符 ⠰ ＋無縮約）。"""
        ...

    def translate_english(self, text: str) -> BrailleResult:
        """言語判定せず、必ず英語（UEB）として点訳する。"""
        ...
