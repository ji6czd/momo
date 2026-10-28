class PredictionResult:
    """予測結果。`Predictor.predict()` の戻り値。"""

    source: str
    """入力文字列（原文）"""
    kana: str
    """変換後のカナ文字列"""
    confidences: list[float]
    """各カナ文字の自信度（0.0〜1.0）"""
    decisions: list[str]
    """各カナ文字の決定根拠（`LR`・`BERT`・`FALLBACK_NUM` など）"""
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
        custom_dict: str | None = None,
        numeric_threshold: float = 0.5,
        bert: str | None = None,
        bert_f32: bool = False,
        bert_threads: int = 0,
    ) -> None:
        """モデルファイルを読み込んで予測器を作成する。

        Args:
            model_path: `.mbm` モデルファイルのパス
            single_char_dict: 単一文字辞書 TSV のパス（省略可）
            custom_dict: カスタム辞書（フレーズ辞書）TSV のパス（省略可）
            numeric_threshold: 数字ルールベース変換を発動させる自信度の上限
            bert: BERT 読みモードのモデルディレクトリ（配布用 zip を展開した `momo-bert/`
                または `export_onnx` の出力、省略可）。feature `bert` 付きでビルドした
                wheel だけが受け付ける
            bert_f32: BERT を float32 版で動かす（既定は int8 版）
            bert_threads: BERT の演算スレッド数（0 は ONNX Runtime の既定）
        """
        ...

    @property
    def has_bert(self) -> bool:
        """BERT 読みモードが有効か。"""
        ...

    @property
    def bert_unknown_labels(self) -> list[str]:
        """BERT のラベルのうち、読みモデルのクラスに無くて使えないもの。"""
        ...

    @classmethod
    def from_bundled(
        cls,
        window: int = 7,
        *,
        single_char_dict: str | None = None,
        custom_dict: str | None = None,
        numeric_threshold: float = 0.5,
    ) -> "Predictor":
        """同梱モデルから予測器を作成する（BERT 読みモードは使えない）。

        Args:
            window: コンテキストウィンドウサイズ（4, 5, 7）。デフォルトは 7。
            single_char_dict: 単一文字辞書 TSV のパス（省略可）
            custom_dict: カスタム辞書（フレーズ辞書）TSV のパス（省略可）
            numeric_threshold: 数字ルールベース変換を発動させる自信度の上限
        """
        ...

    def predict(self, text: str) -> PredictionResult:
        """テキストをカナに変換する。"""
        ...

    def __repr__(self) -> str: ...


class BrailleResult:
    """点字変換結果。`BrailleTranslator.translate()` の戻り値。"""

    language: str
    """点訳した経路（`"japanese"` / `"english"`）"""
    text: str
    """点訳したテキスト（日本語=かな / 英語=英文）"""
    braille: str
    """変換後の点字文字列"""
    text_to_braille: list[int]
    """テキスト文字インデックス → 点字先頭セルインデックス"""
    braille_to_text: list[int]
    """点字セルインデックス → テキスト文字インデックス"""

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
