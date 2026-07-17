import gc
import json
import os
import io
import sys
import zipfile
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, Optional, Any
from importlib import resources
import joblib
import numpy as np

from .features import (
    get_units,
    compute_source_features,
    SourceEntry,
    LABEL_CONTINUE,
    LABEL_SKIP,
    CharType,
)
from .utils import (
    split_on_unescaped_slash,
    CharType,
    normalize_input,
    convert_to_katakana,
    parse_kanji_dict_tsv,
)
from .name_dict import (
    NAME_DICT_FILENAME,
    NameIndex,
    build_name_index,
    compute_name_matches,
    load_name_dict,
    parse_name_dict_text,
)
from .bundle import LRModelBundle, SINGLE_KANJI_DICT_FILENAME

# 小書き仮名（拗音・促音など）。直前の文字に吸収されるべき文字。
_SMALL_KANA = frozenset("ぁぃぅぇぉゃゅょっゎァィゥェォャュョッヮ")

# カタカナでもパススルーせずモデルに委ねる文字（助数詞用小書きカタカナ）
# ヵ (U+30F5), ヶ (U+30F6): 「1ヵ月」「1ヶ所」など、拗音形成に使わない
_KATAKANA_NO_PASSTHROUGH = frozenset("ヵヶ")

# 々繰り返し判定: 先頭音節の連濁マップ（カタカナ）
_RENDAKU_MAP: Dict[str, str] = {
    "カ": "ガ",
    "キ": "ギ",
    "ク": "グ",
    "ケ": "ゲ",
    "コ": "ゴ",
    "サ": "ザ",
    "シ": "ジ",
    "ス": "ズ",
    "セ": "ゼ",
    "ソ": "ゾ",
    "タ": "ダ",
    "チ": "ヂ",
    "ツ": "ヅ",
    "テ": "デ",
    "ト": "ド",
    "ハ": "バ",
    "ヒ": "ビ",
    "フ": "ブ",
    "ヘ": "ベ",
    "ホ": "ボ",
}

# ライマンの法則: 読みにこれらの文字が含まれていれば連濁しない
_VOICED_OBSTRUENTS = frozenset("ガギグゲゴザジズゼゾダヂヅデドバビブベボ")

# LR_LOW 表示用: この自信度未満を「低自信度LR」として色分けする
_LR_LOW_THRESHOLD = 0.5

# 助詞として読む場合にのみ生じる合法的な逸脱: {ひらがな原文: 許容カタカナ予測}
_PARTICLE_DEVIATIONS: Dict[str, str] = {
    "は": "ワ",
    "へ": "エ",
}


def _repeat_rendaku_form(reading: str) -> Optional[str]:
    """々連濁形を返す。連濁できない場合（対象外行・ライマンの法則）は None。"""
    voiced = _RENDAKU_MAP.get(reading[0] if reading else "")
    if voiced is None:
        return None
    if any(c in _VOICED_OBSTRUENTS for c in reading):
        return None
    return voiced + reading[1:]


def _hiragana_direct(char: str) -> str:
    """ひらがなユニットをカタカナに直接変換する。"""
    return "".join(convert_to_katakana(c) for c in char)


def _is_valid_kana_prediction(src_char: str, pred: str, direct: str) -> bool:
    """モデル予測がひらがなに対して合法的かどうかを判定する。"""
    if pred == direct:
        return True
    if _PARTICLE_DEVIATIONS.get(src_char) == pred:
        return True
    if src_char == "う" and pred == "ー":
        return True
    return False


# bypass（素通し）扱いにする文字種
_BYPASS_CTYPES = frozenset(
    {
        CharType.SPACE,
        CharType.SYMBOL,
        CharType.SYMBOL_CLOSE,
        CharType.SYMBOL_OPEN,
        CharType.SYMBOL_STOP,
        CharType.SYMBOL_PAUSE,
        CharType.OTHER,
    }
)


def _load_single_kanji_dict(path: str) -> Dict[str, List[str]]:
    """単一漢字辞書TSVを読み込む。
    フォーマット: 漢字[TAB]読み1[TAB]読み2[TAB]...
    """
    with open(path, encoding="utf-8") as f:
        return parse_kanji_dict_tsv(f.read())


# 🌟 訓読みを誘発しやすい「安全な送り仮名」のリスト
SAFE_OKURIGANA = set(
    "あいうえおかきくけこたちつてとなにぬねのまみむめもやゆよらりるれろわばびぶべぼ"
)

# 🌟 漢数字→アラビア数字の単純置換テーブル
_DIGIT_TABLE = {
    "〇": "0",
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "壱": "1",
    "弐": "2",
    "参": "3",
}

# 🌟 位取り文字の読みテーブル（数字展開しない場合）
_KURAI_READING = {
    "千": "セン",
    "万": "マン",
    "億": "オク",
    "兆": "チョー",
}

# 🌟 前の数字を訓読みに変える助数詞シグナル (助数詞の原文, その読み)。
# 漢数字が低自信度でも、直後がこれらならば数字化せずモデルの読みを信用する
# （例: 三日経って の 三 は、後続の 日→カ を見て ミッ を維持）。
# 助数詞の読み自体が曖昧性を解決する（人→リ＝ふたり は信用、人→ニン＝さんにん は数字化）。
# 前の数字の読みを変える助数詞は少数なので、有限の例外集合として持つ。
_KUN_COUNTER_SIGNALS = {
    ("日", "カ"),  # みっか・ふつか・ようか…
    ("日", "タチ"),  # ついたち（一日）
    ("つ", "ツ"),  # ひとつ・みっつ…
    ("人", "リ"),  # ひとり・ふたり
}


def _is_digit_label(label: str) -> bool:
    """ラベルが（半角/全角）数字のみで構成されるか。"""
    return label != "" and all(("0" <= c <= "9") or ("０" <= c <= "９") for c in label)


# ==========================================
# 🌟 設定
# ==========================================
@dataclass
class PredictorConfig:
    """Predictorの設定をまとめた構造体。

    Attributes:
        model_path: モデルファイルのパス（.zip）
        window: 特徴量ウィンドウサイズ（4, 5, 7）
        single_kanji_dict_path: 単一漢字辞書jsonのパス（省略可）
        custom_dict_path: カスタム辞書ファイルのパス（省略可）
        person_name_dict_path: 人名辞書ファイルのパス（省略可）。
            指定すると辞書マッチによる人名B/Iフラグ特徴量を付与する。
            省略時はモデルファイルと同じディレクトリの person_name_dic.tsv を
            自動検出する。人名特徴量付きで学習したモデルには、学習時と
            同じ辞書を渡すこと。
        numeric_confidence_threshold: JAPANESE_NUMERICルールベース変換を発動させる自信度の上限
        explain_top_n: トレース時に表示する特徴量寄与度の上位件数（0=無効）
    """

    model_path: Optional[str] = None
    window: int = 7
    single_kanji_dict_path: Optional[str] = None
    custom_dict_path: Optional[str] = None
    person_name_dict_path: Optional[str] = None
    numeric_confidence_threshold: float = 0.5
    explain_top_n: int = 8


# ==========================================
# 🌟 決定根拠タグ
# ==========================================
class DecisionSource:
    LR = "LR"  # LRの予測をそのまま採用
    LR_LOW = "LR_LOW"  # 自信度は低いがLRを採用
    FALLBACK_KANJI = "FALLBACK_KANJI"  # KANJI低自信度フォールバック辞書
    FALLBACK_NUMERIC = "FALLBACK_NUM"  # JAPANESE_NUMERICルールベース変換
    FALLBACK_ORPHAN = "FALLBACK_ORPH"  # 文字消失バグ救済ロジック
    FALLBACK_REPEAT = "FALLBACK_々"  # 々の繰り返し処理
    FALLBACK_KANA = "FALLBACK_KANA"  # かな直接変換フォールバック
    DICT = "DICT"  # カスタム辞書による強制置換
    DICT_NAME = "DICT_NAME"  # 人名辞書読み（低自信度フォールバック）
    BYPASS = "BYPASS"  # ASCIIバイパス


# ターミナル表示用ANSIカラー
_ANSI = {
    DecisionSource.LR: "\033[32m",  # 緑
    DecisionSource.LR_LOW: "\033[33m",  # 黄
    DecisionSource.FALLBACK_KANJI: "\033[35m",  # マゼンタ
    DecisionSource.FALLBACK_NUMERIC: "\033[36m",  # シアン
    DecisionSource.FALLBACK_ORPHAN: "\033[31m",  # 赤
    DecisionSource.FALLBACK_REPEAT: "\033[35m",  # マゼンタ
    DecisionSource.FALLBACK_KANA: "\033[96m",  # 明るいシアン
    DecisionSource.DICT: "\033[34m",  # 青
    DecisionSource.DICT_NAME: "\033[94m",  # 明るい青
    DecisionSource.BYPASS: "\033[90m",  # グレー
}
_ANSI_RESET = "\033[0m"
_ANSI_DIM = "\033[2m"


def _kurai_fallback(
    char: str, left_char: str, left_ctype: str, right_ctype: str
) -> str:
    """位取り文字（十百千万億兆）のフォールバック変換ルール。"""
    is_numeric = CharType.JAPANESE_NUMERIC
    left_is_numeric = left_ctype == is_numeric
    right_is_numeric = right_ctype == is_numeric

    if char == "十":
        if left_is_numeric and right_is_numeric:
            return LABEL_SKIP
        if left_is_numeric and not right_is_numeric:
            return "0"
        if not left_is_numeric and right_is_numeric:
            return "1"
        return "10"

    if char == "百":
        if left_is_numeric and right_is_numeric:
            return "0"
        if left_is_numeric and not right_is_numeric:
            return "00"
        if not left_is_numeric and right_is_numeric:
            return "1"
        return "100"

    if char == "千":
        if right_is_numeric:
            return "0"
        return "ゼン" if left_char == "三" else "セン"

    return _KURAI_READING.get(char, char)


# ==========================================
# 🌟 カスタム辞書のロードとインデックス構築
# ==========================================
# カスタム辞書のユニット別エントリ: (読み, 直後で区切るか)
DictUnit = Tuple[str, bool]


def load_custom_dict(path: str) -> Dict[str, List[DictUnit]]:
    """カスタム辞書を読み込む。

    フォーマット: 表層形[TAB]読み。読みは学習データ（basic_raw.tsv）と同じ
    記法で、ユニット別に '/' 区切り、空ブロック（`/ /`）は「直前のユニットの
    後ろで区切る」ことを表す（例: `佐藤太郎  サ/トー/ /タ/ロー`）。

    エントリ内部の境界は辞書が確定する（空ブロックの位置でだけ区切り、
    それ以外は区切らない）。エントリ両端の境界は次に来る語に依存するため
    境界モデルに委ねる（先頭・末尾の区切り指定はエラー）。
    """
    result: Dict[str, List[DictUnit]] = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"カスタム辞書 {path} の {lineno} 行目の形式が不正です: {line!r}"
                )
            surface, reading_str = parts
            blocks = [
                b.replace(r"\/", "/") for b in split_on_unescaped_slash(reading_str)
            ]

            units: List[DictUnit] = []
            for block in blocks:
                if block == " ":
                    if not units:
                        raise ValueError(
                            f"カスタム辞書 {path} の {lineno} 行目: "
                            "エントリ先頭に区切り指定（/ /）は置けません"
                            "（両端の境界は文脈依存のため境界モデルが判定します）。"
                        )
                    if units[-1][1]:
                        raise ValueError(
                            f"カスタム辞書 {path} の {lineno} 行目: "
                            "区切り指定（/ /）が連続しています。"
                        )
                    units[-1] = (units[-1][0], True)
                elif block == "":
                    raise ValueError(
                        f"カスタム辞書 {path} の {lineno} 行目: "
                        f"読みに空のブロックが含まれています: {reading_str!r}"
                    )
                else:
                    units.append((block, False))

            if units and units[-1][1]:
                raise ValueError(
                    f"カスタム辞書 {path} の {lineno} 行目: "
                    "エントリ末尾に区切り指定（/ /）は置けません"
                    "（両端の境界は文脈依存のため境界モデルが判定します）。"
                )

            surface_units = [u[0] for u in get_units(surface)]
            if len(units) != len(surface_units):
                raise ValueError(
                    f"カスタム辞書 {path} の {lineno} 行目: "
                    f"表層形 {surface!r} のユニット数（{len(surface_units)}）と "
                    f"読みブロック数（{len(units)}、区切り指定は数えない）が一致しません。"
                )
            result[surface] = units
    return result


def build_dict_index(
    custom_dict: Dict[str, List[DictUnit]],
) -> Dict[str, List[Tuple[str, List[DictUnit]]]]:
    index: Dict[str, List[Tuple[str, List[DictUnit]]]] = {}
    for surface, readings in custom_dict.items():
        key = surface[0]
        index.setdefault(key, []).append((surface, readings))
    for key in index:
        index[key].sort(key=lambda x: len(x[0]), reverse=True)
    return index


def find_longest_match(
    index: Dict[str, List[Tuple[str, List[DictUnit]]]],
    source_seq: List[SourceEntry],
    pos: int,
) -> Optional[Tuple[int, List[DictUnit]]]:
    char = source_seq[pos][0]
    first_char = char[0]  # 複合ユニットの場合は先頭1文字でインデックス引き
    candidates = index.get(first_char)
    if not candidates:
        return None
    for surface, readings in candidates:
        # surfaceをget_units()でユニット分解してunit単位で比較
        surface_units = [u[0] for u in get_units(surface)]
        length = len(surface_units)
        if pos + length > len(source_seq):
            continue
        if all(source_seq[pos + j][0] == surface_units[j] for j in range(length)):
            return length, readings
    return None


@dataclass
class PredictionResult:
    source_text: str
    kana_text: str
    confidences: List[float]
    kana_to_src_index: List[int]
    src_to_kana_index: List[List[int]]
    decision_sources: List[str] = field(default_factory=list)
    # 各ソース文字位置に対する特徴量寄与度リスト [(特徴量名, 寄与度), ...]
    # explain_top_n=0 のときは空リスト
    feature_contributions: List[List[Tuple[str, float]]] = field(default_factory=list)

    def format_segmented(self) -> str:
        """各ソース文字に対応するカタカナ部分を'/'で区切って返す。

        例: "とても良い天気" → "ト/テ/モ/ /ヨ/イ/テン/キ/"
        境界予測による空白も独立したセグメントとして扱う。
        """
        segments: List[str] = []
        for i in range(len(self.source_text)):
            kana_positions = self.src_to_kana_index[i]
            if not kana_positions:
                continue
            current = ""
            for p in kana_positions:
                ch = self.kana_text[p]
                if ch == " ":
                    if current:
                        segments.append(current)
                        current = ""
                    segments.append(" ")
                else:
                    current += ch
            if current:
                segments.append(current)
        return "/".join(segments)

    def format_source_segmented(self) -> str:
        """点字の分かち書きに沿って原文を分割し、' 'で区切って返す。

        境界モデルが予測した単語境界（仮名列のスペース）に基づいて原文を分割する。
        境界がまったくない場合は原文全体を1セグメントとして返す。

        例:
            "東京都民のために" → "東京 都民の ために"
        """
        segments = self.get_source_segments()
        if not segments:
            return ""
        return "/".join(segments)

    def get_source_segments(self) -> List[str]:
        """点字の分かち書きに沿って原文を分割したリストを返す。

        境界モデルが予測した単語境界（仮名列のスペース）に基づいて原文を分割する。
        境界がまったくない場合は原文全体を含む1要素のリストを返す。

        例:
            "東京都民のために" → ["東京", "都民", "の", "ために"]
        """
        if not self.source_text:
            return []

        segments: List[str] = []
        current_start = 0

        for k, ch in enumerate(self.kana_text):
            if ch == " ":
                # kana_to_src_index[k] は、このスペースを生んだ原文文字のインデックス
                src_i = self.kana_to_src_index[k]
                end = src_i + 1
                seg = self.source_text[current_start:end]
                if seg:
                    segments.append(seg)
                current_start = end

        # 末尾の残りセグメント
        remaining = self.source_text[current_start:]
        if remaining:
            segments.append(remaining)

        return segments

    def to_json(self) -> str:
        text_safe = json.dumps(self.source_text, ensure_ascii=False)
        kana_safe = json.dumps(self.kana_text, ensure_ascii=False)

        conf_str = "[" + ", ".join([f"{c:.3f}" for c in self.confidences]) + "]"
        k2s_str = "[" + ", ".join(map(str, self.kana_to_src_index)) + "]"
        s2k_str = json.dumps(self.src_to_kana_index)
        dec_safe = json.dumps(self.decision_sources, ensure_ascii=False)

        return (
            f"{{\n"
            f'  "text": {text_safe},\n'
            f'  "kana": {kana_safe},\n'
            f'  "kana_to_src_index": {k2s_str},\n'
            f'  "src_to_kana_index": {s2k_str},\n'
            f'  "confidences": {conf_str},\n'
            f'  "decision_sources": {dec_safe}\n'
            f"}}"
        )

    def format_terminal(self, use_color: bool = True) -> str:
        src_len = len(self.source_text)

        src_kana: Dict[int, List[str]] = {i: [] for i in range(src_len)}
        src_conf: Dict[int, List[float]] = {i: [] for i in range(src_len)}
        src_dec: Dict[int, List[str]] = {i: [] for i in range(src_len)}

        for kana_i, (src_i, conf, dec) in enumerate(
            zip(self.kana_to_src_index, self.confidences, self.decision_sources)
        ):
            if 0 <= src_i < src_len:
                src_kana[src_i].append(self.kana_text[kana_i])
                src_conf[src_i].append(conf)
                src_dec[src_i].append(dec)

        rows = []
        for src_i, char in enumerate(self.source_text):
            kanas = src_kana[src_i]
            confs = src_conf[src_i]
            decs = src_dec[src_i]

            if not kanas:
                rows.append(f"  {char}  →  (skip)")
                continue

            kana_str = "".join(kanas)
            conf_val = min(confs)
            dec_tag = decs[0]

            tag_str = f"[{dec_tag:<13}]"
            bar = _confidence_bar(conf_val)

            if dec_tag == DecisionSource.BYPASS:
                line = f"  {char}  →  {kana_str:<6} {tag_str}"
            else:
                line = f"  {char}  →  {kana_str:<6} {tag_str} {bar} {conf_val:.3f}"

            if use_color:
                line = _ANSI.get(dec_tag, "") + line + _ANSI_RESET

            rows.append(line)

            # 特徴量寄与度の表示
            if self.feature_contributions and src_i < len(self.feature_contributions):
                contribs = self.feature_contributions[src_i]
                if contribs:
                    for feat, score in contribs:
                        contrib_line = f"      {feat:<38s} {score:+.4f}"
                        if use_color:
                            contrib_line = _ANSI_DIM + contrib_line + _ANSI_RESET
                        rows.append(contrib_line)

        return "\n".join(rows)


def _confidence_bar(conf: float, width: int = 12) -> str:
    filled = round(conf * width)
    return "█" * filled + "░" * (width - filled)


def _remap_result_to_source(
    result: PredictionResult, source_text: str, norm_to_src: List[int]
) -> None:
    """正規化テキスト基準のインデックスを原文（正規化前）基準へ戻す。

    normalize_input() は欧文リガチャ等で1→N展開を行うため、パイプラインが
    参照する文字位置は原文とずれ得る。source_text を原文へ差し替え、
    kana_to_src_index / src_to_kana_index を原文の文字位置に変換する。
    展開で同じ原文文字に複数の正規化後文字が対応する場合、その仮名位置は
    すべて原文の1文字に集約される。
    """
    if result.source_text == source_text:
        return
    result.kana_to_src_index = [norm_to_src[i] for i in result.kana_to_src_index]
    src_to_kana: List[List[int]] = [[] for _ in source_text]
    for norm_idx, kana_positions in enumerate(result.src_to_kana_index):
        src_to_kana[norm_to_src[norm_idx]].extend(kana_positions)
    result.src_to_kana_index = src_to_kana
    result.source_text = source_text


class Predictor:
    def __init__(self, config: PredictorConfig):
        if config.model_path:
            model_path = config.model_path
            if not model_path.endswith(".zip"):
                model_path += ".zip"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"❌ モデル未検出: {model_path}")
            zip_source: str | io.BytesIO = model_path
        else:
            zip_bytes = (
                resources.files("momo_py") / f"resources/basic_data_{config.window}.zip"
            ).read_bytes()
            zip_source = io.BytesIO(zip_bytes)

        self._config = config
        self._version_info: dict[str, Any] | None = None

        with zipfile.ZipFile(zip_source, "r") as zf:
            namelist = zf.namelist()

            if "version_info.json" not in namelist:
                raise ValueError(
                    f"❌ ZIPファイル内に version_info.json が見つかりません: {zip_source}"
                )

            self._version_info = json.loads(
                zf.read("version_info.json").decode("utf-8")
            )

            bundle_name = self._version_info.get("model_bundle")
            if bundle_name is None:
                raise ValueError(
                    f"❌ version_info.json に model_bundle キーがありません: {zip_source}"
                )
            if bundle_name not in namelist:
                raise ValueError(
                    f"❌ ZIPファイル内に {bundle_name} が見つかりません: {zip_source}"
                )

            # 学習時に同梱された人名辞書（train が書き込む。旧モデルには無い）
            embedded_name_dict_text: Optional[str] = None
            if NAME_DICT_FILENAME in namelist:
                embedded_name_dict_text = zf.read(NAME_DICT_FILENAME).decode("utf-8")

            # 学習時に同梱された単一漢字辞書（同上）
            embedded_single_kanji_text: Optional[str] = None
            if SINGLE_KANJI_DICT_FILENAME in namelist:
                embedded_single_kanji_text = zf.read(SINGLE_KANJI_DICT_FILENAME).decode(
                    "utf-8"
                )

            bundle: LRModelBundle = joblib.load(io.BytesIO(zf.read(bundle_name)))
            self._vectorizer_read = bundle.vectorizer_read
            self._vectorizer_boundary = bundle.vectorizer_boundary
            # 係数を float32 に変換する。変換直後に元の float64 を手放すことで
            # 「float64 と float32 の一時的な共存」によるスパイクを最小化する。
            self._coef_read_sparse = bundle.coef_read_sparse.astype(np.float32)
            del bundle.coef_read_sparse  # type: ignore[attr-defined]
            gc.collect()
            self._intercept_read = bundle.intercept_read.astype(np.float32)
            self._read_classes = bundle.read_classes
            self._model_boundary = bundle.model_boundary
            self._window_size: int = self._version_info.get("window_size", 5)
            del bundle
        # vectorizer の出力を float32 に固定する（推論ごとの astype コピーを回避）
        self._vectorizer_read.dtype = np.float32  # type: ignore[attr-defined]
        self._vectorizer_boundary.dtype = np.float32  # type: ignore[attr-defined]
        gc.collect()

        # 逆引き辞書（explain用）: 実際に explain が呼ばれるまで構築しない
        self._vocab_inv: Optional[Dict[int, str]] = None

        # 単一漢字辞書のロード（常にロードし、推論時の候補制約に使う）
        # 優先順位: 明示指定 > モデルZIP同梱（学習時と同一） > パッケージ内蔵
        if config.single_kanji_dict_path:
            self._single_kanji_dict: Dict[str, List[str]] = _load_single_kanji_dict(
                config.single_kanji_dict_path
            )
        elif embedded_single_kanji_text is not None:
            self._single_kanji_dict = parse_kanji_dict_tsv(embedded_single_kanji_text)
        else:
            tsv_text = (
                resources.files("momo_py") / f"resources/{SINGLE_KANJI_DICT_FILENAME}"
            ).read_text(encoding="utf-8")
            self._single_kanji_dict = parse_kanji_dict_tsv(tsv_text)

        # カスタム辞書のロードとインデックス構築
        self._dict_index: Dict[str, List[Tuple[str, List[DictUnit]]]] = {}
        if config.custom_dict_path:
            custom_dict = load_custom_dict(config.custom_dict_path)
            self._dict_index = build_dict_index(custom_dict)

        # 人名辞書のロード（B/Iフラグ特徴量用。学習時と同じ辞書マッチを適用する）
        # 優先順位:
        #   1. config.person_name_dict_path の明示指定
        #   2. モデルZIPに同梱された辞書（学習時と同一であることが保証される）
        #   3. モデルファイルと同じディレクトリの person_name_dic.tsv
        self._name_index: NameIndex = {}
        if config.person_name_dict_path:
            self._name_index = build_name_index(
                load_name_dict(config.person_name_dict_path)
            )
        elif embedded_name_dict_text is not None:
            self._name_index = build_name_index(
                parse_name_dict_text(embedded_name_dict_text)
            )
        else:
            candidate = (
                os.path.join(
                    os.path.dirname(os.path.abspath(model_path)), NAME_DICT_FILENAME
                )
                if config.model_path
                else None
            )
            if candidate and os.path.isfile(candidate):
                self._name_index = build_name_index(load_name_dict(candidate))
            elif (self._version_info or {}).get("name_dict_entries"):
                print(
                    "⚠️  このモデルは人名辞書特徴量付きで学習されていますが、"
                    "人名辞書が指定されていません。人名まわりの分かち書き精度が低下します。",
                    file=sys.stderr,
                )

    def get_version_info(self) -> dict[str, Any] | None:
        return self._version_info

    # ==========================================
    # 🌟 特徴量寄与度の計算
    # ==========================================
    def _explain_position(
        self,
        src_features: List[dict],
        i: int,
        label: str,
    ) -> List[Tuple[str, float]]:
        """i番目の文字について、予測ラベルへの特徴量寄与度上位を返す。

        寄与度 = 特徴量の値 × そのラベルに対応する重み係数
        正の値はそのラベルを支持し、負の値は抑制することを意味する。
        """
        top_n = self._config.explain_top_n
        if top_n <= 0:
            return []

        idx = np.searchsorted(self._read_classes, label)
        if idx >= len(self._read_classes) or self._read_classes[idx] != label:
            return []

        # 逆引き辞書を初回呼び出し時だけ構築する（遅延初期化）
        if self._vocab_inv is None:
            self._vocab_inv = {  # type: ignore[union-attr]
                v: k
                for k, v in self._vectorizer_read.vocabulary_.items()  # type: ignore[union-attr]
            }

        X = self._vectorizer_read.transform([src_features[i]])
        # そのラベルの重みベクトルを取得
        coef_for_label = np.asarray(self._coef_read_sparse[idx].todense()).flatten()

        # 寄与度 = 特徴量値 × 重み（特徴量値は0/1なので実質的に重みそのもの）
        contrib = X.toarray().flatten() * coef_for_label

        # 上位top_n件（寄与度の絶対値が大きい順）
        nonzero_idx = np.nonzero(contrib)[0]
        if len(nonzero_idx) == 0:
            return []

        sorted_idx = nonzero_idx[np.argsort(contrib[nonzero_idx])[::-1]][:top_n]
        return [
            (self._vocab_inv.get(j, f"feat_{j}"), float(contrib[j])) for j in sorted_idx
        ]

    # ==========================================
    # 🌟 推論パイプライン（The Conductor）
    # ==========================================
    def predict(self, text: str) -> PredictionResult:
        if not text:
            return PredictionResult("", "", [], [], [])
        # 正規化（欧文リガチャ等の1→N展開を含む）。パイプラインは正規化後
        # テキスト基準で動き、最後に _remap_result_to_source() で原文基準に戻す。
        source_text = text
        text, norm_to_src = normalize_input(text)

        source_seq, bypass_indices, ascii_overrides, dict_overrides, dict_boundaries = (
            self._preprocess_text(text)
        )
        if not source_seq:
            return PredictionResult(source_text, "", [], [], [[] for _ in source_text])

        (
            raw_labels,
            boundary_labels,
            max_scores,
            boundary_proba,
            src_features,
            name_readings,
        ) = self._run_inference(source_seq)

        refined_labels, raw_confidences, has_splits, decision_sources = (
            self._refine_predictions(
                source_seq,
                raw_labels,
                boundary_labels,
                max_scores,
                boundary_proba,
                bypass_indices,
                ascii_overrides,
                dict_overrides,
                dict_boundaries,
                name_readings,
            )
        )

        # 特徴量寄与度の計算（explain_top_n > 0 のときのみ）
        feature_contributions: List[List[Tuple[str, float]]] = []
        if self._config.explain_top_n > 0:
            for i, label in enumerate(refined_labels):
                if i in bypass_indices or label in (LABEL_CONTINUE, LABEL_SKIP):
                    feature_contributions.append([])
                else:
                    feature_contributions.append(
                        self._explain_position(src_features, i, label)
                    )

        result = self._assemble_result(
            text,
            source_seq,
            refined_labels,
            raw_confidences,
            has_splits,
            bypass_indices,
            decision_sources,
        )
        result.feature_contributions = feature_contributions
        _remap_result_to_source(result, source_text, norm_to_src)
        return result

    # ==========================================
    # 🌟 ステップ 1: 前処理
    # ==========================================
    def _preprocess_text(
        self, text: str
    ) -> Tuple[
        List[SourceEntry], Set[int], Dict[int, str], Dict[int, str], Dict[int, bool]
    ]:
        units_info = get_units(text)
        source_seq: List[SourceEntry] = []
        bypass_indices: Set[int] = set()
        ascii_overrides: Dict[int, str] = {}

        char_idx = 0
        for val, orig_idx, ctype in units_info:
            # スキップ文字（NBSP・ZWJ など）: 仮名に現れず原文インデックスのみ保持
            if ctype == CharType.SKIP:
                for i, c in enumerate(val):
                    source_seq.append((c, orig_idx + i, ctype))
                    bypass_indices.add(char_idx)
                    ascii_overrides[char_idx] = LABEL_SKIP
                    char_idx += 1
                continue

            # 英数字ブロック、または記号類・空白・未定義文字は bypass 扱い
            is_ascii_bypass = ctype == "ALPHA"
            is_symbol_bypass = ctype in _BYPASS_CTYPES

            if is_ascii_bypass or is_symbol_bypass:
                # bypass は従来通り1文字ずつ展開
                for i, c in enumerate(val):
                    source_seq.append((c, orig_idx + i, ctype))
                    if is_ascii_bypass:
                        bypass_indices.add(char_idx)
                        ascii_overrides[char_idx] = val if i == 0 else LABEL_CONTINUE
                    else:
                        bypass_indices.add(char_idx)
                        ascii_overrides[char_idx] = c  # 記号は1文字ずつそのまま
                    char_idx += 1
            else:
                # 複合ユニット（拗音・漢字語など）は1エントリとして保持（展開しない）
                source_seq.append((val, orig_idx, ctype))
                char_idx += 1

        dict_overrides: Dict[int, str] = {}
        # 辞書エントリ内部の境界（True=区切る / False=区切らない）。
        # エントリ末尾ユニットは登録しない = 境界モデルに委ねる。
        dict_boundaries: Dict[int, bool] = {}
        if self._dict_index:
            i = 0
            while i < len(source_seq):
                if i in bypass_indices:
                    i += 1
                    continue
                match = find_longest_match(self._dict_index, source_seq, i)
                if match:
                    length, readings = match
                    for j, (reading, split_after) in enumerate(readings):
                        dict_overrides[i + j] = reading
                        if j < length - 1:
                            dict_boundaries[i + j] = split_after
                    i += length
                else:
                    i += 1

        return (
            source_seq,
            bypass_indices,
            ascii_overrides,
            dict_overrides,
            dict_boundaries,
        )

    # ==========================================
    # 🌟 ステップ 2: 推論（LinearSVC版）
    # ==========================================
    def _run_inference(
        self, source_seq: List[SourceEntry]
    ) -> Tuple[
        List[str],
        List[str],
        np.ndarray,
        np.ndarray,
        List[Dict[str, float]],
        List[Optional[str]],
    ]:
        """
        読みモデル（LinearSVC）と境界モデル（SGDClassifier）を推論する。

        戻り値:
            raw_labels      : 読みラベル列
            boundary_labels : 境界ラベル列（"0" or "1"）
            max_scores      : 各文字の予測クラスに対する decision score（shape=(n,)）
                              全クラス分の密行列は作らず、argmax クラスのスコアのみ保持する
            boundary_proba  : 境界の確率 shape=(n, 2)
            src_features    : 特徴量辞書リスト（explain用）
            name_readings   : 人名辞書のユニット別読み（低自信度フォールバック用）
        """
        if self._name_index:
            name_flags, name_readings = compute_name_matches(
                source_seq, self._name_index
            )
        else:
            name_flags = None
            name_readings = [None] * len(source_seq)
        src_features = compute_source_features(
            source_seq, window=self._window_size, name_flags=name_flags
        )
        # vectorizer.dtype=float32 をロード時に設定済みなので、transform は直接 float32 を返す
        X_read = self._vectorizer_read.transform(src_features)
        X_boundary = self._vectorizer_boundary.transform(src_features)

        # 読みモデル（LinearSVC）
        # scores_T: (n_classes, n_chars) の疎行列のまま計算する
        scores_T = (self._coef_read_sparse @ X_read.T).toarray()  # type: ignore[union-attr]  # (n_classes, n_chars), float32
        scores_T += self._intercept_read[:, np.newaxis]  # intercept を in-place で加算
        # 漢字辞書制約：辞書に載っている漢字は既知の読み+CONTINUE+SKIPのみを候補とする
        for col_idx, (char, _, ctype) in enumerate(source_seq):
            if ctype == "KANJI" and char in self._single_kanji_dict:
                valid = set(self._single_kanji_dict[char])
                mask = np.array(
                    [cls not in valid for cls in self._read_classes], dtype=bool
                )
                scores_T[mask, col_idx] = -np.inf

        predicted_indices = scores_T.argmax(axis=0)  # (n_chars,)
        # 予測クラスのスコアだけを 1D 配列として取り出し、全クラス分の行列は即座に解放する
        n = scores_T.shape[1]
        max_scores = scores_T[predicted_indices, np.arange(n)]  # (n_chars,), float32
        del scores_T
        raw_labels = list(self._read_classes[predicted_indices])

        # 境界モデル（SGDClassifier）
        boundary_raw = list(self._model_boundary.predict(X_boundary))
        boundary_labels = [str(b) for b in boundary_raw]
        boundary_proba = self._model_boundary.predict_proba(X_boundary)

        return (
            raw_labels,
            boundary_labels,
            max_scores,
            boundary_proba,
            src_features,
            name_readings,
        )

    def _get_read_confidence(self, max_scores: np.ndarray, i: int) -> float:
        """
        読みモデルの予測クラスに対する decision score を sigmoid で 0〜1 に変換して返す。
        max_scores[i] は _run_inference で計算した「予測クラスのスコア」（intercept込み）。
        """
        return float(1.0 / (1.0 + np.exp(-float(max_scores[i]))))

    # ==========================================
    # 🌟 ステップ 3: 後処理
    # ==========================================
    def _refine_predictions(
        self,
        source_seq: List[SourceEntry],
        raw_labels: List[str],
        boundary_labels: List[str],
        max_scores: np.ndarray,
        boundary_proba: np.ndarray,
        bypass_indices: Set[int],
        ascii_overrides: Dict[int, str],
        dict_overrides: Dict[int, str],
        dict_boundaries: Dict[int, bool],
        name_readings: List[Optional[str]],
    ) -> Tuple[List[str], List[float], List[bool], List[str]]:
        refined_labels: List[str] = []
        confidences: List[float] = []
        has_splits: List[bool] = []
        decision_sources: List[str] = []
        last_fallback_reading = ""
        parent_idx = -1

        for i, (char, _, ctype) in enumerate(source_seq):
            raw_clean = raw_labels[i]

            if raw_clean not in (LABEL_CONTINUE, LABEL_SKIP):
                parent_idx = i

            if i in bypass_indices:
                clean_label = ascii_overrides[i]
                confidence = 1.0
                decision = DecisionSource.BYPASS
                last_fallback_reading = ""
                has_split = False

            elif i in dict_overrides:
                clean_label = dict_overrides[i]
                confidence = 1.0
                decision = DecisionSource.DICT
                last_fallback_reading = ""
                # エントリ内部の境界は辞書が確定する（/ / 記法）。
                # エントリ末端の境界は次語に依存するため境界モデルに委ねる。
                if i in dict_boundaries:
                    has_split = dict_boundaries[i]
                else:
                    has_split = boundary_labels[i] == "1"

            else:
                label = raw_labels[i]
                decision = DecisionSource.LR

                confidence = self._get_read_confidence(max_scores, i)

                # 文字消失バグの救済ロジック
                # 親がいない・バイパス・KANJI以外の場合、CONTINUEは不正
                parent_is_kanji = (
                    parent_idx >= 0
                    and parent_idx not in bypass_indices
                    and source_seq[parent_idx][2] == "KANJI"
                )
                if label == LABEL_CONTINUE and not parent_is_kanji:
                    if ctype == "KANJI" and char in self._single_kanji_dict:
                        label = self._single_kanji_dict[char][0]
                    else:
                        label = char
                    confidence = 0.0
                    decision = DecisionSource.FALLBACK_ORPHAN

                elif ctype == CharType.KATAKANA:
                    if char not in _KATAKANA_NO_PASSTHROUGH:
                        label = char
                        confidence = 1.0
                        decision = DecisionSource.FALLBACK_KANA
                    last_fallback_reading = ""

                elif ctype == CharType.HIRAGANA:
                    if label not in (LABEL_CONTINUE, LABEL_SKIP):
                        direct = _hiragana_direct(char)
                        if not _is_valid_kana_prediction(char, label, direct):
                            label = direct
                            confidence = 1.0
                            decision = DecisionSource.FALLBACK_KANA
                    last_fallback_reading = ""

                elif ctype == CharType.JAPANESE_NUMERIC:
                    label, last_fallback_reading, confidence, decision = (
                        self._convert_japanese_numeric(
                            i,
                            char,
                            label,
                            confidence,
                            source_seq,
                            raw_labels,
                            last_fallback_reading,
                        )
                    )
                elif ctype == CharType.NUMERIC:
                    last_fallback_reading = ""
                    label, confidence, decision = self._convert_numeric(
                        i,
                        char,
                        label,
                        confidence,
                        source_seq,
                        raw_labels,
                    )
                else:
                    label, last_fallback_reading, _, decision = (
                        self._apply_kanji_fallback(
                            char,
                            ctype,
                            label,
                            last_fallback_reading,
                        )
                    )
                    # 人名辞書読みフォールバック:
                    # 人名スパン内 かつ モデルが低自信度のときだけ、辞書の固定読みで
                    # 置換する。自信のある予測は尊重するので「湧き出る清水」のような
                    # 同表層の別語（非人名）を壊さない。
                    if (
                        decision == DecisionSource.LR
                        and name_readings[i] is not None
                        and confidence < _LR_LOW_THRESHOLD
                        and label not in (LABEL_CONTINUE, LABEL_SKIP)
                        and label != name_readings[i]
                    ):
                        label = name_readings[i]
                        decision = DecisionSource.DICT_NAME
                        if ctype == "KANJI":
                            last_fallback_reading = label  # 々のために伝播
                    if decision == DecisionSource.LR:
                        if confidence < _LR_LOW_THRESHOLD and ctype == "KANJI":
                            decision = DecisionSource.LR_LOW

                clean_label = label

                # NUMERIC型がSKIP/CONTINUEになった場合、元の文字を素通し
                if (
                    clean_label in (LABEL_SKIP, LABEL_CONTINUE)
                    and ctype == CharType.NUMERIC
                ):
                    clean_label = char
                    confidence = 0.0
                    decision = DecisionSource.FALLBACK_ORPHAN

                # 小書き仮名がCONTINUEになった場合の救済：
                # 本来は親ラベルに小書き仮名が含まれているはずだが、
                # モデルの誤推論で含まれていない場合は親ラベルに追記する
                # （SKIPは意図的なので救済しない）
                elif clean_label == LABEL_CONTINUE and char in _SMALL_KANA:
                    if parent_idx >= 0 and _SMALL_KANA.isdisjoint(
                        refined_labels[parent_idx]
                    ):
                        kana_char = (
                            chr(ord(char) + 0x60) if "ぁ" <= char <= "ん" else char
                        )
                        refined_labels[parent_idx] += kana_char
                        decision_sources[parent_idx] = DecisionSource.FALLBACK_ORPHAN
                        clean_label = LABEL_CONTINUE
                        confidence = 0.0
                        decision = DecisionSource.FALLBACK_ORPHAN

                has_split = boundary_labels[i] == "1"

            refined_labels.append(clean_label)
            confidences.append(confidence)
            has_splits.append(has_split)
            decision_sources.append(decision)

        return refined_labels, confidences, has_splits, decision_sources

    def _convert_japanese_numeric(
        self,
        i: int,
        char: str,
        label: str,
        confidence: float,
        source_seq: List[SourceEntry],
        raw_labels: List[str],
        last_fallback: str,
    ) -> Tuple[str, str, float, str]:
        if confidence >= self._config.numeric_confidence_threshold:
            return label, "", confidence, DecisionSource.LR

        # 助数詞コヒーレンス: 直後が訓読み助数詞（日→カ・つ→ツ・人→リ 等）なら、
        # 低自信度でも数字化せずモデルの読み（ミッ 等）を信用する。
        n = len(source_seq)
        if i + 1 < n and i + 1 < len(raw_labels):
            next_signal = (source_seq[i + 1][0], raw_labels[i + 1])
            if next_signal in _KUN_COUNTER_SIGNALS:
                return label, "", confidence, DecisionSource.LR

        left_char = source_seq[i - 1][0] if i > 0 else ""
        left_ctype = source_seq[i - 1][2] if i > 0 else ""
        right_ctype = source_seq[i + 1][2] if i < len(source_seq) - 1 else ""

        if char in _DIGIT_TABLE:
            return _DIGIT_TABLE[char], "", confidence, DecisionSource.FALLBACK_NUMERIC

        return (
            _kurai_fallback(char, left_char, left_ctype, right_ctype),
            "",
            confidence,
            DecisionSource.FALLBACK_NUMERIC,
        )

    def _convert_numeric(
        self,
        i: int,
        char: str,
        label: str,
        confidence: float,
        source_seq: List[SourceEntry],
        raw_labels: List[str],
    ) -> Tuple[str, float, str]:
        """アラビア数字（NUMERIC）の採否ロジック。

        モデルを信用し、出力の種類で採否を決める:
          - モデルが数字を返した: 原文と一致→採用 / 不一致→原文の数字に戻す（数の崩壊防止）
          - モデルがかなを返した: 直後が既知の訓読み助数詞シグナル（日→カ・タチ、つ→ツ、
                                  人→リ）→採用（みっか・はつか等の助数詞読み） /
                                  それ以外→原文の数字に戻す

        助数詞シグナルが立つのは直後が具体的な訓読みの助数詞のときだけなので、数字が
        隣接する場合（多桁数の内部、例「120日」の 0）や、直後に何もない孤立した数字
        （例「3:5」の 5）は自然に revert される。
        """
        # CONTINUE/SKIP はラベルではないので数字をそのまま素通し。
        if label in (LABEL_CONTINUE, LABEL_SKIP):
            return char, 0.0, DecisionSource.FALLBACK_ORPHAN
        # ① モデルが数字を返した
        if _is_digit_label(label):
            if label != char:
                # 取り違え（例 7→1）→ 原文の数字に戻す
                return char, confidence, DecisionSource.FALLBACK_NUMERIC
            return label, confidence, DecisionSource.LR
        # ② モデルがかなを返した: 直後が訓読み助数詞シグナルのときだけ信用する
        n = len(source_seq)
        if i + 1 < n and i + 1 < len(raw_labels):
            next_signal = (source_seq[i + 1][0], raw_labels[i + 1])
            if next_signal in _KUN_COUNTER_SIGNALS:
                return label, confidence, DecisionSource.LR
        return char, confidence, DecisionSource.FALLBACK_NUMERIC

    def _apply_kanji_fallback(
        self,
        char: str,
        ctype: str,
        label: str,
        last_fallback: str,
    ) -> Tuple[str, str, bool, str]:
        is_applied = False
        new_label = label
        new_last_fallback = last_fallback
        decision = DecisionSource.LR

        if char == "々" and last_fallback:
            rendaku = _repeat_rendaku_form(last_fallback)
            valid: Set[str] = {last_fallback}
            if rendaku is not None:
                valid.add(rendaku)
            if label not in valid:
                new_label = last_fallback
                new_last_fallback = last_fallback
                is_applied = True
                decision = DecisionSource.FALLBACK_REPEAT

        if not is_applied and not ctype.startswith("SYMBOL"):
            if ctype == "KANJI":
                new_last_fallback = label  # LR高自信度でも々のために伝播
            else:
                new_last_fallback = ""

        return new_label, new_last_fallback, is_applied, decision

    # ==========================================
    # 🌟 ステップ 4: 結果の組み立て
    # ==========================================
    def _assemble_result(
        self,
        text: str,
        source_seq: List[SourceEntry],
        refined_labels: List[str],
        raw_confidences: List[float],
        has_splits: List[bool],
        bypass_indices: Set[int],
        decision_sources: List[str],
    ) -> PredictionResult:
        translated = ""
        kana_to_src_index: List[int] = []
        final_confidences: List[float] = []
        final_decision_sources: List[str] = []
        src_to_kana_index: List[List[int]] = [[] for _ in text]
        kana_pos = 0

        for i, (_source_char, orig_idx, _) in enumerate(source_seq):
            clean_label = refined_labels[i]
            confidence = raw_confidences[i]
            decision = decision_sources[i]

            if clean_label == LABEL_SKIP:
                pass
            elif clean_label != LABEL_CONTINUE:
                if i in bypass_indices:
                    for j, ch in enumerate(clean_label):
                        translated += ch
                        target_orig_idx = orig_idx + j
                        kana_to_src_index.append(target_orig_idx)
                        final_confidences.append(confidence)
                        final_decision_sources.append(decision)
                        src_to_kana_index[target_orig_idx].append(kana_pos)
                        kana_pos += 1
                else:
                    for ch in clean_label:
                        translated += ch
                        kana_to_src_index.append(orig_idx)
                        final_confidences.append(confidence)
                        final_decision_sources.append(decision)
                        src_to_kana_index[orig_idx].append(kana_pos)
                        kana_pos += 1

            if has_splits[i]:
                translated += " "
                kana_to_src_index.append(orig_idx)
                final_confidences.append(confidence)
                final_decision_sources.append(decision)
                src_to_kana_index[orig_idx].append(kana_pos)
                kana_pos += 1

        return PredictionResult(
            source_text=text,
            kana_text=translated,
            confidences=final_confidences,
            kana_to_src_index=kana_to_src_index,
            src_to_kana_index=src_to_kana_index,
            decision_sources=final_decision_sources,
        )
