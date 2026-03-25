import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, Optional

import pycrfsuite

from .features import get_units, get_char_type, compute_source_features, SourceEntry, LABEL_CONTINUE, LABEL_SKIP, CharType
from .fallback_dict import FALLBACK_DICT
from .utils import split_on_unescaped_slash


def _is_ascii_printable_block(s: str) -> bool:
    """ASCII印字文字（0x21-0x7E）とスペース/タブのみで構成され、
    先頭・末尾がスペース/タブでないか判定する。"""
    if not s or s[0] in ' \t' or s[-1] in ' \t':
        return False
    return all(0x21 <= ord(c) <= 0x7E or c in ' \t' for c in s)


# 🌟 訓読みを誘発しやすい「安全な送り仮名」のリスト
SAFE_OKURIGANA = set(
    "あいうえおかきくけこたちつてとなにぬねのまみむめもやゆよらりるれろわばびぶべぼ"
)

# 🌟 漢数字→アラビア数字の単純置換テーブル
_DIGIT_TABLE = {
    "〇": "0", "一": "1", "二": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
}

# 🌟 位取り文字の読みテーブル（数字展開しない場合）
_KURAI_READING = {
    "千": "セン", "万": "マン", "億": "オク", "兆": "チョー",
}


# ==========================================
# 🌟 設定
# ==========================================
@dataclass
class PredictorConfig:
    """Predictorの設定をまとめた構造体。

    Attributes:
        model_path: モデルファイルのパス（.zip または .crfsuite）
        custom_dict_path: カスタム辞書ファイルのパス（学習データと同じ形式、省略可）
        confidence_threshold: KANJIフォールバックを発動させる自信度の上限
        numeric_confidence_threshold: JAPANESE_NUMERICルールベース変換を発動させる自信度の上限
    """
    model_path: str
    custom_dict_path: Optional[str] = None
    confidence_threshold: float = 0.3
    numeric_confidence_threshold: float = 0.8


# ==========================================
# 🌟 決定根拠タグ
# ==========================================
class DecisionSource:
    CRF              = "CRF"            # CRFの予測をそのまま採用
    CRF_LOW          = "CRF_LOW"        # 自信度は低いがCRFを採用
    FALLBACK_KANJI   = "FALLBACK_KANJI" # KANJI低自信度フォールバック辞書
    FALLBACK_NUMERIC = "FALLBACK_NUM"   # JAPANESE_NUMERICルールベース変換
    FALLBACK_ORPHAN  = "FALLBACK_ORPH"  # 文字消失バグ救済ロジック
    FALLBACK_REPEAT  = "FALLBACK_々"    # 々の繰り返し処理
    DICT             = "DICT"           # カスタム辞書による強制置換
    BYPASS           = "BYPASS"         # ASCIIバイパス

# ターミナル表示用ANSIカラー
_ANSI = {
    DecisionSource.CRF:              "\033[32m",   # 緑
    DecisionSource.CRF_LOW:          "\033[33m",   # 黄
    DecisionSource.FALLBACK_KANJI:   "\033[35m",   # マゼンタ
    DecisionSource.FALLBACK_NUMERIC: "\033[36m",   # シアン
    DecisionSource.FALLBACK_ORPHAN:  "\033[31m",   # 赤
    DecisionSource.FALLBACK_REPEAT:  "\033[35m",   # マゼンタ
    DecisionSource.DICT:             "\033[34m",   # 青
    DecisionSource.BYPASS:           "\033[90m",   # グレー
}
_ANSI_RESET = "\033[0m"


def _kurai_fallback(char: str, left_char: str, left_ctype: str, right_ctype: str) -> str:
    """
    位取り文字（十百千万億兆）のフォールバック変換ルール。

    十のルール:
        左が JAPANESE_NUMERIC かつ右が JAPANESE_NUMERIC → _
        左が JAPANESE_NUMERIC かつ右が JAPANESE_NUMERIC でない → 0
        左が JAPANESE_NUMERIC でない かつ右が JAPANESE_NUMERIC → 1
        左も右も JAPANESE_NUMERIC でない → 10

    百のルール:
        左が JAPANESE_NUMERIC かつ右が JAPANESE_NUMERIC → 0
        左が JAPANESE_NUMERIC かつ右が JAPANESE_NUMERIC でない → 00
        左が JAPANESE_NUMERIC でない かつ右が JAPANESE_NUMERIC → 1
        左も右も JAPANESE_NUMERIC でない → 100

    千のルール:
        右が JAPANESE_NUMERIC → 0
        右が JAPANESE_NUMERIC でない かつ左が 三 → ゼン
        右が JAPANESE_NUMERIC でない かつ左が 三 以外 → セン

    万億兆のルール:
        常に読み（マン／オク／チョー）
    """
    is_numeric = CharType.JAPANESE_NUMERIC
    left_is_numeric  = (left_ctype  == is_numeric)
    right_is_numeric = (right_ctype == is_numeric)

    if char == "十":
        if left_is_numeric and right_is_numeric:     return LABEL_SKIP
        if left_is_numeric and not right_is_numeric: return "0"
        if not left_is_numeric and right_is_numeric: return "1"
        return "10"

    if char == "百":
        if left_is_numeric and right_is_numeric:     return "0"
        if left_is_numeric and not right_is_numeric: return "00"
        if not left_is_numeric and right_is_numeric: return "1"
        return "100"

    if char == "千":
        if right_is_numeric: return "0"
        return "ゼン" if left_char == "三" else "セン"

    # 万億兆: 常に読み
    return _KURAI_READING.get(char, char)


# ==========================================
# 🌟 カスタム辞書のロードとインデックス構築
# ==========================================
def load_custom_dict(path: str) -> Dict[str, List[str]]:
    """
    カスタム辞書ファイル（学習データと同じ形式）をロードして
    { 表層形: [読みラベル, ...] } の辞書を返す。

    入力形式（例）:
        切明\tキリ/アケ
        三日\tミッ/カ

    出力形式（例）:
        { "切明": ["キリ", "アケ"], "三日": ["ミッ", "カ"] }

    '#' で始まる行と空行はスキップする。
    """
    result: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"カスタム辞書 {path} の {lineno} 行目の形式が不正です "
                    f"（タブ区切りで表層形と読みが必要）: {line!r}"
                )
            surface, reading_str = parts
            readings = [b.replace(r"\/", "/") for b in split_on_unescaped_slash(reading_str)]
            if len(readings) != len(surface):
                raise ValueError(
                    f"カスタム辞書 {path} の {lineno} 行目: "
                    f"表層形 {surface!r} の文字数（{len(surface)}）と "
                    f"読みブロック数（{len(readings)}）が一致しません。"
                )
            result[surface] = readings
    return result


def build_dict_index(custom_dict: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, List[str]]]]:
    """
    最長一致検索のため、先頭文字をキーにしたインデックスを構築する。
    同じ先頭文字を持つエントリは長い順に並べておく（最長一致のため）。

    出力形式:
        { "切": [("切明", ["キリ", "アケ"]), ...], ... }
    """
    index: Dict[str, List[Tuple[str, List[str]]]] = {}
    for surface, readings in custom_dict.items():
        key = surface[0]
        index.setdefault(key, []).append((surface, readings))
    for key in index:
        index[key].sort(key=lambda x: len(x[0]), reverse=True)
    return index


def find_longest_match(
    index: Dict[str, List[Tuple[str, List[str]]]],
    source_seq: List[SourceEntry],
    pos: int,
) -> Optional[Tuple[int, List[str]]]:
    """
    source_seq の pos 番目から始まる最長一致エントリを探す。

    戻り値:
        マッチした場合: (マッチした文字数, 読みラベルのリスト)
        マッチしなかった場合: None
    """
    char = source_seq[pos][0]
    candidates = index.get(char)
    if not candidates:
        return None

    for surface, readings in candidates:
        length = len(surface)
        if pos + length > len(source_seq):
            continue
        if all(source_seq[pos + j][0] == surface[j] for j in range(length)):
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

    def to_json(self) -> str:
        text_safe = json.dumps(self.source_text, ensure_ascii=False)
        kana_safe = json.dumps(self.kana_text, ensure_ascii=False)

        conf_str = "[" + ", ".join([f"{c:.3f}" for c in self.confidences]) + "]"
        k2s_str = "[" + ", ".join(map(str, self.kana_to_src_index)) + "]"
        s2k_str = json.dumps(self.src_to_kana_index)
        dec_safe = json.dumps(self.decision_sources, ensure_ascii=False)

        return (
            f'{{\n'
            f'  "text": {text_safe},\n'
            f'  "kana": {kana_safe},\n'
            f'  "kana_to_src_index": {k2s_str},\n'
            f'  "src_to_kana_index": {s2k_str},\n'
            f'  "confidences": {conf_str},\n'
            f'  "decision_sources": {dec_safe}\n'
            f'}}'
        )

    def format_terminal(self, use_color: bool = True) -> str:
        """
        各ソース文字ごとに決定根拠・読み・自信度をターミナル向けに整形して返す。

        例:
          切  →  キリ   [DICT        ] ████████████ 1.000
          明  →  アケ   [DICT        ] ████████████ 1.000
          の  →  の     [BYPASS      ]
          山  →  サン   [CRF_LOW     ] ███░░░░░░░░░ 0.241
        """
        src_len = len(self.source_text)

        src_kana: Dict[int, List[str]]   = {i: [] for i in range(src_len)}
        src_conf: Dict[int, List[float]] = {i: [] for i in range(src_len)}
        src_dec:  Dict[int, List[str]]   = {i: [] for i in range(src_len)}

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
            decs  = src_dec[src_i]

            if not kanas:
                rows.append(f"  {char}  →  (skip)")
                continue

            kana_str = "".join(kanas)
            conf_val = min(confs)
            dec_tag  = decs[0]

            tag_str = f"[{dec_tag:<13}]"
            bar     = _confidence_bar(conf_val)

            if dec_tag == DecisionSource.BYPASS:
                line = f"  {char}  →  {kana_str:<6} {tag_str}"
            else:
                line = f"  {char}  →  {kana_str:<6} {tag_str} {bar} {conf_val:.3f}"

            if use_color:
                line = _ANSI.get(dec_tag, "") + line + _ANSI_RESET

            rows.append(line)

        return "\n".join(rows)


def _confidence_bar(conf: float, width: int = 12) -> str:
    """自信度を簡易バーグラフで表現する。"""
    filled = round(conf * width)
    return "█" * filled + "░" * (width - filled)


class Predictor:
    def __init__(self, config: PredictorConfig):
        model_path = config.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ モデル未検出: {model_path}")

        self._config = config
        self._version_info: dict | None = None
        self._tmp_dir: str | None = None

        if model_path.endswith(".zip"):
            with zipfile.ZipFile(model_path, "r") as zf:
                namelist = zf.namelist()

                if "version_info.json" not in namelist:
                    raise ValueError(f"❌ ZIPファイル内に version_info.json が見つかりません: {model_path}")

                self._version_info = json.loads(zf.read("version_info.json").decode("utf-8"))

                if "model_read" not in self._version_info or "model_boundary" not in self._version_info:
                    raise ValueError(f"❌ version_info.json に model_read / model_boundary キーがありません: {model_path}")

                read_name  = self._version_info["model_read"]
                bound_name = self._version_info["model_boundary"]
                for name in (read_name, bound_name):
                    if name not in namelist:
                        raise ValueError(f"❌ ZIPファイル内に {name} が見つかりません: {model_path}")

                self._tmp_dir = tempfile.mkdtemp()
                zf.extract(read_name,  self._tmp_dir)
                zf.extract(bound_name, self._tmp_dir)
                path_read  = os.path.join(self._tmp_dir, read_name)
                path_bound = os.path.join(self._tmp_dir, bound_name)

        elif model_path.endswith(".crfsuite"):
            path_read  = model_path
            path_bound = None

        else:
            raise ValueError(f"❌ 未対応のモデル形式です（.zip または .crfsuite を指定してください）: {model_path}")

        self.tagger_read = pycrfsuite.Tagger()
        self.tagger_read.open(path_read)

        self.tagger_boundary: pycrfsuite.Tagger | None = None
        if path_bound:
            self.tagger_boundary = pycrfsuite.Tagger()
            self.tagger_boundary.open(path_bound)

        # カスタム辞書のロードとインデックス構築
        self._dict_index: Dict[str, List[Tuple[str, List[str]]]] = {}
        if config.custom_dict_path:
            custom_dict = load_custom_dict(config.custom_dict_path)
            self._dict_index = build_dict_index(custom_dict)

    def __del__(self) -> None:
        if self._tmp_dir and os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def get_version_info(self) -> dict | None:
        return self._version_info


    # ==========================================
    # 🌟 推論パイプライン（The Conductor）
    # ==========================================
    def predict(self, text: str) -> PredictionResult:
        """メインの推論フロー。各ステップをパイプラインとして呼び出す。"""
        if not text:
            return PredictionResult("", "", [], [], [])

        # 1. 前処理（単位分割・バイパス判定・辞書マッチ）
        source_seq, bypass_indices, ascii_overrides, dict_overrides = self._preprocess_text(text)
        if not source_seq:
            return PredictionResult(text, "", [], [], [[] for _ in text])

        # 2. 推論（CRFによる生のラベル予測）
        raw_labels, boundary_labels = self._run_inference(source_seq)

        # 3. 後処理（バイパス・辞書・フォールバック・自信度の確定）
        refined_labels, raw_confidences, has_splits, decision_sources = self._refine_predictions(
            source_seq, raw_labels, boundary_labels, bypass_indices, ascii_overrides, dict_overrides
        )

        # 4. 結果の組み立て（文字列とインデックスの最終マッピング）
        return self._assemble_result(
            text, source_seq, refined_labels, raw_confidences, has_splits, bypass_indices, decision_sources
        )


    # ==========================================
    # 🌟 ステップ 1: 前処理
    # ==========================================
    def _preprocess_text(
        self, text: str
    ) -> Tuple[List[SourceEntry], Set[int], Dict[int, str], Dict[int, str]]:
        """
        テキストをソース文字系列に展開し、各種オーバーライドを準備する。

        戻り値:
            source_seq     : ソース文字系列
            bypass_indices : ASCIIバイパス対象のインデックス集合
            ascii_overrides: ASCIIバイパス用ラベル辞書
            dict_overrides : カスタム辞書マッチ用ラベル辞書
                             先頭文字インデックス → 読みラベル
                             2文字目以降 → LABEL_CONTINUE
        """
        units_info = get_units(text)
        source_seq: List[SourceEntry] = []
        bypass_indices: Set[int] = set()
        ascii_overrides: Dict[int, str] = {}

        char_idx = 0
        for val, orig_idx, ctype in units_info:
            is_ascii_bypass = (ctype == 'ALPHA' or ctype == 'NUM') and _is_ascii_printable_block(val)
            for i, c in enumerate(val):
                source_seq.append((c, orig_idx + i, ctype))
                if is_ascii_bypass:
                    bypass_indices.add(char_idx)
                    ascii_overrides[char_idx] = val if i == 0 else LABEL_CONTINUE
                char_idx += 1

        # カスタム辞書の最長一致スキャン（ASCIIバイパス済みの文字はスキップ）
        dict_overrides: Dict[int, str] = {}
        if self._dict_index:
            i = 0
            while i < len(source_seq):
                if i in bypass_indices:
                    i += 1
                    continue
                match = find_longest_match(self._dict_index, source_seq, i)
                if match:
                    length, readings = match
                    for j, reading in enumerate(readings):
                        dict_overrides[i + j] = reading  # 各文字に対応する読みをそのまま入れる
                    i += length
                else:
                    i += 1

        return source_seq, bypass_indices, ascii_overrides, dict_overrides


    # ==========================================
    # 🌟 ステップ 2: 推論
    # ==========================================
    def _run_inference(self, source_seq: List[SourceEntry]) -> Tuple[List[str], List[str]]:
        """
        読みモデルと境界モデルをそれぞれ推論する。

        戻り値:
            raw_labels      : 読みラベル列（例: ["カン", "ゼン", "ニ", ...]）
            boundary_labels : 境界ラベル列（例: ["1", "0", "1", ...]）
                              .crfsuite 直接指定（tagger_boundary が None）の場合は
                              すべて "0" を返す（分かち書きなし）。
        """
        src_features = compute_source_features(source_seq)

        self.tagger_read.set(src_features)
        raw_labels = list(self.tagger_read.tag())

        if self.tagger_boundary is not None:
            self.tagger_boundary.set(src_features)
            boundary_labels = list(self.tagger_boundary.tag())
        else:
            boundary_labels = ["0"] * len(raw_labels)

        return raw_labels, boundary_labels


    # ==========================================
    # 🌟 ステップ 3: 後処理（バイパス・辞書・フォールバック）
    # ==========================================
    def _refine_predictions(
        self,
        source_seq: List[SourceEntry],
        raw_labels: List[str],
        boundary_labels: List[str],
        bypass_indices: Set[int],
        ascii_overrides: Dict[int, str],
        dict_overrides: Dict[int, str],
    ) -> Tuple[List[str], List[float], List[bool], List[str]]:
        refined_labels = []
        confidences = []
        has_splits = []
        decision_sources = []
        last_fallback_reading = ""
        parent_idx = -1

        for i, (char, _, ctype) in enumerate(source_seq):
            raw_clean = raw_labels[i]

            if raw_clean not in (LABEL_CONTINUE, LABEL_SKIP):
                parent_idx = i

            if i in bypass_indices:
                # ASCIIバイパス（最優先）
                clean_label = ascii_overrides[i]
                confidence  = 1.0
                decision    = DecisionSource.BYPASS
                last_fallback_reading = ""
                has_split = False

            elif i in dict_overrides:
                # カスタム辞書による強制置換
                clean_label = dict_overrides[i]
                confidence  = 1.0
                decision    = DecisionSource.DICT
                last_fallback_reading = ""
                has_split = (boundary_labels[i] == "1")

            else:
                label      = raw_labels[i]
                confidence = self.tagger_read.marginal(label, i)
                decision   = DecisionSource.CRF

                # 🚨 文字消失バグの救済ロジック
                if label == LABEL_CONTINUE and (parent_idx == -1 or parent_idx in bypass_indices):
                    if ctype == 'KANJI' and char in FALLBACK_DICT:
                        label = FALLBACK_DICT[char]["on"]
                    else:
                        label = char
                    confidence = 0.0
                    decision   = DecisionSource.FALLBACK_ORPHAN

                elif ctype == CharType.JAPANESE_NUMERIC:
                    label, last_fallback_reading, confidence, decision = self._convert_japanese_numeric(
                        i, char, label, confidence, source_seq, last_fallback_reading
                    )
                else:
                    label, last_fallback_reading, _, decision = self._apply_kanji_fallback(
                        i, char, ctype, label, confidence, source_seq, last_fallback_reading
                    )
                    if decision == DecisionSource.CRF:
                        if confidence < self._config.confidence_threshold and ctype == 'KANJI':
                            decision = DecisionSource.CRF_LOW

                clean_label = label
                has_split = (boundary_labels[i] == "1")

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
        last_fallback: str,
    ) -> Tuple[str, str, float, str]:
        """
        JAPANESE_NUMERIC 文字の変換。
        自信度が閾値以上であれば CRF の出力をそのまま使う。
        閾値を下回る場合はルールベース変換にフォールバックする。
        """
        if confidence >= self._config.numeric_confidence_threshold:
            return label, "", confidence, DecisionSource.CRF

        left_char   = source_seq[i - 1][0] if i > 0 else ""
        left_ctype  = source_seq[i - 1][2] if i > 0 else ""
        right_ctype = source_seq[i + 1][2] if i < len(source_seq) - 1 else ""

        if char in _DIGIT_TABLE:
            return _DIGIT_TABLE[char], "", confidence, DecisionSource.FALLBACK_NUMERIC

        return _kurai_fallback(char, left_char, left_ctype, right_ctype), "", confidence, DecisionSource.FALLBACK_NUMERIC

    def _apply_kanji_fallback(
        self, i: int, char: str, ctype: str, label: str, confidence: float,
        source_seq: List[SourceEntry], last_fallback: str
    ) -> Tuple[str, str, bool, str]:
        """KANJI の低自信度処理と々の繰り返し処理。"""
        is_applied = False
        new_label = label
        new_last_fallback = last_fallback
        decision = DecisionSource.CRF

        if ctype == 'KANJI' and confidence < self._config.confidence_threshold and char in FALLBACK_DICT:
            next_char  = source_seq[i + 1][0] if i < len(source_seq) - 1 else ""
            next_ctype = source_seq[i + 1][2] if i < len(source_seq) - 1 else ""

            if next_ctype == 'HIRAGANA' and next_char in SAFE_OKURIGANA:
                replacement_reading = FALLBACK_DICT[char]["kun"]
            else:
                replacement_reading = FALLBACK_DICT[char]["on"]

            new_label = replacement_reading
            new_last_fallback = replacement_reading
            is_applied = True
            decision = DecisionSource.FALLBACK_KANJI

        elif char == '々' and last_fallback:
            new_label = last_fallback
            new_last_fallback = last_fallback
            is_applied = True
            decision = DecisionSource.FALLBACK_REPEAT

        if not is_applied and ctype != 'SYMBOL':
            new_last_fallback = ""

        return new_label, new_last_fallback, is_applied, decision


    # ==========================================
    # 🌟 ステップ 4: 結果の組み立て
    # ==========================================
    def _assemble_result(
        self, text: str, source_seq: List[SourceEntry],
        refined_labels: List[str], raw_confidences: List[float],
        has_splits: List[bool], bypass_indices: Set[int],
        decision_sources: List[str],
    ) -> PredictionResult:
        translated = ""
        kana_to_src_index: List[int] = []
        final_confidences: List[float] = []
        final_decision_sources: List[str] = []
        src_to_kana_index: List[List[int]] = [[] for _ in text]
        kana_pos = 0

        for i, (char, orig_idx, _) in enumerate(source_seq):
            clean_label = refined_labels[i]
            confidence  = raw_confidences[i]
            decision    = decision_sources[i]

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
