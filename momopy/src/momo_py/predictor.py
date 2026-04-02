import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, Optional, Any

import joblib
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer

from .features import get_units, get_char_type, compute_source_features, SourceEntry, LABEL_CONTINUE, LABEL_SKIP, CharType
from .utils import split_on_unescaped_slash


def _is_ascii_printable_block(s: str) -> bool:
    """ASCII印字文字（0x21-0x7E）とスペース/タブのみで構成され、
    先頭・末尾がスペース/タブでないか判定する。"""
    if not s or s[0] in ' \t' or s[-1] in ' \t':
        return False
    return all(0x21 <= ord(c) <= 0x7E or c in ' \t' for c in s)


def _load_single_kanji_dict(path: str) -> Dict[str, Dict[str, str]]:
    """フォールバック辞書TSVを読み込む。
    フォーマット: 漢字[TAB]音読み[TAB]訓読み
    '#' で始まる行と空行はスキップする。
    """
    result: Dict[str, Dict[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            kanji, on, kun = parts
            result[kanji] = {"on": on, "kun": kun}
    return result


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
        model_path: モデルファイルのパス（.zip）
        single_kanji_dict_path: 単一漢字辞書TSVのパス（省略可）
        custom_dict_path: カスタム辞書ファイルのパス（省略可）
        compute_confidence: 自信度を計算するかどうか
            LinearSVC は decision_function のスコアをsigmoidで変換して使用。
            漢数字の JAPANESE_NUMERIC については常に計算する。
        confidence_threshold: KANJIフォールバックを発動させる自信度の上限
        numeric_confidence_threshold: JAPANESE_NUMERICルールベース変換を発動させる自信度の上限
    """
    model_path: str
    single_kanji_dict_path: Optional[str] = None
    custom_dict_path: Optional[str] = None
    compute_confidence: bool = True
    confidence_threshold: float = 0.3
    numeric_confidence_threshold: float = 0.5


# ==========================================
# 🌟 決定根拠タグ
# ==========================================
class DecisionSource:
    LR               = "LR"             # LRの予測をそのまま採用
    LR_LOW           = "LR_LOW"         # 自信度は低いがLRを採用
    FALLBACK_KANJI   = "FALLBACK_KANJI" # KANJI低自信度フォールバック辞書
    FALLBACK_NUMERIC = "FALLBACK_NUM"   # JAPANESE_NUMERICルールベース変換
    FALLBACK_ORPHAN  = "FALLBACK_ORPH"  # 文字消失バグ救済ロジック
    FALLBACK_REPEAT  = "FALLBACK_々"    # 々の繰り返し処理
    DICT             = "DICT"           # カスタム辞書による強制置換
    BYPASS           = "BYPASS"         # ASCIIバイパス

# ターミナル表示用ANSIカラー
_ANSI = {
    DecisionSource.LR:               "\033[32m",   # 緑
    DecisionSource.LR_LOW:           "\033[33m",   # 黄
    DecisionSource.FALLBACK_KANJI:   "\033[35m",   # マゼンタ
    DecisionSource.FALLBACK_NUMERIC: "\033[36m",   # シアン
    DecisionSource.FALLBACK_ORPHAN:  "\033[31m",   # 赤
    DecisionSource.FALLBACK_REPEAT:  "\033[35m",   # マゼンタ
    DecisionSource.DICT:             "\033[34m",   # 青
    DecisionSource.BYPASS:           "\033[90m",   # グレー
}
_ANSI_RESET = "\033[0m"


def _kurai_fallback(char: str, left_char: str, left_ctype: str, right_ctype: str) -> str:
    """位取り文字（十百千万億兆）のフォールバック変換ルール。"""
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

    return _KURAI_READING.get(char, char)


# ==========================================
# 🌟 カスタム辞書のロードとインデックス構築
# ==========================================
def load_custom_dict(path: str) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
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
    filled = round(conf * width)
    return "█" * filled + "░" * (width - filled)


# ==========================================
# 🌟 LRモデルのバンドル（読み＋境界）
# ==========================================
@dataclass
class LRModelBundle:
    """ZIPに格納するモデル一式"""
    vectorizer_read:     DictVectorizer
    coef_read_sparse:    Any
    intercept_read:      Any
    read_classes:        Any
    vectorizer_boundary: DictVectorizer
    model_boundary:      SGDClassifier
    version_info:        dict

class Predictor:
    def __init__(self, config: PredictorConfig):
        model_path = config.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ モデル未検出: {model_path}")

        self._config = config
        self._version_info: dict | None = None
        self._tmp_dir: str | None = None

        if not model_path.endswith(".zip"):
            raise ValueError(f"❌ 未対応のモデル形式です（.zip を指定してください）: {model_path}")

        with zipfile.ZipFile(model_path, "r") as zf:
            namelist = zf.namelist()

            if "version_info.json" not in namelist:
                raise ValueError(f"❌ ZIPファイル内に version_info.json が見つかりません: {model_path}")

            self._version_info = json.loads(zf.read("version_info.json").decode("utf-8"))

            bundle_name = self._version_info.get("model_bundle")
            if bundle_name is None:
                raise ValueError(f"❌ version_info.json に model_bundle キーがありません: {model_path}")
            if bundle_name not in namelist:
                raise ValueError(f"❌ ZIPファイル内に {bundle_name} が見つかりません: {model_path}")

            self._tmp_dir = tempfile.mkdtemp()
            zf.extract(bundle_name, self._tmp_dir)
            bundle_path = os.path.join(self._tmp_dir, bundle_name)

        bundle: LRModelBundle = joblib.load(bundle_path)
        self._vectorizer_read     = bundle.vectorizer_read
        self._coef_read_sparse    = bundle.coef_read_sparse
        self._intercept_read      = bundle.intercept_read
        self._read_classes        = bundle.read_classes
        self._vectorizer_boundary = bundle.vectorizer_boundary
        self._model_boundary      = bundle.model_boundary

        # フォールバック辞書のロード
        self._single_kanji_dict: Dict[str, Dict[str, str]] = {}
        if config.single_kanji_dict_path:
            self._single_kanji_dict = _load_single_kanji_dict(config.single_kanji_dict_path)

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
        if not text:
            return PredictionResult("", "", [], [], [])

        source_seq, bypass_indices, ascii_overrides, dict_overrides = self._preprocess_text(text)
        if not source_seq:
            return PredictionResult(text, "", [], [], [[] for _ in text])

        raw_labels, boundary_labels, decision_scores, boundary_proba = self._run_inference(source_seq)

        refined_labels, raw_confidences, has_splits, decision_sources = self._refine_predictions(
            source_seq, raw_labels, boundary_labels,
            decision_scores, boundary_proba,
            bypass_indices, ascii_overrides, dict_overrides
        )

        return self._assemble_result(
            text, source_seq, refined_labels, raw_confidences, has_splits, bypass_indices, decision_sources
        )


    # ==========================================
    # 🌟 ステップ 1: 前処理
    # ==========================================
    def _preprocess_text(
        self, text: str
    ) -> Tuple[List[SourceEntry], Set[int], Dict[int, str], Dict[int, str]]:
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
                        dict_overrides[i + j] = reading
                    i += length
                else:
                    i += 1

        return source_seq, bypass_indices, ascii_overrides, dict_overrides


    # ==========================================
    # 🌟 ステップ 2: 推論（LinearSVC版）
    # ==========================================
    def _run_inference(
        self, source_seq: List[SourceEntry]
    ) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
        """
        読みモデル（LinearSVC）と境界モデル（SGDClassifier）を推論する。

        戻り値:
            raw_labels      : 読みラベル列
            boundary_labels : 境界ラベル列（"0" or "1"）
            decision_scores : LinearSVC の decision_function 結果
                              shape=(n, n_classes) — sigmoidで確信度に変換して使う
            boundary_proba  : 境界の確率 shape=(n, 2)
        """
        src_features = compute_source_features(source_seq)
        X_read     = self._vectorizer_read.transform(src_features)
        X_boundary = self._vectorizer_boundary.transform(src_features)

        # 読みモデル（LinearSVC）
        decision_scores = (self._coef_read_sparse @ X_read.T).toarray().T + self._intercept_read
        predicted_indices = np.argmax(decision_scores, axis=1)
        raw_labels = list(self._read_classes[predicted_indices])

        # 境界モデル（SGDClassifier）
        boundary_raw    = list(self._model_boundary.predict(X_boundary))
        boundary_labels = [str(b) for b in boundary_raw]
        boundary_proba  = self._model_boundary.predict_proba(X_boundary)  # shape: (n, 2)
        return raw_labels, boundary_labels, decision_scores, boundary_proba

    def _get_read_confidence(self, decision_scores: np.ndarray, label: str, i: int) -> float:
        """
        LinearSVC の decision_function スコアを sigmoid で 0〜1 に変換して返す。
        ラベルがクラスリストにない場合は 0.0 を返す。
        """
        idx = np.searchsorted(self._read_classes, label)
        if idx < len(self._read_classes) and self._read_classes[idx] == label:
            score = decision_scores[i] if decision_scores.ndim == 1 else decision_scores[i, idx]
            return float(1.0 / (1.0 + np.exp(-score)))
        return 0.0

    # ==========================================
    # 🌟 ステップ 3: 後処理
    # ==========================================
    def _refine_predictions(
        self,
        source_seq: List[SourceEntry],
        raw_labels: List[str],
        boundary_labels: List[str],
        decision_scores: np.ndarray,
        boundary_proba: np.ndarray,
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
                clean_label = ascii_overrides[i]
                confidence  = 1.0
                decision    = DecisionSource.BYPASS
                last_fallback_reading = ""
                has_split = False

            elif i in dict_overrides:
                clean_label = dict_overrides[i]
                confidence  = 1.0
                decision    = DecisionSource.DICT
                last_fallback_reading = ""
                has_split = (boundary_labels[i] == "1")

            else:
                label    = raw_labels[i]
                decision = DecisionSource.LR

                if self._config.compute_confidence or ctype == CharType.JAPANESE_NUMERIC:
                    confidence = self._get_read_confidence(decision_scores, label, i)
                else:
                    confidence = 1.0

                # 文字消失バグの救済ロジック
                if label == LABEL_CONTINUE and (parent_idx == -1 or parent_idx in bypass_indices):
                    if ctype == 'KANJI' and char in self._single_kanji_dict:
                        label = self._single_kanji_dict[char]["on"]
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
                    if decision == DecisionSource.LR:
                        if confidence < self._config.confidence_threshold and ctype == 'KANJI':
                            decision = DecisionSource.LR_LOW

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
        if confidence >= self._config.numeric_confidence_threshold:
            return label, "", confidence, DecisionSource.LR

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
        is_applied = False
        new_label = label
        new_last_fallback = last_fallback
        decision = DecisionSource.LR

        if ctype == 'KANJI' and confidence < self._config.confidence_threshold and char in self._single_kanji_dict:
            next_char  = source_seq[i + 1][0] if i < len(source_seq) - 1 else ""
            next_ctype = source_seq[i + 1][2] if i < len(source_seq) - 1 else ""

            if next_ctype == 'HIRAGANA' and next_char in SAFE_OKURIGANA:
                replacement_reading = self._single_kanji_dict[char]["kun"]
            else:
                replacement_reading = self._single_kanji_dict[char]["on"]

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
