import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict

import pycrfsuite

from .features import get_units, get_char_type, compute_source_features, SourceEntry, LABEL_CONTINUE, LABEL_SKIP, CharType
from .fallback_dict import FALLBACK_DICT

# フォールバックを発動させる自信度の境界線
CONFIDENCE_THRESHOLD = 0.3          # KANJI フォールバック用
JAPANESE_NUMERIC_CONFIDENCE_THRESHOLD = 0.8  # JAPANESE_NUMERIC 変換用

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


@dataclass
class PredictionResult:
    source_text: str
    kana_text: str
    confidences: List[float]
    kana_to_src_index: List[int]
    src_to_kana_index: List[List[int]]

    def to_json(self) -> str:
        text_safe = json.dumps(self.source_text, ensure_ascii=False)
        kana_safe = json.dumps(self.kana_text, ensure_ascii=False)

        conf_str = "[" + ", ".join([f"{c:.3f}" for c in self.confidences]) + "]"
        k2s_str = "[" + ", ".join(map(str, self.kana_to_src_index)) + "]"
        s2k_str = json.dumps(self.src_to_kana_index)

        return f'{{\n  "text": {text_safe},\n  "kana": {kana_safe},\n  "kana_to_src_index": {k2s_str},\n  "src_to_kana_index": {s2k_str},\n  "confidences": {conf_str}\n}}'


class Predictor:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ モデル未検出: {model_path}")

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
            # 開発・デバッグ用: 読みモデルを直接指定
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

        # 1. 前処理（単位分割・バイパス判定・特徴量準備）
        source_seq, bypass_indices, ascii_overrides = self._preprocess_text(text)
        if not source_seq:
            return PredictionResult(text, "", [], [], [[] for _ in text])

        # 2. 推論（CRFによる生のラベル予測）
        raw_labels, boundary_labels = self._run_inference(source_seq)

        # 3. 後処理（バイパスの適用・フォールバック・自信度の確定）
        refined_labels, raw_confidences, has_splits = self._refine_predictions(
            source_seq, raw_labels, boundary_labels, bypass_indices, ascii_overrides
        )

        # 4. 結果の組み立て（文字列とインデックスの最終マッピング）
        return self._assemble_result(
            text, source_seq, refined_labels, raw_confidences, has_splits, bypass_indices
        )


    # ==========================================
    # 🌟 ステップ 1: 前処理
    # ==========================================
    def _preprocess_text(self, text: str) -> Tuple[List[SourceEntry], Set[int], Dict[int, str]]:
        units_info = get_units(text)  # 戻り値: List[Tuple[str, int, str]]
        source_seq: List[SourceEntry] = []
        bypass_indices: Set[int] = set()
        ascii_overrides: Dict[int, str] = {}

        char_idx = 0
        for val, orig_idx, ctype in units_info:
            is_ascii_bypass = (ctype == 'ALPHA' or ctype == 'NUM') and bool(
                re.fullmatch(r'[!-~]+(?:[ \t]+[!-~]+)*', val)
            )

            for i, c in enumerate(val):
                source_seq.append((c, orig_idx + i, ctype))
                if is_ascii_bypass:
                    bypass_indices.add(char_idx)
                    ascii_overrides[char_idx] = val if i == 0 else LABEL_CONTINUE
                char_idx += 1

        return source_seq, bypass_indices, ascii_overrides


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
            # .crfsuite 直接指定時は境界モデルなし → 分かち書きなしで動作
            boundary_labels = ["0"] * len(raw_labels)

        return raw_labels, boundary_labels


    # ==========================================
    # 🌟 ステップ 3: 後処理（バイパスとフォールバック）
    # ==========================================
    def _refine_predictions(
        self,
        source_seq: List[SourceEntry],
        raw_labels: List[str],
        boundary_labels: List[str],
        bypass_indices: Set[int],
        ascii_overrides: Dict[int, str],
    ) -> Tuple[List[str], List[float], List[bool]]:
        refined_labels = []
        confidences = []
        has_splits = []
        last_fallback_reading = ""

        parent_idx = -1

        for i, (char, _, ctype) in enumerate(source_seq):
            raw_clean = raw_labels[i]

            if raw_clean not in (LABEL_CONTINUE, LABEL_SKIP):
                parent_idx = i

            if i in bypass_indices:
                clean_label = ascii_overrides[i]
                confidence  = 1.0
                last_fallback_reading = ""
                has_split = False
            else:
                label      = raw_labels[i]
                confidence = self.tagger_read.marginal(label, i)

                # 🚨 文字消失バグの救済ロジック
                if label == LABEL_CONTINUE and (parent_idx == -1 or parent_idx in bypass_indices):
                    if ctype == 'KANJI' and char in FALLBACK_DICT:
                        label = FALLBACK_DICT[char]["on"]
                    else:
                        label = char
                    confidence = 0.0

                if ctype == CharType.JAPANESE_NUMERIC:
                    label, last_fallback_reading, confidence = self._convert_japanese_numeric(
                        i, char, label, confidence, source_seq, last_fallback_reading
                    )
                else:
                    label, last_fallback_reading, _ = self._apply_kanji_fallback(
                        i, char, ctype, label, confidence, source_seq, last_fallback_reading
                    )
                clean_label = label

                has_split = (boundary_labels[i] == "1")

            refined_labels.append(clean_label)
            confidences.append(confidence)
            has_splits.append(has_split)

        return refined_labels, confidences, has_splits

    def _convert_japanese_numeric(
        self,
        i: int,
        char: str,
        label: str,
        confidence: float,
        source_seq: List[SourceEntry],
        last_fallback: str,
    ) -> Tuple[str, str, float]:
        """
        JAPANESE_NUMERIC 文字の変換。
        自信度が閾値以上であれば CRF の出力をそのまま使う（confidence はそのまま）。
        閾値を下回る場合はルールベース変換にフォールバックし、confidence を 1.0 に書き換える。
        （confidence=0.0 は CRF が失敗、confidence=1.0 はルールベースで確定、を意味する）
        """
        if confidence >= JAPANESE_NUMERIC_CONFIDENCE_THRESHOLD:
            return label, "", confidence

        # ルールベース変換（confidence はCRFの値をそのまま引き継ぐ）
        left_char   = source_seq[i - 1][0] if i > 0 else ""
        left_ctype  = source_seq[i - 1][2] if i > 0 else ""
        right_ctype = source_seq[i + 1][2] if i < len(source_seq) - 1 else ""

        if char in _DIGIT_TABLE:
            return _DIGIT_TABLE[char], "", confidence

        # 位取り文字
        return _kurai_fallback(char, left_char, left_ctype, right_ctype), "", confidence

    def _apply_kanji_fallback(self, i: int, char: str, ctype: str, label: str, confidence: float, source_seq: List[SourceEntry], last_fallback: str) -> Tuple[str, str, bool]:
        """
        KANJI の低自信度処理と々の繰り返し処理。
        """
        is_applied = False
        new_label = label
        new_last_fallback = last_fallback

        if ctype == 'KANJI' and confidence < CONFIDENCE_THRESHOLD and char in FALLBACK_DICT:
            next_char  = source_seq[i + 1][0] if i < len(source_seq) - 1 else ""
            next_ctype = source_seq[i + 1][2] if i < len(source_seq) - 1 else ""

            if next_ctype == 'HIRAGANA' and next_char in SAFE_OKURIGANA:
                replacement_reading = FALLBACK_DICT[char]["kun"]
            else:
                replacement_reading = FALLBACK_DICT[char]["on"]

            new_label = replacement_reading
            new_last_fallback = replacement_reading
            is_applied = True

        elif char == '々' and last_fallback:
            new_label = last_fallback
            new_last_fallback = last_fallback
            is_applied = True

        if not is_applied and ctype != 'SYMBOL':
            new_last_fallback = ""

        return new_label, new_last_fallback, is_applied


    # ==========================================
    # 🌟 ステップ 4: 結果の組み立て
    # ==========================================
    def _assemble_result(self, text: str, source_seq: List[SourceEntry], refined_labels: List[str], raw_confidences: List[float], has_splits: List[bool], bypass_indices: Set[int]) -> PredictionResult:
        translated = ""
        kana_to_src_index: List[int] = []
        final_confidences: List[float] = []
        src_to_kana_index: List[List[int]] = [[] for _ in text]
        kana_pos = 0

        for i, (char, orig_idx, _) in enumerate(source_seq):
            clean_label = refined_labels[i]
            confidence  = raw_confidences[i]

            if clean_label == LABEL_SKIP:
                pass
            elif clean_label != LABEL_CONTINUE:
                if i in bypass_indices:
                    for j, ch in enumerate(clean_label):
                        translated += ch
                        target_orig_idx = orig_idx + j
                        kana_to_src_index.append(target_orig_idx)
                        final_confidences.append(confidence)
                        src_to_kana_index[target_orig_idx].append(kana_pos)
                        kana_pos += 1
                else:
                    for ch in clean_label:
                        translated += ch
                        kana_to_src_index.append(orig_idx)
                        final_confidences.append(confidence)
                        src_to_kana_index[orig_idx].append(kana_pos)
                        kana_pos += 1

            if has_splits[i]:
                translated += " "
                kana_to_src_index.append(orig_idx)
                final_confidences.append(confidence)
                src_to_kana_index[orig_idx].append(kana_pos)
                kana_pos += 1

        return PredictionResult(
            source_text=text,
            kana_text=translated,
            confidences=final_confidences,
            kana_to_src_index=kana_to_src_index,
            src_to_kana_index=src_to_kana_index
        )
