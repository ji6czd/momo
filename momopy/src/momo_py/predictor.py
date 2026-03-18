import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict

import pycrfsuite

from .features import get_units, get_char_type, compute_source_features, SourceEntry, MORA_SPLIT
from .fallback_dict import FALLBACK_DICT

# フォールバックを発動させる自信度の境界線（30%）
CONFIDENCE_THRESHOLD = 0.3

# 🌟 訓読みを誘発しやすい「安全な送り仮名」のリスト
SAFE_OKURIGANA = set(
    "あいうえおかきくけこたちつてとなにぬねのまみむめもやゆよらりるれろわばびぶべぼ"
)


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

                if "version_info.json" in namelist:
                    self._version_info = json.loads(zf.read("version_info.json").decode("utf-8"))

                # 🌟 version_info.json からモデルファイル名を取得（新形式）
                # 旧形式（単一モデル）との後方互換性も維持する
                if self._version_info and "model_read" in self._version_info:
                    # 新形式: 読みモデルと境界モデルの2ファイル
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
                else:
                    # 旧形式: 単一の .crfsuite ファイル（後方互換）
                    crfsuite_files = [n for n in namelist if n.endswith(".crfsuite")]
                    if not crfsuite_files:
                        raise ValueError(f"❌ ZIPファイル内に .crfsuite ファイルが見つかりません: {model_path}")
                    self._tmp_dir = tempfile.mkdtemp()
                    zf.extract(crfsuite_files[0], self._tmp_dir)
                    path_read  = os.path.join(self._tmp_dir, crfsuite_files[0])
                    path_bound = None  # 旧形式は境界モデルなし

        else:
            # ZIPなし: 直接 .crfsuite を指定（開発・デバッグ用）
            path_read  = model_path
            path_bound = None

        # 🌟 読みtagger（常に存在）
        self.tagger_read = pycrfsuite.Tagger()
        self.tagger_read.open(path_read)

        # 🌟 境界tagger（新形式のみ。旧形式はNoneのままでフォールバック動作）
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
    # 🌟 ステップ 1: 前処理（無変更）
    # ==========================================
    def _preprocess_text(self, text: str) -> Tuple[List[SourceEntry], Set[int], Dict[int, str]]:
        units_info = get_units(text)
        source_seq: List[SourceEntry] = []
        bypass_indices: Set[int] = set()
        ascii_overrides: Dict[int, str] = {}
        
        char_idx = 0
        for val, orig_idx in units_info:
            is_ascii_bypass = bool(re.fullmatch(r'[!-~]+(?:[ \t]+[!-~]+)*', val))
            
            for i, c in enumerate(val):
                source_seq.append((c, orig_idx + i, get_char_type(c)))
                if is_ascii_bypass:
                    bypass_indices.add(char_idx)
                    ascii_overrides[char_idx] = val if i == 0 else "---"
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
                              旧形式モデル（tagger_boundary が None）の場合は
                              raw_labels の +S を引き継いで互換動作する。
        """
        src_features = compute_source_features(source_seq)

        # 🌟 読みモデルの推論
        self.tagger_read.set(src_features)
        raw_labels = list(self.tagger_read.tag())

        # 🌟 境界モデルの推論
        if self.tagger_boundary is not None:
            # 新形式: 専用の境界モデルで推論
            self.tagger_boundary.set(src_features)
            boundary_labels = list(self.tagger_boundary.tag())
        else:
            # 旧形式との後方互換: 読みラベルの +S から境界を復元
            boundary_labels = ["1" if MORA_SPLIT in lbl else "0" for lbl in raw_labels]

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
        """
        変更点:
            - raw_labels はすでに +S を含まない読みのみのラベル列
            - has_splits は boundary_labels（"0"/"1"）から構築する
            - それ以外のロジックは旧版と同一
        """
        refined_labels = []
        confidences = []
        has_splits = []
        last_fallback_reading = ""

        parent_idx = -1

        for i, (char, _, ctype) in enumerate(source_seq):
            raw_clean = raw_labels[i]  # 🌟 すでに +S なし

            if raw_clean not in ("---", "_"):
                parent_idx = i

            if i in bypass_indices:
                clean_label = ascii_overrides[i]
                confidence  = 1.0
                last_fallback_reading = ""
                has_split = False
            else:
                label      = raw_labels[i]
                confidence = self.tagger_read.marginal(label, i)

                # 🚨 文字消失バグの救済ロジック（旧版と同一）
                if label == "---" and (parent_idx == -1 or parent_idx in bypass_indices):
                    if ctype == 'KANJI' and char in FALLBACK_DICT:
                        label = FALLBACK_DICT[char]["on"]
                    else:
                        label = char
                    confidence = 0.0

                # 通常のフォールバック処理（旧版と同一）
                label, last_fallback_reading, _ = self._apply_fallback(
                    i, char, ctype, label, confidence, source_seq, last_fallback_reading
                )
                clean_label = label  # 🌟 +S はすでにないのでそのまま

                # 🌟 境界は boundary_labels から取得
                has_split = (boundary_labels[i] == "1")

            refined_labels.append(clean_label)
            confidences.append(confidence)
            has_splits.append(has_split)

        return refined_labels, confidences, has_splits

    def _apply_fallback(self, i: int, char: str, ctype: str, label: str, confidence: float, source_seq: List[SourceEntry], last_fallback: str) -> Tuple[str, str, bool]:
        """無変更"""
        is_applied = False
        has_split  = False  # 🌟 +S はラベルに含まれないので常にFalse（境界はboundary_labelsが管理）
        new_label = label
        new_last_fallback = last_fallback

        if ctype == 'KANJI' and confidence < CONFIDENCE_THRESHOLD and char in FALLBACK_DICT:
            next_char  = source_seq[i + 1][0] if i < len(source_seq) - 1 else ""
            next_ctype = source_seq[i + 1][2] if i < len(source_seq) - 1 else ""
            
            if next_ctype == 'HIRAGANA' and next_char in SAFE_OKURIGANA:
                replacement_reading = FALLBACK_DICT[char]["kun"]
            else:
                replacement_reading = FALLBACK_DICT[char]["on"]
            
            new_label = replacement_reading  # 🌟 +S なし
            new_last_fallback = replacement_reading
            is_applied = True
            
        elif char == '々' and last_fallback:
            new_label = last_fallback  # 🌟 +S なし
            new_last_fallback = last_fallback
            is_applied = True
            
        if not is_applied and ctype != 'SYMBOL':
            new_last_fallback = ""

        return new_label, new_last_fallback, is_applied


    # ==========================================
    # 🌟 ステップ 4: 結果の組み立て（無変更）
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
            
            if clean_label == "_":
                pass
            elif clean_label != "---":
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