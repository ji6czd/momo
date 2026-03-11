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
                crfsuite_files = [n for n in namelist if n.endswith(".crfsuite")]
                if not crfsuite_files:
                    raise ValueError(f"❌ ZIPファイル内に .crfsuite ファイルが見つかりません: {model_path}")
                self._tmp_dir = tempfile.mkdtemp()
                zf.extract(crfsuite_files[0], self._tmp_dir)
                actual_model_path = os.path.join(self._tmp_dir, crfsuite_files[0])
        else:
            actual_model_path = model_path

        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(actual_model_path)

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
        raw_labels = self._run_inference(source_seq)

        # 3. 後処理（バイパスの適用・フォールバック・自信度の確定）
        refined_labels, raw_confidences, has_splits = self._refine_predictions(
            source_seq, raw_labels, bypass_indices, ascii_overrides
        )

        # 4. 結果の組み立て（文字列とインデックスの最終マッピング）
        return self._assemble_result(
            text, source_seq, refined_labels, raw_confidences, has_splits, bypass_indices
        )


    # ==========================================
    # 🌟 ステップ 1: 前処理
    # ==========================================
    def _preprocess_text(self, text: str) -> Tuple[List[SourceEntry], Set[int], Dict[int, str]]:
        units_info = get_units(text)
        source_seq: List[SourceEntry] = []
        bypass_indices: Set[int] = set()
        ascii_overrides: Dict[int, str] = {}
        
        char_idx = 0
        for val, orig_idx in units_info:
            # ASCII文字列（英数字・記号・内部スペース）の塊かを判定
            is_ascii_bypass = bool(re.fullmatch(r'[!-~]+(?:[ \t]+[!-~]+)*', val))
            
            for i, c in enumerate(val):
                source_seq.append((c, orig_idx + i, get_char_type(c)))
                # バイパス対象の場合、先頭に全文字列を詰め、以降はプレースホルダー
                if is_ascii_bypass:
                    bypass_indices.add(char_idx)
                    ascii_overrides[char_idx] = val if i == 0 else "---"
                char_idx += 1
                
        return source_seq, bypass_indices, ascii_overrides


    # ==========================================
    # 🌟 ステップ 2: 推論
    # ==========================================
    def _run_inference(self, source_seq: List[SourceEntry]) -> List[str]:
        src_features = compute_source_features(source_seq)
        self.tagger.set(src_features)
        return list(self.tagger.tag())


    # ==========================================
    # 🌟 ステップ 3: 後処理（バイパスとフォールバック）
    # ==========================================
    def _refine_predictions(self, source_seq: List[SourceEntry], raw_labels: List[str], bypass_indices: Set[int], ascii_overrides: Dict[int, str]) -> Tuple[List[str], List[float], List[bool]]:
        refined_labels = []
        confidences = []
        has_splits = []
        last_fallback_reading = ""

        parent_idx = -1  # 🌟 各文字の「読みの親（AIが予測したブロックの先頭）」のインデックス

        for i, (char, _, ctype) in enumerate(source_seq):
            # 現在の文字に対するAIの生の予測（システムタグを除去して判定）
            raw_clean = raw_labels[i].replace(MORA_SPLIT, "")
            
            # 親の更新: 自分が何らかの読みを持っている（"---" や "_" でない）なら、自分が親になる
            if raw_clean not in ("---", "_"):
                parent_idx = i

            if i in bypass_indices:
                # バイパス対象（英語など）: 推論を上書きして100%の自信度を設定
                clean_label = ascii_overrides[i]
                confidence = 1.0
                last_fallback_reading = ""
                has_split = False
            else:
                # 通常対象（日本語）
                label = raw_labels[i]
                confidence = self.tagger.marginal(label, i)
                
                # 🚨 文字消失バグの救済ロジック
                # 自分が "---" (親に吸収される予定) なのに、
                # 親が未定義 または 親がバイパス対象(上書き済み) の場合、自分の読みが消失してしまう！
                raw_clean_current = label.replace(MORA_SPLIT, "")
                if raw_clean_current == "---" and (parent_idx == -1 or parent_idx in bypass_indices):
                    # 自己救済: 辞書にあれば辞書読み（音読み）、なければ文字自身を強制設定
                    if ctype == 'KANJI' and char in FALLBACK_DICT:
                        label = FALLBACK_DICT[char]["on"]
                    else:
                        label = char
                    confidence = 0.0  # フォールバック適用扱いとする
                    
                    # ⚠️ ここで parent_idx は更新しない（連続して消失する文字をすべて救済するため）

                # 通常のフォールバック処理
                label, last_fallback_reading, _ = self._apply_fallback(
                    i, char, ctype, label, confidence, source_seq, last_fallback_reading
                )
                clean_label = label.replace(MORA_SPLIT, "")
                has_split = (MORA_SPLIT in label)

            refined_labels.append(clean_label)
            confidences.append(confidence)
            has_splits.append(has_split)

        return refined_labels, confidences, has_splits

    def _apply_fallback(self, i: int, char: str, ctype: str, label: str, confidence: float, source_seq: List[SourceEntry], last_fallback: str) -> Tuple[str, str, bool]:
        is_applied = False
        has_split = (MORA_SPLIT in label)
        new_label = label
        new_last_fallback = last_fallback

        if ctype == 'KANJI' and confidence < CONFIDENCE_THRESHOLD and char in FALLBACK_DICT:
            next_char = source_seq[i + 1][0] if i < len(source_seq) - 1 else ""
            next_ctype = source_seq[i + 1][2] if i < len(source_seq) - 1 else ""
            
            if next_ctype == 'HIRAGANA' and next_char in SAFE_OKURIGANA:
                replacement_reading = FALLBACK_DICT[char]["kun"]
            else:
                replacement_reading = FALLBACK_DICT[char]["on"]
            
            new_label = replacement_reading + (MORA_SPLIT if has_split else "")
            new_last_fallback = replacement_reading
            is_applied = True
            
        elif char == '々' and last_fallback:
            new_label = last_fallback + (MORA_SPLIT if has_split else "")
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
            confidence = raw_confidences[i]
            
            if clean_label == "_":
                pass
            elif clean_label != "---":
                if i in bypass_indices:
                    # バイパス（英語）は1文字ずつ1:1マッピング
                    for j, ch in enumerate(clean_label):
                        translated += ch
                        target_orig_idx = orig_idx + j
                        kana_to_src_index.append(target_orig_idx)
                        final_confidences.append(confidence)
                        src_to_kana_index[target_orig_idx].append(kana_pos)
                        kana_pos += 1
                else:
                    # 日本語はブロック全体を先頭文字にマッピング
                    for ch in clean_label:
                        translated += ch
                        kana_to_src_index.append(orig_idx)
                        final_confidences.append(confidence)
                        src_to_kana_index[orig_idx].append(kana_pos)
                        kana_pos += 1

            # 分かち書きスペースの追加
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