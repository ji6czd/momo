import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from typing import List, Tuple

import pycrfsuite

from .features import get_units, get_char_type, compute_source_features, SourceEntry, MORA_SPLIT
from .fallback_dict import FALLBACK_DICT

# フォールバックを発動させる自信度の境界線（30%）
CONFIDENCE_THRESHOLD = 0.3

# 🌟 訓読みを誘発しやすい「安全な送り仮名」のリスト
# ※「す」「し」（サ変＋する等）や「に」「を」「は」等の助詞になりやすい文字は除外
SAFE_OKURIGANA = set(
    "あいうえお"
    "かきくけこ"
    "たちつてと"
    "なにぬねの"
    "まみむめも"
    "やゆよ"
    "らりるれろ"
    "わ"
    "ばびぶべぼ"
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

    # 🌟 切り出したフォールバック処理の関数（メソッド）
    def _apply_fallback(self, i: int, char: str, ctype: str, label: str, confidence: float, source_seq: List[SourceEntry], last_fallback: str) -> Tuple[str, str, bool]:
        """
        AIの予測結果に対して、必要に応じて単漢字辞書による上書き（フォールバック）を行う。
        戻り値: (上書き後のラベル, 新しい last_fallback の値, フォールバックが適用されたかどうかのフラグ)
        """
        is_applied = False
        has_split = (MORA_SPLIT in label)
        new_label = label
        new_last_fallback = last_fallback

        if ctype == 'KANJI' and confidence < CONFIDENCE_THRESHOLD and char in FALLBACK_DICT:
            # 次の文字と文字種を取得
            next_char = source_seq[i + 1][0] if i < len(source_seq) - 1 else ""
            next_ctype = source_seq[i + 1][2] if i < len(source_seq) - 1 else ""
            
            # 安全な送り仮名リストに含まれるひらがなか判定
            if next_ctype == 'HIRAGANA' and next_char in SAFE_OKURIGANA:
                replacement_reading = FALLBACK_DICT[char]["kun"]
            else:
                replacement_reading = FALLBACK_DICT[char]["on"]
            
            new_label = replacement_reading + (MORA_SPLIT if has_split else "")
            new_last_fallback = replacement_reading
            is_applied = True
            
        elif char == '々' and last_fallback:
            # 直前が辞書置換されていて「々」が来た場合
            new_label = last_fallback + (MORA_SPLIT if has_split else "")
            new_last_fallback = last_fallback
            is_applied = True
            
        # 辞書が適用されず、記号でもなかった場合は、記憶をリセット
        if not is_applied and ctype != 'SYMBOL':
            new_last_fallback = ""

        return new_label, new_last_fallback, is_applied


    def predict(self, text: str) -> PredictionResult:
        units_info = get_units(text)
        source_seq: List[SourceEntry] = []
        for val, idx in units_info:
            for i, c in enumerate(val):
                source_seq.append((c, idx + i, get_char_type(c)))

        if not source_seq:
            return PredictionResult(text, "", [], [], [[] for _ in text])

        src_features = compute_source_features(source_seq)
        
        self.tagger.set(src_features)
        y_pred = self.tagger.tag()
        
        translated: str = ""
        kana_to_src_index: List[int] = []
        confidences: List[float] = []
        src_to_kana_index: List[List[int]] = [[] for _ in text]
        kana_pos = 0
        
        last_fallback_reading = ""

        for i, (char, orig_idx, ctype) in enumerate(source_seq):
            label = y_pred[i]
            confidence = self.tagger.marginal(label, i)
                            # confidence = self.tagger.marginal(label, i)            
            label, last_fallback_reading, _ = self._apply_fallback(
                i, char, ctype, label, confidence, source_seq, last_fallback_reading
            )

            clean_label = label.replace(MORA_SPLIT, "")

            if clean_label == "_":
                pass
            elif clean_label != "---":
                for ch in clean_label:
                    translated += ch
                    kana_to_src_index.append(orig_idx)
                    confidences.append(confidence)
                    src_to_kana_index[orig_idx].append(kana_pos)
                    kana_pos += 1

            if MORA_SPLIT in label:
                translated += " "
                kana_to_src_index.append(orig_idx)
                confidences.append(confidence)
                src_to_kana_index[orig_idx].append(kana_pos)
                kana_pos += 1

        return PredictionResult(
            source_text=text,
            kana_text=translated,
            confidences=confidences,
            kana_to_src_index=kana_to_src_index,
            src_to_kana_index=src_to_kana_index
        )