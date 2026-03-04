import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from typing import List

# 🌟 joblib と sklearn_crfsuite を削除し、本家 pycrfsuite をインポート
import pycrfsuite

from .features import get_units, get_char_type, compute_source_features, SourceEntry, MORA_SPLIT

@dataclass
class PredictionResult:
    """
    予測結果を保持する専用のデータクラス（入れ物）
    """
    source_text: str
    kana_text: str
    confidences: List[float]
    kana_to_src_index: List[int]
    src_to_kana_index: List[List[int]]

    def to_json(self) -> str:
        """自身を人間が読みやすい綺麗なJSON文字列に変換する"""
        text_safe = json.dumps(self.source_text, ensure_ascii=False)
        kana_safe = json.dumps(self.kana_text, ensure_ascii=False)
        
        # 配列を1行にまとめる
        conf_str = "[" + ", ".join([f"{c:.3f}" for c in self.confidences]) + "]"
        k2s_str = "[" + ", ".join(map(str, self.kana_to_src_index)) + "]"
        s2k_str = json.dumps(self.src_to_kana_index) # リストのリストはjson.dumpsが確実
        
        return f'{{\n  "text": {text_safe},\n  "kana": {kana_safe},\n  "kana_to_src_index": {k2s_str},\n  "src_to_kana_index": {s2k_str},\n  "confidences": {conf_str}\n}}'


class Predictor:
    """
    CRFモデル（ネイティブバイナリ）をロードし、テキストの推論（点訳）を行うエンジンクラス
    """
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ モデル未検出: {model_path}")

        self._version_info: dict | None = None
        self._tmp_dir: str | None = None

        if model_path.endswith(".zip"):
            with zipfile.ZipFile(model_path, "r") as zf:
                namelist = zf.namelist()
                # バージョン情報を読み込む
                if "version_info.json" in namelist:
                    self._version_info = json.loads(
                        zf.read("version_info.json").decode("utf-8")
                    )
                # .crfsuite ファイルを一時ディレクトリに展開
                crfsuite_files = [n for n in namelist if n.endswith(".crfsuite")]
                if not crfsuite_files:
                    raise ValueError(
                        f"❌ ZIPファイル内に .crfsuite ファイルが見つかりません: {model_path}"
                    )
                self._tmp_dir = tempfile.mkdtemp()
                zf.extract(crfsuite_files[0], self._tmp_dir)
                actual_model_path = os.path.join(self._tmp_dir, crfsuite_files[0])
        else:
            actual_model_path = model_path

        # 🌟 pycrfsuite の Tagger を直接使い、バイナリファイルを安全にロード
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(actual_model_path)

    def __del__(self) -> None:
        """一時ディレクトリを削除してクリーンアップする"""
        if self._tmp_dir and os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def get_version_info(self) -> dict | None:
        """
        ZIP形式で読み込んだ場合に保存されているバージョン情報を返す。

        Returns:
            dict: version_info.json の内容（少なくとも ``trained_at`` キーを含む）。
                  .crfsuite ファイルを直接ロードした場合は ``None``。
        """
        return self._version_info

    def predict(self, text: str) -> PredictionResult:
        # 1. 共通関数を使ってソース文字系列を構築
        units_info = get_units(text)
        source_seq: List[SourceEntry] = []
        for val, idx in units_info:
            for i, c in enumerate(val):
                source_seq.append((c, idx + i, get_char_type(c)))

        if not source_seq:
            # 空文字が渡された場合の安全処理
            return PredictionResult(text, "", [], [], [[] for _ in text])

        # 2. 特徴量計算
        src_features = compute_source_features(source_seq)
        
        # 🌟 pycrfsuite による予測と確率（自信度）の取得
        # まず特徴量を Tagger にセットする
        self.tagger.set(src_features)
        
        # 最も確率の高いラベル列を取得
        y_pred = self.tagger.tag()
        
        # 各文字のラベルに対する確率（自信度）を計算
        # （sklearn_crfsuite の predict_marginals_single と同等の処理）
        y_marginals = []
        labels = self.tagger.labels() # モデルが知っている全てのラベル
        for i in range(len(src_features)):
            # その位置(i)での各ラベルの確率を辞書化
            probs = {label: self.tagger.marginal(label, i) for label in labels}
            y_marginals.append(probs)

        # 3. 結果構築用の変数
        translated: str = ""
        kana_to_src_index: List[int] = []
        confidences: List[float] = []
        
        # 原文から仮名への逆引き用（原文の文字数分、空のリストを用意）
        src_to_kana_index: List[List[int]] = [[] for _ in text]
        kana_pos = 0  # 構築中の仮名文字列の現在位置

        # 4. ラベルのデコードとインデックスの同期
        for i, (char, orig_idx, ctype) in enumerate(source_seq):
            label = y_pred[i]
            confidence = y_marginals[i].get(label, 0.0)
            
            clean_label = label.replace(MORA_SPLIT, "")

            if clean_label == "_":
                pass
            elif clean_label != "---":
                for ch in clean_label:
                    translated += ch
                    kana_to_src_index.append(orig_idx)
                    confidences.append(confidence)
                    
                    # 原文 -> 仮名のインデックスを記録
                    src_to_kana_index[orig_idx].append(kana_pos)
                    kana_pos += 1

            if MORA_SPLIT in label:
                translated += " "
                kana_to_src_index.append(orig_idx)
                confidences.append(confidence)
                
                # スペースも仮名の1文字としてインデックスに記録
                src_to_kana_index[orig_idx].append(kana_pos)
                kana_pos += 1

        return PredictionResult(
            source_text=text,
            kana_text=translated,
            confidences=confidences,
            kana_to_src_index=kana_to_src_index,
            src_to_kana_index=src_to_kana_index
        )