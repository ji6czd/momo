"""モデルZIPに詰めるシリアライズ定義。

trainer が学習結果をこの形にまとめて joblib で ZIP に書き込み、exporter が
量子化のために読み戻す。両者が共有する唯一の定義なので、どちらにも依存しない
このモジュールに置く（trainer → exporter の import 方向があるため、exporter が
trainer から取ることはできない）。

joblib（pickle）はクラスの完全修飾名を記録するため、このモジュールを移動・改名
すると既存の ZIP は読めなくなる（＝再学習が必要になる）ことに注意。
"""

from dataclasses import dataclass
from typing import Any

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer

# 単一漢字辞書のファイル名（モデルZIPへの同梱名・パッケージリソース名と共通）
SINGLE_KANJI_DICT_FILENAME = "single_character_dic.tsv"


@dataclass
class LRModelBundle:
    """ZIPに格納するモデル一式"""

    vectorizer_read: DictVectorizer
    coef_read_sparse: Any
    intercept_read: Any
    read_classes: Any
    vectorizer_boundary: DictVectorizer
    model_boundary: SGDClassifier
    version_info: dict
