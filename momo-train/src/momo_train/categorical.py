"""境界モデル（GBDT）向けのカテゴリカル特徴量表現。

`features.compute_source_features` は pycrfsuite ネイティブの
`{"name=value": 1.0}` 形式（DictVectorizerでone-hot展開する前提）で特徴量を返す。
これは線形モデル（LinearSVC/SGDClassifier）には適しているが、GBDT（LightGBM）に
one-hotのまま渡すと特徴量次元が数十万〜百万に膨れ上がり、木の分岐が疎な列を
手探りすることになって過学習しやすい（実験で確認済み）。

ここでは同じ特徴量dictを「特徴量名ごとに1列、値は整数コード」という数十列の
密な表現に変換し、LightGBMのネイティブcategorical_feature機構に渡せるようにする。
列数が少ないぶん、木は「この特徴量のどの値集合が正例寄りか」を効率よく学習できる。
"""

from typing import Dict, List, Tuple

import numpy as np

from .features import F_BIAS, FeatureDict

CategoricalVocab = Dict[str, Dict[str, int]]


def _feature_name_value(key: str) -> Tuple[str, str]:
    """特徴量キー 'name=value' を (name, value) に分解する（値なし系はvalue='1'）。"""
    if "=" in key:
        name, _, val = key.partition("=")
        return name, val
    return key, "1"


def fit_categorical(dicts: List[FeatureDict]) -> Tuple[np.ndarray, List[str], CategoricalVocab]:
    """FeatureDict列から、特徴量名ごとに1列の整数コード行列と語彙表を新規構築する（訓練用）。

    戻り値: (行列 (n, m) float64・NaN=欠損, 列名リスト（ソート済み）, 語彙表 {列名: {値: コード}})
    """
    names = set()
    row_maps = []
    for d in dicts:
        row = {}
        for k in d:
            if k == F_BIAS:
                continue
            name, val = _feature_name_value(k)
            row[name] = val
            names.add(name)
        row_maps.append(row)

    names_sorted = sorted(names)
    vocabs: CategoricalVocab = {name: {} for name in names_sorted}
    n, m = len(row_maps), len(names_sorted)
    mat = np.full((n, m), np.nan, dtype=np.float64)
    for i, row in enumerate(row_maps):
        for j, name in enumerate(names_sorted):
            val = row.get(name)
            if val is None:
                continue
            vocab = vocabs[name]
            mat[i, j] = vocab.setdefault(val, len(vocab))
    return mat, names_sorted, vocabs


def transform_categorical(
    dicts: List[FeatureDict], names_sorted: List[str], vocabs: CategoricalVocab
) -> np.ndarray:
    """fit_categorical で作った語彙表を使って、別データを同じ列・コード体系に変換する（推論/検証用）。

    訓練時に出現しなかった値は NaN（LightGBM の欠損値扱い）にする。
    """
    name_index = {name: j for j, name in enumerate(names_sorted)}
    n, m = len(dicts), len(names_sorted)
    mat = np.full((n, m), np.nan, dtype=np.float64)
    for i, d in enumerate(dicts):
        for k in d:
            if k == F_BIAS:
                continue
            name, val = _feature_name_value(k)
            j = name_index.get(name)
            if j is None:
                continue
            code = vocabs[name].get(val)
            if code is None:
                continue
            mat[i, j] = code
    return mat
