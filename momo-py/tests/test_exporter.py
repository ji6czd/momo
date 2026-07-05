"""
exporter.py の単体テスト
  - parse_feature_key()（人名フラグ系）
  - export()（人名辞書テーブルの書き出し）
"""
import io
import json
import struct
import zipfile

import joblib
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier

from momo_py.exporter import export, parse_feature_key, FT
from momo_py.predictor import LRModelBundle


# ------------------------------------------------------------------ #
# parse_feature_key（人名フラグ系）
# ------------------------------------------------------------------ #
class TestParseNameFlagKeys:
    def test_name_s(self):
        assert parse_feature_key("name_s=B") == (FT.NAME_FLAG_SELF, [], [], 1)
        assert parse_feature_key("name_s=I") == (FT.NAME_FLAG_SELF, [], [], 2)

    def test_name_p1(self):
        assert parse_feature_key("name_p1=B") == (FT.NAME_FLAG_PREV1, [], [], 1)
        assert parse_feature_key("name_p1=I") == (FT.NAME_FLAG_PREV1, [], [], 2)

    def test_name_n1(self):
        assert parse_feature_key("name_n1=B") == (FT.NAME_FLAG_NEXT1, [], [], 1)
        assert parse_feature_key("name_n1=I") == (FT.NAME_FLAG_NEXT1, [], [], 2)

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="未知の特徴量キー"):
            parse_feature_key("name_s=X")


# ------------------------------------------------------------------ #
# export（人名辞書テーブル）
# ------------------------------------------------------------------ #
def _make_model_zip(tmp_path, name_dict_text=None):
    """人名フラグ特徴量を含む最小のモデルZIPを組み立てる。"""
    feats = [
        {"bias": 1.0, "char_s=佐": 1.0, "name_s=B": 1.0, "name_n1=I": 1.0},
        {"bias": 1.0, "char_s=藤": 1.0, "name_s=I": 1.0, "name_p1=B": 1.0},
    ]
    vect = DictVectorizer(sparse=True)
    X = vect.fit_transform(feats)
    X.indices = X.indices.astype(np.int32, copy=False)
    X.indptr = X.indptr.astype(np.int32, copy=False)
    n_feat = len(vect.vocabulary_)

    coef = csr_matrix(
        np.array([[0.5] * n_feat, [-0.25] * n_feat], dtype=np.float32)
    )
    boundary = SGDClassifier(loss="modified_huber", max_iter=10)
    boundary.fit(X, ["0", "1"])

    bundle = LRModelBundle(
        vectorizer_read=vect,
        coef_read_sparse=coef,
        intercept_read=np.zeros(2, dtype=np.float32),
        read_classes=np.array(["サ", "トー"]),
        vectorizer_boundary=vect,
        model_boundary=boundary,
        version_info={},
    )
    buf = io.BytesIO()
    joblib.dump(bundle, buf)

    zip_path = tmp_path / "model.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("model_bundle.pkl", buf.getvalue())
        zf.writestr(
            "version_info.json", json.dumps({"model_bundle": "model_bundle.pkl"})
        )
        if name_dict_text is not None:
            zf.writestr("person_name_dic.tsv", name_dict_text)
    return zip_path


class TestExportNameDict:
    def test_with_name_dict(self, tmp_path):
        zip_path = _make_model_zip(
            tmp_path,
            "# コメント\n#表層形\t読み\t出現回数\n佐藤\tサ/トー\t2\n太郎\t1\n",
        )
        out = tmp_path / "model.mbm"
        export(str(zip_path), str(out))

        data = out.read_bytes()
        assert data[:4] == b"MOMO"
        assert data[4] == 0x04  # version

        # ファイル末尾 = 人名辞書テーブル（表層形 + ユニット別読み）
        expected = bytearray(struct.pack("<I", 2))
        # 佐藤: 読みあり（サ, トー）
        encoded = "佐藤".encode("utf-8")
        expected.append(len(encoded))
        expected += encoded
        expected.append(2)  # n_readings
        for reading in ["サ", "トー"]:
            r_enc = reading.encode("utf-8")
            expected.append(len(r_enc))
            expected += r_enc
        # 太郎: 旧形式（出現回数のみ）→ 読みなし
        encoded = "太郎".encode("utf-8")
        expected.append(len(encoded))
        expected += encoded
        expected.append(0)  # n_readings = 0
        assert data.endswith(bytes(expected))

    def test_without_name_dict(self, tmp_path):
        zip_path = _make_model_zip(tmp_path, None)
        out = tmp_path / "model.mbm"
        export(str(zip_path), str(out))

        data = out.read_bytes()
        assert data[4] == 0x04
        # 辞書なしモデルは n_names = 0
        assert data.endswith(struct.pack("<I", 0))
