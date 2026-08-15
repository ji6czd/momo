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

from momo_py.exporter import (
    FT,
    char32_count,
    chartype_count,
    export,
    export_float,
    is_uint8_payload,
    parse_feature_key,
)
from momo_py.bundle import LRModelBundle


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
def _make_model_zip(tmp_path, name_dict_text=None, single_char_dict_text="漢\tカン\tゲン\n"):
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
        if single_char_dict_text is not None:
            zf.writestr("single_character_dic.tsv", single_char_dict_text)
    return zip_path


def _expected_single_char_section() -> bytes:
    """フィクスチャの単一文字辞書（漢→カン,ゲン）に対応するテーブルバイト列。"""
    expected = bytearray(struct.pack("<I", 1))
    encoded = "漢".encode("utf-8")
    expected.append(len(encoded))
    expected += encoded
    expected.append(2)  # n_readings
    for reading in ["カン", "ゲン"]:
        r_enc = reading.encode("utf-8")
        expected.append(len(r_enc))
        expected += r_enc
    return bytes(expected)


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
        assert data[4] == 0x07  # version

        # 人名辞書テーブル（表層形 + ユニット別読み）+ 末尾に単一文字辞書テーブル
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
        expected += _expected_single_char_section()
        assert data.endswith(bytes(expected))

    def test_without_name_dict(self, tmp_path):
        zip_path = _make_model_zip(tmp_path, None)
        out = tmp_path / "model.mbm"
        export(str(zip_path), str(out))

        data = out.read_bytes()
        assert data[4] == 0x07
        # 辞書なしモデルは n_names = 0、続けて単一文字辞書テーブル
        assert data.endswith(struct.pack("<I", 0) + _expected_single_char_section())


# ------------------------------------------------------------------ #
# export_float（.mbmf: 量子化前 float32 サイドカー）
# ------------------------------------------------------------------ #
def _skip_vocab(buf: bytes, offset: int, n_features: int, has_cat: bool = False) -> int:
    """統合語彙テーブル（version 0x07）を読み飛ばして直後のオフセットを返す。

    feature_id は行番号で暗黙なので格納されない。`has_cat`（GBDT境界）のとき、
    各エントリ末尾に column(u32)+code(u32) が付く。
    """
    for _ in range(n_features):
        ft = buf[offset]
        offset += 1
        offset += chartype_count(ft)
        offset += char32_count(ft) * 4
        if is_uint8_payload(ft):
            offset += 1
        if has_cat:
            offset += 8  # cat_column (u32) + cat_code (u32)
    return offset


def _skip_labels(buf: bytes, offset: int, n_classes: int) -> int:
    """読みラベルテーブルを読み飛ばして直後のオフセットを返す。"""
    for _ in range(n_classes):
        length = buf[offset]
        offset += 1 + length
    return offset


def _read_csc_weights(buf: bytes, offset: int, n_features: int, value_fmt: str):
    """読みモデル重み（CSC）を読む。`.mbm` / `.mbmf` で疎構造は同一、値の型だけが違う。

    `value_fmt` は struct の書式（`.mbm` は "b" = int8、`.mbmf` は "f" = float32）。
    戻り値: (colptr, rowind, data, next_offset)
    """
    (n_nonzero,) = struct.unpack_from("<I", buf, offset)
    offset += 4
    colptr = struct.unpack_from(f"<{n_features + 1}I", buf, offset)
    offset += 4 * (n_features + 1)
    rowind = struct.unpack_from(f"<{n_nonzero}H", buf, offset)
    offset += 2 * n_nonzero
    data = struct.unpack_from(f"<{n_nonzero}{value_fmt}", buf, offset)
    offset += struct.calcsize(f"<{n_nonzero}{value_fmt}")
    return colptr, rowind, data, offset


class TestExportFloat:
    def test_header_magic_and_version(self, tmp_path):
        zip_path = _make_model_zip(tmp_path, None)
        out = tmp_path / "model.mbmf"
        export_float(str(zip_path), str(out))

        data = out.read_bytes()
        assert data[:4] == b"MBMF"
        assert data[4] == 0x07  # version（.mbm と共有。区別は magic だけで行う）

    def test_read_weights_are_plain_float32(self, tmp_path):
        zip_path = _make_model_zip(tmp_path, None)
        out = tmp_path / "model.mbmf"
        export_float(str(zip_path), str(out))

        data = out.read_bytes()
        n_classes, n_features = struct.unpack_from("<II", data, 8)

        offset = _skip_vocab(data, 16, n_features)
        offset = _skip_labels(data, offset, n_classes)

        colptr, rowind, read_data, _ = _read_csc_weights(data, offset, n_features, "f")

        # _make_model_zip の coef は全要素が非ゼロ（0.5 / -0.25 の密行列）
        assert len(read_data) == n_classes * n_features
        assert colptr[-1] == n_classes * n_features
        # CSC なので値は列順に並ぶ。値はクラス (行) で決まるので rowind で引く。
        expected = {0: 0.5, 1: -0.25}
        for row, v in zip(rowind, read_data):
            assert abs(v - expected[row]) < 1e-6

    def test_matches_quantized_export_within_tolerance(self, tmp_path):
        """.mbm（量子化）と .mbmf（非量子化）が同じ .zip から生成されたとき、
        int8 * scale ≈ float32 の関係が読みモデル・境界モデル双方で成り立つこと。"""
        zip_path = _make_model_zip(tmp_path, None)
        mbm_path = tmp_path / "model.mbm"
        mbmf_path = tmp_path / "model.mbmf"
        export(str(zip_path), str(mbm_path))
        export_float(str(zip_path), str(mbmf_path))

        mbm = mbm_path.read_bytes()
        mbmf = mbmf_path.read_bytes()

        n_classes, n_features = struct.unpack_from("<II", mbmf, 8)
        assert struct.unpack_from("<II", mbm, 8) == (n_classes, n_features)

        # --- .mbmf: 読みモデル重み + 境界モデル重み ---
        off_f = _skip_vocab(mbmf, 16, n_features)
        off_f = _skip_labels(mbmf, off_f, n_classes)
        _, rowind_f, read_data_f, off_f = _read_csc_weights(mbmf, off_f, n_features, "f")
        off_f += 4 * n_classes  # intercept_read
        boundary_data_f = struct.unpack_from(f"<{n_features}f", mbmf, off_f)

        # --- .mbm: 読みモデル重み（量子化）+ 境界モデル重み（量子化）---
        off_q = _skip_vocab(mbm, 16, n_features)
        off_q = _skip_labels(mbm, off_q, n_classes)
        scales_r = struct.unpack_from(f"<{n_classes}f", mbm, off_q)
        off_q += 4 * n_classes
        _, rowind_q, data_q, off_q = _read_csc_weights(mbm, off_q, n_features, "b")
        off_q += 4 * n_classes  # intercept_read
        (scale_b,) = struct.unpack_from("<f", mbm, off_q)
        off_q += 4
        boundary_data_q = struct.unpack_from(f"<{n_features}b", mbm, off_q)

        # 疎構造は量子化の有無に関係なく同一のはず
        assert rowind_f == rowind_q

        # 量子化 scale はクラス (行) ごとなので、CSC では rowind から引く
        for j, row in enumerate(rowind_q):
            dequantized = data_q[j] * scales_r[row]
            assert abs(dequantized - read_data_f[j]) < abs(scales_r[row]) + 1e-6

        for q, f in zip(boundary_data_q, boundary_data_f):
            dequantized = q * scale_b
            assert abs(dequantized - f) < abs(scale_b) + 1e-6


# ------------------------------------------------------------------ #
# 統合語彙テーブル（version 0x07）: カテゴリカルの埋め込みと cat⊆読み 強制
# ------------------------------------------------------------------ #
from momo_py.exporter import (  # noqa: E402
    NO_CAT_COLUMN,
    _build_cat_by_feature_id,
    _build_vocab_bytes,
)


class TestUnifiedVocab:
    def test_cat_by_feature_id_maps_reading_ids(self):
        # 読み語彙のキー文字列とカテゴリカルのキー文字列は、正準化すれば一致する。
        vocab = {"bias": 0, "char_s=漢": 1, "char_s=字": 2}
        cat_names = ["char_s"]
        cat_vocabs = {"char_s": {"漢": 0, "字": 1}}
        cat_by_id = _build_cat_by_feature_id(vocab, cat_names, cat_vocabs)
        assert cat_by_id == {1: (0, 0), 2: (0, 1)}

    def test_cat_not_in_reading_vocab_raises(self):
        # カテゴリカルキーが読み語彙に無ければ、統合語彙に居場所がないので失敗させる。
        vocab = {"bias": 0, "char_s=漢": 1}
        cat_names = ["char_s"]
        cat_vocabs = {"char_s": {"雨": 0}}  # char_s=雨 は読み語彙に無い
        with pytest.raises(ValueError, match="読み語彙に存在しません"):
            _build_cat_by_feature_id(vocab, cat_names, cat_vocabs)

    def test_vocab_bytes_omits_feature_id(self):
        # feature_id は行番号で暗黙。線形（cat_by_id=None）では column/code も書かない。
        vocab = {"bias": 0, "char_s=漢": 1}
        payload_only = _build_vocab_bytes(vocab, cat_by_id=None)
        # bias(ft 1byte) + char_s=漢(ft 1 + cp 4) = 1 + 5 = 6 bytes、feature_id なし
        assert len(payload_only) == 1 + (1 + 4)

    def test_vocab_bytes_embeds_cat(self):
        # GBDT（cat_by_id あり）では各エントリ末尾に (column, code)。cat の無い bias は番兵。
        vocab = {"bias": 0, "char_s=漢": 1}
        cat_by_id = {1: (0, 5)}
        data = _build_vocab_bytes(vocab, cat_by_id=cat_by_id)
        # bias: ft(1) + column(4)+code(4)。char_s=漢: ft(1)+cp(4) + column(4)+code(4)
        assert len(data) == (1 + 8) + (1 + 4 + 8)
        # bias(feature_id 0): 番兵、続いて char_s=漢(feature_id 1): (0, 5)
        bias_col, bias_code = struct.unpack_from("<II", data, 1)
        assert (bias_col, bias_code) == (NO_CAT_COLUMN, 0)
        kanji_col, kanji_code = struct.unpack_from("<II", data, (1 + 8) + 1 + 4)
        assert (kanji_col, kanji_code) == (0, 5)
