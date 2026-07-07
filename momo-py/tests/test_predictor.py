"""
predictor.py の単体テスト
  - PredictionResult.to_json()
  - Predictor.__init__() のエラーハンドリング
  - load_custom_dict()（学習データと同じ / / 記法）
"""
import json
from pathlib import Path
import pytest
from momo_py.predictor import (
    PredictionResult,
    Predictor,
    PredictorConfig,
    _remap_result_to_source,
    load_custom_dict,
)


# ------------------------------------------------------------------ #
# load_custom_dict
# ------------------------------------------------------------------ #
class TestLoadCustomDict:
    def _load(self, tmp_path, content):
        path = tmp_path / "custom.tsv"
        path.write_text(content, encoding="utf-8")
        return load_custom_dict(str(path))

    def test_basic_no_split(self, tmp_path):
        # 区切り指定なし = 1語（内部で区切らない）
        d = self._load(tmp_path, "超音波\tチョー/オン/パ\n")
        assert d["超音波"] == [("チョー", False), ("オン", False), ("パ", False)]

    def test_internal_split(self, tmp_path):
        # 空ブロック / / は「直前ユニットの後ろで区切る」
        d = self._load(tmp_path, "佐藤太郎\tサ/トー/ /タ/ロー\n")
        assert d["佐藤太郎"] == [
            ("サ", False),
            ("トー", True),
            ("タ", False),
            ("ロー", False),
        ]

    def test_leading_split_raises(self, tmp_path):
        with pytest.raises(ValueError, match="先頭に区切り指定"):
            self._load(tmp_path, "佐藤\t /サ/トー\n")

    def test_trailing_split_raises(self, tmp_path):
        with pytest.raises(ValueError, match="末尾に区切り指定"):
            self._load(tmp_path, "佐藤\tサ/トー/ \n")

    def test_consecutive_split_raises(self, tmp_path):
        with pytest.raises(ValueError, match="連続"):
            self._load(tmp_path, "佐藤太郎\tサ/トー/ / /タ/ロー\n")

    def test_unit_count_mismatch_raises(self, tmp_path):
        # 区切り指定は読みブロック数に数えない（3ユニットに2読み）
        with pytest.raises(ValueError, match="一致しません"):
            self._load(tmp_path, "超音波\tチョー/ /オンパ\n")

    def test_compound_unit(self, tmp_path):
        # 拗音を含む表層形はユニット単位（きゃ = 1ユニット）
        d = self._load(tmp_path, "きゃく\tキャ/ク\n")
        assert d["きゃく"] == [("キャ", False), ("ク", False)]


# ------------------------------------------------------------------ #
# _remap_result_to_source()
# ------------------------------------------------------------------ #
class TestRemapResultToSource:
    def test_ligature_expansion_remap(self):
        # 原文 "oﬃce"（4文字）→ 正規化 "office"（6文字）のケース。
        # パイプラインは正規化テキスト基準の結果を作る。
        result = PredictionResult(
            source_text="office",
            kana_text="office",
            confidences=[1.0] * 6,
            kana_to_src_index=[0, 1, 2, 3, 4, 5],
            src_to_kana_index=[[0], [1], [2], [3], [4], [5]],
        )
        _remap_result_to_source(result, "oﬃce", [0, 1, 1, 1, 2, 3])
        assert result.source_text == "oﬃce"
        assert result.kana_text == "office"  # 仮名は変わらない
        # ﬃ 由来の f/f/i はすべて原文位置1を指す
        assert result.kana_to_src_index == [0, 1, 1, 1, 2, 3]
        # 原文の1文字（ﬃ）に仮名3文字分が集約される
        assert result.src_to_kana_index == [[0], [1, 2, 3], [4], [5]]

    def test_identity_is_noop(self):
        result = PredictionResult(
            source_text="東京",
            kana_text="トーキョー",
            confidences=[1.0] * 5,
            kana_to_src_index=[0, 0, 1, 1, 1],
            src_to_kana_index=[[0, 1], [2, 3, 4]],
        )
        _remap_result_to_source(result, "東京", [0, 1])
        assert result.source_text == "東京"
        assert result.kana_to_src_index == [0, 0, 1, 1, 1]
        assert result.src_to_kana_index == [[0, 1], [2, 3, 4]]


# ------------------------------------------------------------------ #
# PredictionResult.to_json()
# ------------------------------------------------------------------ #
class TestPredictionResultToJson:
    def _make_result(self, **kwargs) -> PredictionResult:
        defaults = dict(
            source_text="東京",
            kana_text="トウキョウ",
            confidences=[0.9, 0.8, 0.95, 0.7, 0.85],
            kana_to_src_index=[0, 0, 0, 1, 1],
            src_to_kana_index=[[0, 1, 2], [3, 4]],
        )
        defaults.update(kwargs)
        return PredictionResult(**defaults)

    def test_valid_json(self):
        result = self._make_result()
        parsed = json.loads(result.to_json())
        assert parsed["text"] == "東京"
        assert parsed["kana"] == "トウキョウ"

    def test_contains_required_keys(self):
        result = self._make_result()
        parsed = json.loads(result.to_json())
        assert "text" in parsed
        assert "kana" in parsed
        assert "kana_to_src_index" in parsed
        assert "src_to_kana_index" in parsed
        assert "confidences" in parsed

    def test_confidence_length_matches_kana(self):
        result = self._make_result()
        parsed = json.loads(result.to_json())
        assert len(parsed["confidences"]) == len(result.kana_text)

    def test_kana_to_src_index_length(self):
        result = self._make_result()
        parsed = json.loads(result.to_json())
        assert len(parsed["kana_to_src_index"]) == len(result.kana_text)

    def test_src_to_kana_index_length(self):
        result = self._make_result()
        parsed = json.loads(result.to_json())
        assert len(parsed["src_to_kana_index"]) == len(result.source_text)

    def test_non_ascii_preserved(self):
        result = self._make_result(source_text="日本語", kana_text="ニホンゴ",
                                   confidences=[0.9, 0.8, 0.7, 0.95],
                                   kana_to_src_index=[0, 1, 2, 2],
                                   src_to_kana_index=[[0], [1], [2, 3]])
        parsed = json.loads(result.to_json())
        assert parsed["text"] == "日本語"
        assert parsed["kana"] == "ニホンゴ"

    def test_empty_input(self):
        result = PredictionResult(
            source_text="",
            kana_text="",
            confidences=[],
            kana_to_src_index=[],
            src_to_kana_index=[],
        )
        parsed = json.loads(result.to_json())
        assert parsed["text"] == ""
        assert parsed["kana"] == ""
        assert parsed["confidences"] == []


# ------------------------------------------------------------------ #
# Predictor.__init__()
# ------------------------------------------------------------------ #
class TestPredictorInit:
    def test_missing_model_raises(self, tmp_path: Path):
        config = PredictorConfig(model_path=str(tmp_path / "nonexistent.model"))
        with pytest.raises(FileNotFoundError, match="モデル未検出"):
            Predictor(config)
