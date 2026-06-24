"""
predictor.py の単体テスト
  - PredictionResult.to_json()
  - Predictor.__init__() のエラーハンドリング
"""
import json
from pathlib import Path
import pytest
from momo_py.predictor import PredictionResult, Predictor, PredictorConfig


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
