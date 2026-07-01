"""
utils.py の単体テスト
  - normalize_compat_ideographs()
"""
from momo_py.utils import normalize_compat_ideographs


class TestNormalizeCompatIdeographs:
    def test_cjk_radicals_supplement(self):
        assert normalize_compat_ideographs("⺟") == "母"  # ⺟ -> 母
        assert normalize_compat_ideographs("⻳") == "龟"  # ⻳ -> 龟

    def test_kangxi_radical(self):
        assert normalize_compat_ideographs("⼀") == "一"  # ⼀ -> 一

    def test_cjk_compatibility_ideograph(self):
        # U+F900 と U+8C48 は見た目が同じ「豈」だが別コードポイント
        assert normalize_compat_ideographs("豈") == "豈"

    def test_cjk_compatibility_ideograph_supplement(self):
        assert normalize_compat_ideographs("\U0002f800") == "丽"  # 丽

    def test_passthrough_normal_kanji(self):
        assert normalize_compat_ideographs("漢字") == "漢字"

    def test_passthrough_non_kanji(self):
        assert normalize_compat_ideographs("あいうえおABC123") == "あいうえおABC123"
