"""
utils.py の単体テスト
  - normalize_compat_ideographs()
  - normalize_input()
"""
from momo_py.utils import normalize_compat_ideographs, normalize_input


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

    def test_ligature_is_not_expanded(self):
        # 1→1変換専用なので欧文リガチャは対象外（normalize_input が担当）
        assert normalize_compat_ideographs("ﬃ") == "ﬃ"


class TestNormalizeInput:
    def test_identity(self):
        text, index_map = normalize_input("あいうABC123")
        assert text == "あいうABC123"
        assert index_map == list(range(9))

    def test_latin_ligature_expansion(self):
        # ﬃ (U+FB03) → ffi。展開後の3文字すべてが原文の同じ位置を指す
        text, index_map = normalize_input("oﬃce")
        assert text == "office"
        assert index_map == [0, 1, 1, 1, 2, 3]

    def test_all_latin_ligatures(self):
        # U+FB00–FB06 の全リガチャ
        text, _ = normalize_input("ﬀﬁﬂﬃﬄﬅﬆ")
        assert text == "fffiflffifflstst"

    def test_compat_ideograph_keeps_position(self):
        # 互換漢字の1→1変換は位置が変わらない
        text, index_map = normalize_input("⼀二")
        assert text == "一二"
        assert index_map == [0, 1]

    def test_mixed(self):
        text, index_map = normalize_input("⼀ﬀあ")
        assert text == "一ffあ"
        assert index_map == [0, 1, 1, 2]

    def test_empty(self):
        text, index_map = normalize_input("")
        assert text == ""
        assert index_map == []
