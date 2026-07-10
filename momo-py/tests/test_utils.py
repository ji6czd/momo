"""
utils.py の単体テスト
  - normalize_one_to_one()
  - normalize_input()
"""
from momo_py.utils import normalize_one_to_one, normalize_input


class TestNormalizeOneToOne:
    def test_cjk_radicals_supplement(self):
        assert normalize_one_to_one("⺟") == "母"  # ⺟ -> 母
        assert normalize_one_to_one("⻳") == "龟"  # ⻳ -> 龟

    def test_kangxi_radical(self):
        assert normalize_one_to_one("⼀") == "一"  # ⼀ -> 一

    def test_cjk_compatibility_ideograph(self):
        # U+F900 と U+8C48 は見た目が同じ「豈」だが別コードポイント
        assert normalize_one_to_one("豈") == "豈"

    def test_cjk_compatibility_ideograph_supplement(self):
        assert normalize_one_to_one("\U0002f800") == "丽"  # 丽

    def test_fullwidth_digits(self):
        assert normalize_one_to_one("０１２３４５６７８９") == "0123456789"

    def test_fullwidth_alphabet(self):
        assert normalize_one_to_one("ＡＺａｚ") == "AZaz"

    def test_fullwidth_symbols_are_kept(self):
        # 全角記号は点字側で半角と区別が必要なため正規化しない
        assert normalize_one_to_one("（）！？：・％") == "（）！？：・％"

    def test_passthrough_normal_kanji(self):
        assert normalize_one_to_one("漢字") == "漢字"

    def test_passthrough_non_kanji(self):
        assert normalize_one_to_one("あいうえおABC123") == "あいうえおABC123"

    def test_ligature_is_not_expanded(self):
        # 1→1変換専用なので欧文リガチャは対象外（normalize_input が担当）
        assert normalize_one_to_one("ﬃ") == "ﬃ"


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

    def test_fullwidth_alnum_keeps_position(self):
        # 全角英数字の1→1変換は位置が変わらない。全角記号は変換しない
        text, index_map = normalize_input("３冊（Ａ）")
        assert text == "3冊（A）"
        assert index_map == [0, 1, 2, 3, 4]

    def test_mixed(self):
        text, index_map = normalize_input("⼀ﬀあ")
        assert text == "一ffあ"
        assert index_map == [0, 1, 1, 2]

    def test_empty(self):
        text, index_map = normalize_input("")
        assert text == ""
        assert index_map == []
