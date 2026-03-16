"""
features.py の単体テスト
  - get_char_type()
  - get_units()
  - compute_source_features()
"""
import pytest
from momobrl.features import get_char_type, get_units, compute_source_features, SourceEntry


# ------------------------------------------------------------------ #
# get_char_type
# ------------------------------------------------------------------ #
class TestGetCharType:
    def test_hiragana(self):
        assert get_char_type("あ") == "HIRAGANA"
        assert get_char_type("ん") == "HIRAGANA"

    def test_katakana(self):
        assert get_char_type("ア") == "KATAKANA"
        assert get_char_type("ヶ") == "KATAKANA"

    def test_kanji(self):
        assert get_char_type("漢") == "KANJI"
        assert get_char_type("字") == "KANJI"

    def test_alpha(self):
        assert get_char_type("A") == "ALPHA"
        assert get_char_type("z") == "ALPHA"

    def test_digit(self):
        assert get_char_type("0") == "NUM"
        assert get_char_type("9") == "NUM"

    def test_symbol_ascii(self):
        assert get_char_type("!") == "SYMBOL"
        assert get_char_type(".") == "SYMBOL"

    def test_symbol_fullwidth(self):
        assert get_char_type("。") == "SYMBOL"
        assert get_char_type("、") == "SYMBOL"

    def test_space(self):
        assert get_char_type(" ") == "SPACE"
        assert get_char_type("　") == "SPACE"

    def test_empty_string(self):
        assert get_char_type("") == "SPACE"


# ------------------------------------------------------------------ #
# get_units
# ------------------------------------------------------------------ #
class TestGetUnits:
    def test_single_hiragana(self):
        units = get_units("あいう")
        assert units == [("あ", 0), ("い", 1), ("う", 2)]

    def test_compound_mora(self):
        # 拗音（きゃ など）は2文字で1ユニット
        units = get_units("きゃく")
        chars = [u[0] for u in units]
        assert "きゃ" in chars

    def test_alphanumeric_block(self):
        # 英数字の連続は1ブロックにまとめる
        units = get_units("abc123")
        assert len(units) == 1
        assert units[0] == ("abc123", 0)

    def test_alphanumeric_with_hyphen(self):
        units = get_units("abc-123")
        assert len(units) == 1
        assert units[0][0] == "abc-123"

    def test_bracket_notation(self):
        # [xxx] 形式はブラケットを除いた中身が1ユニット
        units = get_units("[キャ]")
        assert units[0][0] == "キャ"
        assert units[0][1] == 1  # 原文インデックスはブラケット内の先頭

    def test_mixed_text(self):
        units = get_units("東京abc")
        chars = [u[0] for u in units]
        assert "東" in chars
        assert "京" in chars
        assert "abc" in chars

    def test_space_preserved(self):
        units = get_units("あ い")
        chars = [u[0] for u in units]
        assert " " in chars

    def test_index_accuracy(self):
        units = get_units("abc東")
        # "abc" は idx=0、"東" は idx=3
        assert units[0] == ("abc", 0)
        assert units[1] == ("東", 3)

    def test_empty_string(self):
        assert get_units("") == []


# ------------------------------------------------------------------ #
# compute_source_features
# ------------------------------------------------------------------ #
class TestComputeSourceFeatures:
    def _make_seq(self, text: str) -> list[SourceEntry]:
        from momobrl.features import get_char_type
        return [(c, i, get_char_type(c)) for i, c in enumerate(text)]

    def test_length_matches_input(self):
        seq = self._make_seq("あいう")
        features = compute_source_features(seq)
        assert len(features) == 3

    def test_bias_present(self):
        seq = self._make_seq("あ")
        features = compute_source_features(seq)
        assert features[0]["bias"] == 1.0

    def test_bos_flag_on_first(self):
        seq = self._make_seq("あい")
        features = compute_source_features(seq)
        assert features[0].get("BOS") == 1.0
        assert "BOS" not in features[1]

    def test_eos_flag_on_last(self):
        seq = self._make_seq("あい")
        features = compute_source_features(seq)
        assert features[-1].get("EOS") == 1.0
        assert "EOS" not in features[0]

    def test_context_features_middle(self):
        seq = self._make_seq("あいう")
        features = compute_source_features(seq)
        mid = features[1]
        # pycrfsuiteネイティブ形式: 値がキーに埋め込まれる
        assert mid.get("-1:char=あ") == 1.0
        assert mid.get("+1:char=う") == 1.0

    def test_kanji_run_len(self):
        seq = self._make_seq("漢字")
        features = compute_source_features(seq)
        assert features[0]["kanji_run_len"] == 2
        assert features[1]["kanji_run_len"] == 2

    def test_kanji_pos_first(self):
        seq = self._make_seq("漢字")
        features = compute_source_features(seq)
        assert features[0]["kanji_pos_first"] == 1.0
        assert "kanji_pos_first" not in features[1]

    def test_type_transition(self):
        seq = self._make_seq("あ字")
        features = compute_source_features(seq)
        # pycrfsuiteネイティブ形式: 値がキーに埋め込まれる
        assert features[1].get("type_transition=HIRAGANA->KANJI") == 1.0

    def test_single_char(self):
        seq = self._make_seq("あ")
        features = compute_source_features(seq)
        assert features[0].get("BOS") == 1.0
        assert features[0].get("EOS") == 1.0
