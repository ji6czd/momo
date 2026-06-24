"""
features.py の単体テスト
  - get_char_type()
  - get_units()
  - compute_source_features()
"""
import pytest
from momo_py.features import get_char_type, get_units, compute_source_features, SourceEntry


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

    def test_symbol_stop(self):
        # 文末記号類は SYMBOL_STOP
        assert get_char_type("!") == "SYMBOL_STOP"
        assert get_char_type(".") == "SYMBOL_STOP"
        assert get_char_type("。") == "SYMBOL_STOP"

    def test_symbol_pause(self):
        # 読点・中点類は SYMBOL_PAUSE
        assert get_char_type("、") == "SYMBOL_PAUSE"
        assert get_char_type(",") == "SYMBOL_PAUSE"

    def test_symbol_open(self):
        # 開き括弧類は SYMBOL_OPEN
        assert get_char_type("「") == "SYMBOL_OPEN"
        assert get_char_type("(") == "SYMBOL_OPEN"

    def test_symbol_close(self):
        # 閉じ括弧類は SYMBOL_CLOSE
        assert get_char_type("」") == "SYMBOL_CLOSE"
        assert get_char_type(")") == "SYMBOL_CLOSE"

    def test_symbol_other(self):
        # いずれの特殊カテゴリにも属さない記号は SYMBOL
        assert get_char_type("@") == "SYMBOL"
        assert get_char_type("#") == "SYMBOL"

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
        assert units == [("あ", 0, "HIRAGANA"), ("い", 1, "HIRAGANA"), ("う", 2, "HIRAGANA")]

    def test_compound_mora(self):
        # 拗音（きゃ など）は2文字で1ユニット
        units = get_units("きゃく")
        chars = [u[0] for u in units]
        assert "きゃ" in chars

    def test_alphanumeric_block(self):
        # 英字はまとめるが、数字は1文字ずつ個別ユニットになる
        units = get_units("abc123")
        chars = [u[0] for u in units]
        assert "abc" in chars          # 英字ブロックはまとめる
        assert "1" in chars and "2" in chars and "3" in chars  # 数字は個別

    def test_alphanumeric_with_hyphen(self):
        # ハイフンは英字と同じグループ、数字は個別
        units = get_units("abc-123")
        chars = [u[0] for u in units]
        assert "abc-" in chars         # 英字+ハイフンはまとめる
        assert "1" in chars and "2" in chars and "3" in chars

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
        assert units[0] == ("abc", 0, "ALPHA")
        assert units[1] == ("東", 3, "KANJI")

    def test_empty_string(self):
        assert get_units("") == []


# ------------------------------------------------------------------ #
# compute_source_features
# ------------------------------------------------------------------ #
class TestComputeSourceFeatures:
    def _make_seq(self, text: str) -> list[SourceEntry]:
        from momo_py.features import get_char_type
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
        # 先頭文字は直前コンテキスト特徴量を持たない
        seq = self._make_seq("あい")
        features = compute_source_features(seq)
        assert not any(k.startswith("char_p1=") for k in features[0])
        assert any(k.startswith("char_p1=") for k in features[1])

    def test_eos_flag_on_last(self):
        # 末尾文字は直後コンテキスト特徴量を持たない
        seq = self._make_seq("あい")
        features = compute_source_features(seq)
        assert not any(k.startswith("char_n1=") for k in features[-1])
        assert any(k.startswith("char_n1=") for k in features[0])

    def test_context_features_middle(self):
        seq = self._make_seq("あいう")
        features = compute_source_features(seq)
        mid = features[1]
        assert mid.get("char_p1=あ") == 1.0
        assert mid.get("char_n1=う") == 1.0

    def test_kanji_run_len(self):
        seq = self._make_seq("漢字")
        features = compute_source_features(seq)
        assert features[0].get("kanji_run=2") == 1.0
        assert features[1].get("kanji_run=2") == 1.0

    def test_kanji_pos_first(self):
        seq = self._make_seq("漢字")
        features = compute_source_features(seq)
        assert features[0]["kanji_pos_first"] == 1.0
        assert "kanji_pos_first" not in features[1]

    def test_type_transition(self):
        seq = self._make_seq("あ字")
        features = compute_source_features(seq)
        assert features[1].get("type_trans_p1_s=HIRAGANA->KANJI") == 1.0

    def test_single_char(self):
        # 1文字だけの場合、前後のコンテキスト特徴量ともに持たない
        seq = self._make_seq("あ")
        features = compute_source_features(seq)
        assert not any(k.startswith("char_p1=") for k in features[0])
        assert not any(k.startswith("char_n1=") for k in features[0])
