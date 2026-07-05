"""
name_dict.py の単体テスト
  - parse_name_marks()
  - name_flag_for_unit()
  - extract_names()
  - build_name_index() / compute_name_flags()
  - load_name_dict()
"""
import pytest
from momo_py.name_dict import (
    parse_name_marks,
    name_flag_for_unit,
    extract_names,
    build_name_index,
    compute_name_flags,
    compute_name_matches,
    load_name_dict,
    parse_name_dict_text,
)
from momo_py.features import get_units


# ------------------------------------------------------------------ #
# parse_name_marks
# ------------------------------------------------------------------ #
class TestParseNameMarks:
    def test_basic(self):
        clean, spans = parse_name_marks("{佐藤}さんにあった。")
        assert clean == "佐藤さんにあった。"
        assert spans == [(0, 2)]

    def test_multiple_spans(self):
        # 姓と名は別スパン
        clean, spans = parse_name_marks("{佐藤}{太郎}さん")
        assert clean == "佐藤太郎さん"
        assert spans == [(0, 2), (2, 4)]

    def test_span_in_middle(self):
        clean, spans = parse_name_marks("昨日{山田}さんと")
        assert clean == "昨日山田さんと"
        assert spans == [(2, 4)]

    def test_no_marks(self):
        clean, spans = parse_name_marks("東京へ行く")
        assert clean == "東京へ行く"
        assert spans == []

    def test_empty_mark_raises(self):
        with pytest.raises(ValueError, match="空の人名マーク"):
            parse_name_marks("{}さん")

    def test_unmatched_open_raises(self):
        with pytest.raises(ValueError, match="対応の取れていない"):
            parse_name_marks("{佐藤さん")

    def test_unmatched_close_raises(self):
        with pytest.raises(ValueError, match="対応の取れていない"):
            parse_name_marks("佐藤}さん")


# ------------------------------------------------------------------ #
# name_flag_for_unit
# ------------------------------------------------------------------ #
class TestNameFlagForUnit:
    def test_begin(self):
        assert name_flag_for_unit(0, 1, [(0, 2)]) == "B"

    def test_inside(self):
        assert name_flag_for_unit(1, 1, [(0, 2)]) == "I"

    def test_outside(self):
        assert name_flag_for_unit(2, 1, [(0, 2)]) == "O"

    def test_no_spans(self):
        assert name_flag_for_unit(0, 1, []) == "O"

    def test_partial_overlap_raises(self):
        # ユニット（位置1〜3）がスパン（0〜2）の途中で切れている
        with pytest.raises(ValueError, match="音節ユニットの途中"):
            name_flag_for_unit(1, 2, [(0, 2)])


# ------------------------------------------------------------------ #
# extract_names
# ------------------------------------------------------------------ #
class TestExtractNames:
    def test_basic(self):
        assert extract_names("{佐藤}{太郎}さんと{鈴木}さん") == ["佐藤", "太郎", "鈴木"]

    def test_no_marks(self):
        assert extract_names("東京へ行く") == []


# ------------------------------------------------------------------ #
# build_name_index / compute_name_flags
# ------------------------------------------------------------------ #
class TestComputeNameFlags:
    def test_surname_and_given_name(self):
        index = build_name_index(["佐藤", "太郎"])
        seq = get_units("佐藤太郎さんにあった")
        flags = compute_name_flags(seq, index)
        # 佐=B 藤=I 太=B 郎=I 以降O
        assert flags[:4] == ["B", "I", "B", "I"]
        assert all(f == "O" for f in flags[4:])

    def test_no_match(self):
        index = build_name_index(["佐藤"])
        seq = get_units("鈴木さん")
        assert compute_name_flags(seq, index) == ["O"] * len(seq)

    def test_longest_match_wins(self):
        # 「佐藤原」と「佐藤」が両方登録されている場合は長い方を採用
        index = build_name_index(["佐藤", "佐藤原"])
        seq = get_units("佐藤原さん")
        flags = compute_name_flags(seq, index)
        assert flags[:3] == ["B", "I", "I"]

    def test_compound_unit_name(self):
        # 拗音を含むひらがな名は複合ユニット単位で照合される
        index = build_name_index(["りょう"])
        seq = get_units("りょうさん")  # → りょ / う / さ / ん
        flags = compute_name_flags(seq, index)
        assert flags == ["B", "I", "O", "O"]

    def test_empty_index(self):
        seq = get_units("佐藤さん")
        assert compute_name_flags(seq, {}) == ["O"] * len(seq)


# ------------------------------------------------------------------ #
# load_name_dict / parse_name_dict_text
# ------------------------------------------------------------------ #
class TestLoadNameDict:
    def test_load_with_readings(self, tmp_path):
        path = tmp_path / "person_name_dic.tsv"
        path.write_text(
            "# コメント行\n#表層形\t読み\t出現回数\n"
            "渡辺\tワタ/ナベ\t3\n太郎\tタ/ロー\t1\n渡辺\tワタ/ナベ\t3\n\n",
            encoding="utf-8",
        )
        entries = load_name_dict(str(path))
        # 重複は除去される
        assert entries == [("渡辺", ["ワタ", "ナベ"]), ("太郎", ["タ", "ロー"])]

    def test_old_format_without_readings(self):
        # 旧形式（表層形\t出現回数）は読みなしエントリとして読める
        entries = parse_name_dict_text("佐藤\t3\n太郎\t1\n")
        assert entries == [("佐藤", None), ("太郎", None)]

    def test_reading_unit_mismatch_is_dropped(self):
        # ユニット数と読みブロック数が合わない場合は読みを捨てる
        entries = parse_name_dict_text("佐藤\tサトー\t1\n")
        assert entries == [("佐藤", None)]

    def test_compound_unit_reading(self):
        # 「りょう」は りょ/う の2ユニット
        entries = parse_name_dict_text("りょう\tリョ/ー\t1\n")
        assert entries == [("りょう", ["リョ", "ー"])]


# ------------------------------------------------------------------ #
# compute_name_matches（読み付き照合）
# ------------------------------------------------------------------ #
class TestComputeNameMatches:
    def test_readings_assigned_per_unit(self):
        index = build_name_index([("渡辺", ["ワタ", "ナベ"]), ("太郎", None)])
        seq = get_units("渡辺太郎さん")
        flags, readings = compute_name_matches(seq, index)
        assert flags[:4] == ["B", "I", "B", "I"]
        # 読みが登録されているスパンだけ readings が入る
        assert readings[:4] == ["ワタ", "ナベ", None, None]
        assert readings[4:] == [None, None]

    def test_no_dict_readings(self):
        index = build_name_index(["佐藤"])  # 文字列のみ（読みなし）
        seq = get_units("佐藤さん")
        flags, readings = compute_name_matches(seq, index)
        assert flags[:2] == ["B", "I"]
        assert all(r is None for r in readings)
