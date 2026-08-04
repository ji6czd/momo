"""
bracket.py の単体テスト
  - lookup() / is_pair()
  - strip_for_training()
  - tokenize_for_training()
"""
from momo_py.bracket import (
    ASIDE_TOKEN,
    INLINE_CLOSE_TOKEN,
    INLINE_OPEN_TOKEN,
    LABEL_CONTEXT_ONLY,
    Role,
    Treatment,
    is_pair,
    lookup,
    strip_for_training,
    tokenize_for_training,
)


# ------------------------------------------------------------------ #
# lookup / is_pair
# ------------------------------------------------------------------ #
class TestLookup:
    def test_inline_quotes(self):
        for c in "「」『』“”‘’【】":
            assert lookup(c)[1] == Treatment.INLINE

    def test_aside_brackets(self):
        for c in "（）()｛｝":
            assert lookup(c)[1] == Treatment.ASIDE

    def test_roles(self):
        for c in "「『“‘（(｛【":
            assert lookup(c)[0] == Role.OPEN
        for c in "」』”’）)｝】":
            assert lookup(c)[0] == Role.CLOSE

    def test_non_bracket_is_none(self):
        assert lookup("あ") is None
        assert lookup("。") is None

    def test_is_pair(self):
        assert is_pair("「", "」") is True
        assert is_pair("（", "）") is True
        assert is_pair("「", "）") is False


# ------------------------------------------------------------------ #
# strip_for_training
# ------------------------------------------------------------------ #
def _row(char, label, ctype="KANJI", idx=0, name_flag="O"):
    return [char, label, ctype, str(idx), name_flag]


class TestStripForTraining:
    def test_no_brackets_passthrough(self):
        rows = [_row("東", "トー"), _row("京", "キョー")]
        main, subs = strip_for_training(rows)
        assert main == rows
        assert subs == []

    def test_inline_brackets_are_removed(self):
        rows = [
            _row("A", "エー"),
            _row("「", "_", ctype="SYMBOL_OPEN"),
            _row("B", "ビー"),
            _row("」", "_", ctype="SYMBOL_CLOSE"),
            _row("C", "シー"),
        ]
        main, subs = strip_for_training(rows)
        assert [r[0] for r in main] == ["A", "B", "C"]
        assert subs == []
        # 括弧自体に+Sが無ければ、隣接ラベルは変化しない
        assert main[0][1] == "エー"
        assert main[1][1] == "ビー"

    def test_inline_open_space_stays_on_preceding_char(self):
        # 開き括弧の直前スペースは、そもそも直前の実文字の行に付いている
        # （読みアノテーション上、開き括弧自身の行に+Sが乗ることはない）。
        # strip後もそのまま保持されることだけを確認する。
        rows = [
            _row("A", "エー+S"),
            _row("「", "_", ctype="SYMBOL_OPEN"),
            _row("B", "ビー"),
        ]
        main, _ = strip_for_training(rows)
        assert [r[0] for r in main] == ["A", "B"]
        assert main[0][1] == "エー+S"

    def test_inline_close_space_moves_to_preceding_char(self):
        # 閉じ括弧行自身に+Sが付いている(未対応寄りだが移設で吸収する)場合、
        # strip後に見える直前の実文字へ+Sを移す。
        rows = [
            _row("A", "エー"),
            _row("「", "_", ctype="SYMBOL_OPEN"),
            _row("B", "ビー"),
            _row("」", "_+S", ctype="SYMBOL_CLOSE"),
            _row("C", "シー"),
        ]
        main, subs = strip_for_training(rows)
        assert [r[0] for r in main] == ["A", "B", "C"]
        assert main[1][1] == "ビー+S"
        assert subs == []

    def test_aside_span_becomes_independent_sentence(self):
        rows = [
            _row("A", "エー"),
            _row("（", "_", ctype="SYMBOL_OPEN"),
            _row("B", "ビー"),
            _row("）", "_", ctype="SYMBOL_CLOSE"),
            _row("C", "シー"),
        ]
        main, subs = strip_for_training(rows)
        assert [r[0] for r in main] == ["A", "C"]
        assert len(subs) == 1
        assert [r[0] for r in subs[0]] == ["B"]

    def test_aside_close_space_moves_to_preceding_main_char(self):
        rows = [
            _row("A", "エー"),
            _row("（", "_", ctype="SYMBOL_OPEN"),
            _row("B", "ビー"),
            _row("）", "_+S", ctype="SYMBOL_CLOSE"),
            _row("C", "シー"),
        ]
        main, subs = strip_for_training(rows)
        assert [r[0] for r in main] == ["A", "C"]
        assert main[0][1] == "エー+S"
        assert [r[0] for r in subs[0]] == ["B"]

    def test_nested_aside_is_flattened(self):
        rows = [
            _row("A", "エー"),
            _row("（", "_", ctype="SYMBOL_OPEN"),
            _row("B", "ビー"),
            _row("（", "_", ctype="SYMBOL_OPEN"),
            _row("C", "シー"),
            _row("）", "_", ctype="SYMBOL_CLOSE"),
            _row("）", "_", ctype="SYMBOL_CLOSE"),
            _row("D", "ディー"),
        ]
        main, subs = strip_for_training(rows)
        assert [r[0] for r in main] == ["A", "D"]
        assert len(subs) == 2
        # 外側Asideの中身(B, 内側Aside)から、内側Asideがさらに独立文として
        # 抜き取られるため、外側の副文には B だけが残る（Rust側の再帰独立
        # 推論と対応: 入れ子は各階層が独立した文になる）。
        assert [r[0] for r in subs[0]] == ["B"]
        assert [r[0] for r in subs[1]] == ["C"]

    def test_unmatched_open_bracket_passthrough_with_warning(self, capsys):
        rows = [
            _row("A", "エー"),
            _row("（", "_", ctype="SYMBOL_OPEN"),
            _row("B", "ビー"),
        ]
        main, subs = strip_for_training(rows)
        assert [r[0] for r in main] == ["A", "（", "B"]
        assert subs == []
        assert "対応する閉じ括弧が見つかりません" in capsys.readouterr().out

    def test_empty_aside_content_yields_no_sub_sentence(self):
        rows = [
            _row("A", "エー"),
            _row("（", "_", ctype="SYMBOL_OPEN"),
            _row("）", "_", ctype="SYMBOL_CLOSE"),
            _row("B", "ビー"),
        ]
        main, subs = strip_for_training(rows)
        assert [r[0] for r in main] == ["A", "B"]
        assert subs == []


# ------------------------------------------------------------------ #
# tokenize_for_training（圧縮アイデンティティ・トークン方式）
# ------------------------------------------------------------------ #
class TestTokenizeForTraining:
    def test_no_brackets_passthrough(self):
        rows = [_row("東", "トー"), _row("京", "キョー")]
        main, subs = tokenize_for_training(rows)
        assert main == rows
        assert subs == []

    def test_inline_brackets_become_tokens_but_keep_own_label(self):
        # strip_for_trainingと違い、行は削除せず残す（隣接文字のbigram/
        # trigram/kanji_runがsource_seq経由で括弧の存在を認識できるように）。
        # ASIDEと違い、char列だけをトークンに差し替え、ラベルには触れない
        # （中身と一緒に本文の流れに残るため、開き・閉じの行自身がRust推論側
        # でも実際の境界判定対象になる＝LABEL_CONTEXT_ONLYにはしない）。
        rows = [
            _row("A", "エー"),
            _row("「", "_", ctype="SYMBOL_OPEN"),
            _row("B", "ビー"),
            _row("」", "_", ctype="SYMBOL_CLOSE"),
            _row("C", "シー"),
        ]
        main, subs = tokenize_for_training(rows)
        assert [r[0] for r in main] == ["A", INLINE_OPEN_TOKEN, "B", INLINE_CLOSE_TOKEN, "C"]
        assert subs == []
        # プレースホルダ行のラベルは元の"_"のまま（LABEL_CONTEXT_ONLYにはしない）
        assert main[1][1] == "_"
        assert main[3][1] == "_"
        # ctypeは元のSYMBOL_OPEN/SYMBOL_CLOSEのまま
        assert main[1][2] == "SYMBOL_OPEN"
        assert main[3][2] == "SYMBOL_CLOSE"
        # 括弧に挟まれた実文字の読みラベルは変化しない
        assert main[0][1] == "エー"
        assert main[2][1] == "ビー"
        assert main[4][1] == "シー"

    def test_inline_close_space_stays_on_its_own_row(self):
        # ASIDEと違い、閉じ括弧行の+Sは直前の実文字行へ移設しない。閉じ括弧
        # 自身がRust推論側で実際の境界判定対象になるため、自分の+Sは自分の
        # 行に残す必要がある。
        rows = [
            _row("A", "エー"),
            _row("「", "_", ctype="SYMBOL_OPEN"),
            _row("B", "ビー"),
            _row("」", "_+S", ctype="SYMBOL_CLOSE"),
            _row("C", "シー"),
        ]
        main, _ = tokenize_for_training(rows)
        assert [r[0] for r in main] == ["A", INLINE_OPEN_TOKEN, "B", INLINE_CLOSE_TOKEN, "C"]
        # +Sは移設されず、閉じ括弧トークン自身の行に残る
        assert main[2][1] == "ビー"
        assert main[3][1] == "_+S"

    def test_aside_open_becomes_token_close_and_content_are_removed(self):
        rows = [
            _row("A", "エー"),
            _row("（", "_", ctype="SYMBOL_OPEN"),
            _row("B", "ビー"),
            _row("）", "_", ctype="SYMBOL_CLOSE"),
            _row("C", "シー"),
        ]
        main, subs = tokenize_for_training(rows)
        assert [r[0] for r in main] == ["A", ASIDE_TOKEN, "C"]
        assert main[1][1] == LABEL_CONTEXT_ONLY
        assert len(subs) == 1
        assert [r[0] for r in subs[0]] == ["B"]

    def test_aside_close_space_moves_to_preceding_main_char(self):
        rows = [
            _row("A", "エー"),
            _row("（", "_", ctype="SYMBOL_OPEN"),
            _row("B", "ビー"),
            _row("）", "_+S", ctype="SYMBOL_CLOSE"),
            _row("C", "シー"),
        ]
        main, subs = tokenize_for_training(rows)
        assert [r[0] for r in main] == ["A", ASIDE_TOKEN, "C"]
        assert main[0][1] == "エー+S"
        assert [r[0] for r in subs[0]] == ["B"]

    def test_nested_aside_is_flattened(self):
        rows = [
            _row("A", "エー"),
            _row("（", "_", ctype="SYMBOL_OPEN"),
            _row("B", "ビー"),
            _row("（", "_", ctype="SYMBOL_OPEN"),
            _row("C", "シー"),
            _row("）", "_", ctype="SYMBOL_CLOSE"),
            _row("）", "_", ctype="SYMBOL_CLOSE"),
            _row("D", "ディー"),
        ]
        main, subs = tokenize_for_training(rows)
        assert [r[0] for r in main] == ["A", ASIDE_TOKEN, "D"]
        assert len(subs) == 2
        assert [r[0] for r in subs[0]] == ["B", ASIDE_TOKEN]
        assert [r[0] for r in subs[1]] == ["C"]

    def test_unmatched_open_bracket_passthrough_with_warning(self, capsys):
        rows = [
            _row("A", "エー"),
            _row("（", "_", ctype="SYMBOL_OPEN"),
            _row("B", "ビー"),
        ]
        main, subs = tokenize_for_training(rows)
        assert [r[0] for r in main] == ["A", "（", "B"]
        assert subs == []
        assert "対応する閉じ括弧が見つかりません" in capsys.readouterr().out
