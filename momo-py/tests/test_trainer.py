"""
trainer.py の単体テスト
  - is_suspicious()
  - process_line_to_tsv()
  - KUTOUTEN 定数
"""
import pytest
from momo_py.bracket import ASIDE_TOKEN, INLINE_CLOSE_TOKEN, INLINE_OPEN_TOKEN, LABEL_CONTEXT_ONLY
from momo_py.trainer import (
    KUTOUTEN,
    _check_japanese_reading,
    _check_non_ascii_identity,
    _load_sentences,
    _tokenize_bracket_sentences,
    create_data,
    is_suspicious,
    process_line_to_tsv,
)


# ------------------------------------------------------------------ #
# is_suspicious
# ------------------------------------------------------------------ #
class TestIsSuspicious:
    # 正常ケース（疑わしくない）
    def test_katakana_identity(self):
        assert is_suspicious("ア", "ア") is False

    def test_hiragana_to_katakana(self):
        assert is_suspicious("か", "カ") is False

    def test_hiragana_double_mora(self):
        assert is_suspicious("きゃ", "キャ") is False

    def test_particle_ha_wa(self):
        # 助詞「は」は「ワ」も許可
        assert is_suspicious("は", "ワ") is False
        assert is_suspicious("は", "ハ") is False

    def test_particle_he_e(self):
        # 助詞「へ」は「エ」も許可
        assert is_suspicious("へ", "エ") is False
        assert is_suspicious("へ", "ヘ") is False

    def test_u_to_prolonged(self):
        # 「う」→「ー」（長音変換）は許可
        assert is_suspicious("う", "ー") is False

    def test_reading_with_split_marker(self):
        # +S が含まれていても判定できる
        assert is_suspicious("か", "カ+S") is False

    def test_skip_marker(self):
        # "_" は常にFalse
        assert is_suspicious("！", "_") is False

    def test_empty_raw_space_read(self):
        assert is_suspicious("", " ") is False

    # 異常ケース（疑わしい）
    def test_katakana_mismatch(self):
        assert is_suspicious("ア", "イ") is True

    def test_hiragana_mismatch(self):
        assert is_suspicious("か", "サ") is True


# ------------------------------------------------------------------ #
# process_line_to_tsv
# ------------------------------------------------------------------ #
class TestProcessLineToTsv:
    def test_basic(self):
        # 漢字1文字=1ユニットなのでラベルは「トウ/キョウ」の中区切り形式
        rows = process_line_to_tsv("東京\tトウ/キョウ", 1)
        assert len(rows) == 2
        assert rows[0].startswith("東\t")
        assert rows[1].startswith("京\t")

    def test_no_tab_raises(self):
        with pytest.raises(ValueError, match="タブが見つかりません"):
            process_line_to_tsv("タブなし行", 1)

    def test_multiple_tabs_raises(self):
        with pytest.raises(ValueError, match="タブが複数"):
            process_line_to_tsv("A\tB\tC", 1)

    def test_too_many_labels_raises(self):
        # 読みブロックが原文より多い
        with pytest.raises(ValueError, match="読みラベル過多"):
            process_line_to_tsv("あ\tア/イ", 1)

    def test_too_few_labels_raises(self):
        # 原文が読みより多い
        with pytest.raises(ValueError, match="原文余り"):
            process_line_to_tsv("あい\tア", 1)

    def test_kutouten_is_skipped(self):
        # 句読点（SYMBOL_STOP/PAUSE）はバイパス文字として LABEL_SKIP="_" で格納される
        rows = process_line_to_tsv("東京。\tトウ/キョウ/。", 1)
        kutouten_row = next(r for r in rows if r.startswith("。\t"))
        label = kutouten_row.split("\t")[1]
        assert label == "_"

    def test_space_label_appends_split_marker(self):
        # 読み部分のスペースは直前の行に +S を付ける
        # 「東 京」は「東」「 」「京」の3ユニット、「ToU/ /KyoU」のラベル形式
        rows = process_line_to_tsv("東 京\tトウ/ /キョウ", 1)
        labels = [r.split("\t")[1] for r in rows]
        assert any("+S" in lbl for lbl in labels)

    def test_single_char_row_format(self):
        # 1文字は「原文\t読み\t文字種\t原文位置\t人名フラグ」の5列で返る
        rows = process_line_to_tsv("あ\tア", 1)
        cols = rows[0].split("\t")
        assert cols[0] == "あ"
        assert cols[1] == "ア"
        assert cols[2] == "HIRAGANA"
        assert cols[3] == "0"
        assert cols[4] == "O"

    def test_compound_mora_is_single_row(self):
        # 拗音「きゃ」は1ユニット → LABEL_CONTINUE を使わず1行で格納される
        rows = process_line_to_tsv("きゃく\tキャ/ク", 1)
        assert len(rows) == 2
        assert rows[0].startswith("きゃ\t")
        assert rows[1].startswith("く\t")

    def test_name_marks(self):
        # {…} マークは除去され、人名フラグ列（B/I/O）が付与される
        rows = process_line_to_tsv("{佐藤}さんだ。\tサ/トー/ /サ/ン/ダ/。", 1)
        cells = [r.split("\t") for r in rows]
        assert [c[0] for c in cells] == ["佐", "藤", "さ", "ん", "だ", "。"]
        assert [c[4] for c in cells] == ["B", "I", "O", "O", "O", "O"]
        # 人名と敬称の間には +S（空白）が付く
        assert cells[1][1] == "トー+S"

    def test_name_marks_surname_and_given(self):
        rows = process_line_to_tsv(
            "{佐藤}{太郎}さんだ。\tサ/トー/ /タ/ロー/ /サ/ン/ダ/。", 1
        )
        cells = [r.split("\t") for r in rows]
        assert [c[4] for c in cells] == ["B", "I", "B", "I", "O", "O", "O", "O"]

    def test_unmatched_name_mark_raises(self):
        with pytest.raises(ValueError, match="対応の取れていない"):
            process_line_to_tsv("{佐藤さんだ。\tサ/トー/ /サ/ン/ダ/。", 1)

    def test_fullwidth_alnum_is_normalized_in_output(self, capsys):
        # 注釈は全角のまま書き、学習データ出力時に半角へ正規化される
        # （検証は注釈原文に対して行われるため警告は出ない）
        rows = process_line_to_tsv("３冊のＱ\t３/サツ/ノ/Ｑ", 1)
        assert "🚨 警告" not in capsys.readouterr().out
        cells = [r.split("\t") for r in rows]
        assert [c[0] for c in cells] == ["3", "冊", "の", "Q"]
        assert cells[0][1] == "3"  # 恒等ラベルも半角化
        assert cells[0][2] == "NUM"
        assert cells[3][1] == "_"  # ALPHA は素通し（ラベルは LABEL_SKIP）

    def test_fullwidth_halfwidth_produce_same_rows(self, capsys):
        # 正規化により "３冊" と "3冊" は同一の学習データになる
        rows_fw = process_line_to_tsv("３冊\t３/サツ", 1)
        rows_hw = process_line_to_tsv("3冊\t3/サツ", 1)
        capsys.readouterr()
        assert rows_fw == rows_hw


# ------------------------------------------------------------------ #
# _check_non_ascii_identity
# ------------------------------------------------------------------ #
class TestCheckNonAsciiIdentity:
    WARN = "🚨 警告"

    # 警告されるケース：全角→半角の置換は点字変換側で行う
    def test_fullwidth_digit_to_halfwidth_warns(self, capsys):
        _check_non_ascii_identity("３", "NUM", "3", 1)
        assert self.WARN in capsys.readouterr().out

    def test_fullwidth_symbol_to_halfwidth_warns(self, capsys):
        _check_non_ascii_identity("％", "SYMBOL", "%", 1)
        assert self.WARN in capsys.readouterr().out

    def test_fullwidth_alpha_to_halfwidth_warns(self, capsys):
        _check_non_ascii_identity("Ａ", "ALPHA", "A", 1)
        assert self.WARN in capsys.readouterr().out

    def test_other_char_with_kana_reading_warns(self, capsys):
        # OTHER（丸数字など）はバイパスして点字変換側で処理するため恒等であるべき
        _check_non_ascii_identity("②", "OTHER", "ニ", 1)
        assert self.WARN in capsys.readouterr().out

    # 警告されないケース
    def test_identity_is_ok(self, capsys):
        _check_non_ascii_identity("％", "SYMBOL", "％", 1)
        assert capsys.readouterr().out == ""

    def test_identity_with_split_marker_is_ok(self, capsys):
        _check_non_ascii_identity("％", "SYMBOL", "％+S", 1)
        assert capsys.readouterr().out == ""

    def test_skip_label_is_ok(self, capsys):
        _check_non_ascii_identity("％", "SYMBOL", "_", 1)
        assert capsys.readouterr().out == ""

    def test_ascii_char_is_not_checked(self, capsys):
        # ASCII範囲の文字は対象外（既存の注釈スタイルを許容）
        _check_non_ascii_identity("%", "SYMBOL", "パーセント", 1)
        assert capsys.readouterr().out == ""

    def test_japanese_char_is_not_checked(self, capsys):
        # かな・漢字・漢数字は読みを学習する文字種なので対象外
        _check_non_ascii_identity("東", "KANJI", "トウ", 1)
        _check_non_ascii_identity("三", "JAPANESE_NUMERIC", "ミッ", 1)
        assert capsys.readouterr().out == ""

    def test_fullwidth_digit_kana_reading_is_ok(self, capsys):
        # 単一数字のかな読み（"６"→"ムイ"＝六日）は半角数字と同様に許容
        _check_non_ascii_identity("６", "NUM", "ムイ", 1)
        assert capsys.readouterr().out == ""

    def test_process_line_integration(self, capsys):
        # process_line_to_tsv 経由でも警告が出る
        process_line_to_tsv("３個\t3/コ", 1)
        assert self.WARN in capsys.readouterr().out


# ------------------------------------------------------------------ #
# _check_japanese_reading
# ------------------------------------------------------------------ #
class TestCheckJapaneseReading:
    WARN = "🚨 警告"

    # 警告されるケース：日本語ユニットの読みに半角文字
    def test_japanese_numeric_halfwidth_digit_warns(self, capsys):
        _check_japanese_reading("三", "JAPANESE_NUMERIC", "3", 1)
        assert self.WARN in capsys.readouterr().out

    def test_kanji_halfwidth_digit_warns(self, capsys):
        # 位取り文字単独（昇格せず KANJI 扱い）の半角数字読み
        _check_japanese_reading("十", "KANJI", "10", 1)
        assert self.WARN in capsys.readouterr().out

    def test_partially_halfwidth_warns(self, capsys):
        # 1文字でも半角が混ざれば警告
        _check_japanese_reading("十", "JAPANESE_NUMERIC", "1０", 1)
        assert self.WARN in capsys.readouterr().out

    def test_halfwidth_with_split_marker_warns(self, capsys):
        # +S を除去した上で判定する
        _check_japanese_reading("三", "JAPANESE_NUMERIC", "3+S", 1)
        assert self.WARN in capsys.readouterr().out

    def test_stray_space_in_label_warns(self, capsys):
        # '/' 抜けでラベルに残った半角スペースも検出する
        _check_japanese_reading("尋", "KANJI", " ジン", 1)
        assert self.WARN in capsys.readouterr().out

    def test_hiragana_stray_space_warns(self, capsys):
        _check_japanese_reading("か", "HIRAGANA", "カ ", 1)
        assert self.WARN in capsys.readouterr().out

    # 警告されないケース
    def test_katakana_reading_is_ok(self, capsys):
        _check_japanese_reading("三", "JAPANESE_NUMERIC", "ミッ", 1)
        _check_japanese_reading("東", "KANJI", "トウ", 1)
        assert capsys.readouterr().out == ""

    def test_fullwidth_digit_reading_is_ok(self, capsys):
        _check_japanese_reading("三", "JAPANESE_NUMERIC", "３", 1)
        _check_japanese_reading("十", "KANJI", "１０", 1)
        assert capsys.readouterr().out == ""

    def test_fullwidth_with_split_marker_is_ok(self, capsys):
        _check_japanese_reading("三", "JAPANESE_NUMERIC", "３+S", 1)
        assert capsys.readouterr().out == ""

    def test_skip_label_is_ok(self, capsys):
        _check_japanese_reading("〇", "JAPANESE_NUMERIC", "_", 1)
        assert capsys.readouterr().out == ""

    def test_continue_label_is_ok(self, capsys):
        # 手動入力の '---'（読み継続）は正当なラベル
        _check_japanese_reading("日", "KANJI", "---", 1)
        process_line_to_tsv("今日\tキョー/---", 1)
        assert "🚨 警告" not in capsys.readouterr().out

    def test_kanji_numeral_skip_label_is_ok(self, capsys):
        # 位取り文字のスキップ（"二十五"→"２/_/５" の '十'）は警告しない
        rows = process_line_to_tsv("二十五\t２/_/５", 1)
        assert "🚨 警告" not in capsys.readouterr().out
        labels = {r.split("\t")[0]: r.split("\t")[1] for r in rows}
        # 全角数字の読みは学習データ出力時に半角へ正規化される
        assert labels == {"二": "2", "十": "_", "五": "5"}

    def test_non_japanese_ctype_is_not_checked(self, capsys):
        # 日本語以外の文字種は対象外（NUMERIC の恒等注釈 "3"→"3" など）
        _check_japanese_reading("3", "NUM", "3", 1)
        _check_japanese_reading("A", "ALPHA", "A", 1)
        assert capsys.readouterr().out == ""

    def test_process_line_integration(self, capsys):
        # process_line_to_tsv 経由でも警告が出る
        process_line_to_tsv("三個\t3/コ", 1)
        assert self.WARN in capsys.readouterr().out


# ------------------------------------------------------------------ #
# _load_sentences
# ------------------------------------------------------------------ #
class TestLoadSentences:
    def test_space_source_char_row_is_preserved(self, tmp_path):
        # 原文列が空白文字の行（ASCII連内のスペース）で列がずれないこと。
        # strip() でパースすると先頭列が消え、文字種列に OrigIdx が入る
        # 列ズレが起き、export が不正な特徴量キーで失敗する（回帰テスト）。
        tsv = tmp_path / "data.tsv"
        tsv.write_text(
            "#原文\t読み\t文字種\tOrigIdx\t人名\n"
            "N\t_\tALPHA\t0\tO\n"
            " \t_\tALPHA\t1\tO\n"
            "B\t_\tALPHA\t2\tO\n",
            encoding="utf-8",
        )
        sentences = _load_sentences(str(tsv))
        assert len(sentences) == 1
        rows = sentences[0]
        assert len(rows) == 3
        assert rows[1][0] == " "  # 原文列に空白文字が残る
        assert rows[1][2] == "ALPHA"  # 文字種列がずれていない

    def test_sentence_split_on_blank_line(self, tmp_path):
        tsv = tmp_path / "data.tsv"
        tsv.write_text(
            "あ\tア\tHIRAGANA\t0\tO\n\nい\tイ\tHIRAGANA\t0\tO\n",
            encoding="utf-8",
        )
        sentences = _load_sentences(str(tsv))
        assert len(sentences) == 2


# ------------------------------------------------------------------ #
# create_data（人名辞書の自動生成）
# ------------------------------------------------------------------ #
class TestCreateDataNameDict:
    def test_creates_name_dict(self, tmp_path):
        raw = tmp_path / "basic_raw.tsv"
        raw.write_text(
            "{佐藤}さんだ。\tサ/トー/ /サ/ン/ダ/。\n"
            "{佐藤}{花子}さんだ。\tサ/トー/ /ハナ/コ/ /サ/ン/ダ/。\n",
            encoding="utf-8",
        )
        out = tmp_path / "basic_data.tsv"
        create_data(str(raw), str(out))

        tsv = out.read_text(encoding="utf-8")
        assert tsv.splitlines()[0] == "#原文\t読み\t文字種\tOrigIdx\t人名"
        assert "佐\tサ\tKANJI\t0\tB" in tsv
        assert "藤\tトー+S\tKANJI\t1\tI" in tsv

        dict_path = tmp_path / "person_name_dic.tsv"
        assert dict_path.exists()
        content = dict_path.read_text(encoding="utf-8")
        # 読みは +S 除去済みのユニット別 '/' 区切りで自動抽出される
        assert "佐藤\tサ/トー\t2" in content
        assert "花子\tハナ/コ\t1" in content

    def test_no_marks_no_dict_file(self, tmp_path):
        raw = tmp_path / "basic_raw.tsv"
        raw.write_text("東京だ。\tトー/キョー/ダ/。\n", encoding="utf-8")
        out = tmp_path / "basic_data.tsv"
        create_data(str(raw), str(out))
        assert not (tmp_path / "person_name_dic.tsv").exists()


# ------------------------------------------------------------------ #
# 括弧のトークン化（学習/推論のstrip不一致解消。bracket.py参照）
# ------------------------------------------------------------------ #
class TestTokenizeBracketSentences:
    def test_no_bracket_returns_single_sentence(self):
        rows = process_line_to_tsv("東京だ。\tトー/キョー/ダ/。", 1)
        sentences = _tokenize_bracket_sentences(rows)
        assert len(sentences) == 1
        assert sentences[0] == rows

    def test_inline_bracket_becomes_token_not_removed(self):
        rows = process_line_to_tsv("先「あ」だ。\tセン/「/ア/」/ダ/。", 1)
        sentences = _tokenize_bracket_sentences(rows)
        assert len(sentences) == 1
        chars = [line.split("\t")[0] for line in sentences[0]]
        assert chars == ["先", INLINE_OPEN_TOKEN, "あ", INLINE_CLOSE_TOKEN, "だ", "。"]

    def test_aside_bracket_becomes_separate_sentence(self):
        rows = process_line_to_tsv(
            "東京（首都）です。\tトー/キョー/（/シュ/ト/）/デ/ス/。", 1
        )
        sentences = _tokenize_bracket_sentences(rows)
        assert len(sentences) == 2
        main_chars = [line.split("\t")[0] for line in sentences[0]]
        sub_chars = [line.split("\t")[0] for line in sentences[1]]
        assert main_chars == ["東", "京", ASIDE_TOKEN, "で", "す", "。"]
        assert sub_chars == ["首", "都"]


class TestCreateDataBracketTokenize:
    def test_aside_content_written_as_separate_block(self, tmp_path):
        raw = tmp_path / "basic_raw.tsv"
        raw.write_text(
            "東京（首都）です。\tトー/キョー/（/シュ/ト/）/デ/ス/。\n",
            encoding="utf-8",
        )
        out = tmp_path / "basic_data.tsv"
        create_data(str(raw), str(out))

        sentences = _load_sentences(str(out))
        assert len(sentences) == 2
        assert [row[0] for row in sentences[0]] == ["東", "京", ASIDE_TOKEN, "で", "す", "。"]
        assert [row[0] for row in sentences[1]] == ["首", "都"]
        # プレースホルダ行のラベルはLABEL_CONTEXT_ONLY（学習対象から除外する目印）
        aside_row = [row for row in sentences[0] if row[0] == ASIDE_TOKEN][0]
        assert aside_row[1] == LABEL_CONTEXT_ONLY
        # 括弧の実文字そのものは出力TSVに残らない（token化されている）
        tsv = out.read_text(encoding="utf-8")
        assert "（\t" not in tsv
        assert "）\t" not in tsv


# ------------------------------------------------------------------ #
# KUTOUTEN 定数
# ------------------------------------------------------------------ #
class TestKutouten:
    def test_is_frozenset(self):
        assert isinstance(KUTOUTEN, frozenset)

    def test_contains_japanese_punctuation(self):
        assert "。" in KUTOUTEN
        assert "、" in KUTOUTEN
        assert "？" in KUTOUTEN
        assert "！" in KUTOUTEN

    def test_contains_ascii_punctuation(self):
        assert "." in KUTOUTEN
        assert "," in KUTOUTEN
