"""
trainer.py の単体テスト
  - is_suspicious()
  - process_line_to_tsv()
  - KUTOUTEN 定数
"""
import pytest
from momo_py.trainer import is_suspicious, process_line_to_tsv, KUTOUTEN


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

    def test_kutouten_gets_split_marker(self):
        rows = process_line_to_tsv("東京。\tトウ/キョウ/。", 1)
        # 「。」のラベルに +S が付いていること
        kutouten_row = next(r for r in rows if r.startswith("。\t"))
        label = kutouten_row.split("\t")[1]
        assert "+S" in label

    def test_space_label_appends_split_marker(self):
        # 読み部分のスペースは直前の行に +S を付ける
        # 「東 京」は「東」「 」「京」の3ユニット、「ToU/ /KyoU」のラベル形式
        rows = process_line_to_tsv("東 京\tトウ/ /キョウ", 1)
        labels = [r.split("\t")[1] for r in rows]
        assert any("+S" in lbl for lbl in labels)

    def test_single_char_tag_is_s(self):
        rows = process_line_to_tsv("あ\tア", 1)
        tag = rows[0].split("\t")[3]
        assert tag == "S"

    def test_multi_char_bio_tags(self):
        # 拗音「きゃ」は2文字1ユニット → B/E タグが付く
        rows = process_line_to_tsv("きゃく\tキャ/ク", 1)
        tags = [r.split("\t")[3] for r in rows]
        # 「き」→B、「ゃ」→E、「く」→S
        assert tags[0] == "B"
        assert tags[1] == "E"
        assert tags[2] == "S"


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
