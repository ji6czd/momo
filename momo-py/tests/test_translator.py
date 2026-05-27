"""
translator.py の単体テスト
  - _convert_prolonged_sound_mark()
"""
import pytest
from unittest.mock import MagicMock

from momo_py.translator import Translator


def make_morpheme(surface: str, reading: str, pos: tuple) -> MagicMock:
    """テスト用の Morpheme モックオブジェクトを生成する。"""
    m = MagicMock()
    m.surface.return_value = surface
    m.reading_form.return_value = reading
    m.part_of_speech.return_value = pos
    return m


@pytest.fixture(scope="module")
def translator():
    return Translator()


# ------------------------------------------------------------------ #
# _convert_prolonged_sound_mark
# ------------------------------------------------------------------ #
class TestConvertProlongedSoundMark:
    """_convert_prolonged_sound_mark() のテスト

    変換が行われる条件: not 動詞  OR  意志推量形
    変換対象:
      - surface が複数漢字のみ   → 各漢字の読みの2文字目以降の 'ウ' を 'ー' に置換
      - surface が連続ひらがなのみ → reading の2文字目以降の 'ウ' を 'ー' に置換
      - それ以外                  → 変換しない
    """

    # --- 動詞（意志推量形でない）→ 条件を満たさないので変換しない ---

    def test_verb_plain_no_conversion(self, translator):
        """動詞終止形: 条件を満たさないので 'ウ' を変換しない"""
        m = make_morpheme("思う", "オモウ", ("動詞", "一般", "*", "*", "五段-ワア行", "終止形-一般"))
        assert translator._convert_prolonged_sound_mark(m) == "オモウ"

    def test_verb_continuative_no_conversion(self, translator):
        """動詞連用形: 条件を満たさないので reading をそのまま返す"""
        m = make_morpheme("行く", "イク", ("動詞", "一般", "*", "*", "五段-カ行", "連用形-一般"))
        assert translator._convert_prolonged_sound_mark(m) == "イク"

    # --- 動詞の意志推量形 → 変換条件を満たす ---

    def test_verb_volitional_hiragana_converts(self, translator):
        """動詞意志推量形 + surface がひらがなのみ: reading の2文字目以降の 'ウ' → 'ー'"""
        m = make_morpheme("いこう", "イコウ", ("動詞", "一般", "*", "*", "五段-カ行", "意志推量形"))
        assert translator._convert_prolonged_sound_mark(m) == "イコー"

    def test_verb_volitional_mixed_surface_converts(self, translator):
        """動詞意志推量形で surface が漢字+ひらがな混在: 末尾ひらがな2文字以上なら 'ウ' → 'ー'"""
        m = make_morpheme("行こう", "イコウ", ("動詞", "一般", "*", "*", "五段-カ行", "意志推量形"))
        assert translator._convert_prolonged_sound_mark(m) == "イコー"

    def test_kanji_compound_unresolvable_fallback(self, translator):
        """読みが各漢字に分割できない場合はひらがな扱いにフォールバック: 今日→キョー"""
        m = make_morpheme("今日", "キョウ", ("名詞", "普通名詞", "副詞可能", "*", "*", "*"))
        assert translator._convert_prolonged_sound_mark(m) == "キョー"

    def test_kanji_compound_u_before_vowel_no_conversion(self, translator):
        """ウの直後が別の母音のときは長音変換しない: 井上→イノウエ"""
        m = make_morpheme("井上", "イノウエ", ("名詞", "固有名詞", "人名", "姓", "*", "*"))
        assert translator._convert_prolonged_sound_mark(m) == "イノウエ"

    # --- 非動詞 + 複数漢字 surface → 各漢字の読みで変換 ---

    def test_kanji_compound_u_in_second_char(self, translator):
        """複数漢字(名詞): 動=ドウ → ドー, 路=ロ → ロ: 「道路」→「ドーロ」"""
        m = make_morpheme("道路", "ドウロ", ("名詞", "普通名詞", "一般", "*", "*", "*"))
        assert translator._convert_prolonged_sound_mark(m) == "ドーロ"

    def test_kanji_compound_multiple_u(self, translator):
        """複数漢字: 運=ウン(変換なし), 動=ドウ → ドー: 「運動」→「ウンドー」"""
        m = make_morpheme("運動", "ウンドウ", ("名詞", "普通名詞", "一般", "*", "*", "*"))
        assert translator._convert_prolonged_sound_mark(m) == "ウンドー"

    def test_kanji_single_char_no_conversion(self, translator):
        """漢字1文字: 複数漢字の条件(len>=2)を満たさないので変換しない"""
        m = make_morpheme("道", "ミチ", ("名詞", "普通名詞", "一般", "*", "*", "*"))
        assert translator._convert_prolonged_sound_mark(m) == "ミチ"

    # --- 非動詞 + 連続ひらがな surface → reading の2文字目以降を変換 ---

    def test_hiragana_only_converts_u_from_second(self, translator):
        """連続ひらがな: reading の2文字目以降の 'ウ' → 'ー': 「どうぞ」→「ドーゾ」"""
        m = make_morpheme("どうぞ", "ドウゾ", ("副詞", "*", "*", "*", "*", "*"))
        assert translator._convert_prolonged_sound_mark(m) == "ドーゾ"

    def test_hiragana_first_char_u_preserved(self, translator):
        """連続ひらがな: 1文字目の 'ウ' は置換しない: 「ういろう」→「ウイロー」"""
        m = make_morpheme("ういろう", "ウイロウ", ("名詞", "普通名詞", "一般", "*", "*", "*"))
        assert translator._convert_prolonged_sound_mark(m) == "ウイロー"

    def test_hiragana_single_char_no_conversion(self, translator):
        """ひらがな1文字: 2文字目以降がないので変換しない"""
        m = make_morpheme("う", "ウ", ("感動詞", "*", "*", "*", "*", "*"))
        assert translator._convert_prolonged_sound_mark(m) == "ウ"

    def test_hiragana_no_u_unchanged(self, translator):
        """連続ひらがなで reading に 'ウ' がなければ変換なし"""
        m = make_morpheme("たまに", "タマニ", ("副詞", "*", "*", "*", "*", "*"))
        assert translator._convert_prolonged_sound_mark(m) == "タマニ"

    # --- それ以外(カタカナ・混在など) → 変換しない ---

    def test_katakana_surface_no_conversion(self, translator):
        """surface がカタカナ: ひらがな・漢字の条件を満たさないので変換しない"""
        m = make_morpheme("ドウ", "ドウ", ("名詞", "普通名詞", "一般", "*", "*", "*"))
        assert translator._convert_prolonged_sound_mark(m) == "ドウ"

    def test_mixed_kanji_hiragana_no_conversion(self, translator):
        """漢字+ひらがな混在: どちらの条件も満たさないので変換しない"""
        m = make_morpheme("動く", "ウゴク", ("動詞", "一般", "*", "*", "五段-カ行", "終止形-一般"))
        assert translator._convert_prolonged_sound_mark(m) == "ウゴク"
