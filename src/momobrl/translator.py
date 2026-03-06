from typing import Optional
import re
from google.protobuf import text_format
from importlib import resources

from loguru import logger
import sys
from sudachipy import tokenizer
from sudachipy import dictionary
from sudachipy import Morpheme
from .braille_rules_pb2 import BrailleRules
from .braille_rules_pb2 import PartOfSpeech
from . import pybraille

class Translator:
    """
    日本語テキストを点字（かな変換・分かち書き変換）に変換するクラス。
    公開メソッド: segment_braille_rule(), convert_to_kana(), convert_to_braille()
    """

    def __init__(self):
        logger.debug("Starting MOMO!")
        logger.remove()
        self._mode = tokenizer.Tokenizer.SplitMode.B
        self._dic_obj = dictionary.Dictionary()
        self._tokenizer_obj = self._dic_obj.create(self._mode)
        self._rules = BrailleRules()
        self._load_braille_rules()

    # --- 初期化 ---

    def _load_braille_rules(self) -> None:
        try:
            with resources.open_text(
                "momobrl", "proto/braille_rules.textproto", encoding="utf-8"
            ) as f:
                logger.debug("Loading braille rules from file.")
                text_format.Parse(f.read(), self._rules)
        except FileNotFoundError:
            print(
                "Error: The file 'braille_rules.textproto' was not found.", file=sys.stderr
            )
        except IOError as e:
            print(f"Error: An I/O error occurred: {e}", file=sys.stderr)

    # --- ユーティリティ（プライベート） ---

    @staticmethod
    def _is_english_alphanumeric(word: str) -> bool:
        return word.isalnum() and all(ch.isascii() for ch in word)

    @staticmethod
    def _sound_len(s: str) -> int:
        count: int = 0
        for c in s:
            if c not in "ァィゥェォャュョ":
                count += 1
        return count

    @staticmethod
    def _has_item_in_list(lst, item: str) -> bool:
        if not lst:
            return True
        for i in lst:
            if i == item or i == "*":
                return True
        return False

    def _score_part_of_speech(self, morpheme: Morpheme, pos: PartOfSpeech) -> Optional[int]:
        src_pos = morpheme.part_of_speech()
        src_surface = morpheme.surface()
        src_reading_form = morpheme.reading_form()
        for index, m_pos in enumerate(src_pos):
            if (
                index <= len(src_pos)
                and self._has_item_in_list(pos.name, m_pos)
                and self._has_item_in_list(pos.word_match, src_surface)
                and (
                    pos.reading_word_length_less == 0
                    or (self._sound_len(src_reading_form) <= pos.reading_word_length_less
                        and len(src_surface) <= pos.surface_word_length_less))
                and (
                    pos.reading_word_length_greater == 0
                    or (self._sound_len(src_reading_form) >= pos.reading_word_length_greater
                        and len(src_surface) >= pos.surface_word_length_greater)
                )
            ):
                return index
        return None

    def _search_braille_rules(self, morpheme: Morpheme) -> Optional[int]:
        max_score = -1
        rule_index = -1
        for index, rule in enumerate(self._rules.rule):
            score = self._score_part_of_speech(morpheme, rule.current_pos)
            if score is not None and score > max_score:
                max_score = score
                rule_index = index
        if rule_index < 0:
            return None
        logger.debug(
            f"current_rule: {rule_index}, {morpheme.surface()} {self._rules.rule[rule_index].current_pos.name}"
        )
        return rule_index

    def _search_next_rule(self, morpheme: Morpheme, rule) -> Optional[int]:
        max_score = -1
        next_index = -1
        for index, n_rule in enumerate(rule.next_pos):
            score = self._score_part_of_speech(morpheme, n_rule)
            if score is not None and score > max_score:
                max_score = score
                next_index = index
        if next_index < 0:
            logger.debug(f"next_rule: None, {morpheme.surface()},")
            return None
        logger.debug(
            f"next_rule: {next_index}, {morpheme.surface()}, {rule.next_pos[next_index].name}"
        )
        return next_index

    @staticmethod
    def _has_part_of_speech(morpheme: Morpheme, target_pos: str) -> bool:
        src_pos = morpheme.part_of_speech()
        for m_pos in src_pos:
            if m_pos == target_pos:
                return True
        return False

    def _is_space_required(self, current_morpheme: Morpheme, next_morpheme: Morpheme) -> bool:
        space_flag: bool = True
        rule_index = self._search_braille_rules(current_morpheme)
        if rule_index is not None:
            rule = self._rules.rule[rule_index]
            rule_index = self._search_next_rule(next_morpheme, rule)
            if rule_index is not None:
                space_flag = rule.next_pos[rule_index].before_space
            else:
                space_flag = False
        else:
            space_flag = False
        return space_flag

    def _is_kana_conversion_required(self, morphe: Morpheme) -> bool:
        if (
            self._has_part_of_speech(morphe, "補助記号")
            or self._has_part_of_speech(morphe, "空白")
            or morphe.reading_form() == morphe.surface()
            or self._is_english_alphanumeric(morphe.surface())
        ):
            return False
        return True

    def _convert_prolonged_sound_mark(self, morpheme: Morpheme) -> str:
        reading = morpheme.reading_form()
        if (not self._has_part_of_speech(morpheme, "動詞")
                or self._has_part_of_speech(morpheme, "意志推量形")):
            surface = morpheme.surface()
            # 複数の漢字で構成されているか確認
            if (len(surface) >= 2
                    and all('\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf'
                            for c in surface)):
                segments = self._segment_reading_by_kanji(surface, reading)
                if segments is not None:
                    result = ""
                    for seg in segments:
                        if len(seg) >= 2:
                            result += seg[0] + re.sub(r'ウ(?![アイエオァィェォ])', 'ー', seg[1:])
                        else:
                            result += seg
                    return result
                else:
                    # 各漢字の読みが特定できない場合はひらがな扱いにフォールバック
                    if len(reading) >= 2:
                        return reading[0] + re.sub(r'ウ(?![アイエオァィェォ])', 'ー', reading[1:])
            # 連続するひらがなで構成されているか確認
            elif all('\u3041' <= c <= '\u3096' for c in surface):
                if len(reading) >= 2:
                    return reading[0] + re.sub(r'ウ(?![アイエオァィェォ])', 'ー', reading[1:])
            else:
                # 末尾の連続ひらがな部分について変換
                trailing_hiragana = ""
                for c in reversed(surface):
                    if '\u3041' <= c <= '\u3096':
                        trailing_hiragana = c + trailing_hiragana
                    else:
                        break
                if len(trailing_hiragana) >= 2:
                    suffix_reading = reading[-len(trailing_hiragana):]
                    prefix_reading = reading[:-len(trailing_hiragana)]
                    return prefix_reading + suffix_reading[0] + re.sub(r'ウ(?![アイエオァィェォ])', 'ー', suffix_reading[1:])
        return reading

    def _segment_reading_by_kanji(self, surface: str, reading: str) -> Optional[list[str]]:
        """漢字列 surface の各文字に対応するカタカナ reading のセグメントを返す。
        マッチしない場合は None を返す。"""
        if not surface:
            return [] if not reading else None
        for r in self._get_reading_form_in_dictionary(surface[0]):
            if reading.startswith(r):
                rest = self._segment_reading_by_kanji(surface[1:], reading[len(r):])
                if rest is not None:
                    return [r] + rest
        return None


    def _correct_counter_suffix_reading(
        self, morpheme_num: Morpheme, morpheme_counter: Morpheme
    ) -> str:
        reading = morpheme_counter.reading_form()
        if self._has_part_of_speech(morpheme_counter, "助数詞可能") or self._has_part_of_speech(
            morpheme_counter, "助数詞"
        ):
            reading = morpheme_counter.reading_form()
            number = morpheme_num.surface()
            if morpheme_counter.surface() in "匹":
                if number[len(number) - 1] in "168":
                    reading = chr(ord(reading[0]) + 2) + reading[1:]
                elif number[len(number) - 1] == "3":
                    reading = chr(ord(reading[0]) + 1) + reading[1:]
            elif morpheme_counter.surface() == "本":
                if number[len(number) - 1] in "24579":
                    reading = chr(ord(reading[0]) - 2) + reading[1:]
                elif number[len(number) - 1] in "3":
                    reading = chr(ord(reading[0]) - 1) + reading[1:]
            elif morpheme_counter.surface() == "版":
                if number[len(number) - 1] in "24579":
                    reading = chr(ord(reading[0]) - 1) + reading[1:]
                elif number[len(number) - 1] in "168":
                    reading = chr(ord(reading[0]) + 1) + reading[1:]
        else:
            reading = ""
        return reading

    def _has_reading_in_dictionary(self, surface: str, reading: str) -> bool:
        reading_list = self._dic_obj.lookup(surface)
        for m in reading_list:
            if m.reading_form() == reading:
                return True
        return False

    def _get_reading_form_in_dictionary(self, surface: str) -> list[str]:
        m_list = self._dic_obj.lookup(surface)
        if not m_list:
            return []
        reading_list = []
        for m in m_list:
            reading_list.append(m.reading_form())
        return reading_list

    # --- 公開メソッド ---

    def segment_source_text(self, src_string: str) -> str:
        """分かち書きルールに従ってテキストを分割する。"""
        segmented_string: str = ""
        tokenized_list = self._tokenizer_obj.tokenize(src_string)
        for m_index, m in enumerate(tokenized_list):
            segmented_string += m.surface()
            if m_index < len(tokenized_list) - 1:
                if self._is_space_required(m, tokenized_list[m_index + 1]):
                    segmented_string += " "
        return segmented_string

    def convert_to_kana(self, src_string: str) -> str:
        """テキストをかな（カタカナ）読みに変換する。"""
        kana_str: str = ""
        tokenized_list = self._tokenizer_obj.tokenize(src_string)
        for m_index, m in enumerate(tokenized_list):
            if m.part_of_speech()[0] == "助詞":
                if m.reading_form() == "ハ":
                    kana_str += "ワ"
                elif m.reading_form() == "ヘ":
                    kana_str += "エ"
                else:
                    kana_str += m.reading_form()
            elif not self._is_kana_conversion_required(m):
                kana_str += m.surface()
            else:
                counter = self._correct_counter_suffix_reading(tokenized_list[m_index - 1], m)
                if counter != "":
                    kana_str += counter
                else:
                    if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] == "数詞":
                        kana_str += m.normalized_form()
                    else:
                        kana_str += self._convert_prolonged_sound_mark(m)
            if m_index < len(tokenized_list) - 1:
                if self._is_space_required(m, tokenized_list[m_index + 1]):
                    kana_str += " "
        return kana_str

    def segment_kana_string(self, src_string: str) -> str:
        """ソーステキストを仮名に変換しながら、ソーステキストの各文字に対する仮名を'/'で区切って出力する"""
        segmented_kana_string: str = ""
        tokenized_list = self._tokenizer_obj.tokenize(src_string)
        tokenized_list = self._tokenizer_obj.tokenize(src_string)
        for m_index, m in enumerate(tokenized_list):
            if m.part_of_speech()[0] == "助詞":
                if m.reading_form() == "ハ":
                    segmented_kana_string += "ワ/"
                elif m.reading_form() == "ヘ":
                    segmented_kana_string += "エ/"
                else:
                    segmented_kana_string += m.reading_form() + "/"
            elif not self._is_kana_conversion_required(m):
                segmented_kana_string += m.surface() + "/"
            else:
                counter = self._correct_counter_suffix_reading(tokenized_list[m_index - 1], m)
                if counter != "":
                    segmented_kana_string += counter + "/"
                else:
                    if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] == "数詞":
                        segmented_kana_string += m.normalized_form() + "/"
                    else:
                        segmented_kana_string += self._convert_prolonged_sound_mark(m) + "/"
            if m_index < len(tokenized_list) - 1:
                if self._is_space_required(m, tokenized_list[m_index + 1]):
                    segmented_kana_string += " /"
        return segmented_kana_string

    def convert_to_braille(self, src: str) -> str:
        """テキストを点字に変換する。"""
        kana_str = self.convert_to_kana(src)
        return pybraille.to_jp_braille(kana_str)
