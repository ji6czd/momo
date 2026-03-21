import re
import json
from typing import Optional, List
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
from .features import get_basic_char_category, CharType, _has_vowel
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
        self._single_char_dic = self._load_single_char_dic()

    # --- 初期化 ---

    def _load_braille_rules(self) -> None:
        try:
            with resources.open_text(
                "momo_py", "proto/braille_rules.textproto", encoding="utf-8"
            ) as f:
                logger.debug("Loading braille rules from file.")
                text_format.Parse(f.read(), self._rules)
        except FileNotFoundError:
            print(
                "Error: The file 'braille_rules.textproto' was not found.", file=sys.stderr
            )
        except IOError as e:
            print(f"Error: An I/O error occurred: {e}", file=sys.stderr)

    def _load_single_char_dic(self) -> dict[str, list[str]]:
        """単一漢字辞書（JSON）をロードして返す。
        ひらがなを含むエントリーがあれば警告を出す。
        
        Returns:
            dict[str, list[str]]: キーが漢字、値が読みのリスト
        """
        try:
            with resources.open_text(
                "momo_py", "single_character_dic.json", encoding="utf-8"
            ) as f:
                logger.debug("Loading single character dictionary from file.")
                dic = json.load(f)
                # ひらがなを含むエントリーをチェック
                self._check_hiragana_in_dic(dic)
                return dic
        except FileNotFoundError:
            print(
                "Error: The file 'single_character_dic.json' was not found.", file=sys.stderr
            )
            return {}
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON file: {e}", file=sys.stderr)
            return {}
        except IOError as e:
            print(f"Error: An I/O error occurred: {e}", file=sys.stderr)
            return {}

    def _check_hiragana_in_dic(self, dic: dict[str, list[str]]) -> None:
        """辞書内のひらがなをチェックして警告を出す。
        
        Args:
            dic: チェック対象の辞書
        """
        issues = []
        for kanji, readings in dic.items():
            for reading in readings:
                for char in reading:
                    if get_basic_char_category(char) == CharType.HIRAGANA:
                        issues.append((kanji, reading))
                        break
        
        if issues:
            logger.warning(f"Found {len(issues)} entries with hiragana in single_character_dic.json:")
            for kanji, reading in issues[:10]:
                logger.warning(f"  '{kanji}': {repr(reading)}")
            if len(issues) > 10:
                logger.warning(f"  ... and {len(issues) - 10} more")

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

    @staticmethod
    def _kana_equal(a: Optional[str], b: Optional[str]) -> bool:
        """ひらがなとカタカナを同一視して1文字同士を比較する。"""
        if a is None or b is None:
            return False
        if a == b:
            return True
        # ひらがな(U+3041–U+3096)とカタカナ(U+30A1–U+30F6)はオフセット0x60で対応
        if '\u3041' <= a <= '\u3096' and '\u30A1' <= b <= '\u30F6':
            return ord(a) + 0x60 == ord(b)
        if '\u30A1' <= a <= '\u30F6' and '\u3041' <= b <= '\u3096':
            return ord(a) - 0x60 == ord(b)
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

    def _convert_prolonged_sound_mark(self, morpheme: Morpheme, reading: str, split: bool = False) -> str:
        delimiter = '/' if split else ''
        # ソーステキストの一文字対して分割された読みを取得
        segmented_list = self._segment_reading_string(morpheme)
        if segmented_list:
            # 動詞であれば変換しない
            if self._has_part_of_speech(morpheme, "動詞"):
                return delimiter.join(segmented_list) + delimiter

            # 分割された読みの中で'ウ'を点字のルールに従って'ー'に変換する。
            for i, seg in enumerate(segmented_list):
                seg_len = len(seg)
                if (seg_len > 1 and seg[seg_len-1] == 'ウ' and
                    (_has_vowel(seg[seg_len-2], 'オ')
                    or _has_vowel(seg[seg_len-2], 'ウ')
                    or _has_vowel(seg[seg_len-1], 'ョ')
                    or _has_vowel(seg[seg_len-2], 'ュ'))
                ):
                    segmented_list[i] = seg[:-1] + 'ー'

            return delimiter.join(segmented_list) + delimiter


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
        if self._has_part_of_speech(morpheme_counter, "助数詞可能") or self._has_part_of_speech(morpheme_counter, "助数詞"):
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

    def _get_reading_from_single_char_dic(self, surface: str) -> list[str]:
        """単一漢字辞書から読みを取得する"""
        if surface in self._single_char_dic:
            return self._single_char_dic[surface]
        return []

    def _get_reading_form_in_dictionary(self, surface: str) -> list[str]:
        m_list = self._dic_obj.lookup(surface)
        reading_list = []
        
        # 通常の辞書から読みを取得
        for m in m_list:
            reading = m.reading_form()
            if reading not in reading_list:
                reading_list.append(reading)
        # 単一漢字辞書からも読みを取得して追加
        single_char_readings = self._get_reading_from_single_char_dic(surface)
        for reading in single_char_readings:
            if reading not in reading_list:
                reading_list.append(reading)
            
        return reading_list

    def _get_char_candidates(self, char: str) -> List[str]:
        """
        1文字に対する読み候補リストを返す。
        ひらがな・カタカナは変換した1文字のみ、漢字等は辞書から取得。
        """
        if '\u3041' <= char <= '\u3096':
            return [chr(ord(char) + 0x60)]   # ひらがな→カタカナ
        if '\u30A1' <= char <= '\u30F6':
            return [char]                      # カタカナはそのまま
        readings = self._get_reading_form_in_dictionary(char)
        return readings if readings else [char]

    def _resolve_segment(
        self,
        surface_seg: str,
        reading_seg: str,
        candidates_seg: List[List[str]],
    ) -> Optional[List[str]]:
        """
        surface_seg の各文字を reading_seg に過不足なく割り当てる。
        解が一意に存在すればそのリストを返し、なければ None を返す。
        """
        if not surface_seg:
            return [] if not reading_seg else None
        results = []
        for r in candidates_seg[0]:
            if reading_seg.startswith(r):
                rest = self._resolve_segment(
                    surface_seg[1:],
                    reading_seg[len(r):],
                    candidates_seg[1:],
                )
                if rest is not None:
                    results.append([r] + rest)
        # 解が1つだけなら確定、0または複数なら None
        return results[0] if len(results) == 1 else None

    def _segment_reading_string(self, m: Morpheme) -> List[str]:
        """
        形態素 m の reading_form() を、surface の各文字に対応するように
        '/' で区切った文字列を返す。

        アルゴリズム:
            1. 各文字の読み候補を収集する。
            2. 前向きフルスキャン: reading 全体を見渡して、各文字の読みが
               一意に確定できる（＝錨になれる）かどうかを判定する。
               途中で詰まっても止まらず最後まで走る。
            3. 錨と錨の間の未確定区間を _resolve_segment() で充填する。
            4. 最終フォールバック: それでも確定できない文字は最短一致で埋める。

        例:
            surface="水い動う", reading="スイイドウウ"
            前向きスキャン: 「動」→「ドウ」が錨として確定
            後向き伝播:     「い」→「ドウ」直前の「イ」で確定
            残り充填:       「水」→「スイ」で確定、「う」→「ウ」で確定
            結果: "スイ/イ/ドウ/ウ"

        Returns:
            各文字に対応する読みのリスト。分割できない場合は reading_form() をそのまま含むリストを返す。
        """
        surface = m.surface()
        reading = m.reading_form()

        n = len(surface)
        if n == 0:
            return reading

        # 各文字の読み候補を収集
        candidates: List[List[str]] = [self._get_char_candidates(c) for c in surface]

        # resolved[i] = (reading上の開始位置, 読み文字列) or None
        resolved: List[Optional[Tuple[int, str]]] = [None] * n

        # ==========================================
        # 前向きフルスキャン: 錨を見つける
        # ==========================================
        # reading上の各位置から始まる候補を全文字についてチェックする。
        # 「この文字がこの位置から始まる」という仮定のもとで、
        # reading全体との整合が取れる割り当てが一意かどうかを判定する。

        def find_anchors() -> bool:
            """前向きスキャンで錨を探す。1つでも確定したら True を返す。"""
            changed = False

            # 確定済みの錨からreading上の位置制約を計算する
            # anchor_constraints[i] = (pos_min, pos_max)
            # その文字がreading上で取りうる開始位置の範囲
            pos_min = [0] * n
            pos_max = [len(reading)] * n

            # 左から: 確定済みの錨を使って pos_min を絞り込む
            cur_min = 0
            for i in range(n):
                if resolved[i] is not None:
                    cur_min = resolved[i][0]
                pos_min[i] = max(pos_min[i], cur_min)
                # 確定していない場合は最短候補分だけ最低限進む
                if resolved[i] is not None:
                    cur_min = resolved[i][0] + len(resolved[i][1])
                else:
                    min_len = min(len(r) for r in candidates[i])
                    cur_min += min_len

            # 右から: 確定済みの錨を使って pos_max を絞り込む
            cur_max = len(reading)
            for i in range(n - 1, -1, -1):
                if resolved[i] is not None:
                    cur_max = resolved[i][0] + len(resolved[i][1])
                pos_max[i] = min(pos_max[i], cur_max)
                if resolved[i] is not None:
                    cur_max = resolved[i][0]
                else:
                    min_len = min(len(r) for r in candidates[i])
                    cur_max -= min_len

            # 各未確定文字について、位置範囲内で一意に確定できるか試みる
            for i in range(n):
                if resolved[i] is not None:
                    continue

                matched = []
                for r in candidates[i]:
                    rlen = len(r)
                    # この候補がpos_min[i]〜pos_max[i]の範囲内でreadingと一致するか
                    for pos in range(pos_min[i], pos_max[i] - rlen + 1):
                        if reading[pos:pos + rlen] == r:
                            matched.append((pos, r))

                if len(matched) == 1:
                    # 一意確定 → 錨
                    resolved[i] = matched[0]
                    changed = True

            return changed

        # 収束するまで繰り返す
        for _ in range(n):
            if not find_anchors():
                break
            if all(r is not None for r in resolved):
                break

        # ==========================================
        # 錨間の未確定区間を充填する
        # ==========================================
        # 確定済みの錨をソートして、区間ごとに _resolve_segment() を試みる
        anchors = [(i, resolved[i][0], resolved[i][1])
                   for i in range(n)
                   if resolved[i] is not None]

        # 錨の間の未確定区間を充填
        def fill_gap(start_char: int, end_char: int, start_pos: int, end_pos: int) -> None:
            """surface[start_char:end_char] を reading[start_pos:end_pos] で充填する。
            分割できない場合は読み文字列をそのまま先頭文字に割り当て、文字消失を防ぐ。
            """
            if start_char >= end_char:
                return
            seg_surface    = surface[start_char:end_char]
            seg_reading    = reading[start_pos:end_pos]
            seg_candidates = candidates[start_char:end_char]
            result = self._resolve_segment(seg_surface, seg_reading, seg_candidates)
            if result is not None:
                pos = start_pos
                for k, r in enumerate(result):
                    resolved[start_char + k] = (pos, r)
                    pos += len(r)
            else:
                # 分割できない（熟字訓など）→ 読みをそのまま先頭文字に割り当て、
                # 残りの文字は空文字で埋めて文字消失を防ぐ
                resolved[start_char] = (start_pos, seg_reading)
                for k in range(1, end_char - start_char):
                    resolved[start_char + k] = (end_pos, "")

        # 先頭〜最初の錨
        if anchors:
            first_i, first_pos, first_r = anchors[0]
            fill_gap(0, first_i, 0, first_pos)

            # 錨と錨の間
            for idx in range(len(anchors) - 1):
                i1, pos1, r1 = anchors[idx]
                i2, pos2, r2 = anchors[idx + 1]
                fill_gap(i1 + 1, i2, pos1 + len(r1), pos2)

            # 最後の錨〜末尾
            last_i, last_pos, last_r = anchors[-1]
            fill_gap(last_i + 1, n, last_pos + len(last_r), len(reading))
        else:
            # 錨が1つも見つからなかった場合は全体を _resolve_segment() で試みる
            result = self._resolve_segment(surface, reading, candidates)
            if result is not None:
                pos = 0
                for i, r in enumerate(result):
                    resolved[i] = (pos, r)
                    pos += len(r)
            else:
                # 全体が分割不能（熟字訓など）→ 読みをそのまま先頭文字に割り当て
                resolved[0] = (0, reading)
                for i in range(1, n):
                    resolved[i] = (len(reading), "")

        return [r for _, r in resolved if r is not None]

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
                    kana_str += self._convert_prolonged_sound_mark(m, counter)
                else:
                    if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] == "数詞":
                        kana_str += m.normalized_form()
                    else:
                        kana_str += self._convert_prolonged_sound_mark(m, m.reading_form(), False)
            if m_index < len(tokenized_list) - 1:
                if self._is_space_required(m, tokenized_list[m_index + 1]):
                    kana_str += " "
        return kana_str

    def _split_kanastr(self, s: str) -> str:
        """カタカナ文字列を1文字ずつ分割する。"""
        l = list(s)
        # 文字と文字の間に'/'を挿入して結合する
        return '/'.join(l)

    def _split_kana(self, m: Morpheme) -> str:
        """ソーステキストと読みを比較して、後ろから一致する部分を１文字ずつデリミタで区切って出力する"""

        # 後ろから比較したいので、文字列を逆順にしてリスト化する
        r = list(reversed(m.reading_form()))
        s = list(reversed(m.surface()))
        res = ""
        i: int = 0 # 次の２つのリープで使う
        for i, r_char in enumerate(r):
            if i < len(s):
                if self._kana_equal(r_char, s[i]):
                    res = r_char + "/" + res
                else:
                    break
        # まだ残っている読みがある
        if i < len(r) and not self._kana_equal(r[i], s[i]):
            res = '/' + res
            # 残りの文字列を追加
            res = ''.join(reversed(r[i:])) + res
        return res

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
                    segmented_kana_string += self._split_kana(m)
            elif not self._is_kana_conversion_required(m):
                segmented_kana_string +=  self._split_kana(m)
            else:
                counter = self._correct_counter_suffix_reading(tokenized_list[m_index - 1], m)
                if counter != "":
                    segmented_kana_string += self._convert_prolonged_sound_mark(m, counter, True)
                else:
                    if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] == "数詞":
                        segmented_kana_string += m.normalized_form() + "/"
                    else:
                        segmented_kana_string += self._convert_prolonged_sound_mark(m, m.reading_form(), True)
            if m_index < len(tokenized_list) - 1:
                if self._is_space_required(m, tokenized_list[m_index + 1]):
                    segmented_kana_string += " /"
        return segmented_kana_string

    def convert_to_braille(self, src: str) -> str:
        """テキストを点字に変換する。"""
        kana_str = self.convert_to_kana(src)
        return pybraille.to_jp_braille(kana_str)

