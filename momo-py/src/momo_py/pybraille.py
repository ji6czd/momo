"""日本語点字変換モジュール（momors-braille の Python バックポート）。

テーブルは resources/japanese_grade1_braille.toml から読み込む。
"""

from __future__ import annotations
import tomllib
from importlib import resources
from typing import Any

BRAILLE_SPACE = "⠀"  # ⠀


# ============================================================
# BrailleTable
# ============================================================

class BrailleTable:
    """japanese_grade1_braille.toml から構築する変換テーブル。"""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.kana_compound: dict[str, str] = raw["table"]["kana"]["compound"]
        self.kana_single: dict[str, str] = raw["table"]["kana"]["single"]
        self.digit: dict[str, str] = raw["table"]["digit"]
        self.latin: dict[str, str] = raw["table"]["latin"]

        def _parse_punct(src: dict[str, Any]) -> dict[str, tuple[str, str]]:
            out: dict[str, tuple[str, str]] = {}
            for k, v in src.items():
                if isinstance(v, str):
                    out[k] = (v, "none")
                else:
                    out[k] = (v["braille"], v.get("class", "none"))
            return out

        self.punct_jp = _parse_punct(raw["table"]["punct"]["jp"])
        self.punct_latin = _parse_punct(raw["table"]["punct"]["latin"])

        flags = raw["flags"]
        self.flag_digit: dict[str, Any] = flags["digit"]
        self.flag_foreign_word: dict[str, Any] = flags["foreign_word"]
        self.flag_capital: dict[str, Any] = flags["capital"]

        # クラス遷移 (from, to) → 挿入する点字スペース数（[transitions]）
        self.transitions: dict[tuple[str, str], int] = {}
        for key, spaces in raw.get("transitions", {}).items():
            frm, _, to = key.partition("->")
            self.transitions[(frm.strip(), to.strip())] = int(spaces)

    @classmethod
    def load(cls) -> BrailleTable:
        data = (
            resources.files("momo_py") / "resources" / "japanese_grade1_braille.toml"
        ).read_bytes()
        return cls(tomllib.loads(data.decode("utf-8")))


# ============================================================
# BrailleConverter
# ============================================================

class BrailleConverter:
    """点字変換器。momors-braille の BrailleConverter に対応する。"""

    def __init__(self, table: BrailleTable) -> None:
        self._t = table
        # 数字テーブルの各値の先頭セル集合（explicit_exit 挿入判定用）
        self._digit_cells: frozenset[str] = frozenset(
            brl[0] for brl in table.digit.values() if brl
        )

    @classmethod
    def from_table(cls) -> BrailleConverter:
        return cls(BrailleTable.load())

    def convert(self, kana_text: str) -> str:
        """かな文字列を点字文字列に変換する。

        momors-core がバイパスした全角英数字を ASCII に正規化してから処理する。
        """
        # 全角英数字 → ASCII 正規化
        chars: list[str] = []
        for c in kana_text:
            cp = ord(c)
            if 0xFF10 <= cp <= 0xFF19:
                chars.append(chr(cp - 0xFF10 + 0x30))   # ０-９ → 0-9
            elif 0xFF21 <= cp <= 0xFF3A:
                chars.append(chr(cp - 0xFF21 + 0x41))   # Ａ-Ｚ → A-Z
            elif 0xFF41 <= cp <= 0xFF5A:
                chars.append(chr(cp - 0xFF41 + 0x61))   # ａ-ｚ → a-z
            else:
                chars.append(c)

        n = len(chars)
        t = self._t
        out: list[str] = []

        in_digit = False
        in_foreign_word = False
        in_capital_word = False
        in_double_capital = False

        # 直前に出力した文字のクラス（[transitions] の判定に使う）。テキスト先頭は None。
        prev_class: str | None = None

        i = 0
        while i < n:
            c = chars[i]

            if ord(c) < 0x80:
                # ==================== ASCII ====================
                if c.isalpha():
                    in_digit = False
                    self._push_transition(out, prev_class, "latin")
                    if not in_foreign_word:
                        out.append(t.flag_foreign_word["entry_prefix"])
                        in_foreign_word = True
                    if c.isupper():
                        if not in_capital_word:
                            out.append(t.flag_capital["entry_prefix"])
                        in_capital_word = True
                        if i + 1 < n and chars[i + 1].isupper() and not in_double_capital:
                            out.append(t.flag_capital["entry_prefix"])
                            in_double_capital = True
                    elif in_capital_word:
                        in_capital_word = False
                        in_double_capital = False
                    out.append(t.latin.get(c.lower(), BRAILLE_SPACE))
                    prev_class = "latin"

                elif c.isdigit():
                    in_foreign_word = False
                    in_capital_word = False
                    in_double_capital = False
                    self._push_transition(out, prev_class, "digit")
                    if not in_digit:
                        out.append(t.flag_digit["entry_prefix"])
                        in_digit = True
                    out.append(t.digit.get(c, BRAILLE_SPACE))
                    prev_class = "digit"

                elif c in (".", ","):
                    # 数字の小数点・桁区切りの可能性: 数字モードを終了させない
                    entry = (
                        t.punct_latin.get(c)
                        if in_foreign_word
                        else (t.punct_jp.get(c) or t.punct_latin.get(c))
                    )
                    cur = entry[1] if entry else "none"
                    self._push_transition(out, prev_class, cur)
                    self._emit_punct(out, entry)
                    prev_class = cur

                else:
                    # その他 ASCII: 全フラグをリセット
                    in_foreign_word = False
                    in_digit = False
                    in_capital_word = False
                    in_double_capital = False
                    entry = t.punct_latin.get(c)
                    cur = entry[1] if entry else "none"
                    self._push_transition(out, prev_class, cur)
                    self._emit_punct(out, entry)
                    prev_class = cur

                i += 1

            else:
                # ==================== 非 ASCII（日本語） ====================

                # 外来語モード終了: exit_suffix（⠀）を挿入
                if in_foreign_word:
                    out.append(t.flag_foreign_word.get("exit_suffix", ""))

                # 数字モード終了: 次の点字先頭セルが数字セルと衝突するなら ⠤ を挿入
                if in_digit:
                    first = self._peek_first_kana_cell(c, chars[i + 1] if i + 1 < n else None)
                    if first in self._digit_cells:
                        out.append(t.flag_digit.get("explicit_exit", ""))

                in_digit = False
                in_foreign_word = False
                in_capital_word = False
                in_double_capital = False

                # 複合音（2文字キー）を先に試みる
                if i + 1 < n:
                    key2 = c + chars[i + 1]
                    if key2 in t.kana_compound:
                        self._push_transition(out, prev_class, "kana")
                        out.append(t.kana_compound[key2])
                        prev_class = "kana"
                        i += 2
                        continue

                # 単音・日本語記号
                if c in t.kana_single:
                    self._push_transition(out, prev_class, "kana")
                    out.append(t.kana_single[c])
                    prev_class = "kana"
                elif c in t.punct_jp:
                    brl, cls = t.punct_jp[c]
                    self._push_transition(out, prev_class, cls)
                    out.append(brl)
                    prev_class = cls
                else:
                    out.append(BRAILLE_SPACE)
                    prev_class = "none"

                i += 1

        return "".join(out)

    def _emit_punct(self, out: list[str], entry: tuple[str, str] | None) -> None:
        if entry:
            out.append(entry[0])
        else:
            out.append(BRAILLE_SPACE)

    def _transition_spaces(self, frm: str, to: str) -> int:
        """クラス遷移 frm → to で挿入する点字スペース数。

        完全一致（明示 0 の抑制を含む）が最優先。なければワイルドカード
        "frm -> *" と "* -> to" の最大値。どちらもなければ 0。
        """
        t = self._t.transitions
        exact = t.get((frm, to))
        if exact is not None:
            return exact
        return max(t.get((frm, "*"), 0), t.get(("*", to), 0))

    def _push_transition(self, out: list[str], prev: str | None, cur: str) -> None:
        """クラス遷移 prev → cur に [transitions] で宣言されたスペースを挿入する。

        テキスト先頭（prev が None）では発火しない。直前の出力がすでに
        点字スペースで終わっていれば挿入しない（二重スペース防止）。
        """
        if prev is None:
            return
        spaces = self._transition_spaces(prev, cur)
        if spaces and not self._ends_with_space(out):
            out.append(BRAILLE_SPACE * spaces)

    @staticmethod
    def _ends_with_space(out: list[str]) -> bool:
        # exit_suffix が空文字列のときなど、空要素は読み飛ばして判定する
        for s in reversed(out):
            if s:
                return s.endswith(BRAILLE_SPACE)
        return False

    def _peek_first_kana_cell(self, c: str, next_c: str | None) -> str:
        """数字モード終了時の explicit_exit 判定用: 次の点字先頭セルを返す。"""
        if next_c is not None:
            key2 = c + next_c
            if key2 in self._t.kana_compound:
                brl = self._t.kana_compound[key2]
                return brl[0] if brl else BRAILLE_SPACE
        if c in self._t.kana_single:
            brl = self._t.kana_single[c]
            return brl[0] if brl else BRAILLE_SPACE
        if c in self._t.punct_jp:
            brl, _ = self._t.punct_jp[c]
            return brl[0] if brl else BRAILLE_SPACE
        return BRAILLE_SPACE


# ============================================================
# モジュールレベル API
# ============================================================

_converter: BrailleConverter | None = None


def _get_converter() -> BrailleConverter:
    global _converter
    if _converter is None:
        _converter = BrailleConverter.from_table()
    return _converter


def to_jp_braille(text: str) -> str:
    """日本語かな文字列（カタカナ・ASCII 混在可）を点字文字列に変換する。"""
    return _get_converter().convert(text)


def to_braille(text: str) -> str:
    """点字文字列に変換する（to_jp_braille の別名）。"""
    return _get_converter().convert(text)


if __name__ == "__main__":
    print(to_jp_braille("Hello World!."))
    print(to_jp_braille("コンニチワ、セカイ！"))
    print(to_jp_braille("12345"))
    print(to_jp_braille("３ネン ２クミ"))
    print(
        to_jp_braille(
            "ワタシワ ベンキョーガ キライナノデチュ。ヴァイオリンノ レンシューワ モット キライニャニェニョーー。"
        )
    )
    print(to_jp_braille("NHK ラジオ ダイ１ノ シューハスーワ ５９４kHz デス。"))
    print(to_jp_braille("33min45sec"))
    print(to_jp_braille("エンシューリツワ、3.141592 グライデショー。"))
