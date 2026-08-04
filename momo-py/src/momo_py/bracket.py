"""括弧類の構造分類テーブル（momors-core/src/bracket.rs の移植）。

Rust側(momors-core)の推論時strip/splice処理と同じ分類をPython学習側でも
使えるようにするための、依存の少ない小さなモジュール。判定基準・分類は
bracket.rsと同一に保つこと（どちらかを直したら両方直す）。

- Inline（引用系「」『』""''【】）: 中身は文の項。中身は本文の流れに残す。
- Aside（注釈系（）()｛｝）: 中身は傍注。span ごと本文から抜く。
"""

from enum import Enum
from typing import List, Optional, Tuple

from .features import MORA_SPLIT


class Role(str, Enum):
    """括弧の食いつき方向。"""

    OPEN = "OPEN"
    CLOSE = "CLOSE"

    def __str__(self) -> str:
        return self.value


class Treatment(str, Enum):
    """括弧の中身を本文と一緒に扱ってよいか。"""

    INLINE = "INLINE"
    ASIDE = "ASIDE"

    def __str__(self) -> str:
        return self.value


# bracket.rs の lookup() と同一の対応表。
_TABLE = {
    # 引用系: 中身は文の項
    "「": (Role.OPEN, Treatment.INLINE),
    "」": (Role.CLOSE, Treatment.INLINE),
    "『": (Role.OPEN, Treatment.INLINE),
    "』": (Role.CLOSE, Treatment.INLINE),
    "“": (Role.OPEN, Treatment.INLINE),  # “
    "”": (Role.CLOSE, Treatment.INLINE),  # ”
    "‘": (Role.OPEN, Treatment.INLINE),  # ‘
    "’": (Role.CLOSE, Treatment.INLINE),  # ’
    "【": (Role.OPEN, Treatment.INLINE),
    "】": (Role.CLOSE, Treatment.INLINE),
    # 注釈・挿入系: 中身は傍注
    "（": (Role.OPEN, Treatment.ASIDE),
    "）": (Role.CLOSE, Treatment.ASIDE),
    "(": (Role.OPEN, Treatment.ASIDE),
    ")": (Role.CLOSE, Treatment.ASIDE),
    "｛": (Role.OPEN, Treatment.ASIDE),
    "｝": (Role.CLOSE, Treatment.ASIDE),
}

_PAIRS = {
    ("「", "」"),
    ("『", "』"),
    ("“", "”"),
    ("‘", "’"),
    ("（", "）"),
    ("(", ")"),
    ("｛", "｝"),
    ("【", "】"),
}


def lookup(c: str) -> Optional[Tuple[Role, Treatment]]:
    """`c` が括弧なら (役割, 中身の扱い) を返す。該当しなければ None。"""
    return _TABLE.get(c)


# 対応表に載っている全括弧文字（文中に括弧が含まれるかの判定に使う）。
ALL_BRACKET_CHARS = frozenset(_TABLE.keys())


# 特徴量計算専用の圧縮アイデンティティ・トークン（Private Use Area）。
#
# 個々の括弧字形（「『""''【 など）はbigram/trigram特徴量にとって希少すぎて
# 汎化しない。かといって完全にstripすると、隣接文字のkanji_run/kanji_pos_first/
# bigram/trigramが「括弧がそこにあった」ことを一切知りえず、区切りとして
# 認識できない（実測: momo-inspect trace で「、」が読み分けに効く主因は
# bi_p1_s/type_tri_p2_p1_s のような隣接ベースの特徴量だった）。
#
# そこで、特徴量計算専用の系列（出力・マスあけを決めるstrip/Atom/splice経路とは
# 別）でだけ、括弧の位置をこの3値の圧縮トークンに置き換えて残す。ctypeは
# 既存のSYMBOL_OPEN/SYMBOL_CLOSE（get_char_type）のままでよい（役割はそちらで
# 十分圧縮済み）。この位置自体には読み/境界のラベルは立てない。
#
# 将来Rust側（momors-core/src/bracket.rs）でも同じコードポイントを使うこと。
# ずれるとモデルの語彙（char_s等の特徴量キー）がPython/Rustで食い違う。
INLINE_OPEN_TOKEN = ""
INLINE_CLOSE_TOKEN = ""
ASIDE_TOKEN = ""


def is_pair(open_c: str, close_c: str) -> bool:
    """`open_c` と `close_c` が同じ括弧ペアか。"""
    return (open_c, close_c) in _PAIRS


# 学習データ用の行表現: [char, label, ctype, orig_idx, name_flag, ...]
Row = List[str]
Sentence = List[Row]


def _has_space(label: str) -> bool:
    return MORA_SPLIT in label


def _set_space(label: str, want_space: bool) -> str:
    base = label.replace(MORA_SPLIT, "")
    return base + MORA_SPLIT if want_space else base


def _with_label(row: Row, new_label: str) -> Row:
    return row[:1] + [new_label] + row[2:]


def _with_char_and_label(row: Row, new_char: str, new_label: str) -> Row:
    return [new_char, new_label] + row[2:]


# trainer.py側でこのラベルの行を実際の学習例（Y_read/Y_boundary）から除外する
# ためのセンチネル。source_seq（ひいてはbigram/trigram/kanji_run等の特徴量
# 計算）には残す＝隣接する実文字が「括弧がここにあった」ことを認識できる
# ようにするが、この行自体には読み/境界を予測させない。
#
# ASIDE括弧（開き括弧の行）にのみ使う。中身がspanごと本文から抜かれて独立文
# になるため、開き括弧の行自体は「そこにASIDEがあった」ことを示す特徴量専用の
# プレースホルダでしかなく、実際の出力・マスあけの決定対象になり得ない
# （Rust推論側のextract_aside_spans/splice_atomsが別途決める）。
#
# INLINE括弧には使わない（tokenize_for_training参照）。中身と一緒に本文の
# 流れに残るため、開き・閉じの行自体もRust推論側で実際の境界判定対象になる
# ─ その行自身のラベル（+Sの有無）をそのまま学習に使う。
LABEL_CONTEXT_ONLY = "@"


def strip_for_training(rows: Sentence) -> Tuple[Sentence, List[Sentence]]:
    """学習データの行列から、bracket.rs（momors-core）の推論時strip/splice処理と
    同じ規則で括弧を取り除く。

    従来のtrainer.pyは括弧を生のまま学習データに残していたため、Rust推論側が
    常に括弧をstripしてから予測する実際の入力と食い違っていた
    （bigram/trigram文脈やAsideの独立推論構造が学習/推論で不一致）。
    utilities/bracket_boundary_experiment.py での実データ検証で、この不整合の
    解消だけで実際のBOUNDARY（分かち書き）誤り2件が直ることを確認済み。

    - INLINE括弧（「」『』""''【】）: その場でstrip。閉じ括弧行に付いていた
      +S（空白ラベル）は、strip後に地の文で直前に来る実文字の行へ移す
      （bracket.rsのstrip_bracket_charsは括弧を跨いでモデルの空白判断に触れない
      ため、学習側も同じ隣接関係で+Sを持たせる必要がある）。
    - ASIDE括弧（（）()｛｝）: spanごと本文から抜き取り、中身を独立した副文
      として返す（Rust側predict_with_bracketsの再帰独立推論に対応）。閉じ括弧
      行の+Sは、開き括弧の直前に来る実文字の行へ移す。

    戻り値: (stripしたメイン文, 副文のリスト。入れ子Asideは再帰的に平坦化)
    """
    out_rows: Sentence = []
    sub_sentences: List[Sentence] = []
    i, n = 0, len(rows)
    while i < n:
        row = rows[i]
        ch = row[0]
        info = lookup(ch) if len(ch) == 1 else None

        if info is None:
            out_rows.append(row)
            i += 1
            continue

        role, treatment = info

        if treatment == Treatment.INLINE:
            if role == Role.OPEN:
                if _has_space(row[1]):
                    print(f"⚠️  開き括弧行に+S(未対応パターン、無視): {row}")
            else:  # CLOSE
                if out_rows and _has_space(row[1]):
                    out_rows[-1] = _with_label(out_rows[-1], _set_space(out_rows[-1][1], True))
                elif _has_space(row[1]):
                    print(f"⚠️  文頭付近の閉じ括弧+Sを移設できません(無視): {row}")
            i += 1
            continue

        # Aside: 対応する閉じまでdepth追跡して span ごと抜き取る。
        depth, j = 1, i + 1
        content_rows: Sentence = []
        close_row: Optional[Row] = None
        while j < n:
            ch2 = rows[j][0]
            info2 = lookup(ch2) if len(ch2) == 1 else None
            if info2 is not None and info2[1] == Treatment.ASIDE:
                if info2[0] == Role.OPEN:
                    depth += 1
                    content_rows.append(rows[j])
                else:
                    depth -= 1
                    if depth == 0:
                        close_row = rows[j]
                        j += 1
                        break
                    content_rows.append(rows[j])
            else:
                content_rows.append(rows[j])
            j += 1

        if close_row is None:
            print(f"⚠️  対応する閉じ括弧が見つかりません(素通し): {row}")
            out_rows.append(row)
            i += 1
            continue

        if out_rows and _has_space(close_row[1]):
            out_rows[-1] = _with_label(out_rows[-1], _set_space(out_rows[-1][1], True))
        elif _has_space(close_row[1]):
            print(f"⚠️  文頭の注釈括弧の閉じ+Sを移設できません(無視): {close_row}")

        if content_rows:
            sub_main, sub_subs = strip_for_training(content_rows)
            sub_sentences.append(sub_main)
            sub_sentences.extend(sub_subs)

        i = j

    return out_rows, sub_sentences


def tokenize_for_training(rows: Sentence) -> Tuple[Sentence, List[Sentence]]:
    """学習データの行列に対し、括弧の行を削除せず残しつつ、charを特徴量計算
    専用の圧縮トークン（INLINE_OPEN_TOKEN/INLINE_CLOSE_TOKEN/ASIDE_TOKEN）に
    置き換える（strip_for_trainingの後継、圧縮アイデンティティ・トークン方式）。

    strip_for_trainingとの違い: 括弧の行そのものは削除しない。char列だけを
    トークンへ差し替える。個々の括弧字形はbigram/trigramにとって希少すぎて
    汎化しないが、Role×Treatmentを3値に圧縮したトークンなら汎化する
    （実測: momo-inspect traceで「、」が読み分けに効く主因はbi_p1_s/
    type_tri_p2_p1_sのような隣接ベースの特徴量だった。kanji_run/
    kanji_pos_firstはsource_seq上の隣接をそのまま辿る実装のため、
    strip_for_trainingのように行ごと削除すると括弧の存在を一切認識できない）。
    utilities/bracket_identity_token_experiment.pyでの実データ検証で、
    READING誤り2件（海→ウミ、楽→ガッ）が直ることを確認済み。

    - INLINE括弧: その場でトークンに置換するだけで、ラベル（`+S`の有無を
      含む）には一切触れない。中身と一緒に本文の流れに残り、Rust推論側
      （momors-core::prediction）でも実文字のまま境界判定の対象になる
      （strip/spliceでは外さない）ため、開き・閉じの行自身が持つ`+S`を
      そのまま学習に使えばよく、直前行への移設は不要（旧strip_for_training
      時代の名残）。
    - ASIDE括弧: 開き括弧の行だけを`ASIDE_TOKEN`＋`LABEL_CONTEXT_ONLY`に
      置換し、閉じ括弧の行と中身は本文から除く。中身は独立した副文として
      返す（Rust側predict_with_bracketsの再帰独立推論に対応）。中身は
      span ごと本文から抜かれるため、閉じ括弧の`+S`はRust側と同じく
      開き括弧の直前に来る実文字行へ移す（Aside側は従来通り）。

    戻り値: (トークン化したメイン文, 副文のリスト。入れ子Asideは再帰的に平坦化)
    """
    out_rows: Sentence = []
    sub_sentences: List[Sentence] = []
    i, n = 0, len(rows)
    while i < n:
        row = rows[i]
        ch = row[0]
        info = lookup(ch) if len(ch) == 1 else None

        if info is None:
            out_rows.append(row)
            i += 1
            continue

        role, treatment = info

        if treatment == Treatment.INLINE:
            token = INLINE_OPEN_TOKEN if role == Role.OPEN else INLINE_CLOSE_TOKEN
            out_rows.append(_with_char_and_label(row, token, row[1]))
            i += 1
            continue

        # Aside: 対応する閉じまでdepth追跡。開き行をASIDEトークンに置換し、
        # 閉じ行・中身は本文から除く（中身は独立した副文として再帰処理）。
        depth, j = 1, i + 1
        content_rows: Sentence = []
        close_row: Optional[Row] = None
        while j < n:
            ch2 = rows[j][0]
            info2 = lookup(ch2) if len(ch2) == 1 else None
            if info2 is not None and info2[1] == Treatment.ASIDE:
                if info2[0] == Role.OPEN:
                    depth += 1
                    content_rows.append(rows[j])
                else:
                    depth -= 1
                    if depth == 0:
                        close_row = rows[j]
                        j += 1
                        break
                    content_rows.append(rows[j])
            else:
                content_rows.append(rows[j])
            j += 1

        if close_row is None:
            print(f"⚠️  対応する閉じ括弧が見つかりません(素通し): {row}")
            out_rows.append(row)
            i += 1
            continue

        if out_rows and _has_space(close_row[1]):
            out_rows[-1] = _with_label(out_rows[-1], _set_space(out_rows[-1][1], True))
        elif _has_space(close_row[1]):
            print(f"⚠️  文頭の注釈括弧の閉じ+Sを移設できません(無視): {close_row}")

        out_rows.append(_with_char_and_label(row, ASIDE_TOKEN, LABEL_CONTEXT_ONLY))

        if content_rows:
            sub_main, sub_subs = tokenize_for_training(content_rows)
            sub_sentences.append(sub_main)
            sub_sentences.extend(sub_subs)

        i = j

    return out_rows, sub_sentences
