import re
from typing import List, Dict, Tuple

from .utils import (
    CharType,
    get_char_type,
    KURAI_CHARS,
)

# --- [共通定義・型定義] ---
FeatureDict = Dict[str, float]
# ソース文字系列 = [(char, orig_idx, ctype), ...]
SourceEntry = Tuple[str, int, str]

# --- [共通定数] ---
MORA_SPLIT = "+S"
LABEL_CONTINUE = "---"
LABEL_SKIP = "_"

# --- [特徴量キー定数] ---
# 実際のキーは f'{F_xxx}={value}' の形式（ペイロードなし系は定数そのまま）。
F_BIAS = "bias"
F_CHAR_S = "char_s"
F_CHAR_P1 = "char_p1"
F_CHAR_P2 = "char_p2"
F_CHAR_P3 = "char_p3"
F_CHAR_N1 = "char_n1"
F_CHAR_N2 = "char_n2"
F_CHAR_N3 = "char_n3"
F_TYPE_S = "type_s"
F_TYPE_P1 = "type_p1"
F_TYPE_P2 = "type_p2"
F_TYPE_P3 = "type_p3"
F_TYPE_N1 = "type_n1"
F_TYPE_N2 = "type_n2"
F_TYPE_N3 = "type_n3"
F_BI_P1_S = "bi_p1_s"
F_BI_P2_P1 = "bi_p2_p1"
F_BI_P3_P2 = "bi_p3_p2"
F_BI_S_N1 = "bi_s_n1"
F_BI_N1_N2 = "bi_n1_n2"
F_BI_N2_N3 = "bi_n2_n3"
F_TRI_P3_P2_P1 = "tri_p3_p2_p1"
F_TRI_P2_P1_S = "tri_p2_p1_s"
F_TRI_P1_S_N1 = "tri_p1_s_n1"
F_TRI_S_N1_N2 = "tri_s_n1_n2"
F_TRI_N1_N2_N3 = "tri_n1_n2_n3"
F_TYPE_TRI_P3_P2_P1 = "type_tri_p3_p2_p1"
F_TYPE_TRI_P2_P1_S = "type_tri_p2_p1_s"
F_TYPE_TRI_P1_S_N1 = "type_tri_p1_s_n1"
F_TYPE_TRI_S_N1_N2 = "type_tri_s_n1_n2"
F_TYPE_TRI_N1_N2_N3 = "type_tri_n1_n2_n3"
F_TYPE_TRANS_P1_S = "type_trans_p1_s"
F_KANJI_RUN = "kanji_run"
F_KANJI_POS_FIRST = "kanji_pos_first"
F_JNUM_RUN = "jnum_run"
F_JNUM_RUN_P1 = "jnum_run_p1"
# 人名辞書マッチフラグ（値は B/I。name_dict.compute_name_flags が生成する）
F_NAME_S = "name_s"
F_NAME_P1 = "name_p1"
F_NAME_N1 = "name_n1"

# 特徴量として発火させる人名フラグ値（O は発火させない）
_NAME_ACTIVE_FLAGS = ("B", "I")


def get_units(text: str) -> List[Tuple[str, int, str]]:
    """
    テキストを音節ユニットに分解し、(文字列, 原文開始位置, 文字種) のリストを返す。

    ひらがな+小書き（拗音など）は複数文字ブロックとしてまとめる。
    ASCII文字（英数字・記号）は1文字ずつ個別ユニットとして扱う（RAWアノテーションで
    各文字に個別ラベルを付けられるようにするため）。
    [...]構文で明示的にブロック化できる（例: [Hello] → 1ユニット）。
    JAPANESE_NUMERIC については以下の2段階で文字種を確定する:
      1. 漢数字（〇一二三…）は無条件で JAPANESE_NUMERIC。
      2. 位取り文字（十百千万億兆）は左右いずれかに JAPANESE_NUMERIC が
         隣接する場合のみ昇格させる。
         （「万全」の「万」は昇格しない。「三万円」の「万」は昇格する。）
    ブロック化は行わず、各要素は1文字（または既存の複数文字ブロック）のまま。
    """
    # 拗音・特殊音の複合ユニット: 1文字目は小書き仮名以外のかな、2文字目は小書き仮名1文字
    # 1文字目の負lookbehindで小書き仮名（ぁぃぅぇぉっゃゅょゎ / ァィゥェォッャュョヮ）を除外する
    # 英字・記号（!-/は記号、:-~は英字+記号）はまとめる。数字(0-9)は除外して個別ユニットにする。
    regex = r"\[(.*?)\]|([ぁ-んァ-ヶ](?<![ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ])[ぁぃぅぇぉゃゅょゎァィゥェォャュョヮヵヶ])|([!-/:-~]+(?:[ \t]+[!-/:-~]+)*)|(\s+)|(.)"
    raw_units: List[Tuple[str, int]] = []
    for m in re.finditer(regex, text):
        if (g := m.group(1)) is not None:
            raw_units.append((g, m.start(1)))
        elif (g := m.group(2)) is not None:
            raw_units.append((g, m.start(2)))
        elif (g := m.group(3)) is not None:
            raw_units.append((g, m.start(3)))
        elif (g := m.group(4)) is not None:
            raw_units.append((g, m.start(4)))
        else:
            raw_units.append((m.group(5) or "", m.start(5)))

    # --- 位取り文字の昇格判定 ---
    n = len(raw_units)
    is_numeric: List[bool] = [
        (len(val) == 1 and get_char_type(val) == CharType.JAPANESE_NUMERIC)
        for val, _ in raw_units
    ]

    # 左→右パス: 左隣が numeric なら位取り文字を昇格
    for i in range(1, n):
        val = raw_units[i][0]
        if len(val) == 1 and val in KURAI_CHARS and is_numeric[i - 1]:
            is_numeric[i] = True

    # 右→左パス: 右隣が numeric なら位取り文字を昇格（「十一」の「十」など）
    for i in range(n - 2, -1, -1):
        val = raw_units[i][0]
        if len(val) == 1 and val in KURAI_CHARS and is_numeric[i + 1]:
            is_numeric[i] = True

    # --- ctype を確定して返す（ブロック化しない） ---
    units: List[Tuple[str, int, str]] = []
    for i, (val, orig_idx) in enumerate(raw_units):
        if is_numeric[i]:
            ctype = CharType.JAPANESE_NUMERIC
        else:
            ctype = get_char_type(val[0]) if val else CharType.OTHER
        units.append((val, orig_idx, ctype))

    return units


def compute_source_features(
    source_seq: List[SourceEntry],
    window: int = 7,
    name_flags: List[str] | None = None,
) -> List[FeatureDict]:
    """
    ソース文字系列全体に対して、各文字の文脈特徴量を一括計算する。
    pycrfsuiteネイティブの { "feature_name=value": 1.0 } 形式で出力する。

    window: 参照する文字数。4, 5, 7 を指定する。
      7 (デフォルト): 前後3文字まで参照（bigram×6, trigram×5）
      5             : 前後2文字まで参照（bigram×4, trigram×3）
      4             : 前1・後2文字参照 [-1,0,+1,+2]（bigram×3, trigram×2）
    name_flags: 人名辞書マッチフラグ（B/I/O）の系列。source_seq と同じ長さ。
      None の場合は人名特徴量を出力しない（人名辞書なしの従来動作）。
      学習・推論とも name_dict.compute_name_flags() で生成すること
      （学習データの正解マークを直接渡してはならない）。
    """
    if window not in (4, 5, 7):
        raise ValueError(f"window は 4, 5, 7 のいずれかでなければなりません: {window}")
    if name_flags is not None and len(name_flags) != len(source_seq):
        raise ValueError(
            f"name_flags の長さ（{len(name_flags)}）が "
            f"source_seq（{len(source_seq)}）と一致しません"
        )
    result: List[FeatureDict] = []
    n = len(source_seq)

    for i, (char, _orig_idx, ctype) in enumerate(source_seq):
        # 文脈特徴量（bigram/trigram/char_p*/n*）は各エントリの先頭1文字を使用。
        # 複合ユニット（例: "きゃ"）でも先頭文字 "き" のみを文脈に使う。
        # char_s のみ複合ユニット文字列全体を使用（C++側 CHAR_SELF_COMPOUND と対応）。
        c = char[0] if char else ""  # 文脈特徴量用

        prev3_char = source_seq[i - 3][0] if i > 2 else ""
        prev3_ctype = source_seq[i - 3][2] if i > 2 else ""
        prev2_char = source_seq[i - 2][0] if i > 1 else ""
        prev2_ctype = source_seq[i - 2][2] if i > 1 else ""
        prev_char = source_seq[i - 1][0] if i > 0 else ""
        prev_ctype = source_seq[i - 1][2] if i > 0 else ""
        next_char = source_seq[i + 1][0] if i < n - 1 else ""
        next_ctype = source_seq[i + 1][2] if i < n - 1 else ""
        next2_char = source_seq[i + 2][0] if i < n - 2 else ""
        next2_ctype = source_seq[i + 2][2] if i < n - 2 else ""
        next3_char = source_seq[i + 3][0] if i < n - 3 else ""
        next3_ctype = source_seq[i + 3][2] if i < n - 3 else ""

        # 各エントリの先頭1文字（文脈用）
        pc  = prev_char[0]  if prev_char  else ""
        pc2 = prev2_char[0] if prev2_char else ""
        pc3 = prev3_char[0] if prev3_char else ""
        nc  = next_char[0]  if next_char  else ""
        nc2 = next2_char[0] if next2_char else ""
        nc3 = next3_char[0] if next3_char else ""

        features: FeatureDict = {
            F_BIAS: 1.0,
            f"{F_CHAR_S}={char}": 1.0,  # 複合ユニット全体（C++: CHAR_SELF_COMPOUND）
            f"{F_TYPE_S}={ctype}": 1.0,
        }

        if i > 0:
            features[f"{F_CHAR_P1}={pc}"] = 1.0
            features[f"{F_TYPE_P1}={prev_ctype}"] = 1.0
            features[f"{F_BI_P1_S}={pc}{c}"] = 1.0
            if window >= 5 and i > 1:
                features[f"{F_CHAR_P2}={pc2}"] = 1.0
                features[f"{F_TYPE_P2}={prev2_ctype}"] = 1.0
                features[f"{F_BI_P2_P1}={pc2}{pc}"] = 1.0
                if window >= 7 and i > 2:
                    features[f"{F_CHAR_P3}={pc3}"] = 1.0
                    features[f"{F_TYPE_P3}={prev3_ctype}"] = 1.0
                    features[f"{F_BI_P3_P2}={pc3}{pc2}"] = 1.0
        if i < n - 1:
            features[f"{F_CHAR_N1}={nc}"] = 1.0
            features[f"{F_TYPE_N1}={next_ctype}"] = 1.0
            features[f"{F_BI_S_N1}={c}{nc}"] = 1.0
            if window >= 4 and i < n - 2:
                features[f"{F_CHAR_N2}={nc2}"] = 1.0
                features[f"{F_TYPE_N2}={next2_ctype}"] = 1.0
                features[f"{F_BI_N1_N2}={nc}{nc2}"] = 1.0
                if window >= 7 and i < n - 3:
                    features[f"{F_CHAR_N3}={nc3}"] = 1.0
                    features[f"{F_TYPE_N3}={next3_ctype}"] = 1.0
                    features[f"{F_BI_N2_N3}={nc2}{nc3}"] = 1.0
        if i > 0:
            features[f"{F_TYPE_TRANS_P1_S}={prev_ctype}->{ctype}"] = 1.0

        # trigram: 前3-前2-前1（window=7 のみ）
        if window >= 7 and i > 2:
            features[f"{F_TRI_P3_P2_P1}={pc3}{pc2}{pc}"] = 1.0
            features[
                f"{F_TYPE_TRI_P3_P2_P1}={prev3_ctype}-{prev2_ctype}-{prev_ctype}"
            ] = 1.0

        # trigram: 前2-前1-対象（window=5 以上）
        if window >= 5 and i > 1:
            features[f"{F_TRI_P2_P1_S}={pc2}{pc}{c}"] = 1.0
            features[f"{F_TYPE_TRI_P2_P1_S}={prev2_ctype}-{prev_ctype}-{ctype}"] = 1.0

        # trigram: 前1-対象-後1
        if i > 0 and i < n - 1:
            features[f"{F_TRI_P1_S_N1}={pc}{c}{nc}"] = 1.0
            features[f"{F_TYPE_TRI_P1_S_N1}={prev_ctype}-{ctype}-{next_ctype}"] = 1.0

        # trigram: 対象-後1-後2（window=5 以上。window=4 はバイグラムで代替）
        if window >= 5 and i < n - 2:
            features[f"{F_TRI_S_N1_N2}={c}{nc}{nc2}"] = 1.0
            features[f"{F_TYPE_TRI_S_N1_N2}={ctype}-{next_ctype}-{next2_ctype}"] = 1.0

        # trigram: 後1-後2-後3（window=7 のみ）
        if window >= 7 and i < n - 3:
            features[f"{F_TRI_N1_N2_N3}={nc}{nc2}{nc3}"] = 1.0
            features[
                f"{F_TYPE_TRI_N1_N2_N3}={next_ctype}-{next2_ctype}-{next3_ctype}"
            ] = 1.0

        if ctype == "KANJI":
            run = 1
            j = i + 1
            while j < n and source_seq[j][2] == "KANJI":
                run += 1
                j += 1
            j = i - 1
            while j >= 0 and source_seq[j][2] == "KANJI":
                run += 1
                j -= 1

            run_key = str(run) if run <= 4 else "5+"
            features[f"{F_KANJI_RUN}={run_key}"] = 1.0

            if i == 0 or source_seq[i - 1][2] != "KANJI":
                features[F_KANJI_POS_FIRST] = 1.0

            # 左隣の JAPANESE_NUMERIC 連続長を「日」などの隣接 KANJI に伝える
            if i > 0 and source_seq[i - 1][2] == "JAPANESE_NUMERIC":
                num_run = 0
                j = i - 1
                while j >= 0 and source_seq[j][2] == "JAPANESE_NUMERIC":
                    num_run += 1
                    j -= 1
                num_run_key = str(num_run) if num_run <= 4 else "5+"
                features[f"{F_JNUM_RUN_P1}={num_run_key}"] = 1.0

        if ctype == "JAPANESE_NUMERIC":
            run = 1
            j = i + 1
            while j < n and source_seq[j][2] == "JAPANESE_NUMERIC":
                run += 1
                j += 1
            j = i - 1
            while j >= 0 and source_seq[j][2] == "JAPANESE_NUMERIC":
                run += 1
                j -= 1

            run_key = str(run) if run <= 4 else "5+"
            features[f"{F_JNUM_RUN}={run_key}"] = 1.0

        # 人名辞書マッチフラグ（自分・前1・後1）。
        # 「name_s=I かつ name_n1 なし」= 人名の右端、のように B/I の
        # 位置の組み合わせで人名の境界が線形モデルに見えるようにする。
        if name_flags is not None:
            if name_flags[i] in _NAME_ACTIVE_FLAGS:
                features[f"{F_NAME_S}={name_flags[i]}"] = 1.0
            if i > 0 and name_flags[i - 1] in _NAME_ACTIVE_FLAGS:
                features[f"{F_NAME_P1}={name_flags[i - 1]}"] = 1.0
            if i < n - 1 and name_flags[i + 1] in _NAME_ACTIVE_FLAGS:
                features[f"{F_NAME_N1}={name_flags[i + 1]}"] = 1.0

        result.append(features)

    return result
