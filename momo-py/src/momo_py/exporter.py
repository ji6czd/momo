"""
exporter.py  ―  momo モデルを C++ 推論エンジン向けバイナリ (.mbm) に変換する。

バイナリフォーマット概要
========================

[ファイルヘッダ]          16 bytes
  magic        : uint8[4]   "MOMO"
  version      : uint8      0x04
  _reserved    : uint8[3]   0x00 x3
  n_classes    : uint32     読みラベル数
  n_features   : uint32     特徴量次元数（語彙サイズ）

[語彙テーブル]            n_features エントリ
  feature_type : uint8      FeatureType 値
  chartype[N]  : uint8 × N  N = chartype_count(feature_type)
  char32[M]    : uint32 × M M = char32_count(feature_type)
  uint8_val    : uint8      is_uint8_payload(feature_type) のときのみ
  feature_id   : uint32     DictVectorizer の feature_id（ソート用インデックス）

[読みラベルテーブル]      n_classes エントリ
  len          : uint8      UTF-8バイト長
  utf8         : uint8[len] ラベル文字列（UTF-8）

[読みモデル重み（CSR・int8量子化・クラスごとscale）]
  quant_scale  : float32 × n_classes   クラス(行)ごとの量子化スケール係数
  n_nonzero    : uint32     非ゼロ要素数
  indptr       : uint32 × (n_classes + 1)
  indices      : uint32 × n_nonzero
  data         : int8  × n_nonzero

[読みモデル intercept]
  intercept    : float32 × n_classes

[境界モデル重み（int8量子化）]
  quant_scale  : float32
  data         : int8 × n_features   （クラス1の重みベクトル）
  intercept    : float32 × 2         （クラス0, クラス1）

[人名辞書テーブル]        version 0x03 で追加、0x04 で読みを追加
  n_names      : uint32     人名エントリ数（辞書なしモデルは 0）
  以下 n_names エントリ:
    len        : uint8      UTF-8バイト長
    utf8       : uint8[len] 表層形（UTF-8）
    n_readings : uint8      ユニット別読みの個数（0=読みなし。
                            非0なら表層形のユニット数と一致する）
    以下 n_readings 個:
      len      : uint8      UTF-8バイト長
      utf8     : uint8[len] ユニット読み（カタカナ、UTF-8）
  推論側はこの辞書に対して get_units() 相当のユニット単位で最長一致を行い、
  文字ごとの人名フラグ（1=B: スパン先頭, 2=I: 継続）を NAME_FLAG_* 特徴量
  として発火させる（Python 側 name_dict.compute_name_matches と同一の手順）。
  読みが登録されているスパンでは、読みモデルが低自信度のとき辞書読みで
  置換するフォールバックに使う（人名の読みは文脈で変化しないため固定辞書）。

[単一漢字辞書テーブル]    version 0x04 の途中（アルファ期間）で追加
  n_entries    : uint32     エントリ数
  以下 n_entries エントリ:
    len        : uint8      UTF-8バイト長
    utf8       : uint8[len] 漢字（1文字、UTF-8）
    n_readings : uint8      既知の読みの個数
    以下 n_readings 個:
      len      : uint8      UTF-8バイト長
      utf8     : uint8[len] 読み（カタカナ、UTF-8）
  読みモデルの候補制約（辞書に載っている漢字は既知の読み+CONTINUE+SKIPのみを
  argmax 候補とする）に使う。モデル動作に必須のデータなので .mbm に同梱する。

バージョン履歴
  0x01: 初版
  0x02: 読みモデルの量子化scaleをクラスごと(n_classes個)に変更
  0x03: 人名辞書テーブルと NAME_FLAG_* 特徴量（0xC3-0xC5）を追加
  0x04: 人名辞書テーブルにユニット別読みを追加（低自信度フォールバック用）。
        アルファ期間中に単一漢字辞書テーブルも追加（バージョン番号は据え置き。
        追加前の 0x04 ファイルは読み込み時にエラーになるため再エクスポートすること）
"""

import re
import struct
import zipfile
import tempfile
import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np

from .name_dict import NAME_DICT_FILENAME, parse_name_dict_text
from .predictor import SINGLE_KANJI_DICT_FILENAME, _parse_kanji_dict_tsv


# =====================================================================
# CharType 文字列 → uint8 対応表
# =====================================================================
CHARTYPE_TO_INT: dict[str, int] = {
    "SPACE":            0x00,
    "ALPHA":            0x10,
    "NUM":              0x11,   # Python 側は CharType.NUMERIC.value == 'NUM'
    "SYMBOL":           0x30,
    "SYMBOL_CLOSE":     0x31,
    "SYMBOL_OPEN":      0x32,
    "SYMBOL_STOP":      0x33,
    "SYMBOL_PAUSE":     0x34,
    "HIRAGANA":         0x40,
    "KATAKANA":         0x41,
    "KANJI":            0x42,
    "JAPANESE_NUMERIC": 0x43,
    "OTHER":            0xFF,
}


# =====================================================================
# FeatureType
#
# bit7-6 : ペイロード種別
#   0b00 (0x0_) = なし
#   0b01 (0x4_-0x7_) = CharType
#   0b10 (0x8_-0xB_) = char32_t
#   0b11 (0xC_) = uint8
#
# bit5-4 : 個数（CharType または char32_t の個数）
#   0b00 = 0 個（または uint8×1 固定）
#   0b01 = 1 個
#   0b10 = 2 個
#   0b11 = 3 個
# =====================================================================
class FT:
    BIAS                          = 0x00
    KANJI_POS_FIRST               = 0x01

    TYPE_SELF                     = 0x50   # CharType×1
    TYPE_PREV1                    = 0x51
    TYPE_PREV2                    = 0x52
    TYPE_NEXT1                    = 0x53
    TYPE_NEXT2                    = 0x54
    TYPE_PREV3                    = 0x55
    TYPE_NEXT3                    = 0x56

    TYPE_TRANSITION               = 0x60   # CharType×2

    TYPE_TRI_PREV2_PREV1_SELF     = 0x70   # CharType×3  (前2-前1-対象)
    TYPE_TRI_PREV1_SELF_NEXT1     = 0x71   # CharType×3  (前1-対象-後1)
    TYPE_TRI_SELF_NEXT1_NEXT2     = 0x72   # CharType×3  (対象-後1-後2)
    TYPE_TRI_PREV3_PREV2_PREV1    = 0x73   # CharType×3  (前3-前2-前1)
    TYPE_TRI_NEXT1_NEXT2_NEXT3    = 0x74   # CharType×3  (後1-後2-後3)

    CHAR_SELF                     = 0x90   # char32_t×1
    CHAR_PREV1                    = 0x91
    CHAR_PREV2                    = 0x92
    CHAR_NEXT1                    = 0x93
    CHAR_NEXT2                    = 0x94
    CHAR_PREV3                    = 0x95
    CHAR_NEXT3                    = 0x96

    BIGRAM_PREV1_SELF             = 0xA0   # char32_t×2
    BIGRAM_PREV2_PREV1            = 0xA1
    BIGRAM_SELF_NEXT1             = 0xA2
    BIGRAM_NEXT1_NEXT2            = 0xA3
    BIGRAM_PREV3_PREV2            = 0xA4
    BIGRAM_NEXT2_NEXT3            = 0xA5
    CHAR_SELF_COMPOUND_2          = 0xA6   # char32_t×2: 2文字複合ユニットの char_s

    TRIGRAM_PREV2_PREV1_SELF      = 0xB0   # char32_t×3  (前2-前1-対象)
    TRIGRAM_PREV1_SELF_NEXT1      = 0xB1   # char32_t×3  (前1-対象-後1)
    TRIGRAM_SELF_NEXT1_NEXT2      = 0xB2   # char32_t×3  (対象-後1-後2)
    TRIGRAM_PREV3_PREV2_PREV1     = 0xB3   # char32_t×3  (前3-前2-前1)
    TRIGRAM_NEXT1_NEXT2_NEXT3     = 0xB4   # char32_t×3  (後1-後2-後3)
    CHAR_SELF_COMPOUND_3          = 0xB5   # char32_t×3: 3文字複合ユニットの char_s

    KANJI_RUN_LEN                 = 0xC0   # uint8
    JAPANESE_NUMERIC_RUN_LEN      = 0xC1
    PREV_JAPANESE_NUMERIC_RUN_LEN = 0xC2
    NAME_FLAG_SELF                = 0xC3   # uint8: 1=B, 2=I
    NAME_FLAG_PREV1               = 0xC4
    NAME_FLAG_NEXT1               = 0xC5


def chartype_count(ft: int) -> int:
    if (ft & 0xC0) != 0x40:
        return 0
    return (ft >> 4) & 0x03


def char32_count(ft: int) -> int:
    if (ft & 0xC0) != 0x80:
        return 0
    return (ft >> 4) & 0x03


def is_uint8_payload(ft: int) -> bool:
    return (ft & 0xC0) == 0xC0


# =====================================================================
# Python 特徴量キー文字列 → (FeatureType, chartype_list, char32_list, uint8_val)
# =====================================================================

# run_len の文字列値 → uint8
_RUN_LEN_MAP = {"1": 1, "2": 2, "3": 3, "4": 4, "5+": 5}

# 人名フラグの文字列値 → uint8
_NAME_FLAG_MAP = {"B": 1, "I": 2}


def parse_feature_key(key: str) -> Tuple[int, list, list, int | None]:
    """
    DictVectorizer の vocabulary_ キーを解析して
    (feature_type, chartype_vals, char32_vals, uint8_val) を返す。

    chartype_vals : list[int]   CharType の uint8 値列
    char32_vals   : list[int]   Unicode コードポイント列
    uint8_val     : int | None  run_len 系のときのみ
    """

    # --- ペイロードなし系 ---
    if key == "bias":
        return FT.BIAS, [], [], None
    if key == "kanji_pos_first":
        return FT.KANJI_POS_FIRST, [], [], None

    # --- run 系（uint8ペイロード）---
    m = re.fullmatch(r'kanji_run=(.+)', key)
    if m:
        return FT.KANJI_RUN_LEN, [], [], _RUN_LEN_MAP[m.group(1)]

    m = re.fullmatch(r'jnum_run=(.+)', key)
    if m:
        return FT.JAPANESE_NUMERIC_RUN_LEN, [], [], _RUN_LEN_MAP[m.group(1)]

    m = re.fullmatch(r'jnum_run_p1=(.+)', key)
    if m:
        return FT.PREV_JAPANESE_NUMERIC_RUN_LEN, [], [], _RUN_LEN_MAP[m.group(1)]

    # --- 人名フラグ系（uint8ペイロード: 1=B, 2=I）---
    m = re.fullmatch(r'name_s=([BI])', key)
    if m:
        return FT.NAME_FLAG_SELF, [], [], _NAME_FLAG_MAP[m.group(1)]

    m = re.fullmatch(r'name_p1=([BI])', key)
    if m:
        return FT.NAME_FLAG_PREV1, [], [], _NAME_FLAG_MAP[m.group(1)]

    m = re.fullmatch(r'name_n1=([BI])', key)
    if m:
        return FT.NAME_FLAG_NEXT1, [], [], _NAME_FLAG_MAP[m.group(1)]

    # --- type_tri （CharType×3）---
    m = re.fullmatch(r'type_tri_p2_p1_s=(.+)-(.+)-(.+)', key)
    if m:
        cts = [CHARTYPE_TO_INT[m.group(i)] for i in (1, 2, 3)]
        return FT.TYPE_TRI_PREV2_PREV1_SELF, cts, [], None

    m = re.fullmatch(r'type_tri_p1_s_n1=(.+)-(.+)-(.+)', key)
    if m:
        cts = [CHARTYPE_TO_INT[m.group(i)] for i in (1, 2, 3)]
        return FT.TYPE_TRI_PREV1_SELF_NEXT1, cts, [], None

    m = re.fullmatch(r'type_tri_s_n1_n2=(.+)-(.+)-(.+)', key)
    if m:
        cts = [CHARTYPE_TO_INT[m.group(i)] for i in (1, 2, 3)]
        return FT.TYPE_TRI_SELF_NEXT1_NEXT2, cts, [], None

    m = re.fullmatch(r'type_tri_p3_p2_p1=(.+)-(.+)-(.+)', key)
    if m:
        cts = [CHARTYPE_TO_INT[m.group(i)] for i in (1, 2, 3)]
        return FT.TYPE_TRI_PREV3_PREV2_PREV1, cts, [], None

    m = re.fullmatch(r'type_tri_n1_n2_n3=(.+)-(.+)-(.+)', key)
    if m:
        cts = [CHARTYPE_TO_INT[m.group(i)] for i in (1, 2, 3)]
        return FT.TYPE_TRI_NEXT1_NEXT2_NEXT3, cts, [], None

    # --- type_trans_p1_s （CharType×2）---
    m = re.fullmatch(r'type_trans_p1_s=(.+)->(.+)', key)
    if m:
        cts = [CHARTYPE_TO_INT[m.group(1)], CHARTYPE_TO_INT[m.group(2)]]
        return FT.TYPE_TRANSITION, cts, [], None

    # --- type 系（CharType×1）---
    m = re.fullmatch(r'type_s=(.+)', key)
    if m:
        return FT.TYPE_SELF, [CHARTYPE_TO_INT[m.group(1)]], [], None
    m = re.fullmatch(r'type_p1=(.+)', key)
    if m:
        return FT.TYPE_PREV1, [CHARTYPE_TO_INT[m.group(1)]], [], None
    m = re.fullmatch(r'type_p2=(.+)', key)
    if m:
        return FT.TYPE_PREV2, [CHARTYPE_TO_INT[m.group(1)]], [], None
    m = re.fullmatch(r'type_n1=(.+)', key)
    if m:
        return FT.TYPE_NEXT1, [CHARTYPE_TO_INT[m.group(1)]], [], None
    m = re.fullmatch(r'type_n2=(.+)', key)
    if m:
        return FT.TYPE_NEXT2, [CHARTYPE_TO_INT[m.group(1)]], [], None
    m = re.fullmatch(r'type_p3=(.+)', key)
    if m:
        return FT.TYPE_PREV3, [CHARTYPE_TO_INT[m.group(1)]], [], None
    m = re.fullmatch(r'type_n3=(.+)', key)
    if m:
        return FT.TYPE_NEXT3, [CHARTYPE_TO_INT[m.group(1)]], [], None

    # --- trigram（char32_t×3）---
    m = re.fullmatch(r'tri_p2_p1_s=(.)(.)(.)', key)
    if m:
        cps = [ord(m.group(i)) for i in (1, 2, 3)]
        return FT.TRIGRAM_PREV2_PREV1_SELF, [], cps, None

    m = re.fullmatch(r'tri_p1_s_n1=(.)(.)(.)', key)
    if m:
        cps = [ord(m.group(i)) for i in (1, 2, 3)]
        return FT.TRIGRAM_PREV1_SELF_NEXT1, [], cps, None

    m = re.fullmatch(r'tri_s_n1_n2=(.)(.)(.)', key)
    if m:
        cps = [ord(m.group(i)) for i in (1, 2, 3)]
        return FT.TRIGRAM_SELF_NEXT1_NEXT2, [], cps, None

    m = re.fullmatch(r'tri_p3_p2_p1=(.)(.)(.)', key)
    if m:
        cps = [ord(m.group(i)) for i in (1, 2, 3)]
        return FT.TRIGRAM_PREV3_PREV2_PREV1, [], cps, None

    m = re.fullmatch(r'tri_n1_n2_n3=(.)(.)(.)', key)
    if m:
        cps = [ord(m.group(i)) for i in (1, 2, 3)]
        return FT.TRIGRAM_NEXT1_NEXT2_NEXT3, [], cps, None

    # --- bigram（char32_t×2）---
    m = re.fullmatch(r'bi_p1_s=(.)(.)', key)
    if m:
        return FT.BIGRAM_PREV1_SELF, [], [ord(m.group(1)), ord(m.group(2))], None
    m = re.fullmatch(r'bi_p2_p1=(.)(.)', key)
    if m:
        return FT.BIGRAM_PREV2_PREV1, [], [ord(m.group(1)), ord(m.group(2))], None
    m = re.fullmatch(r'bi_s_n1=(.)(.)', key)
    if m:
        return FT.BIGRAM_SELF_NEXT1, [], [ord(m.group(1)), ord(m.group(2))], None
    m = re.fullmatch(r'bi_n1_n2=(.)(.)', key)
    if m:
        return FT.BIGRAM_NEXT1_NEXT2, [], [ord(m.group(1)), ord(m.group(2))], None
    m = re.fullmatch(r'bi_p3_p2=(.)(.)', key)
    if m:
        return FT.BIGRAM_PREV3_PREV2, [], [ord(m.group(1)), ord(m.group(2))], None
    m = re.fullmatch(r'bi_n2_n3=(.)(.)', key)
    if m:
        return FT.BIGRAM_NEXT2_NEXT3, [], [ord(m.group(1)), ord(m.group(2))], None

    # --- char_s（複合ユニット対応: 1〜3文字）---
    m = re.fullmatch(r'char_s=(.{3})', key)
    if m:
        return FT.CHAR_SELF_COMPOUND_3, [], [ord(c) for c in m.group(1)], None
    m = re.fullmatch(r'char_s=(.{2})', key)
    if m:
        return FT.CHAR_SELF_COMPOUND_2, [], [ord(c) for c in m.group(1)], None
    m = re.fullmatch(r'char_s=(.)', key)
    if m:
        return FT.CHAR_SELF, [], [ord(m.group(1))], None

    # --- char 系（char32_t×1）---
    m = re.fullmatch(r'char_p1=(.)', key)
    if m:
        return FT.CHAR_PREV1, [], [ord(m.group(1))], None
    m = re.fullmatch(r'char_p2=(.)', key)
    if m:
        return FT.CHAR_PREV2, [], [ord(m.group(1))], None
    m = re.fullmatch(r'char_n1=(.)', key)
    if m:
        return FT.CHAR_NEXT1, [], [ord(m.group(1))], None
    m = re.fullmatch(r'char_n2=(.)', key)
    if m:
        return FT.CHAR_NEXT2, [], [ord(m.group(1))], None
    m = re.fullmatch(r'char_p3=(.)', key)
    if m:
        return FT.CHAR_PREV3, [], [ord(m.group(1))], None
    m = re.fullmatch(r'char_n3=(.)', key)
    if m:
        return FT.CHAR_NEXT3, [], [ord(m.group(1))], None

    raise ValueError(f"未知の特徴量キー: {key!r}")


# =====================================================================
# int8 量子化
# =====================================================================

def quantize_to_int8(
    arr: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    float32 配列を int8 に量子化する。
    スケール係数は |最大値| / 127.0 。
    戻り値: (scale, int8_array)
    推論時のスコア: int8_val * scale ≈ 元の float32 値
    """
    max_abs = np.max(np.abs(arr))
    if max_abs == 0.0:
        return 1.0, np.zeros_like(arr, dtype=np.int8)
    scale = float(max_abs / 127.0)
    quantized = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return scale, quantized


def quantize_csr_per_row_to_int8(
    csr,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CSR 行列を行 (クラス) ごとに独立した scale で int8 量子化する。

    全体で 1 つの scale を使う `quantize_to_int8` と違い、行ごとに
    |最大値| / 127.0 を計算するため、重みの大きさがクラス間で偏っていても
    各クラスがほぼフルの int8 分解能を使える。

    戻り値: (scales[n_classes], int8_data[n_nonzero])
    推論時のスコア: data[j] * scales[row(j)] ≈ 元の float32 値
    """
    n_rows = csr.shape[0]
    data_f32 = csr.data.astype(np.float32)
    quantized = np.zeros_like(data_f32, dtype=np.int8)
    scales = np.zeros(n_rows, dtype=np.float32)
    for row in range(n_rows):
        start, end = csr.indptr[row], csr.indptr[row + 1]
        if start == end:
            scales[row] = 1.0
            continue
        scale, row_int8 = quantize_to_int8(data_f32[start:end])
        scales[row] = scale
        quantized[start:end] = row_int8
    return scales, quantized


# =====================================================================
# エクスポート本体
# =====================================================================

def export(zip_path: str, out_path: str) -> None:
    """
    momo の .zip モデルを C++ 向けバイナリ (.momo) に変換して書き出す。
    """
    # --- モデルのロード ---
    print(f"📦 モデル読み込み中: {zip_path}")
    tmp_dir = tempfile.mkdtemp()
    name_entries: list = []  # [(表層形, ユニット別読み or None), ...]
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            import json
            version_info = json.loads(zf.read("version_info.json").decode("utf-8"))
            bundle_name = version_info["model_bundle"]
            zf.extract(bundle_name, tmp_dir)
            # 学習時に同梱された人名辞書（なければ空 = n_names 0 で書き出す）
            if NAME_DICT_FILENAME in zf.namelist():
                name_entries = parse_name_dict_text(
                    zf.read(NAME_DICT_FILENAME).decode("utf-8")
                )
            # 学習時に同梱された単一漢字辞書（旧ZIPで無い場合はパッケージ内蔵で代替）
            if SINGLE_KANJI_DICT_FILENAME in zf.namelist():
                single_kanji_text = zf.read(SINGLE_KANJI_DICT_FILENAME).decode("utf-8")
            else:
                from importlib import resources
                single_kanji_text = (
                    resources.files("momo_py")
                    / f"resources/{SINGLE_KANJI_DICT_FILENAME}"
                ).read_text(encoding="utf-8")
                print("⚠️  ZIPに単一漢字辞書が同梱されていないため、パッケージ内蔵の辞書を使用します。")
        bundle = joblib.load(os.path.join(tmp_dir, bundle_name))
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    vocab        = bundle.vectorizer_read.vocabulary_   # {key_str: feature_id}
    coef_sparse  = bundle.coef_read_sparse              # CSR (n_classes × n_features)
    intercept_r  = bundle.intercept_read                # float32 (n_classes,)
    read_classes = bundle.read_classes                  # str array (n_classes,)
    model_b      = bundle.model_boundary                # SGDClassifier

    n_classes  = len(read_classes)
    n_features = len(vocab)

    print(f"   クラス数    : {n_classes}")
    print(f"   特徴量次元数: {n_features}")

    # --- 語彙テーブルをfeature_id順に並べる ---
    # vocab は {key_str: feature_id} なので逆引きして id 順に並べる
    id_to_key = {v: k for k, v in vocab.items()}
    sorted_keys = [id_to_key[i] for i in range(n_features)]

    # --- 語彙テーブルのバイナリ構築 ---
    print("🔨 語彙テーブル変換中...")
    vocab_bytes = bytearray()
    for key in sorted_keys:
        try:
            ft, ct_vals, cp_vals, u8_val = parse_feature_key(key)
        except (ValueError, KeyError) as e:
            # KeyError は type_* 系のペイロードが CharType 名でない場合など。
            # 学習データの列ズレ等が原因なので、キーを添えて原因調査できるようにする。
            raise ValueError(
                f"特徴量キーの解析に失敗しました: {key!r} ({e!r})。"
                "学習TSVの列ズレや不正な文字種が混入していないか確認してください。"
            ) from None
        vocab_bytes.append(ft)
        for ct in ct_vals:
            vocab_bytes.append(ct)
        for cp in cp_vals:
            vocab_bytes += struct.pack('<I', cp)   # uint32 LE
        if u8_val is not None:
            vocab_bytes.append(u8_val)
        vocab_bytes += struct.pack('<I', vocab[key])  # feature_id

    # --- 読みラベルテーブル ---
    print("🔨 読みラベルテーブル変換中...")
    label_bytes = bytearray()
    for label in read_classes:
        encoded = label.encode("utf-8")
        assert len(encoded) <= 255, f"ラベルが長すぎます: {label!r}"
        label_bytes.append(len(encoded))
        label_bytes += encoded

    # --- 読みモデル重み（CSR → int8量子化・クラスごとscale）---
    print("🔨 読みモデル重み量子化中...")
    # coef_sparse は CSR (n_classes × n_features)
    csr = coef_sparse.tocsr()
    scales_r, data_int8 = quantize_csr_per_row_to_int8(csr)

    read_weight_bytes = bytearray()
    read_weight_bytes += struct.pack(f'<{n_classes}f', *scales_r.tolist())
    read_weight_bytes += struct.pack('<I', len(data_int8))
    read_weight_bytes += struct.pack('<' + 'I' * (n_classes + 1), *csr.indptr.tolist())
    read_weight_bytes += struct.pack('<' + 'I' * len(csr.indices), *csr.indices.tolist())
    read_weight_bytes += bytes(data_int8.tobytes())

    # --- 読みモデル intercept ---
    intercept_r_f32 = intercept_r.astype(np.float32)
    intercept_r_bytes = intercept_r_f32.tobytes()   # float32 × n_classes

    # --- 境界モデル重み（int8量子化）---
    print("🔨 境界モデル重み量子化中...")
    # SGDClassifier の coef_ は (1, n_features) または (2, n_features)
    # modified_huber の2値分類では coef_[0] がクラス1の重みベクトル
    b_coef = model_b.coef_.astype(np.float32)
    if b_coef.ndim == 2:
        b_coef = b_coef[0]                          # shape: (n_features,)
    scale_b, b_int8 = quantize_to_int8(b_coef)
    b_intercept = model_b.intercept_.astype(np.float32)  # shape: (2,) or (1,)
    if b_intercept.shape[0] == 1:
        # 2値分類で intercept が1要素のことがある
        b_intercept = np.array([0.0, float(b_intercept[0])], dtype=np.float32)

    boundary_bytes = bytearray()
    boundary_bytes += struct.pack('<f', scale_b)
    boundary_bytes += bytes(b_int8.tobytes())
    boundary_bytes += struct.pack('<ff', b_intercept[0], b_intercept[1])

    # --- 人名辞書テーブル ---
    print(f"🔨 人名辞書テーブル変換中... ({len(name_entries)} エントリ)")
    name_dict_bytes = bytearray()
    name_dict_bytes += struct.pack('<I', len(name_entries))
    for surface, readings in name_entries:
        encoded = surface.encode("utf-8")
        assert len(encoded) <= 255, f"人名が長すぎます: {surface!r}"
        name_dict_bytes.append(len(encoded))
        name_dict_bytes += encoded
        if readings is None:
            name_dict_bytes.append(0)
        else:
            assert len(readings) <= 255
            name_dict_bytes.append(len(readings))
            for reading in readings:
                r_enc = reading.encode("utf-8")
                assert len(r_enc) <= 255, f"読みが長すぎます: {reading!r}"
                name_dict_bytes.append(len(r_enc))
                name_dict_bytes += r_enc

    # --- 単一漢字辞書テーブル ---
    single_kanji_dict = _parse_kanji_dict_tsv(single_kanji_text)
    print(f"🔨 単一漢字辞書テーブル変換中... ({len(single_kanji_dict)} エントリ)")
    kanji_dict_bytes = bytearray()
    kanji_dict_bytes += struct.pack('<I', len(single_kanji_dict))
    for kanji in sorted(single_kanji_dict):
        readings = single_kanji_dict[kanji]
        encoded = kanji.encode("utf-8")
        assert len(encoded) <= 255, f"漢字キーが長すぎます: {kanji!r}"
        kanji_dict_bytes.append(len(encoded))
        kanji_dict_bytes += encoded
        assert len(readings) <= 255, f"読みが多すぎます: {kanji!r}"
        kanji_dict_bytes.append(len(readings))
        for reading in readings:
            r_enc = reading.encode("utf-8")
            assert len(r_enc) <= 255, f"読みが長すぎます: {reading!r}"
            kanji_dict_bytes.append(len(r_enc))
            kanji_dict_bytes += r_enc

    # --- ファイルヘッダ ---
    # version 0x04: 人名辞書テーブルにユニット別読みを追加
    #               （アルファ期間中に単一漢字辞書テーブルも追加、番号据え置き）
    header = struct.pack(
        '<4sBBBBII',
        b'MOMO',          # magic
        0x04,             # version
        0x00, 0x00, 0x00, # reserved
        n_classes,
        n_features,
    )

    # --- 書き出し ---
    print(f"💾 書き出し中: {out_path}")  # 推奨拡張子: .mbm
    with open(out_path, 'wb') as f:
        f.write(header)
        f.write(vocab_bytes)
        f.write(label_bytes)
        f.write(read_weight_bytes)
        f.write(intercept_r_bytes)
        f.write(boundary_bytes)
        f.write(name_dict_bytes)
        f.write(kanji_dict_bytes)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"✅ 完了: {size_mb:.1f} MB")
    print(f"   語彙テーブル  : {len(vocab_bytes):>10,} bytes")
    print(f"   読みラベル    : {len(label_bytes):>10,} bytes")
    print(f"   読みモデル重み: {len(read_weight_bytes):>10,} bytes  (scale: min={scales_r.min():.6f} max={scales_r.max():.6f})")
    print(f"   読み intercept: {len(intercept_r_bytes):>10,} bytes")
    print(f"   境界モデル    : {len(boundary_bytes):>10,} bytes  (scale={scale_b:.6f})")
    print(f"   人名辞書      : {len(name_dict_bytes):>10,} bytes  ({len(name_entries)} エントリ)")
    print(f"   単一漢字辞書  : {len(kanji_dict_bytes):>10,} bytes  ({len(single_kanji_dict)} エントリ)")


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="momo モデル (.zip) を C++ 向けバイナリ (.mbm) に変換する"
    )
    parser.add_argument("zip_path",  help="入力: momo モデル ZIP ファイル")
    parser.add_argument("out_path",  help="出力: バイナリファイル (.momo)")
    args = parser.parse_args()

    export(args.zip_path, args.out_path)
