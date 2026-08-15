#!/usr/bin/env python3
"""
gen_fixture_mbm.py
================
loader のテスト用 `.mbm` ファイルを生成する。

scikit-learn は使わず、ハードコードされた小さなモデルを `.mbm` バイナリで
書き出す。バイナリフォーマットは `momopy/src/momo_py/exporter.py` と同じ。

生成されるモデル:
  - 3 クラス: ["カ", "キ", "ク"]
  - 5 特徴量:
      0: bias                       (FT.BIAS)
      1: char_s=漢                  (FT.CHAR_SELF, cp=0x6F22)
      2: char_s=字                  (FT.CHAR_SELF, cp=0x5B57)
      3: type_s=KANJI               (FT.TYPE_SELF, ct=0x42)
      4: kanji_run=2                (FT.KANJI_RUN_LEN, u8=2)
  - 重みは決定的なテストデータ
"""

import struct
from pathlib import Path

# ----------------------------------------------------------------------
# 定数
# ----------------------------------------------------------------------
# FeatureType の値（feature.rs / feature_type.hpp と同期）
FT_BIAS          = 0x00
FT_TYPE_SELF     = 0x50
FT_CHAR_SELF     = 0x90
FT_KANJI_RUN_LEN = 0xC0

# CharType の値
CT_KANJI = 0x42

# ファイル識別情報（exporter.py の MAGIC_MBM / VERSION と同期）
MAGIC = b'MOMO'
# バージョンは `.mbmf`・GBDT フィクスチャと共有する（それぞれ base.VERSION を使う）
VERSION = 0x07

# ヘッダ flags（version 0x07）。bit0 = 統合語彙が GBDT カテゴリカルを持つ。
# このフィクスチャは線形境界なので flags=0（vocab に column/code を書かない）。
FLAG_VOCAB_HAS_CAT = 0x01

# カテゴリカル列なしの番兵（loader.rs の NO_CAT_COLUMN と一致）
NO_CAT_COLUMN = 0xFFFFFFFF

# 境界モデルの algo_tag（このフィクスチャは線形のみ）
BOUNDARY_ALGO_LINEAR = 0x00

# 出力先
OUT_PATH = Path(__file__).parent.parent / "testdata" / "fixture.mbm"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# モデル定義
# ----------------------------------------------------------------------
N_CLASSES = 3
N_FEATURES = 5
READ_CLASSES = ["カ", "キ", "ク"]

# 語彙テーブル: 各エントリ = (feature_type, [ct_vals], [cp_vals], u8_val, feature_id)
VOCAB = [
    (FT_BIAS,          [],         [],                None, 0),
    (FT_CHAR_SELF,     [],         [0x6F22],          None, 1),  # 漢
    (FT_CHAR_SELF,     [],         [0x5B57],          None, 2),  # 字
    (FT_TYPE_SELF,     [CT_KANJI], [],                None, 3),
    (FT_KANJI_RUN_LEN, [],         [],                2,    4),
]

# 読みモデル重み (n_classes × n_features)
# 定義はクラス (行) ごとに書き、書き出し時に `to_csc()` で CSC に転置する
# クラス0=カ: feature 0,1,3 にスコア
# クラス1=キ: feature 0,2,3 にスコア
# クラス2=ク: feature 0,4 にスコア
# 値は int8 量子化済みとして直接書く
# version 0x02 以降、量子化scaleはクラス(行)ごと。
# あえて 3 クラスとも異なる値にして per-row scale が正しく
# インデックスされることをテストで検証できるようにする。
QUANT_SCALES_READ = [0.01, 0.02, 0.005]  # 推論時の実値 = data * QUANT_SCALES_READ[class_id]

# CSR: 行 (クラス) ごとの非ゼロエントリ
CSR_ROWS = [
    # (col_idx, int8_val)
    [(0, 50),  (1, 80),  (3, 30)],    # カ
    [(0, 40),  (2, 70),  (3, 20)],    # キ
    [(0, 10),  (4, 90)],              # ク
]

INTERCEPT_READ = [0.1, 0.05, -0.05]  # f32 × n_classes

# 境界モデル
QUANT_SCALE_BOUNDARY = 0.005
BOUNDARY_DATA = [10, -5, 20, 15, -3]  # int8 × n_features
BOUNDARY_INTERCEPT = [0.2, -0.2]      # f32 × 2

# 人名辞書 (version 0x03 で追加、0x04 でユニット別読みを追加)
# (表層形, ユニット別読みリスト or None)
NAME_DICT = [
    ("佐藤", ["サ", "トー"]),
    ("太郎", None),
]

# 単一文字辞書 (version 0x04 の途中で追加)
# (文字, 読みリスト)
SINGLE_CHAR_DICT = [
    ("漢", ["カン"]),
    ("字", ["ジ", "アザ"]),
]


# ----------------------------------------------------------------------
# バイナリ構築
# ----------------------------------------------------------------------
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


def build_header(flags: int = 0x00) -> bytes:
    return struct.pack(
        '<4sBBBBII',
        MAGIC, VERSION, flags, 0x00, 0x00,
        N_CLASSES, N_FEATURES,
    )


def build_vocab(cat_map: dict | None = None) -> bytes:
    """統合語彙テーブル（version 0x07）を組む。

    feature_id はファイル内の並び順（VOCAB の順）で暗黙に決まるため書き出さない。
    VOCAB は feature_id 昇順で定義してある前提。

    `cat_map` を渡すと（GBDT フィクスチャ用、flags bit0=1）、各エントリ末尾に
    `(column, code)` を書き出す。`cat_map[feature_id]` が無いキーは番兵で埋める。
    None のとき（線形フィクスチャ、flags=0）は書き出さない。
    """
    buf = bytearray()
    for fid, (ft, ct_vals, cp_vals, u8_val, defined_fid) in enumerate(VOCAB):
        assert defined_fid == fid, f"VOCAB は feature_id 昇順で定義すること: {defined_fid} != {fid}"
        # 整合性チェック
        assert len(ct_vals) == chartype_count(ft), f"FT {ft:#x} ct count"
        assert len(cp_vals) == char32_count(ft), f"FT {ft:#x} cp count"
        assert (u8_val is not None) == is_uint8_payload(ft), f"FT {ft:#x} u8"

        buf.append(ft)
        for ct in ct_vals:
            buf.append(ct)
        for cp in cp_vals:
            buf += struct.pack('<I', cp)
        if u8_val is not None:
            buf.append(u8_val)
        if cat_map is not None:
            column, code = cat_map.get(fid, (NO_CAT_COLUMN, 0))
            buf += struct.pack('<II', column, code)
    return bytes(buf)


def build_labels() -> bytes:
    buf = bytearray()
    for label in READ_CLASSES:
        encoded = label.encode('utf-8')
        assert len(encoded) <= 255
        buf.append(len(encoded))
        buf += encoded
    return bytes(buf)


def to_csc(rows: list) -> tuple:
    """行 (クラス) ごとの非ゼロエントリを CSC の (colptr, rowind, data) に転置する。

    `CSR_ROWS` はクラスごとに書いたほうが読みやすいのでその形で定義してあるが、
    ファイルフォーマット (version 0x05 以降) は CSC なのでここで転置する。
    列ごとに行インデックス昇順で並べる（scipy の `tocsc()` と同じ並び）。
    """
    colptr = [0]
    rowind = []
    data = []
    for col in range(N_FEATURES):
        for row_idx, row in enumerate(rows):
            for c, val in row:
                if c == col:
                    rowind.append(row_idx)
                    data.append(val)
        colptr.append(len(rowind))
    return colptr, rowind, data


def build_read_weights() -> bytes:
    """CSC フォーマット: quant_scale[n_classes] + n_nonzero + colptr + rowind + data"""
    colptr, rowind, data = to_csc(CSR_ROWS)

    n_nonzero = len(data)
    buf = bytearray()
    buf += struct.pack(f'<{N_CLASSES}f', *QUANT_SCALES_READ)
    buf += struct.pack('<I', n_nonzero)
    buf += struct.pack(f'<{len(colptr)}I', *colptr)
    buf += struct.pack(f'<{n_nonzero}H', *rowind)
    buf += struct.pack(f'<{n_nonzero}b', *data)
    return bytes(buf)


def build_intercept_read() -> bytes:
    return struct.pack(f'<{N_CLASSES}f', *INTERCEPT_READ)


def build_boundary() -> bytes:
    buf = bytearray()
    buf.append(BOUNDARY_ALGO_LINEAR)
    buf += struct.pack('<f', QUANT_SCALE_BOUNDARY)
    buf += struct.pack(f'<{N_FEATURES}b', *BOUNDARY_DATA)
    buf += struct.pack('<ff', *BOUNDARY_INTERCEPT)
    return bytes(buf)


def build_name_dict() -> bytes:
    """人名辞書テーブル:
    n_names(u32) + [len(u8) + utf8表層形 + n_readings(u8) + [len(u8) + utf8読み]*]*
    """
    buf = bytearray()
    buf += struct.pack('<I', len(NAME_DICT))
    for surface, readings in NAME_DICT:
        encoded = surface.encode('utf-8')
        assert len(encoded) <= 255
        buf.append(len(encoded))
        buf += encoded
        if readings is None:
            buf.append(0)
        else:
            buf.append(len(readings))
            for reading in readings:
                r_enc = reading.encode('utf-8')
                assert len(r_enc) <= 255
                buf.append(len(r_enc))
                buf += r_enc
    return bytes(buf)


def build_single_char_dict() -> bytes:
    """単一文字辞書テーブル:
    n_entries(u32) + [len(u8) + utf8文字 + n_readings(u8) + [len(u8) + utf8読み]*]*
    """
    buf = bytearray()
    buf += struct.pack('<I', len(SINGLE_CHAR_DICT))
    for ch, readings in SINGLE_CHAR_DICT:
        encoded = ch.encode('utf-8')
        buf.append(len(encoded))
        buf += encoded
        buf.append(len(readings))
        for reading in readings:
            r_enc = reading.encode('utf-8')
            buf.append(len(r_enc))
            buf += r_enc
    return bytes(buf)


# ----------------------------------------------------------------------
# 書き出し
# ----------------------------------------------------------------------
def main() -> None:
    parts = {
        'header'        : build_header(),
        'vocab'         : build_vocab(),
        'labels'        : build_labels(),
        'read_weights'  : build_read_weights(),
        'intercept_r'   : build_intercept_read(),
        'boundary'      : build_boundary(),
        'name_dict'     : build_name_dict(),
        'single_char_dict': build_single_char_dict(),
    }

    blob = b''.join(parts.values())
    OUT_PATH.write_bytes(blob)

    print(f'Generated: {OUT_PATH}')
    print(f'Total size: {len(blob)} bytes')
    print()
    print('Section sizes:')
    for name, data in parts.items():
        print(f'  {name:<14}: {len(data):>4} bytes')


if __name__ == '__main__':
    main()
