#!/usr/bin/env python3
"""
gen_dummy_mbm.py
================
loader のテスト用ダミー `.mbm` ファイルを生成する。

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

# 出力先
OUT_PATH = Path(__file__).parent.parent / "testdata" / "dummy.mbm"
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

# 読みモデル重み (n_classes × n_features) → CSR
# クラス0=カ: feature 0,1,3 にスコア
# クラス1=キ: feature 0,2,3 にスコア
# クラス2=ク: feature 0,4 にスコア
# 値は int8 量子化済みとして直接書く
QUANT_SCALE_READ = 0.01  # 推論時の実値 = data * 0.01

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


def build_header() -> bytes:
    return struct.pack(
        '<4sBBBBII',
        b'MOMO', 0x01, 0x00, 0x00, 0x00,
        N_CLASSES, N_FEATURES,
    )


def build_vocab() -> bytes:
    buf = bytearray()
    for ft, ct_vals, cp_vals, u8_val, fid in VOCAB:
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
        buf += struct.pack('<I', fid)
    return bytes(buf)


def build_labels() -> bytes:
    buf = bytearray()
    for label in READ_CLASSES:
        encoded = label.encode('utf-8')
        assert len(encoded) <= 255
        buf.append(len(encoded))
        buf += encoded
    return bytes(buf)


def build_read_weights() -> bytes:
    """CSR フォーマット: indptr + indices + data"""
    indptr = [0]
    indices = []
    data = []
    for row in CSR_ROWS:
        for col, val in row:
            indices.append(col)
            data.append(val)
        indptr.append(len(indices))

    n_nonzero = len(data)
    buf = bytearray()
    buf += struct.pack('<f', QUANT_SCALE_READ)
    buf += struct.pack('<I', n_nonzero)
    buf += struct.pack(f'<{len(indptr)}I', *indptr)
    buf += struct.pack(f'<{n_nonzero}I', *indices)
    buf += struct.pack(f'<{n_nonzero}b', *data)
    return bytes(buf)


def build_intercept_read() -> bytes:
    return struct.pack(f'<{N_CLASSES}f', *INTERCEPT_READ)


def build_boundary() -> bytes:
    buf = bytearray()
    buf += struct.pack('<f', QUANT_SCALE_BOUNDARY)
    buf += struct.pack(f'<{N_FEATURES}b', *BOUNDARY_DATA)
    buf += struct.pack('<ff', *BOUNDARY_INTERCEPT)
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
