#!/usr/bin/env python3
"""
gen_fixture_mbmf.py
====================
float_loader のテスト用 `.mbmf` ファイルを生成する。

`gen_fixture_mbm.py` と同じ語彙・ラベル・人名辞書・単一漢字辞書を共有し、
読みモデル重み・境界モデル重みだけを「量子化前の float32 実値」
（`gen_fixture_mbm.py` の `int8_val * scale` を機械的に計算したもの）に
差し替える。これにより `MomoModel`（`.mbm`）と `FloatMomoModel`（`.mbmf`）が
同一フィクスチャに対して厳密に同じスコアを出すことをテストで断言できる。

バイナリフォーマットは `momo_py/exporter.py` の `export_float()` と同じ
（`momors-core/src/float_loader.rs` のドキュメントコメントも参照）。
"""

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gen_fixture_mbm as base  # noqa: E402

MAGIC = b'MBMF'
VERSION = 0x01

OUT_PATH = Path(__file__).parent.parent / "testdata" / "fixture.mbmf"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_header() -> bytes:
    return struct.pack(
        '<4sBBBBII',
        MAGIC, VERSION, 0x00, 0x00, 0x00,
        base.N_CLASSES, base.N_FEATURES,
    )


def build_read_weights_float() -> bytes:
    """CSR フォーマット（量子化なし）: n_nonzero + indptr + indices + data(f32)

    値は `gen_fixture_mbm.py` の CSR_ROWS（int8）を対応する
    QUANT_SCALES_READ（クラスごと）で dequantize した実値そのもの。
    """
    indptr = [0]
    indices = []
    data = []
    for row_idx, row in enumerate(base.CSR_ROWS):
        scale = base.QUANT_SCALES_READ[row_idx]
        for col, int8_val in row:
            indices.append(col)
            data.append(int8_val * scale)
        indptr.append(len(indices))

    n_nonzero = len(data)
    buf = bytearray()
    buf += struct.pack('<I', n_nonzero)
    buf += struct.pack(f'<{len(indptr)}I', *indptr)
    buf += struct.pack(f'<{n_nonzero}I', *indices)
    buf += struct.pack(f'<{n_nonzero}f', *data)
    return bytes(buf)


def build_boundary_float() -> bytes:
    """境界モデル重み（量子化なし）: data(f32 × n_features) + intercept(f32 × 2)"""
    data = [v * base.QUANT_SCALE_BOUNDARY for v in base.BOUNDARY_DATA]
    buf = bytearray()
    buf += struct.pack(f'<{base.N_FEATURES}f', *data)
    buf += struct.pack('<ff', *base.BOUNDARY_INTERCEPT)
    return bytes(buf)


def main() -> None:
    parts = {
        'header'        : build_header(),
        'vocab'         : base.build_vocab(),
        'labels'        : base.build_labels(),
        'read_weights'  : build_read_weights_float(),
        'intercept_r'   : base.build_intercept_read(),
        'boundary'      : build_boundary_float(),
        'name_dict'     : base.build_name_dict(),
        'kanji_dict'    : base.build_kanji_dict(),
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
