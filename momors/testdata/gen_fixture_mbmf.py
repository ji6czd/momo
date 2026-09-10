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
# バージョン番号は `.mbm` と共有する（magic だけで区別する）
VERSION = base.VERSION

OUT_PATH = Path(__file__).parent.parent / "testdata" / "fixture.mbmf"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_header(flags: int = 0x00) -> bytes:
    return struct.pack(
        '<4sBBBBII',
        MAGIC, VERSION, flags, 0x00, 0x00,
        base.N_CLASSES, base.N_FEATURES,
    )


def build_read_weights_float() -> bytes:
    """CSC フォーマット（量子化なし・version 0x09）:
    n_nonzero + col_len(uint16 × n_features) + rowind + data(f32)

    値は `gen_fixture_mbm.py` の CSR_ROWS（int8）を対応する
    QUANT_SCALES_READ（クラスごと）で dequantize した実値そのもの。
    転置と列の並べ替えは `.mbm` と同じ `base.to_csc()` を使うので、両フィクスチャの
    col_len / rowind は必ず一致する。
    """
    dequantized = [
        [(col, int8_val * base.QUANT_SCALES_READ[row_idx]) for col, int8_val in row]
        for row_idx, row in enumerate(base.CSR_ROWS)
    ]
    col_len, rowind, data = base.to_csc(dequantized)

    n_nonzero = len(data)
    buf = bytearray()
    buf += struct.pack('<I', n_nonzero)
    buf += struct.pack(f'<{base.N_FEATURES}H', *col_len)
    buf += struct.pack(f'<{n_nonzero}H', *rowind)
    buf += struct.pack(f'<{n_nonzero}f', *data)
    return bytes(buf)


def build_boundary_float() -> bytes:
    """境界モデル（algo_tag=線形、量子化なし）: data(f32 × n_features) + intercept(f32 × 2)"""
    # 線形境界の重みは feature_id で引くので、CSC の列と同じ順に並べ替える。
    data = [
        base.BOUNDARY_DATA[c] * base.QUANT_SCALE_BOUNDARY for c in base.csc_column_order()
    ]
    buf = bytearray()
    buf.append(base.BOUNDARY_ALGO_LINEAR)
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
        'single_char_dict': base.build_single_char_dict(),
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
