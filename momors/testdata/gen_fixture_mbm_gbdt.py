#!/usr/bin/env python3
"""
gen_fixture_mbm_gbdt.py
========================
GBDT境界モデル（algo_tag=0x01、木のアンサンブル）を持つ `.mbm`/`.mbmf` テスト用
フィクスチャを生成する。`gen_fixture_mbm.py` と同じ読みモデル（語彙・ラベル・重み・
人名辞書・単一漢字辞書）を再利用し、境界モデルセクションだけを木のアンサンブルに
差し替える。

境界モデルのカテゴリカル列:
  列0: type_s   値 KANJI->コード0, HIRAGANA->コード1
  列1: char_p1  値 漢(U+6F22)->コード0

木（2本）:
  木0: 列0を分岐。コード{0}(=KANJI)なら左(leaf=0.5)、それ以外/欠損なら右(leaf=-0.5)
       （default_left=False）
  木1: 定数の葉（leaf=0.25、分岐なし）

期待されるスコア（全木のleaf値の合計）:
  type_s=KANJI のみ    : 0.5 + 0.25 = 0.75
  type_s=HIRAGANA のみ : -0.5 + 0.25 = -0.25
  type_s キーなし（欠損）: -0.5 + 0.25 = -0.25 （default_left=False）
"""

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gen_fixture_mbm as base  # noqa: E402

MAGIC_MBM = b'MOMO'
MAGIC_MBMF = b'MBMF'
VERSION = base.VERSION

BOUNDARY_ALGO_TREE = 0x01

FT_TYPE_SELF = 0x50
FT_CHAR_PREV1 = 0x91
CT_KANJI = 0x42
CT_HIRAGANA = 0x40

OUT_MBM = Path(__file__).parent.parent / "testdata" / "fixture_gbdt.mbm"
OUT_MBMF = Path(__file__).parent.parent / "testdata" / "fixture_gbdt.mbmf"


def build_cat_vocab() -> bytes:
    # (feature_type, ct_vals, cp_vals, u8_val, column_index, code)
    entries = [
        (FT_TYPE_SELF, [CT_KANJI], [], None, 0, 0),
        (FT_TYPE_SELF, [CT_HIRAGANA], [], None, 0, 1),
        (FT_CHAR_PREV1, [], [0x6F22], None, 1, 0),
    ]
    buf = bytearray()
    buf += struct.pack('<II', 2, len(entries))  # n_cat_columns, n_cat_vocab_entries
    for ft, ct_vals, cp_vals, u8_val, col, code in entries:
        buf.append(ft)
        for ct in ct_vals:
            buf.append(ct)
        for cp in cp_vals:
            buf += struct.pack('<I', cp)
        if u8_val is not None:
            buf.append(u8_val)
        buf += struct.pack('<II', col, code)
    return bytes(buf)


def build_leaf(value: float) -> bytes:
    return struct.pack('<Bf', 0, value)


def build_split(column: int, default_left: bool, cats: list, left: bytes, right: bytes) -> bytes:
    buf = bytearray()
    buf.append(1)  # node_tag: split
    buf += struct.pack('<I', column)
    buf.append(1 if default_left else 0)
    buf += struct.pack('<I', len(cats))
    for c in cats:
        buf += struct.pack('<I', c)
    buf += left
    buf += right
    return bytes(buf)


def build_trees() -> bytes:
    tree0 = build_split(0, False, [0], build_leaf(0.5), build_leaf(-0.5))
    tree1 = build_leaf(0.25)
    buf = bytearray()
    buf += struct.pack('<I', 2)  # n_trees
    buf += tree0
    buf += tree1
    return bytes(buf)


def build_boundary_tree() -> bytes:
    buf = bytearray()
    buf.append(BOUNDARY_ALGO_TREE)
    buf += build_cat_vocab()
    buf += build_trees()
    return bytes(buf)


def build_header(magic: bytes) -> bytes:
    return struct.pack(
        '<4sBBBBII',
        magic, VERSION, 0x00, 0x00, 0x00,
        base.N_CLASSES, base.N_FEATURES,
    )


def main() -> None:
    boundary = build_boundary_tree()

    mbm_parts = {
        'header': build_header(MAGIC_MBM),
        'vocab': base.build_vocab(),
        'labels': base.build_labels(),
        'read_weights': base.build_read_weights(),
        'intercept_r': base.build_intercept_read(),
        'boundary': boundary,
        'name_dict': base.build_name_dict(),
        'kanji_dict': base.build_kanji_dict(),
    }
    OUT_MBM.write_bytes(b''.join(mbm_parts.values()))
    print(f'Generated: {OUT_MBM}')

    # .mbmf は読みモデル重みだけ float32・量子化なしにする。境界モデル（木）は
    # 量子化しないため .mbm と完全に同一バイト列（boundary をそのまま再利用する）。
    import gen_fixture_mbmf as basef  # noqa: E402

    mbmf_parts = {
        'header': build_header(MAGIC_MBMF),
        'vocab': base.build_vocab(),
        'labels': base.build_labels(),
        'read_weights': basef.build_read_weights_float(),
        'intercept_r': base.build_intercept_read(),
        'boundary': boundary,
        'name_dict': base.build_name_dict(),
        'kanji_dict': base.build_kanji_dict(),
    }
    OUT_MBMF.write_bytes(b''.join(mbmf_parts.values()))
    print(f'Generated: {OUT_MBMF}')


if __name__ == '__main__':
    main()
