#!/usr/bin/env python3
"""
gen_fixture_mbm_gbdt.py
========================
GBDT境界モデル（algo_tag=0x01、木のアンサンブル）を持つ `.mbm`/`.mbmf` テスト用
フィクスチャを生成する。`gen_fixture_mbm.py` と同じ読みモデル（語彙・ラベル・重み・
人名辞書・単一漢字辞書）を再利用し、境界モデルセクションだけを木のアンサンブルに
差し替える。

version 0x09（タイプ別セクション）:
  カテゴリカル列は統合語彙のセクションヘッダが持ち、コードは
  `cat_code_base + セクション内の添字` で決まる（ヘッダ flags bit0=1）。
  境界モデルセクションは n_columns + 木だけを持つ。**カテゴリカルキーは
  読み語彙の部分集合**という本番の不変条件に合わせ、読み語彙にあるキー
  （char_s=漢 / char_s=字）だけを使う。

境界モデルのカテゴリカル列:
  列0: char_s。コードはキー順（コードポイント順）に振られるので
       字(U+5B57)->0、漢(U+6F22)->1 になる。ここは並び順で決まるため、
       木が参照するコードもハードコードせず `base.cat_code_of()` から引く。

木（2本）:
  木0: 列0を分岐。漢のコードなら左(leaf=0.5)、それ以外/欠損なら右(leaf=-0.5)
       （default_left=False）
  木1: 定数の葉（leaf=0.25、分岐なし）

期待されるスコア（全木のleaf値の合計）:
  char_s=漢 のみ    : 0.5 + 0.25 = 0.75
  char_s=字 のみ    : -0.5 + 0.25 = -0.25 （字のコードは分岐集合に含まれない）
  char_s キーなし（欠損）: -0.5 + 0.25 = -0.25 （default_left=False）
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

# カテゴリカル列の割り当て: feature_type -> column。
# version 0x09 ではコードはセクション内の並び順で決まるので、ここでは列だけ決める。
CAT_COLUMNS = {base.FT_CHAR_SELF: 0}
N_CAT_COLUMNS = 1

# 木が「漢」を指すためのコード。並び順で決まるので base から引く
# （base.VOCAB の feature_id 1 が char_s=漢）。
KANJI_CODE = base.cat_code_of(CAT_COLUMNS)[1][1]


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
    tree0 = build_split(0, False, [KANJI_CODE], build_leaf(0.5), build_leaf(-0.5))
    tree1 = build_leaf(0.25)
    buf = bytearray()
    buf += struct.pack('<I', 2)  # n_trees
    buf += tree0
    buf += tree1
    return bytes(buf)


def build_boundary_tree() -> bytes:
    """version 0x07: algo_tag + n_columns + n_trees + 木（cat_vocab は持たない）。"""
    buf = bytearray()
    buf.append(BOUNDARY_ALGO_TREE)
    buf += struct.pack('<I', N_CAT_COLUMNS)
    buf += build_trees()
    return bytes(buf)


def build_header(magic: bytes) -> bytes:
    # flags bit0 = 統合語彙が GBDT カテゴリカルを持つ。
    return struct.pack(
        '<4sBBBBII',
        magic, VERSION, base.FLAG_VOCAB_HAS_CAT, 0x00, 0x00,
        base.N_CLASSES, base.N_FEATURES,
    )


def main() -> None:
    boundary = build_boundary_tree()
    # 統合語彙（flags=1 → column/code 付き）。読みモデルの重み等は base を再利用。
    vocab = base.build_vocab(cat_columns=CAT_COLUMNS)

    mbm_parts = {
        'header': build_header(MAGIC_MBM),
        'vocab': vocab,
        'labels': base.build_labels(),
        'read_weights': base.build_read_weights(),
        'intercept_r': base.build_intercept_read(),
        'boundary': boundary,
        'name_dict': base.build_name_dict(),
        'single_char_dict': base.build_single_char_dict(),
    }
    out_mbm = Path(__file__).parent.parent / "testdata" / "fixture_gbdt.mbm"
    out_mbm.write_bytes(b''.join(mbm_parts.values()))
    print(f'Generated: {out_mbm}')

    # .mbmf は読みモデル重みだけ float32・量子化なしにする。境界モデル（木）は
    # 量子化しないため .mbm と完全に同一バイト列（boundary をそのまま再利用する）。
    import gen_fixture_mbmf as basef  # noqa: E402

    mbmf_parts = {
        'header': build_header(MAGIC_MBMF),
        'vocab': vocab,
        'labels': base.build_labels(),
        'read_weights': basef.build_read_weights_float(),
        'intercept_r': base.build_intercept_read(),
        'boundary': boundary,
        'name_dict': base.build_name_dict(),
        'single_char_dict': base.build_single_char_dict(),
    }
    out_mbmf = Path(__file__).parent.parent / "testdata" / "fixture_gbdt.mbmf"
    out_mbmf.write_bytes(b''.join(mbmf_parts.values()))
    print(f'Generated: {out_mbmf}')


if __name__ == '__main__':
    main()
