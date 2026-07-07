#!/usr/bin/env python3
"""
gen_normalize_table.py
=======================
Python 側の入力正規化（momo_py/src/momo_py/utils.py）を真実の源として、
Rust 用の正規化テーブル `table.rs` を生成する。

2種類のテーブルを生成する:
  1. 互換漢字テーブル（1→1変換）: normalize_compat_ideographs() 相当
  2. 展開テーブル（1→N展開）  : normalize_input() の _EXPANSION_RANGES 相当

対象コードポイント範囲は utils.py の _COMPAT_IDEOGRAPH_RANGES /
_EXPANSION_RANGES をファイルパス指定で直接読み込む（モノリポ前提）。
utils.py に範囲を追加したら、このスクリプトを再実行するだけでよい。
"""

import importlib.util
import unicodedata
from pathlib import Path

# ----------------------------------------------------------------------
# 真実の源: momo_py/src/momo_py/utils.py の範囲定義を直接読み込む。
# momo_py パッケージとして import すると __init__.py 経由で numpy/sklearn
# を引き込むため、utils.py 単体（標準ライブラリのみに依存）をロードする。
# ----------------------------------------------------------------------
_UTILS_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "momo-py"
    / "src"
    / "momo_py"
    / "utils.py"
)


def _load_utils():
    spec = importlib.util.spec_from_file_location("_momo_py_utils", _UTILS_PATH)
    assert spec is not None and spec.loader is not None, _UTILS_PATH
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_utils = _load_utils()
# 1→1変換のみのカテゴリ
_COMPAT_IDEOGRAPH_RANGES = _utils._COMPAT_IDEOGRAPH_RANGES
# 1→N展開を含むカテゴリ（カテゴリ名: レンジリスト）
_EXPANSION_RANGES = _utils._EXPANSION_RANGES

# ----------------------------------------------------------------------
# 互換漢字テーブル（1→1）: 対象範囲を走査し、NFKC で畳み込まれるものだけを収集
# ----------------------------------------------------------------------
entries: list[tuple[int, int]] = []
for start, end in _COMPAT_IDEOGRAPH_RANGES:
    for cp in range(start, end + 1):
        c = chr(cp)
        if unicodedata.category(c) == "Cn":
            continue
        nfkc = unicodedata.normalize("NFKC", c)
        if nfkc == c:
            continue
        # 1文字→1文字の畳み込みのみをテーブル化する。
        # (対象4レンジは全エントリが1:1であることを確認済み。
        #  1:1でない変換が混入した場合は展開テーブル側で扱う必要があるため、
        #  ここで検出して失敗させる。)
        if len(nfkc) != 1:
            raise ValueError(f"U+{cp:04X} ({c!r}) は複数文字へ変換される: {nfkc!r}")
        entries.append((cp, ord(nfkc)))

entries.sort(key=lambda x: x[0])

# ----------------------------------------------------------------------
# 展開テーブル（1→N）: NFKC 展開結果を文字列として収集
# ----------------------------------------------------------------------
expand_entries: list[tuple[int, str]] = []
for _category, ranges in _EXPANSION_RANGES.items():
    for start, end in ranges:
        for cp in range(start, end + 1):
            c = chr(cp)
            if unicodedata.category(c) == "Cn":
                continue
            nfkc = unicodedata.normalize("NFKC", c)
            if nfkc == c:
                continue
            expand_entries.append((cp, nfkc))

expand_entries.sort(key=lambda x: x[0])

# ----------------------------------------------------------------------
# Rust ソース出力
# ----------------------------------------------------------------------
OUT_PATH = (
    Path(__file__).parent.parent
    / "crates"
    / "momors-core"
    / "src"
    / "normalize"
    / "table.rs"
)
unicode_version = unicodedata.unidata_version

COLS = 8


def format_block(values: list[str], indent: str = "    ") -> str:
    lines = []
    for i in range(0, len(values), COLS):
        lines.append(indent + ", ".join(values[i : i + COLS]) + ",")
    return "\n".join(lines)


def rust_str_literal(s: str) -> str:
    """Rust の文字列リテラルとしてエスケープする（非ASCIIは \\u{...} 形式）。"""
    out = []
    for c in s:
        if c == '"':
            out.append('\\"')
        elif c == "\\":
            out.append("\\\\")
        elif 0x20 <= ord(c) < 0x7F:
            out.append(c)
        else:
            out.append(f"\\u{{{ord(c):04X}}}")
    return '"' + "".join(out) + '"'


from_strs = [f"0x{cp:06X}" for cp, _ in entries]
to_strs = [f"0x{cp:06X}" for _, cp in entries]
expand_from_strs = [f"0x{cp:06X}" for cp, _ in expand_entries]
expand_to_strs = [rust_str_literal(s) for _, s in expand_entries]
compat_ranges = " / ".join(
    f"U+{lo:04X}-{hi:04X}" for lo, hi in _COMPAT_IDEOGRAPH_RANGES
)
expansion_categories = " / ".join(
    f"{name}({', '.join(f'U+{lo:04X}-{hi:04X}' for lo, hi in ranges)})"
    for name, ranges in _EXPANSION_RANGES.items()
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(
        f"""\
// AUTO-GENERATED — DO NOT EDIT MANUALLY
// Source : tools/gen_normalize_table.py
// Truth  : momo_py/src/momo_py/utils.py (normalize_compat_ideographs / normalize_input)
// Unicode    : {unicode_version} (実行時 Python の unicodedata)
// Ranges     : {compat_ranges}
// Expansion  : {expansion_categories}
// Entry count: {len(entries)} (1:1) + {len(expand_entries)} (1:N)

/// 変換テーブル: コードポイントの昇順（変換前）
#[rustfmt::skip]
static NORMALIZE_FROM_CP: &[u32] = &[
{format_block(from_strs)}
];

/// 変換テーブル: 対応する変換後コードポイント
#[rustfmt::skip]
static NORMALIZE_TO_CP: &[u32] = &[
{format_block(to_strs)}
];

/// 展開テーブル: コードポイントの昇順（変換前）
#[rustfmt::skip]
static EXPAND_FROM_CP: &[u32] = &[
{format_block(expand_from_strs)}
];

/// 展開テーブル: 対応する展開後文字列
#[rustfmt::skip]
static EXPAND_TO: &[&str] = &[
{format_block(expand_to_strs)}
];

/// `cp` が互換漢字系ならNFKC正規化後のコードポイントを返す。
/// 該当しなければ `None`。
///
/// Python の `unicodedata.normalize("NFKC", c)` による畳み込みと等価
/// （対象4レンジ内、1文字→1文字変換のみ）。2分探索で O(log n) アクセス。
pub(crate) fn normalize_lookup(cp: u32) -> Option<u32> {{
    match NORMALIZE_FROM_CP.binary_search(&cp) {{
        Ok(idx) => Some(NORMALIZE_TO_CP[idx]),
        Err(_) => None,
    }}
}}

/// `cp` が展開対象（欧文リガチャ等）ならNFKC展開後の文字列を返す。
/// 該当しなければ `None`。
///
/// Python の `normalize_input()` の 1→N 展開と等価。
pub(crate) fn expand_lookup(cp: u32) -> Option<&'static str> {{
    match EXPAND_FROM_CP.binary_search(&cp) {{
        Ok(idx) => Some(EXPAND_TO[idx]),
        Err(_) => None,
    }}
}}
"""
    )

print(
    f"Generated {len(entries)} (1:1) + {len(expand_entries)} (1:N) entries -> {OUT_PATH}"
)
