#!/usr/bin/env python3
"""
gen_char_type_rs.py
===================
Python 側の get_char_type()（momo-py/src/momo_py/utils.py）を真実の源として、
Rust 用の記号文字種テーブル `crates/momors-core/src/char_type/table.rs` を
生成する。

utils.py をファイルパス指定で直接読み込み、get_char_type() が SYMBOL 系を
返すコードポイントをそのままテーブル化する（モノリポ前提）。
utils.py の記号セットや判定ロジックを変更したら、このスクリプトを
再実行するだけでよい。

注意: テーブル内容は実行する Python の unicodedata（Unicodeバージョン）に
依存する。Python/Rust の挙動一致のため、momo-py が使うのと同じ Python で
実行すること（例: `cd momo-py && uv run python ../momors/tools/gen_char_type_rs.py`）。
"""

import importlib.util
import unicodedata
from pathlib import Path

# ----------------------------------------------------------------------
# 真実の源: momo_py/src/momo_py/utils.py の get_char_type() を直接読み込む。
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

# CharType enum の u8 値（char_type.rs と一致させること）
_SYMBOL_CT_VALUES = {
    _utils.CharType.SYMBOL: 0x30,
    _utils.CharType.SYMBOL_CLOSE: 0x31,
    _utils.CharType.SYMBOL_OPEN: 0x32,
    _utils.CharType.SYMBOL_STOP: 0x33,
    _utils.CharType.SYMBOL_PAUSE: 0x34,
}

# ----------------------------------------------------------------------
# BMP 全域走査: get_char_type() が SYMBOL 系を返すものだけをテーブル化
# ----------------------------------------------------------------------
entries: list[tuple[int, int]] = []
for cp in range(0x10000):
    ct = _utils.get_char_type(chr(cp))
    sub = _SYMBOL_CT_VALUES.get(ct)
    if sub is not None:
        entries.append((cp, sub))

entries.sort(key=lambda x: x[0])

# ----------------------------------------------------------------------
# Rust ソース出力
# ----------------------------------------------------------------------
OUT_PATH = (
    Path(__file__).parent.parent
    / "crates"
    / "momors-core"
    / "src"
    / "char_type"
    / "table.rs"
)
unicode_version = unicodedata.unidata_version

COLS = 12


def format_block(values: list[str], indent: str = "    ") -> str:
    lines = []
    for i in range(0, len(values), COLS):
        lines.append(indent + ", ".join(values[i : i + COLS]) + ",")
    return "\n".join(lines)


cp_strs = [f"0x{cp:04X}" for cp, _ in entries]
ct_strs = [f"0x{sub:02X}" for _, sub in entries]

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(f"""\
// AUTO-GENERATED — DO NOT EDIT MANUALLY
// Source : tools/gen_char_type_rs.py
// Truth  : momo-py/src/momo_py/utils.py (get_char_type)
// Unicode    : {unicode_version} (実行時 Python の unicodedata)
// Codepoints : BMP (U+0000–U+FFFF), SYMBOL 系のみ
// Entry count: {len(entries)}

use crate::char_type::CharType;

/// 記号テーブル: コードポイントの昇順
#[rustfmt::skip]
static SYMBOL_CP: &[u16] = &[
{format_block(cp_strs)}
];

/// 記号テーブル: 対応する CharType の u8 値
#[rustfmt::skip]
static SYMBOL_CT: &[u8] = &[
{format_block(ct_strs)}
];

/// `cp` が SYMBOL 系ならその CharType を返す。それ以外は CharType::Other。
///
/// Python の `unicodedata.category()` による P*/S* 判定と等価。
/// 2 分探索で O(log n) アクセス。
pub(crate) fn symbol_type_lookup(c: char) -> CharType {{
    let cp = c as u32;
    if cp > 0xFFFF {{
        return CharType::Other;
    }}
    let key = cp as u16;

    match SYMBOL_CP.binary_search(&key) {{
        Ok(idx) => {{
            // u8 → CharType: 値が SYMBOL系のいずれかであることが保証されているので unsafe で
            // 変換することもできるが、安全のため明示的な match を使う。
            match SYMBOL_CT[idx] {{
                0x30 => CharType::Symbol,
                0x31 => CharType::SymbolClose,
                0x32 => CharType::SymbolOpen,
                0x33 => CharType::SymbolStop,
                0x34 => CharType::SymbolPause,
                _ => CharType::Other, // 到達不能だが念のため
            }}
        }}
        Err(_) => CharType::Other,
    }}
}}
""")

print(f"Generated {len(entries)} entries -> {OUT_PATH}")
