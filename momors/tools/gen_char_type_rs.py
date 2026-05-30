#!/usr/bin/env python3
"""
gen_char_type_rs.py
===================
Python 側の get_char_type()（momopy/src/momo_py/utils.py）を真実の源として、
Rust 用の記号文字種テーブル `char_type_table.rs` を生成する。

C++ 版の tools/gen_char_type.py の Rust 版。出力フォーマットだけ差し替えている。
真実の源（Python の get_char_type ロジック）はこのスクリプト内に inline で
複製してある（momopy を依存ライブラリ化していないため）。
"""

import unicodedata
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------
# Python 側の get_char_type() のロジックを inline 複製
# （momopy/src/momo_py/utils.py と同期させること）
# ----------------------------------------------------------------------

_CLOSE_SYMBOLS = frozenset('」』)）】〕}｝〉》"\'')
_OPEN_SYMBOLS  = frozenset('「『(（【〔{｛〈《')
_STOP_SYMBOLS  = frozenset('。！？.!?')  # 注: Python版と完全一致させること
_PAUSE_SYMBOLS = frozenset('、・,')

# CharType enum 値（char_type.hpp / char_type.rs と一致）
SYMBOL       = 0x30
SYMBOL_CLOSE = 0x31
SYMBOL_OPEN  = 0x32
SYMBOL_STOP  = 0x33
SYMBOL_PAUSE = 0x34


def classify_symbol(c: str) -> int | None:
    """記号系（P*/S*）に該当すればそのサブカテゴリ値を返す。それ以外は None。"""
    cat = unicodedata.category(c)
    if not (cat.startswith('P') or cat.startswith('S')):
        return None
    if c in _CLOSE_SYMBOLS: return SYMBOL_CLOSE
    if c in _OPEN_SYMBOLS:  return SYMBOL_OPEN
    if c in _STOP_SYMBOLS:  return SYMBOL_STOP
    if c in _PAUSE_SYMBOLS: return SYMBOL_PAUSE
    return SYMBOL


# ----------------------------------------------------------------------
# BMP 全域走査
# ----------------------------------------------------------------------
entries: list[tuple[int, int]] = []
for cp in range(0x10000):
    c = chr(cp)
    if c.isspace():
        continue
    if c.isdigit() or ('0' <= c <= '9'):
        continue
    sub = classify_symbol(c)
    if sub is not None:
        entries.append((cp, sub))

entries.sort(key=lambda x: x[0])

# ----------------------------------------------------------------------
# Rust ソース出力
# ----------------------------------------------------------------------
OUT_PATH = Path(__file__).parent / "char_type_table.rs"
timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

COLS = 12
def format_block(values: list[str], indent: str = '    ') -> str:
    lines = []
    for i in range(0, len(values), COLS):
        lines.append(indent + ', '.join(values[i:i + COLS]) + ',')
    return '\n'.join(lines)

cp_strs = [f'0x{cp:04X}' for cp, _ in entries]
ct_strs = [f'0x{sub:02X}' for _, sub in entries]

with open(OUT_PATH, 'w', encoding='utf-8', newline='\n') as f:
    f.write(f"""\
// AUTO-GENERATED — DO NOT EDIT MANUALLY
// Source : tools/gen_char_type_rs.py
// Truth  : momopy/src/momo_py/utils.py (get_char_type / _*_SYMBOLS sets)
// Generated  : {timestamp}
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

print(f'Generated {len(entries)} entries → {OUT_PATH}')
