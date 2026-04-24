#!/usr/bin/env python3
"""
tools/gen_char_type.py
======================
Python 側の get_char_type()（momo_py/utils.py）を唯一の真実源として、
C++ 用のシンボル文字種テーブルを生成する。

生成先: ../src/char_type_generated.hpp

使い方:
    cd momo.cxx/tools
    python gen_char_type.py
"""
import sys
import os
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '../../momopy/src'))

from momo_py.utils import get_char_type, CharType

# Python CharType → C++ enum 値（char_type.hpp の値と一致）
_CT_VAL: dict[CharType, int] = {
    CharType.SYMBOL:       0x30,
    CharType.SYMBOL_CLOSE: 0x31,
    CharType.SYMBOL_OPEN:  0x32,
    CharType.SYMBOL_STOP:  0x33,
    CharType.SYMBOL_PAUSE: 0x34,
}
_SYMBOL_TYPES = frozenset(_CT_VAL.keys())

# BMP 全域を走査して SYMBOL 系エントリを収集
entries: list[tuple[int, CharType]] = []
for cp in range(0x10000):
    try:
        ct = get_char_type(chr(cp))
    except Exception:
        continue
    if ct in _SYMBOL_TYPES:
        entries.append((cp, ct))

entries.sort(key=lambda x: x[0])

# 出力ファイル
out_path = os.path.join(_HERE, '../src/char_type_generated.hpp')
timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

COLS = 12  # 1行あたりのエントリ数


def format_block(values: list[str], indent: str = '    ') -> str:
    lines = []
    for i in range(0, len(values), COLS):
        lines.append(indent + ', '.join(values[i:i + COLS]) + ',')
    return '\n'.join(lines)


cp_strs = [f'0x{cp:04X}' for cp, _ in entries]
ct_strs = [f'0x{_CT_VAL[ct]:02X}' for _, ct in entries]

with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(f"""\
// AUTO-GENERATED — DO NOT EDIT MANUALLY
// Source : tools/gen_char_type.py
// Truth  : momopy/src/momo_py/utils.py (get_char_type / _*_SYMBOLS sets)
// Generated : {timestamp}
// Codepoints: BMP (U+0000–U+FFFF), SYMBOL 系のみ
// Entry count: {len(entries)}
//
// 再生成:
//   cd momo.cxx/tools && python gen_char_type.py
#pragma once
#include "char_type.hpp"
#include <cstddef>
#include <cstdint>

// --------------------------------------------------------------------------
// kSymbolCp / kSymbolCt : Python の get_char_type() が SYMBOL 系を返す
//                          BMP コードポイントの昇順テーブル
// --------------------------------------------------------------------------
// clang-format off
static const uint16_t kSymbolCp[] = {{
{format_block(cp_strs)}
}};
static const uint8_t kSymbolCt[] = {{
{format_block(ct_strs)}
}};
// clang-format on
static const std::size_t kSymbolTableSize = {len(entries)};

// --------------------------------------------------------------------------
// symbol_type_lookup
//   cp が SYMBOL 系ならその CharType を返す。それ以外は CharType::OTHER。
//   Python の unicodedata.category() による P*/S* 判定と等価。
// --------------------------------------------------------------------------
inline CharType symbol_type_lookup(char32_t cp) noexcept {{
    if (cp > 0xFFFF) return CharType::OTHER;
    const auto key = static_cast<uint16_t>(cp);
    std::size_t lo = 0, hi = kSymbolTableSize;
    while (lo < hi) {{
        const std::size_t mid = lo + (hi - lo) / 2;
        if      (kSymbolCp[mid] < key) lo = mid + 1;
        else if (kSymbolCp[mid] > key) hi = mid;
        else return static_cast<CharType>(kSymbolCt[mid]);
    }}
    return CharType::OTHER;
}}
""")

print(f'Generated {len(entries)} entries -> {os.path.abspath(out_path)}')
