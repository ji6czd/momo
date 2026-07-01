#!/usr/bin/env python3
"""
gen_normalize_table.py
=======================
Python 側の normalize_compat_ideographs()（momo_py/src/momo_py/utils.py）を
真実の源として、Rust 用の互換漢字正規化テーブル `table.rs` を生成する。

真実の源（対象コードポイント範囲・NFKC 変換結果）はこのスクリプト内に inline
で複製してある（momo_py を依存ライブラリ化していないため）。
momo_py/src/momo_py/utils.py の対象範囲を変更したときは、このスクリプトの
_COMPAT_IDEOGRAPH_RANGES も同期させること。
"""

import unicodedata
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------
# Python 側の normalize_compat_ideographs() の対象範囲を inline 複製
# （momo_py/src/momo_py/utils.py と同期させること）
# ----------------------------------------------------------------------
_COMPAT_IDEOGRAPH_RANGES = [
    (0x2E80, 0x2EFF),  # CJK部首補助
    (0x2F00, 0x2FD5),  # 康煕部首
    (0xF900, 0xFAFF),  # CJK互換漢字
    (0x2F800, 0x2FA1F),  # CJK互換漢字補助
]

# ----------------------------------------------------------------------
# 対象範囲を走査し、NFKC で実際に畳み込まれるコードポイントだけを収集する
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
        #  1:1でない変換が混入した場合は原文位置の対応が崩れるため、
        #  ここで検出して失敗させる。)
        if len(nfkc) != 1:
            raise ValueError(
                f"U+{cp:04X} ({c!r}) は複数文字へ変換される: {nfkc!r}"
            )
        entries.append((cp, ord(nfkc)))

entries.sort(key=lambda x: x[0])

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
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

COLS = 8


def format_block(values: list[str], indent: str = "    ") -> str:
    lines = []
    for i in range(0, len(values), COLS):
        lines.append(indent + ", ".join(values[i : i + COLS]) + ",")
    return "\n".join(lines)


from_strs = [f"0x{cp:06X}" for cp, _ in entries]
to_strs = [f"0x{cp:06X}" for _, cp in entries]

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(
        f"""\
// AUTO-GENERATED — DO NOT EDIT MANUALLY
// Source : tools/gen_normalize_table.py
// Truth  : momo_py/src/momo_py/utils.py (normalize_compat_ideographs)
// Generated  : {timestamp}
// Ranges     : CJK部首補助(U+2E80-2EFF) / 康煕部首(U+2F00-2FD5) /
//              CJK互換漢字(U+F900-FAFF) / CJK互換漢字補助(U+2F800-2FA1F)
// Entry count: {len(entries)}

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
"""
    )

print(f"Generated {len(entries)} entries -> {OUT_PATH}")
