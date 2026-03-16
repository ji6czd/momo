import json
import os
import re
import unicodedata
import zipfile
from datetime import datetime, timezone
from typing import List
from collections import defaultdict

import pycrfsuite

from .features import (
    get_units, get_char_type, compute_source_features,
    SourceEntry, FeatureDict,
)

KUTOUTEN = frozenset(["。", "、", "？", "！", ".", ","])


# ==========================================
# 🌟 1. 統計データ構築
# ==========================================
def build_stats_from_tsv(tsvdata: str) -> dict:
    """過去のTSVファイルから安全な統計辞書を構築する"""
    stats = defaultdict(lambda: defaultdict(int))
    if not os.path.exists(tsvdata):
        return stats
        
    with open(tsvdata, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            
            parts = line.split('\t')
            if len(parts) >= 2:
                char, reading = parts[0], parts[1]
                clean_read = reading.replace("+S", "")
                if clean_read != "---" and clean_read not in ("_", "-"):
                    stats[char][clean_read] += 1
    return stats


# ==========================================
# 🌟 2. バリデーション（検査）ロジック
# ==========================================
def _is_basic_suspicious(raw: str, read: str) -> bool:
    """ひらがな/カタカナの単純な対応ミスを検知"""
    if (raw == "" and read == " ") or read == "_": return False
    clean_read = read.replace("+S", "")

    if all('KATAKANA' in unicodedata.name(c, "") for c in raw):
        return raw != clean_read

    PARTICLE_EXCEPTIONS = {"は": ["ハ", "ワ"], "へ": ["ヘ", "エ"]}
    if raw in PARTICLE_EXCEPTIONS:
        return clean_read not in PARTICLE_EXCEPTIONS[raw]

    if all('HIRAGANA' in unicodedata.name(c, "") for c in raw):
        expected = "".join([chr(ord(c) + 0x60) for c in raw])
        if expected == 'ウ' and clean_read == 'ー': return False
        return expected != clean_read

    return False

def _validate_label_chars(r_label: str, line_num: int) -> None:
    """読みに漢字やひらがなが混入していないかチェック"""
    clean_r_label = r_label.replace("+S", "")
    for c in clean_r_label:
        if c in ("_", " ", "-"): continue
        ctype = get_char_type(c)
        if ctype == 'KANJI':
            print(f"🚨 警告 (Line {line_num}): 読みに漢字が混入しています！ -> '{c}' (in '{r_label}')")
        elif ctype == 'HIRAGANA':
            print(f"🚨 警告 (Line {line_num}): 読みにひらがなが混入しています！ -> '{c}' (in '{r_label}')")

def _check_alignment_anomalies(target_chars: str, r_label: str, orig_idx: int, label_idx: int, line_num: int, stats: dict) -> None:
    """統計的異常や単純なミスマッチを警告する"""
    # 1. 過去のTSV実績に基づく自己学習型バリデーション
    if stats and target_chars in stats:
        if any(get_char_type(c) == 'KANJI' for c in target_chars):
            clean_label = r_label.replace("+S", "")
            total_occurrences = sum(stats[target_chars].values())
            current_occurrences = stats[target_chars].get(clean_label, 0)
            if total_occurrences >= 3 and current_occurrences == 0:
                print(f"⚠️  Statistical Anomaly (Line {line_num}): '{target_chars}' が過去の実績にない読み '{clean_label}' になっています。ズレていませんか？")
    
    # 2. 旧ルールの基本的なチェック
    if _is_basic_suspicious(target_chars, r_label):
        print(f"⚠️  Suspicious (Line {line_num}): 読みインデックス [{label_idx}] '{target_chars}' -> '{r_label}' (原文インデックス: {orig_idx})")


# ==========================================
# 🌟 3. TSV行の生成（フォーマッタ）
# ==========================================
def _create_biose_rows(target_chars: str, r_label: str, orig_idx: int) -> List[str]:
    """1ブロック分の文字列とラベルから、BIOSEタグ付きのTSV行リストを生成する"""
    if target_chars in KUTOUTEN and "+S" not in r_label:
        r_label += "+S"

    rows = []
    block_len = len(target_chars)
    for i, char in enumerate(target_chars):
        ctype = get_char_type(char)
        r_val = r_label if i == 0 else "---"
        tag = "S" if block_len == 1 else ("B" if i == 0 else ("E" if i == block_len - 1 else "I"))
        rows.append(f"{char}\t{r_val}\t{ctype}\t{tag}\t{orig_idx + i}")
    return rows


# ==========================================
# 🌟 4. 行処理パイプライン（The Conductor）
# ==========================================
def process_line_to_tsv(line: str, line_num: int, stats: dict = None) -> List[str]:
    line = line.strip()
    parts = line.split('\t')

    if len(parts) < 2:
        raise ValueError(f"(Line {line_num}): タブが見つかりません。")
    elif len(parts) > 2:
        raise ValueError(f"(Line {line_num}): タブが複数含まれています。\n -> タブは「1つだけ」にしてください。")

    raw_part, read_full = parts[0], parts[1]

    if re.search(r'(?<!\\)//', read_full):
        print(f"⚠️  Warning (Line {line_num}): 読み部分に連続した '/' が含まれています: '{read_full}'")

    read_blocks_raw = re.split(r'(?<!\\)/', read_full)
    read_blocks = [b.replace(r'\/', '/').replace(r'\_', '_') for b in read_blocks_raw]

    raw_units_info = get_units(raw_part)
    tsv_rows, raw_ptr = [], 0

    for label_idx, r_label in enumerate(read_blocks):
        # 読みに漢字やひらがなが混ざっていないかチェック
        _validate_label_chars(r_label, line_num)
        
        # 空白ブロックの処理
        if r_label == " ":
            if tsv_rows:
                last_parts = tsv_rows[-1].split('\t')
                if "+S" not in last_parts[1]:
                    last_parts[1] += "+S"
                    tsv_rows[-1] = "\t".join(last_parts)
            while raw_ptr < len(raw_units_info) and raw_units_info[raw_ptr][0].isspace():
                raw_ptr += 1
            continue

        while raw_ptr < len(raw_units_info) and raw_units_info[raw_ptr][0].isspace():
            raw_ptr += 1

        if raw_ptr >= len(raw_units_info):
            raise ValueError(f"(Line {line_num}): 読みラベル過多。\n -> 読みインデックス [{label_idx}] '{r_label}' に対応する原文がありません！")

        target_chars, orig_idx = raw_units_info[raw_ptr]

        # アライメントの異常チェック
        _check_alignment_anomalies(target_chars, r_label, orig_idx, label_idx, line_num, stats)

        # TSV行の生成と追加
        rows = _create_biose_rows(target_chars, r_label, orig_idx)
        tsv_rows.extend(rows)
        raw_ptr += 1

    remaining = [u for u in raw_units_info[raw_ptr:] if not u[0].isspace()]
    if remaining:
        raise ValueError(f"(Line {line_num}): 原文余り。\n -> 原文 '{remaining[0][0]}' 以降に対応する読みラベルがありません！")

    return tsv_rows


# ==========================================
# 🌟 5. メインルーチン
# ==========================================
def create_data(rawdata: str, tsvdata: str) -> None:
    print(f"📊 過去の実績 ({tsvdata}) から統計辞書を構築中...")
    stats = build_stats_from_tsv(tsvdata)

    with open(rawdata, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    all_tsv = ["#原文\t読み\t文字種\tタグ\tOrigIdx"]
    success = 0
    for i, line in enumerate(lines, 1):
        if not line.strip() or line.startswith('#'): continue
        rows = process_line_to_tsv(line, i, stats)
        if rows:
            all_tsv.extend(rows)
            all_tsv.append("")
            success += 1
            
    with open(tsvdata, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_tsv))
    print(f"✅ TSV作成完了 ({success}行): {tsvdata}")


def train(tsvdata: str) -> None:
    sentences, current = [], []
    with open(tsvdata, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                if current:
                    sentences.append(current)
                current = []
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                current.append(parts)
    if current:
        sentences.append(current)
        
    X: List[List[FeatureDict]] = []
    y: List[List[str]] = []
    for sentence in sentences:
        source_seq: List[SourceEntry] = [
            (p[0], int(p[4]) if len(p) > 4 and p[4].lstrip('-').isdigit() else idx, p[2])
            for idx, p in enumerate(sentence)
        ]
        labels = [p[1] for p in sentence]
        X.append(compute_source_features(source_seq))
        y.append(labels)

    trainer = pycrfsuite.Trainer(verbose=True)
    for xseq, yseq in zip(X, y):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 0.1,
        'c2': 0.01,
        'max_iterations': 70,
        'feature.possible_transitions': False
    })

    model_path = tsvdata.rsplit('.', 1)[0] + ".crfsuite"
    trainer.train(model_path)
    print(f"💾 学習完了: {model_path} にネイティブバイナリを保存しました！")

    zip_path = model_path.rsplit(".", 1)[0] + ".zip"
    version_info = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_file": os.path.basename(model_path),
    }
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, os.path.basename(model_path))
        zf.writestr("version_info.json", json.dumps(version_info, ensure_ascii=False, indent=2))
    print(f"📦 ZIPパッケージ作成完了: {zip_path}")
