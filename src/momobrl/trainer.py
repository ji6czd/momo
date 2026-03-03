import joblib
import unicodedata

import sklearn_crfsuite
from typing import List

from .features import (
    get_units, get_char_type, compute_source_features,
    SourceEntry, FeatureDict,
)

KUTOUTEN = frozenset(["。", "、", "？", "！", ".", ","])


def is_suspicious(raw: str, read: str) -> bool:
    """
    ひらがな/カタカナの単純な対応ミスを検知。
    「は/ワ」「へ/エ」は点訳ルールとして許可。
    """
    if (raw == "" and read == " ") or read == "_": return False
    clean_read = read.replace("+S", "")

    # カタカナ -> カタカナの完全一致チェック
    if all('KATAKANA' in unicodedata.name(c, "") for c in raw):
        return raw != clean_read

    # ひらがな -> カタカナの一致チェック（例外あり）
    PARTICLE_EXCEPTIONS = {"は": ["ハ", "ワ"], "へ": ["ヘ", "エ"]}
    if raw in PARTICLE_EXCEPTIONS:
        return clean_read not in PARTICLE_EXCEPTIONS[raw]

    if all('HIRAGANA' in unicodedata.name(c, "") for c in raw):
        expected = "".join([chr(ord(c) + 0x60) for c in raw])
        # 点字の長音変換と文字が一致したときはfalse
        if expected == 'ウ' and clean_read == 'ー':
            return False
        else:
            return expected != clean_read

    return False


def process_line_to_tsv(line: str, line_num: int) -> List[str]:
    line = line.strip()
    parts = line.split('\t')

    if len(parts) < 2:
        raise ValueError(f"(Line {line_num}): タブが見つかりません。")
    elif len(parts) > 2:
        raise ValueError(
            f"(Line {line_num}): タブが複数（{len(parts)-1}個）含まれています。\n"
            f"   -> 原文と読みを区切るためのタブは「1つだけ」にしてください。分かち書き等にタブが混ざっていないか確認してください。\n"
            f"   余分な列の内容: {parts[2]}"
        )

    raw_part, read_full = parts[0], parts[1]

    # '//' の連続チェック
    if '//' in read_full:
        print(f"⚠️  Warning (Line {line_num}): 読み部分に連続した '/' が含まれています: '{read_full}'")

    read_blocks = read_full.split('/')
    raw_units_info = get_units(raw_part)

    tsv_rows, raw_ptr = [], 0

    # 0始まりでブロックをカウント
    for label_idx, r_label in enumerate(read_blocks):
        if r_label == " ":
            if tsv_rows:
                parts = tsv_rows[-1].split('\t')
                if "+S" not in parts[1]:
                    parts[1] += "+S"
                    tsv_rows[-1] = "\t".join(parts)
            while raw_ptr < len(raw_units_info) and raw_units_info[raw_ptr][0].isspace():
                raw_ptr += 1
            continue

        while raw_ptr < len(raw_units_info) and raw_units_info[raw_ptr][0].isspace():
            raw_ptr += 1

        if raw_ptr >= len(raw_units_info):
            raise ValueError(
                f"(Line {line_num}): 読みラベル過多。\n"
                f"   -> 読みインデックス [{label_idx}] のブロック '{r_label}' に対応する原文がありません！"
            )

        target_chars, orig_idx = raw_units_info[raw_ptr]

        if is_suspicious(target_chars, r_label):
            print(f"⚠️  Suspicious (Line {line_num}): 読みインデックス [{label_idx}] '{target_chars}' -> '{r_label}' (原文インデックス: {orig_idx})")

        # 句読点を見つけたら、読みラベルに自動で「+S」を補う
        if target_chars in KUTOUTEN and "+S" not in r_label:
            r_label += "+S"

        block_len = len(target_chars)
        for i, char in enumerate(target_chars):
            ctype = get_char_type(char)
            r_val = r_label if i == 0 else "---"
            tag = "S" if block_len == 1 else ("B" if i == 0 else ("E" if i == block_len - 1 else "I"))
            # TSVにも正確な0オリジンのインデックスを記録
            tsv_rows.append(f"{char}\t{r_val}\t{ctype}\t{tag}\t{orig_idx + i}")
        raw_ptr += 1

    remaining = [u for u in raw_units_info[raw_ptr:] if not u[0].isspace()]
    if remaining:
        first_leftover_val, first_leftover_idx = remaining[0]
        raise ValueError(
            f"(Line {line_num}): 原文余り。\n"
            f"   -> 原文インデックス [{first_leftover_idx}] の '{first_leftover_val}' 以降に対応する読みラベルがありません！"
        )

    return tsv_rows


def create_data(rawdata: str, tsvdata: str) -> None:
    """
    原データファイル(rawdata)を処理して、TSVファイル(tsvdata)に出力する。
    """
    with open(rawdata, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    all_tsv = ["#原文\t読み\t文字種\tタグ\tOrigIdx"]
    success = 0
    for i, line in enumerate(lines, 1):
        # '#'で始まる行はコメント行としてスキップする（TSVのヘッダー行もこれに含まれる）
        if not line.strip() or line.startswith('#'):
            continue
        rows = process_line_to_tsv(line, i)
        if rows:
            all_tsv.extend(rows)
            all_tsv.append("")
            success += 1
    with open(tsvdata, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_tsv))
    print(f"✅ TSV作成完了 ({success}行): {tsvdata}")


def train(tsvdata: str) -> None:
    """
    TSVファイルを読み込み、CRFモデルを学習して保存する。
    """
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
        # TSV列: [char, reading, ctype, tag, orig_idx]
        source_seq: List[SourceEntry] = [
            (p[0], int(p[4]) if len(p) > 4 and p[4].lstrip('-').isdigit() else idx,
             p[2])
            for idx, p in enumerate(sentence)
        ]
        labels = [p[1] for p in sentence]

        X.append(compute_source_features(source_seq))
        y.append(labels)

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,   # L1正則化: 低情報特徴量の重みをゼロに近づける
        c2=0.01,  # L2正則化: 重みを全体的に小さく抑える
        max_iterations=70,
        all_possible_transitions=False,  # あり得ない遷移パターンの計算をスキップ
        verbose=True
    )
    crf.fit(X, y)

    model_path = tsvdata.rsplit('.', 1)[0] + ".model"
    joblib.dump(crf, model_path)
    print("💾 学習完了")
