import io
import json
import os
import unicodedata
import zipfile
from datetime import datetime, timezone
from typing import List
from collections import defaultdict

import joblib
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import csr_matrix

from .features import (
    get_units, get_char_type, compute_source_features,
    SourceEntry, FeatureDict, LABEL_CONTINUE, LABEL_SKIP,
)
from .utils import split_on_unescaped_slash, CharType
from .predictor import LRModelBundle
from .exporter import export
KUTOUTEN = frozenset(["。", "、", "？", "！", ".", ","])

# ラベルを '_'（LABEL_SKIP）にして素通しする文字種
_SKIP_CTYPES = frozenset({
    CharType.ALPHA,
    CharType.NUMERIC,
    CharType.SYMBOL,
    CharType.SYMBOL_CLOSE,
    CharType.SYMBOL_OPEN,
    CharType.SYMBOL_STOP,
    CharType.SYMBOL_PAUSE,
})


# ==========================================
# 🌟 1. 統計データ構築
# ==========================================
def build_stats_from_tsv(tsvdata: str) -> dict:
    """過去のTSVファイルから安全な統計辞書を構築する"""
    stats = defaultdict(lambda: defaultdict(int))
    if not os.path.exists(tsvdata):
        print("⚠️  注意: 過去のTSVファイルが見つかりません。初期辞書を作ります。")
        stats['切']['キリ'] = 1
        return stats

    with open(tsvdata, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue

            parts = line.split('\t')
            if len(parts) >= 2:
                char, reading = parts[0], parts[1]
                clean_read = reading.replace("+S", "")
                if clean_read != LABEL_CONTINUE and clean_read not in (LABEL_SKIP, "-"):
                    stats[char][clean_read] += 1
    return stats


# ==========================================
# 🌟 2. バリデーション（検査）ロジック
# ==========================================
def _is_basic_suspicious(raw: str, read: str) -> bool:
    """ひらがな/カタカナの単純な対応ミスを検知"""
    if (raw == "" and read == " ") or read == LABEL_SKIP: return False
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

# 後方互換のためエイリアスを残す
is_suspicious = _is_basic_suspicious

def _validate_label_chars(r_label: str, line_num: int) -> None:
    """読みに漢字やひらがなが混入していないかチェック"""
    clean_r_label = r_label.replace("+S", "")
    for c in clean_r_label:
        if c in (LABEL_SKIP, " ", "-"): continue
        ctype = get_char_type(c)
        if ctype == 'KANJI':
            print(f"🚨 警告 (Line {line_num}): 読みに漢字が混入しています！ -> '{c}' (in '{r_label}')")
        elif ctype == 'HIRAGANA':
            print(f"🚨 警告 (Line {line_num}): 読みにひらがなが混入しています！ -> '{c}' (in '{r_label}')")

def _check_alignment_anomalies(target_chars: str, ctype: str, r_label: str, orig_idx: int, label_idx: int, line_num: int, stats: dict) -> None:
    """統計的異常や単純なミスマッチを警告する"""
    if ctype == 'KANJI':
        clean_label = r_label.replace("+S", "")
        if stats and target_chars in stats:
            total_occurrences = sum(stats[target_chars].values())
            current_occurrences = stats[target_chars].get(clean_label, 0)
            if total_occurrences > 0 and current_occurrences == 0:
                print(f"⚠️  Statistical Anomaly (Line {line_num}): '{target_chars}' が過去の実績にない読み '{clean_label}' になっています。ズレていませんか？")
        else:
            print(f"⚠️ Line {line_num}: '{target_chars}' は '{clean_label}' として学習されます。")

    if _is_basic_suspicious(target_chars, r_label):
        print(f"⚠️  Suspicious (Line {line_num}): 読みインデックス [{label_idx}] '{target_chars}' -> '{r_label}' (原文インデックス: {orig_idx})")


# ==========================================
# 🌟 3. TSV行の生成（フォーマッタ）
# ==========================================
def _create_tsv_rows(target_chars: str, ctype: str, r_label: str, orig_idx: int) -> List[str]:
    """1ブロック分の文字列とラベルから、TSV行リストを生成する"""
    if ctype in _SKIP_CTYPES:
        return [f"{char}\t{LABEL_SKIP}\t{ctype}\t{orig_idx + i}"
                for i, char in enumerate(target_chars)]
    # 複合ユニット（拗音・漢字語など）は1行にまとめる。LABEL_CONTINUE で分割しない。
    return [f"{target_chars}\t{r_label}\t{ctype}\t{orig_idx}"]


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

    read_blocks_raw = split_on_unescaped_slash(read_full)
    if any(b == "" for b in read_blocks_raw[1:-1]):
        print(f"⚠️  Warning (Line {line_num}): 読み部分に連続した '/' が含まれています: '{read_full}'")
    read_blocks = [b.replace(r'\/', '/').replace(r'\_', '_') for b in read_blocks_raw]

    raw_units_info = get_units(raw_part)
    tsv_rows, raw_ptr = [], 0

    for label_idx, r_label in enumerate(read_blocks):
        _validate_label_chars(r_label, line_num)

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

        target_chars, orig_idx, ctype = raw_units_info[raw_ptr]
        _check_alignment_anomalies(target_chars, ctype, r_label, orig_idx, label_idx, line_num, stats)

        rows = _create_tsv_rows(target_chars, ctype, r_label, orig_idx)
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

    all_tsv = ["#原文\t読み\t文字種\tOrigIdx"]
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


# ==========================================
# 🌟 6. ラベル分離ユーティリティ
# ==========================================
def _split_labels(raw_labels: List[str]) -> tuple:
    """
    TSVの読みラベル列を読みモデル用と境界モデル用に分離する。
    """
    y_read     = [label.replace("+S", "") for label in raw_labels]
    y_boundary = ["0" if label == LABEL_CONTINUE else ("1" if "+S" in label else "0")
                  for label in raw_labels]
    return y_read, y_boundary


# ==========================================
# 🌟 7. train()
# ==========================================
def train(tsvdata: str, model_file: str | None, window: int = 7, dry_run: bool = False) -> None:
    """
    TSVから読みモデル（LinearSVC）と境界モデル（SGDClassifier）を学習し、
    1つのZIPにまとめる。

    モデル構成:
        読みモデル  : LinearSVC
            - 多クラス分類（ラベル数1000超）でも高速な推論が可能
            - predict_proba は使えないが decision_function でスコアを取得できる
        境界モデル  : SGDClassifier(loss='modified_huber')
            - 2値分類（0/1）
            - predict_proba が使える（漢数字フォールバック判定に使用）

    出力ファイル:
        basename_bundle.pkl  - モデル一式（joblib）
        basename.zip         - 上記をまとめたパッケージ
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

    X_dicts:    List[FeatureDict] = []
    Y_read:     List[str]         = []
    Y_boundary: List[str]         = []
    compound_units: list[str]     = []  # 複合ユニット（拗音を除く漢字語など）

    for sentence in sentences:
        source_seq: List[SourceEntry] = [
            (p[0], int(p[3]) if len(p) > 3 and p[3].lstrip('-').isdigit() else idx, p[2])
            for idx, p in enumerate(sentence)
        ]
        raw_labels = [p[1] for p in sentence]

        # 2文字以上の非skipユニットを複合ユニット辞書に登録（拗音は推論時にアルゴリズム検出）
        for entry in source_seq:
            val = entry[0]
            if len(val) >= 2 and not all(
                'ぁ' <= c <= 'ん' or 'ァ' <= c <= 'ヶ'
                for c in val
            ):
                compound_units.append(val)

        features = compute_source_features(source_seq, window=window)
        X_dicts.extend(features)

        y_read, y_boundary = _split_labels(raw_labels)
        Y_read.extend(y_read)
        Y_boundary.extend(y_boundary)

    print(f"\n📊 学習サンプル数: {len(X_dicts)}")

    # ==========================================
    # 読みモデル: LinearSVC
    # ==========================================
    print("\n🏋️  [読みモデル] ベクトル化中...")
    vect_read = DictVectorizer(sparse=True)
    X_read = vect_read.fit_transform(X_dicts)
    print(f"   特徴量ベクトル次元数: {X_read.shape[1]}")
    print(f"   読みラベル種類数: {len(set(Y_read))}")
    if dry_run:
        print("⚠️  dry_run=True のため、ここまでで終了します。")
        return

    print("🏋️  [読みモデル] 学習中 (LinearSVC)...")
    # windowに応じてパラメータだけ切り替え
    params = {
        7: dict(C=1.0, max_iter=2000, tol=1e-4, verbose=0),
        5: dict(C=1.0, max_iter=2000, tol=1e-4, verbose=0),
        4: dict(C=1.0, max_iter=2000, tol=1e-4, verbose=0),
    }

    model_read = LinearSVC(**params[window])
    model_read.fit(X_read, Y_read)
    model_read.coef_ = model_read.coef_.astype(np.float32)
    model_read.intercept_ = model_read.intercept_.astype(np.float32)

    # 枝刈り→sparse化
    coef = model_read.coef_.copy()
    coef[np.abs(coef) < 0.01] = 0.0
    coef_sparse = csr_matrix(coef)
    print(f"疎性: {(coef == 0).sum() / coef.size:.1%}")
    print(f"推定サイズ: {coef_sparse.data.nbytes / 1024**2:.1f}MB")
    print("💾 [読みモデル] 学習完了")

    # ==========================================
    # 境界モデル: SGDClassifier
    # ==========================================
    print("\n🏋️  [境界モデル] ベクトル化中...")
    vect_boundary = DictVectorizer(sparse=True)
    X_boundary = vect_boundary.fit_transform(X_dicts)

    print("🏋️  [境界モデル] 学習中 (SGDClassifier)...")
    model_boundary = SGDClassifier(
        loss='modified_huber',  # predict_proba が使える
        max_iter=200,
        random_state=42,
        verbose=0,
    )
    model_boundary.fit(X_boundary, Y_boundary)
    model_boundary.coef_ = model_boundary.coef_.astype(np.float32)
    model_boundary.intercept_ = model_boundary.intercept_.astype(np.float32)
    print("💾 [境界モデル] 学習完了")

    # ==========================================
    # ZIPにまとめて保存
    # ==========================================
    # model_fileが指定されていればそれを使う。
    if model_file:
        base = model_file
    else:
        base = tsvdata.rsplit('.', 1)[0]
    bundle_name = os.path.basename(base) + "_bundle.pkl"
    zip_path    = base + f"_{window}.zip"
    mbm_path    = base + f"_{window}.mbm"

    out_dir = os.path.dirname(zip_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    bundle = LRModelBundle(
        vectorizer_read     = vect_read,
        coef_read_sparse    = coef_sparse,
        intercept_read      = model_read.intercept_.astype(np.float32),
        read_classes        = np.array(model_read.classes_),
        vectorizer_boundary = vect_boundary,
        model_boundary      = model_boundary,
        version_info        = {},
    )

    version_info = {
        "trained_at":     datetime.now(timezone.utc).isoformat(),
        "model_bundle":   bundle_name,
        "algorithm":      "LinearSVC+SGD",
        "window_size":    window,
        "compound_units": sorted(set(compound_units)),
    }
    bundle_buf = io.BytesIO()
    joblib.dump(bundle, bundle_buf, compress=0)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(bundle_name, bundle_buf.getvalue())
        zf.writestr("version_info.json", json.dumps(version_info, ensure_ascii=False, indent=2))

    print(f"\n📦 ZIPパッケージ作成完了: {zip_path}")
    print(f"   ├ {bundle_name}")
    print(f"   └ version_info.json")
    export(zip_path, mbm_path)
    print(f"量子化モデル (MBM) エクスポート完了: {mbm_path}")