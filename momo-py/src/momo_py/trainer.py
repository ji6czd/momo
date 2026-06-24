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
from joblib import Parallel, delayed
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import csr_matrix, vstack

from .features import (
    get_units,
    get_char_type,
    compute_source_features,
    SourceEntry,
    FeatureDict,
    LABEL_CONTINUE,
    LABEL_SKIP,
)
from .utils import split_on_unescaped_slash, CharType
from .predictor import LRModelBundle
from .exporter import export

KUTOUTEN = frozenset(["。", "、", "？", "！", ".", ","])

# ラベルを '_'（LABEL_SKIP）にして素通しする文字種
# 注: NUMERIC は含めない。数字は恒等またはかな読みとして学習対象にする
#     （_create_tsv_rows を参照）。英字（ALPHA）は引き続き素通し。
_SKIP_CTYPES = frozenset(
    {
        CharType.ALPHA,
        CharType.SYMBOL,
        CharType.SYMBOL_CLOSE,
        CharType.SYMBOL_OPEN,
        CharType.SYMBOL_STOP,
        CharType.SYMBOL_PAUSE,
    }
)


# ==========================================
# 🌟 1. 統計データ構築
# ==========================================
def build_stats_from_tsv(tsvdata: str) -> dict:
    """過去のTSVファイルから安全な統計辞書を構築する"""
    stats = defaultdict(lambda: defaultdict(int))
    if not os.path.exists(tsvdata):
        print("⚠️  注意: 過去のTSVファイルが見つかりません。初期辞書を作ります。")
        stats["切"]["キリ"] = 1
        return stats

    with open(tsvdata, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
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
    if (raw == "" and read == " ") or read == LABEL_SKIP:
        return False
    clean_read = read.replace("+S", "")

    if all("KATAKANA" in unicodedata.name(c, "") for c in raw):
        return raw != clean_read

    PARTICLE_EXCEPTIONS = {"は": ["ハ", "ワ"], "へ": ["ヘ", "エ"]}
    if raw in PARTICLE_EXCEPTIONS:
        return clean_read not in PARTICLE_EXCEPTIONS[raw]

    if all("HIRAGANA" in unicodedata.name(c, "") for c in raw):
        expected = "".join([chr(ord(c) + 0x60) for c in raw])
        if expected == "ウ" and clean_read == "ー":
            return False
        return expected != clean_read

    return False


# 後方互換のためエイリアスを残す
is_suspicious = _is_basic_suspicious


def _validate_label_chars(r_label: str, line_num: int) -> None:
    """読みに漢字やひらがなが混入していないかチェック"""
    clean_r_label = r_label.replace("+S", "")
    for c in clean_r_label:
        if c in (LABEL_SKIP, " ", "-"):
            continue
        ctype = get_char_type(c)
        if ctype == "KANJI":
            print(
                f"🚨 警告 (Line {line_num}): 読みに漢字が混入しています！ -> '{c}' (in '{r_label}')"
            )
        elif ctype == "HIRAGANA":
            print(
                f"🚨 警告 (Line {line_num}): 読みにひらがなが混入しています！ -> '{c}' (in '{r_label}')"
            )


def _check_alignment_anomalies(
    target_chars: str,
    ctype: str,
    r_label: str,
    orig_idx: int,
    label_idx: int,
    line_num: int,
    stats: dict,
) -> None:
    """統計的異常や単純なミスマッチを警告する"""
    if ctype == "KANJI":
        clean_label = r_label.replace("+S", "")
        if stats and target_chars in stats:
            total_occurrences = sum(stats[target_chars].values())
            current_occurrences = stats[target_chars].get(clean_label, 0)
            if total_occurrences > 0 and current_occurrences == 0:
                print(
                    f"⚠️  Statistical Anomaly (Line {line_num}): '{target_chars}' が過去の実績にない読み '{clean_label}' になっています。ズレていませんか？"
                )
        else:
            print(
                f"⚠️ Line {line_num}: '{target_chars}' は '{clean_label}' として学習されます。"
            )

    if _is_basic_suspicious(target_chars, r_label):
        print(
            f"⚠️  Suspicious (Line {line_num}): 読みインデックス [{label_idx}] '{target_chars}' -> '{r_label}' (原文インデックス: {orig_idx})"
        )


# ==========================================
# 🌟 3. TSV行の生成（フォーマッタ）
# ==========================================
def _create_tsv_rows(
    target_chars: str, ctype: str, r_label: str, orig_idx: int
) -> List[str]:
    """1ブロック分の文字列とラベルから、TSV行リストを生成する"""
    if ctype in _SKIP_CTYPES:
        return [
            f"{char}\t{LABEL_SKIP}\t{ctype}\t{orig_idx + i}"
            for i, char in enumerate(target_chars)
        ]
    # 数字（NUMERIC）は1文字ずつ学習する（推論側も1文字に展開するため整合させる）。
    #   単一数字 : 注釈をそのまま採用（恒等 "3"→"3"、またはかな "3"→"ミッ"＝みっか）
    #   多桁     : 桁ごとに恒等（"120"→ 1,2,0）。多桁のかな読み（はつか等）は扱わず
    #              漢数字（二十日）で表現する。
    if ctype == CharType.NUMERIC:
        return [f"{target_chars}\t{r_label}\t{ctype}\t{orig_idx}"]
    # 拗音（ひらがな/カタカナ複合ユニット）は1行にまとめる。
    # 推論時も get_units() が同じユニットとして認識するため整合が取れる。
    # 漢字などは LABEL_CONTINUE で1文字ずつに分割する（推論は1文字単位のため）。
    if len(target_chars) == 1 or ctype in (CharType.HIRAGANA, CharType.KATAKANA):
        return [f"{target_chars}\t{r_label}\t{ctype}\t{orig_idx}"]
    rows = []
    for i, char in enumerate(target_chars):
        r_val = r_label if i == 0 else LABEL_CONTINUE
        rows.append(f"{char}\t{r_val}\t{ctype}\t{orig_idx + i}")
    return rows


# ==========================================
# 🌟 4. 行処理パイプライン（The Conductor）
# ==========================================
def process_line_to_tsv(line: str, line_num: int, stats: dict = None) -> List[str]:
    line = line.strip()
    parts = line.split("\t")

    if len(parts) < 2:
        raise ValueError(f"(Line {line_num}): タブが見つかりません。")
    elif len(parts) > 2:
        raise ValueError(
            f"(Line {line_num}): タブが複数含まれています。\n -> タブは「1つだけ」にしてください。"
        )

    raw_part, read_full = parts[0], parts[1]

    read_blocks_raw = split_on_unescaped_slash(read_full)
    if any(b == "" for b in read_blocks_raw[1:-1]):
        print(
            f"⚠️  Warning (Line {line_num}): 読み部分に連続した '/' が含まれています: '{read_full}'"
        )
    read_blocks = [b.replace(r"\/", "/").replace(r"\_", "_") for b in read_blocks_raw]

    raw_units_info = get_units(raw_part)
    tsv_rows, raw_ptr = [], 0

    for label_idx, r_label in enumerate(read_blocks):
        _validate_label_chars(r_label, line_num)

        if r_label == " ":
            if tsv_rows:
                last_parts = tsv_rows[-1].split("\t")
                if "+S" not in last_parts[1]:
                    last_parts[1] += "+S"
                    tsv_rows[-1] = "\t".join(last_parts)
            while (
                raw_ptr < len(raw_units_info) and raw_units_info[raw_ptr][0].isspace()
            ):
                raw_ptr += 1
            continue

        while raw_ptr < len(raw_units_info) and raw_units_info[raw_ptr][0].isspace():
            raw_ptr += 1

        if raw_ptr >= len(raw_units_info):
            raise ValueError(
                f"(Line {line_num}): 読みラベル過多。\n -> 読みインデックス [{label_idx}] '{r_label}' に対応する原文がありません！"
            )

        target_chars, orig_idx, ctype = raw_units_info[raw_ptr]
        _check_alignment_anomalies(
            target_chars, ctype, r_label, orig_idx, label_idx, line_num, stats
        )

        rows = _create_tsv_rows(target_chars, ctype, r_label, orig_idx)
        tsv_rows.extend(rows)
        raw_ptr += 1

    remaining = [u for u in raw_units_info[raw_ptr:] if not u[0].isspace()]
    if remaining:
        raise ValueError(
            f"(Line {line_num}): 原文余り。\n -> 原文 '{remaining[0][0]}' 以降に対応する読みラベルがありません！"
        )

    return tsv_rows


# ==========================================
# 🌟 5. メインルーチン
# ==========================================
def create_data(rawdata: str, tsvdata: str) -> None:
    print(f"📊 過去の実績 ({tsvdata}) から統計辞書を構築中...")
    stats = build_stats_from_tsv(tsvdata)

    with open(rawdata, "r", encoding="utf-8") as f:
        lines = f.readlines()

    all_tsv = ["#原文\t読み\t文字種\tOrigIdx"]
    success = 0
    for i, line in enumerate(lines, 1):
        if not line.strip() or line.startswith("#"):
            continue
        rows = process_line_to_tsv(line, i, stats)
        if rows:
            all_tsv.extend(rows)
            all_tsv.append("")
            success += 1

    with open(tsvdata, "w", encoding="utf-8") as f:
        f.write("\n".join(all_tsv))
    print(f"✅ TSV作成完了 ({success}行): {tsvdata}")


# ==========================================
# 🌟 6. ラベル分離ユーティリティ
# ==========================================
def _split_labels(raw_labels: List[str]) -> tuple:
    """
    TSVの読みラベル列を読みモデル用と境界モデル用に分離する。
    """
    y_read = [label.replace("+S", "") for label in raw_labels]
    y_boundary = [
        "0" if label == LABEL_CONTINUE else ("1" if "+S" in label else "0")
        for label in raw_labels
    ]
    return y_read, y_boundary


# ==========================================
# 🌟 6.5 読みモデルの One-vs-Rest 並列学習
# ==========================================
PRUNE_THRESHOLD = 0.01


def _fit_ovr_batch(X, y_arr, class_chunk, use_svc, svc_params):
    """与えられたクラス群について、それぞれ「当該クラス vs その他」の
    2値分類器を学習し、枝刈り済み sparse 重み行 + 切片を返す。

    1クラスずつ2値で学習するため、liblinear/SGD が内部確保する密 coef_ は
    常に (1, n_features) = 数MB に収まり、(n_classes, n_features) の巨大配列
    （2000×2M×float64 ≒ 32GB）を一度も確保しない。
    """
    rows = []
    intercepts = []
    for c in class_chunk:
        y_bin = (y_arr == c).astype(np.int8)
        if use_svc:
            clf = LinearSVC(**svc_params)
        else:
            clf = SGDClassifier(
                loss="hinge", max_iter=2000, tol=1e-4, random_state=42, verbose=0
            )
        clf.fit(X, y_bin)
        # 2値なので classes_=[0,1]、coef_[0] は正例(=当該クラス)の重み
        w = clf.coef_[0].astype(np.float32)
        w[np.abs(w) < PRUNE_THRESHOLD] = 0.0
        rows.append(csr_matrix(w))
        intercepts.append(np.float32(clf.intercept_[0]))
    return rows, intercepts


# ==========================================
# 🌟 7. train()
# ==========================================
def train(
    tsvdata: str,
    model_file: str | None,
    window: int = 7,
    dry_run: bool = False,
    use_svc: bool = True,
    n_jobs: int = 4,
) -> None:
    """
    TSVから読みモデルと境界モデル（SGDClassifier）を学習し、1つのZIPにまとめる。

    モデル構成:
        読みモデル  : LinearSVC (use_svc=True) または SGDClassifier/hinge (use_svc=False)
            - LinearSVC: 精度優先。メモリが (クラス数×特徴量次元)×float64 必要
            - SGDClassifier: メモリ節約。大規模データで LinearSVC がOOMになる場合に使用
        境界モデル  : SGDClassifier(loss='modified_huber')
            - 2値分類（0/1）
            - predict_proba が使える（漢数字フォールバック判定に使用）

    出力ファイル:
        basename_bundle.pkl  - モデル一式（joblib）
        basename.zip         - 上記をまとめたパッケージ
    """
    sentences, current = [], []
    with open(tsvdata, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                if current:
                    sentences.append(current)
                current = []
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                current.append(parts)
    if current:
        sentences.append(current)

    X_dicts: List[FeatureDict] = []
    Y_read: List[str] = []
    Y_boundary: List[str] = []

    for sentence in sentences:
        source_seq: List[SourceEntry] = [
            (
                p[0],
                int(p[3]) if len(p) > 3 and p[3].lstrip("-").isdigit() else idx,
                p[2],
            )
            for idx, p in enumerate(sentence)
        ]
        raw_labels = [p[1] for p in sentence]

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
    X_read.data = X_read.data.astype(np.float32, copy=False)  # type: ignore[union-attr]
    X_read.indices = X_read.indices.astype(np.int32, copy=False)  # type: ignore[union-attr]
    X_read.indptr = X_read.indptr.astype(np.int32, copy=False)  # type: ignore[union-attr]
    print(f"   特徴量ベクトル次元数: {X_read.shape[1]}")
    print(f"   読みラベル種類数: {len(set(Y_read))}")
    if dry_run:
        print("⚠️  dry_run=True のため、ここまでで終了します。")
        return

    # windowに応じてパラメータだけ切り替え（LinearSVC用）
    svc_params = {
        7: dict(C=1.0, max_iter=2000, tol=1e-4, verbose=0),
        5: dict(C=1.0, max_iter=2000, tol=1e-4, verbose=0),
        4: dict(C=0.1, max_iter=2000, tol=1e-2, verbose=0),
    }[window]

    algo_name = "LinearSVC" if use_svc else "SGDClassifier loss=hinge"
    # クラスごとに「当該クラス vs その他」を並列で学習（One-vs-Rest）。
    # 密 coef_ (n_classes×n_features) を一括確保せずメモリを抑える。
    y_arr = np.asarray(Y_read)
    classes = np.unique(y_arr)
    print(
        f"🏋️  [読みモデル] 学習中 ({algo_name}, One-vs-Rest "
        f"{len(classes)}クラス, n_jobs={n_jobs})..."
    )

    # ラウンドロビンでクラスをn_jobs個のバッチに分割し、各バッチを1タスクに。
    # これにより X はワーカ数ぶんだけpickleされ（クラス数ぶんではない）、
    # 学習コストの偏り（高頻度かな vs 低頻度）も平準化される。
    n_batches = max(1, n_jobs)
    chunks = [list(classes[i::n_batches]) for i in range(n_batches)]
    chunks = [c for c in chunks if c]

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_fit_ovr_batch)(X_read, y_arr, chunk, use_svc, svc_params)
        for chunk in chunks
    )

    # 結果をクラス→(重み行,切片)に展開し、ソート済みクラス順で再整列する。
    # predictor 側が read_classes に np.searchsorted を使うため、行・切片・
    # read_classes はソート順（= np.unique の順）で揃える必要がある。
    row_by_class = {}
    inter_by_class = {}
    for chunk, (rows, intercepts) in zip(chunks, results):
        for c, r, b in zip(chunk, rows, intercepts):
            row_by_class[c] = r
            inter_by_class[c] = b

    read_classes = classes  # np.unique 済みでソート済み
    coef_sparse = vstack([row_by_class[c] for c in classes]).tocsr()
    coef_sparse.data = coef_sparse.data.astype(np.float32, copy=False)
    intercept_read = np.array([inter_by_class[c] for c in classes], dtype=np.float32)

    total = coef_sparse.shape[0] * coef_sparse.shape[1]
    nnz = coef_sparse.nnz
    print(f"疎性: {(total - nnz) / total:.1%}")
    print(f"推定サイズ: {coef_sparse.data.nbytes / 1024**2:.1f}MB")
    print("💾 [読みモデル] 学習完了")

    # ==========================================
    # 境界モデル: SGDClassifier
    # ==========================================
    print("\n🏋️  [境界モデル] ベクトル化中...")
    vect_boundary = DictVectorizer(sparse=True)
    X_boundary = vect_boundary.fit_transform(X_dicts)
    X_boundary.data = X_boundary.data.astype(np.float32, copy=False)  # type: ignore[union-attr]
    X_boundary.indices = X_boundary.indices.astype(np.int32, copy=False)  # type: ignore[union-attr]
    X_boundary.indptr = X_boundary.indptr.astype(np.int32, copy=False)  # type: ignore[union-attr]

    print("🏋️  [境界モデル] 学習中 (SGDClassifier)...")
    model_boundary = SGDClassifier(
        loss="modified_huber",  # predict_proba が使える
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
        base = tsvdata.rsplit(".", 1)[0]
    bundle_name = os.path.basename(base) + "_bundle.pkl"
    zip_path = base + f"_{window}.zip"
    mbm_path = base + f"_{window}.mbm"

    out_dir = os.path.dirname(zip_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    bundle = LRModelBundle(
        vectorizer_read=vect_read,
        coef_read_sparse=coef_sparse,
        intercept_read=intercept_read,
        read_classes=read_classes,
        vectorizer_boundary=vect_boundary,
        model_boundary=model_boundary,
        version_info={},
    )

    version_info = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_bundle": bundle_name,
        "algorithm": "LinearSVC+SGD",
        "window_size": window,
    }
    bundle_buf = io.BytesIO()
    joblib.dump(bundle, bundle_buf, compress=0)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(bundle_name, bundle_buf.getvalue())
        zf.writestr(
            "version_info.json", json.dumps(version_info, ensure_ascii=False, indent=2)
        )

    print(f"\n📦 ZIPパッケージ作成完了: {zip_path}")
    print(f"   ├ {bundle_name}")
    print(f"   └ version_info.json")
    export(zip_path, mbm_path)
    print(f"量子化モデル (MBM) エクスポート完了: {mbm_path}")
