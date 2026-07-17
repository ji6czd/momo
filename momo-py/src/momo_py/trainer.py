import io
import json
import os
import unicodedata
import zipfile
from datetime import datetime, timezone
from importlib import resources as il_resources
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
from .utils import (
    split_on_unescaped_slash,
    CharType,
    get_basic_char_category,
    normalize_one_to_one,
)
from .name_dict import (
    NAME_DICT_FILENAME,
    NAME_FLAG_BEGIN,
    NAME_FLAG_INSIDE,
    NAME_FLAG_OUT,
    build_name_index,
    compute_name_flags,
    load_name_dict,
    name_flag_for_unit,
    parse_name_marks,
)
from .bundle import LRModelBundle, SINGLE_KANJI_DICT_FILENAME
from .exporter import export, export_float

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
            # strip() ではなく rstrip("\n"): 原文列が空白文字の行
            # （ASCII連内のスペース等）の列ズレを防ぐ
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
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


# 読みが原文と異なってよい（=読みを学習する）日本語の文字種。
# これ以外の非ASCII文字（全角記号・全角英数字・丸数字など）は
# 点字変換側でパターン置換する方針のため、読みは原文と同一でなければならない。
_READING_CTYPES = frozenset(
    {
        CharType.HIRAGANA,
        CharType.KATAKANA,
        CharType.KANJI,
        CharType.JAPANESE_NUMERIC,
    }
)


def _check_non_ascii_identity(
    target_chars: str, ctype: str, r_label: str, line_num: int
) -> None:
    """非ASCIIかつ日本語（かな・漢字・漢数字）でない文字の読みが原文と一致するか検査する。

    全角記号などを半角に置き換えた読みはデータ作成時の誤り
    （半角化・パターン置換は点字変換側で行う方針のため）。
    """
    if ctype in _READING_CTYPES or ctype in (CharType.SPACE, CharType.SKIP):
        return
    if all(ord(c) <= 0x7F for c in target_chars):
        return
    clean_label = r_label.replace("+S", "")
    if clean_label in (LABEL_SKIP, LABEL_CONTINUE, target_chars):
        return
    # 単一数字のかな読み（"３"→"ミッ" 等）は半角数字（"3"→"ミッ"）と同様に許容する
    if ctype == CharType.NUMERIC and clean_label and all(
        get_basic_char_category(c) == CharType.KATAKANA for c in clean_label
    ):
        return
    print(
        f"🚨 警告 (Line {line_num}): 非ASCII文字 '{target_chars}' の読みが "
        f"'{clean_label}' になっています。\n"
        f" -> 半角化などの置換は点字変換側で行うため、読みには原文と同じ文字を書いてください。"
    )


def _check_japanese_reading(
    target_chars: str, ctype: str, r_label: str, line_num: int
) -> None:
    """日本語（かな・漢字・漢数字）ユニットの読みに半角文字が混入していないか検査する。

    これらの読みはカタカナ（"三"→"ミッ"）または全角数字（"三"→"３"）で書く。
    半角数字（"三"→"3"）は全角→半角変換を学習してしまうためデータ作成時の誤り。
    ラベル内の半角スペース（"ワ/ ジン" のような '/' 抜け）もここで検出される。
    """
    if ctype not in _READING_CTYPES:
        return
    clean_label = r_label.replace("+S", "")
    if clean_label in (LABEL_SKIP, LABEL_CONTINUE):
        return
    if any(ord(c) <= 0x7F for c in clean_label):
        print(
            f"🚨 警告 (Line {line_num}): '{target_chars}' の読み "
            f"'{clean_label}' に半角文字が含まれています。\n"
            f" -> 日本語の読みはカタカナまたは全角数字で書いてください。"
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

    _check_non_ascii_identity(target_chars, ctype, r_label, line_num)
    _check_japanese_reading(target_chars, ctype, r_label, line_num)


# ==========================================
# 🌟 3. TSV行の生成（フォーマッタ）
# ==========================================
def _expand_name_flag(name_flag: str, i: int) -> str:
    """ユニットを複数行に展開するとき、i 文字目のフラグを返す（先頭以外は I）。"""
    if i == 0:
        return name_flag
    return NAME_FLAG_INSIDE if name_flag != NAME_FLAG_OUT else NAME_FLAG_OUT


def _create_tsv_rows(
    target_chars: str,
    ctype: str,
    r_label: str,
    orig_idx: int,
    name_flag: str = NAME_FLAG_OUT,
) -> List[str]:
    """1ブロック分の文字列とラベルから、TSV行リストを生成する"""
    if ctype in _SKIP_CTYPES:
        return [
            f"{char}\t{LABEL_SKIP}\t{ctype}\t{orig_idx + i}\t{_expand_name_flag(name_flag, i)}"
            for i, char in enumerate(target_chars)
        ]
    # 数字（NUMERIC）は1文字ずつ学習する（推論側も1文字に展開するため整合させる）。
    #   単一数字 : 注釈をそのまま採用（恒等 "3"→"3"、またはかな "3"→"ミッ"＝みっか）
    #   多桁     : 桁ごとに恒等（"120"→ 1,2,0）。多桁のかな読み（はつか等）は扱わず
    #              漢数字（二十日）で表現する。
    if ctype == CharType.NUMERIC:
        return [f"{target_chars}\t{r_label}\t{ctype}\t{orig_idx}\t{name_flag}"]
    # 拗音（ひらがな/カタカナ複合ユニット）は1行にまとめる。
    # 推論時も get_units() が同じユニットとして認識するため整合が取れる。
    # 漢字などは LABEL_CONTINUE で1文字ずつに分割する（推論は1文字単位のため）。
    if len(target_chars) == 1 or ctype in (CharType.HIRAGANA, CharType.KATAKANA):
        return [f"{target_chars}\t{r_label}\t{ctype}\t{orig_idx}\t{name_flag}"]
    rows = []
    for i, char in enumerate(target_chars):
        r_val = r_label if i == 0 else LABEL_CONTINUE
        rows.append(
            f"{char}\t{r_val}\t{ctype}\t{orig_idx + i}\t{_expand_name_flag(name_flag, i)}"
        )
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

    # {…} 人名マークを除去し、人名スパン（除去後座標）を得る。
    # マークは読み列には書かないので、除去後の原文と読みの1:1整列は保たれる。
    try:
        raw_part, name_spans = parse_name_marks(raw_part)
    except ValueError as e:
        raise ValueError(f"(Line {line_num}): {e}") from None

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

        try:
            name_flag = name_flag_for_unit(orig_idx, len(target_chars), name_spans)
        except ValueError as e:
            raise ValueError(f"(Line {line_num}): {e}") from None
        # 学習データは推論入力の正規化と揃える: 全角英数字は半角へ畳んで出力する
        # （検証は上で注釈の原文に対して実施済み。注釈の書き方は変えない）。
        rows = _create_tsv_rows(
            normalize_one_to_one(target_chars),
            ctype,
            normalize_one_to_one(r_label),
            orig_idx,
            name_flag,
        )
        tsv_rows.extend(rows)
        raw_ptr += 1

    remaining = [u for u in raw_units_info[raw_ptr:] if not u[0].isspace()]
    if remaining:
        raise ValueError(
            f"(Line {line_num}): 原文余り。\n -> 原文 '{remaining[0][0]}' 以降に対応する読みラベルがありません！"
        )

    return tsv_rows


# ==========================================
# 🌟 4.5 人名エントリ抽出（辞書構築用）
# ==========================================
def _extract_name_entries(rows: List[str]) -> List[tuple]:
    """TSV行リストから (人名表層形, ユニット別読み '/'区切り) を抽出する。

    人名フラグ列（B/I）の連続を1スパンとし、読みは +S を除去した
    ラベル列（LABEL_CONTINUE は直前ユニットの継続なのでスキップ）。
    """
    entries: List[tuple] = []
    surface = ""
    blocks: List[str] = []

    def flush() -> None:
        nonlocal surface, blocks
        if surface:
            entries.append((surface, "/".join(blocks)))
        surface, blocks = "", []

    for row in rows:
        cols = row.split("\t")
        flag = cols[4] if len(cols) > 4 else NAME_FLAG_OUT
        if flag == NAME_FLAG_BEGIN:
            flush()
        if flag in (NAME_FLAG_BEGIN, NAME_FLAG_INSIDE):
            surface += cols[0]
            clean = cols[1].replace("+S", "")
            if clean != LABEL_CONTINUE:
                blocks.append(clean)
        else:
            flush()
    flush()
    return entries


# ==========================================
# 🌟 5. メインルーチン
# ==========================================
def create_data(
    rawdata: str, tsvdata: str, name_dict_path: str | None = None
) -> None:
    print(f"📊 過去の実績 ({tsvdata}) から統計辞書を構築中...")
    stats = build_stats_from_tsv(tsvdata)

    with open(rawdata, "r", encoding="utf-8") as f:
        lines = f.readlines()

    all_tsv = ["#原文\t読み\t文字種\tOrigIdx\t人名"]
    success = 0
    # (表層形, 読み) → 出現回数
    name_counts: dict[tuple, int] = defaultdict(int)
    for i, line in enumerate(lines, 1):
        if not line.strip() or line.startswith("#"):
            continue
        rows = process_line_to_tsv(line, i, stats)
        if rows:
            all_tsv.extend(rows)
            all_tsv.append("")
            success += 1
            # {…} マークから人名と読みを収集（辞書構築用）
            for surface, reading in _extract_name_entries(rows):
                name_counts[(surface, reading)] += 1

    with open(tsvdata, "w", encoding="utf-8") as f:
        f.write("\n".join(all_tsv))
    print(f"✅ TSV作成完了 ({success}行): {tsvdata}")

    # 人名辞書の書き出し。
    # フラグ特徴量は学習・推論ともこの辞書へのマッチで計算する。
    # 読みは推論時の低自信度フォールバック（固定読み）に使う。
    if name_counts:
        if name_dict_path is None:
            name_dict_path = os.path.join(
                os.path.dirname(tsvdata) or ".", NAME_DICT_FILENAME
            )
        # 表層形ごとに読みを集計し、最頻の読みを採用する
        by_surface: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for (surface, reading), count in name_counts.items():
            by_surface[surface][reading] += count

        dict_lines = [
            "# 自動生成: 学習データの {…} 人名マークから抽出（createdata 実行時に上書きされる）",
            "#表層形\t読み\t出現回数",
        ]
        for surface in sorted(by_surface):
            readings = by_surface[surface]
            total = sum(readings.values())
            reading = max(readings, key=lambda r: readings[r])
            if len(readings) > 1:
                others = ", ".join(
                    f"{r}×{c}"
                    for r, c in sorted(readings.items(), key=lambda x: -x[1])
                )
                print(
                    f"⚠️  人名 {surface!r} に複数の読みがあります（{others}）"
                    f" → 最頻の {reading!r} を採用します。誤記でないか確認してください。"
                )
            blocks = reading.split("/")
            if reading and LABEL_SKIP not in blocks and len(blocks) == len(
                get_units(surface)
            ):
                dict_lines.append(f"{surface}\t{reading}\t{total}")
            else:
                # 読みがユニットと整合しない場合は表層形のみ登録（フラグ特徴量には使える）
                print(
                    f"⚠️  人名 {surface!r} の読み {reading!r} がユニット数と一致しない"
                    "ため、読みなしで登録します。"
                )
                dict_lines.append(f"{surface}\t{total}")
        with open(name_dict_path, "w", encoding="utf-8") as f:
            f.write("\n".join(dict_lines) + "\n")
        print(f"👤 人名辞書作成完了 ({len(by_surface)} エントリ): {name_dict_path}")


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
# 🌟 6.2 学習用TSVの読み込み
# ==========================================
def _load_sentences(tsvdata: str) -> List[List[List[str]]]:
    """学習用TSVを文（空行/コメント行区切り）ごとの行リストに読み込む。

    注意: 行を strip() してはならない。原文列が空白文字の行
    （ASCII連内のスペース等、例 " \\t_\\tALPHA\\t16\\tO"）で先頭列が
    消えて列がずれ、文字種列に OrigIdx が入った状態で学習されてしまう。
    """
    sentences: List[List[List[str]]] = []
    current: List[List[str]] = []
    with open(tsvdata, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line.strip():
                if current:
                    sentences.append(current)
                current = []
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                current.append(parts)
    if current:
        sentences.append(current)
    return sentences


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
    name_dict: str | None = None,
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

    人名特徴量:
        name_dict に人名辞書のパスを指定すると（None のときは tsvdata と同じ
        ディレクトリの person_name_dic.tsv を自動検出）、推論時と同一の辞書マッチで
        B/I フラグ特徴量を付与して学習する。TSVの人名列（正解マーク）は使わない。

    出力ファイル:
        basename_bundle.pkl  - モデル一式（joblib）
        basename.zip         - 上記をまとめたパッケージ
        basename.mbm         - C++/Rust 向け量子化バイナリ（推論用）
        basename.mbmf        - 量子化前 float32 バイナリ（.mbm との比較用サイドカー）
    """
    name_index: dict = {}
    name_dict_entries = 0
    name_dict_path = name_dict
    if name_dict_path is None:
        candidate = os.path.join(os.path.dirname(tsvdata) or ".", NAME_DICT_FILENAME)
        if os.path.isfile(candidate):
            name_dict_path = candidate
    if name_dict_path:
        names = load_name_dict(name_dict_path)
        name_index = build_name_index(names)
        name_dict_entries = len(names)
        print(f"👤 人名辞書: {name_dict_entries} エントリ ({name_dict_path})")

    sentences = _load_sentences(tsvdata)

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

        name_flags = compute_name_flags(source_seq, name_index) if name_index else None
        features = compute_source_features(
            source_seq, window=window, name_flags=name_flags
        )
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
    mbmf_path = base + f"_{window}.mbmf"

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
        "name_dict_entries": name_dict_entries,
    }
    bundle_buf = io.BytesIO()
    joblib.dump(bundle, bundle_buf, compress=0)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(bundle_name, bundle_buf.getvalue())
        zf.writestr(
            "version_info.json", json.dumps(version_info, ensure_ascii=False, indent=2)
        )
        # 学習に使った人名辞書をそのまま同梱する。
        # 推論（Python の Predictor / .mbm 経由の Rust）はこの同梱辞書を使うことで、
        # 学習時と完全に同一の辞書マッチが保証される。
        if name_dict_path and name_dict_entries:
            with open(name_dict_path, encoding="utf-8") as f:
                zf.writestr(NAME_DICT_FILENAME, f.read())
        # 単一漢字辞書も同梱する。読みモデルの候補制約として必須のデータであり、
        # モデルとペアで配布することで配置ミスによるサイレント劣化を防ぐ。
        # TSVと同じディレクトリの辞書を優先し、なければパッケージ内蔵を使う。
        single_dict_candidate = os.path.join(
            os.path.dirname(tsvdata) or ".", SINGLE_KANJI_DICT_FILENAME
        )
        if os.path.isfile(single_dict_candidate):
            with open(single_dict_candidate, encoding="utf-8") as f:
                single_dict_text = f.read()
            print(f"📖 単一漢字辞書を同梱: {single_dict_candidate}")
        else:
            single_dict_text = (
                il_resources.files("momo_py") / f"resources/{SINGLE_KANJI_DICT_FILENAME}"
            ).read_text(encoding="utf-8")
            print("📖 単一漢字辞書を同梱: パッケージ内蔵リソース")
        zf.writestr(SINGLE_KANJI_DICT_FILENAME, single_dict_text)

    print(f"\n📦 ZIPパッケージ作成完了: {zip_path}")
    print(f"   ├ {bundle_name}")
    if name_dict_path and name_dict_entries:
        print(f"   ├ {NAME_DICT_FILENAME}")
    print(f"   ├ {SINGLE_KANJI_DICT_FILENAME}")
    print(f"   └ version_info.json")
    export(zip_path, mbm_path)
    print(f"量子化モデル (MBM) エクスポート完了: {mbm_path}")

    # 量子化前の float32 サイドカーも一緒に書き出す（.mbm との比較用）。
    export_float(zip_path, mbmf_path)
    print(f"非量子化モデル (MBMF) エクスポート完了: {mbmf_path}")
