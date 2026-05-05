#!/usr/bin/env python3
"""
analyze_coverage.py

学習データ (basic_raw.tsv) と常用漢字表 JSON を突き合わせ、
カバレッジを分析するスタンドアロンスクリプト。

使い方:
    python analyze_coverage.py \
        --raw /path/to/basic_raw.tsv \
        --joyo /path/to/joyo_kanji.json \
        --appendix /path/to/joyo_kanji_appendix.json \
        [--out report.txt]

将来的には momo analyze サブコマンドへ取り込む想定。
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
HIRAGANA_RE = re.compile(r'[\u3041-\u309F]')
KANJI_RE    = re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF]')


# ---------------------------------------------------------------------------
# 入力ロード
# ---------------------------------------------------------------------------
def load_raw_tsv(path: Path) -> List[str]:
    """basic_raw.tsv から原文（タブ前）のみを抽出する。"""
    sentences: List[str] = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            sentences.append(parts[0])
    return sentences


def load_joyo(path: Path) -> List[dict]:
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def load_appendix(path: Path) -> List[dict]:
    with path.open(encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 共通ユーティリティ
# ---------------------------------------------------------------------------
def get_kanji_char(entry: dict) -> str:
    """joyo_kanji.json の1エントリから通用字体の漢字を取り出す。"""
    k = entry.get('漢字', {})
    if isinstance(k, dict):
        return k.get('通用字体', '')
    return k or ''


def normalize_example(ex: str) -> str:
    """
    用例文字列を「漢字部分 + 直後のひらがな送り仮名」の形に正規化する。
    例:
      "縦"           -> "縦"
      "縦書き"       -> "縦書き"
      "生まれる"     -> "生まれる"
      "生の野菜"     -> "生" (ひらがな以降は送り仮名扱いだが「の野菜」は別語と見なし切る)

    今回の目的では、原文に含まれているかどうかを substring match で見るので
    そのまま使ってよい。
    """
    return ex.strip()


def collect_all_examples(joyo_data: List[dict]) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    全用例を平らに集めると同時に、漢字 -> [読みごとのバリエーション] という
    マッピングも作る。

    Returns:
        all_examples: 全用例（重複除去）
        kanji_to_readings: { 漢字: [(読み, [例文,...]), ...] }
    """
    seen: Set[str] = set()
    all_examples: List[str] = []
    kanji_to_readings: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)

    for entry in joyo_data:
        kanji = get_kanji_char(entry)
        if not kanji:
            continue
        for ot in entry.get('音訓', []):
            yomi = ot.get('読み', '')
            ex_list = [normalize_example(e) for e in ot.get('例', [])]
            kanji_to_readings[kanji].append((yomi, ex_list))
            for ex in ex_list:
                if ex and ex not in seen:
                    seen.add(ex)
                    all_examples.append(ex)
    return all_examples, dict(kanji_to_readings)


# ---------------------------------------------------------------------------
# 分析: 用例カバレッジ
# ---------------------------------------------------------------------------
def analyze_example_coverage(
    examples: List[str],
    sentences: List[str],
) -> Tuple[List[str], List[str]]:
    """
    各用例が学習データの原文中に substring として出現するかを判定する。
    Returns: (covered, uncovered)
    """
    full_text = '\n'.join(sentences)
    covered, uncovered = [], []
    for ex in examples:
        if ex and ex in full_text:
            covered.append(ex)
        else:
            uncovered.append(ex)
    return covered, uncovered


# ---------------------------------------------------------------------------
# 分析: 漢字出現回数
# ---------------------------------------------------------------------------
def count_kanji(sentences: List[str]) -> Counter:
    counts: Counter = Counter()
    for s in sentences:
        for c in s:
            if KANJI_RE.match(c):
                counts[c] += 1
    return counts


def analyze_kanji_coverage(
    kanji_to_readings: Dict[str, List[Tuple[str, List[str]]]],
    counts: Counter,
) -> Dict[str, int]:
    """常用漢字ごとの出現回数（学習データに無ければ0）"""
    return {k: counts.get(k, 0) for k in kanji_to_readings}


# ---------------------------------------------------------------------------
# 分析: 読みバリエーションの充足度
# ---------------------------------------------------------------------------
def analyze_reading_variety(
    kanji_to_readings: Dict[str, List[Tuple[str, List[str]]]],
    sentences: List[str],
) -> List[Tuple[str, List[Tuple[str, bool]]]]:
    """
    各漢字について、JSONに登録された読みバリエーションのうち、
    対応する用例がいずれか1つでも学習データに出現しているかを判定する。

    Returns:
        [(漢字, [(読み, 出現したか), ...]), ...]
    """
    full_text = '\n'.join(sentences)
    result = []
    for k, readings in kanji_to_readings.items():
        per_reading = []
        for yomi, ex_list in readings:
            hit = any(ex and ex in full_text for ex in ex_list)
            per_reading.append((yomi, hit))
        result.append((k, per_reading))
    return result


# ---------------------------------------------------------------------------
# レポート整形
# ---------------------------------------------------------------------------
def format_report(
    sentences: List[str],
    examples: List[str],
    covered: List[str],
    uncovered: List[str],
    kanji_to_readings: Dict[str, List[Tuple[str, List[str]]]],
    kanji_counts: Dict[str, int],
    reading_variety: List[Tuple[str, List[Tuple[str, bool]]]],
    appendix: List[dict],
    appendix_covered: List[str],
    appendix_uncovered: List[str],
) -> str:
    lines: List[str] = []
    P = lines.append

    total_chars = sum(len(s) for s in sentences)
    P("=" * 70)
    P("📊 常用漢字熟語カバレッジ分析")
    P("=" * 70)
    P(f"学習データ文数: {len(sentences):>6,} 文")
    P(f"学習データ字数: {total_chars:>6,} 字")
    P("")

    # --- 用例カバレッジ ---
    total_ex = len(examples)
    cov_ex   = len(covered)
    P("【常用漢字 用例カバレッジ】")
    P(f"  ユニーク用例数: {total_ex:>6,}")
    P(f"  学習データに出現: {cov_ex:>6,} ({cov_ex/total_ex*100:>5.1f}%)")
    P(f"  未出現:           {len(uncovered):>6,} ({len(uncovered)/total_ex*100:>5.1f}%)")
    P("")

    # --- 漢字出現回数の分布 ---
    counts_sorted = sorted(kanji_counts.values())
    P("【常用漢字 出現回数分布】")
    P(f"  対象漢字数: {len(kanji_counts):,} 字")
    bins = [(0, '未出現'), (1, '1回'), (2, '2回'), (3, '3-4回'),
            (5, '5-9回'), (10, '10-29回'), (30, '30-99回'), (100, '100回以上')]
    bin_counts = [0] * len(bins)
    for c in counts_sorted:
        if   c == 0:      bin_counts[0] += 1
        elif c == 1:      bin_counts[1] += 1
        elif c == 2:      bin_counts[2] += 1
        elif c <= 4:      bin_counts[3] += 1
        elif c <= 9:      bin_counts[4] += 1
        elif c <= 29:     bin_counts[5] += 1
        elif c <= 99:     bin_counts[6] += 1
        else:             bin_counts[7] += 1
    for (_, label), cnt in zip(bins, bin_counts):
        bar = '█' * int(cnt / 30)
        P(f"  {label:>10}: {cnt:>4} 字 {bar}")
    P("")

    # --- 出現が極端に少ない漢字（弱点候補） ---
    weak = sorted(
        ((k, c) for k, c in kanji_counts.items() if c <= 2),
        key=lambda x: (x[1], x[0])
    )
    P(f"【出現2回以下の常用漢字】 {len(weak)} 字")
    if weak:
        # 0回 / 1回 / 2回 で分類
        for threshold, label in [(0, '未出現'), (1, '出現1回'), (2, '出現2回')]:
            chars = [k for k, c in weak if c == threshold]
            if chars:
                P(f"  {label} ({len(chars)} 字):")
                # 50字ずつ折り返し
                for i in range(0, len(chars), 50):
                    P("    " + " ".join(chars[i:i+50]))
    P("")

    # --- 読みバリエーション充足度 ---
    incomplete = []
    for k, per in reading_variety:
        missing = [y for y, hit in per if not hit]
        total   = len(per)
        if missing and total >= 2:
            incomplete.append((k, total, missing, per))
    incomplete.sort(key=lambda x: (-(len(x[2])/x[1]), -x[1], x[0]))

    P(f"【読みバリエーションが手薄な漢字】 上位 30 字")
    P(f"  (登録読み数 ≥ 2 で 1つ以上の読みが未学習のもの)")
    P(f"  全体: {len(incomplete)} 字 / 登録読み数≥2 漢字 中")
    P("")
    for k, total, missing, per in incomplete[:30]:
        readings_disp = ", ".join(
            f"{y}{'✗' if not hit else '○'}" for y, hit in per
        )
        cnt = kanji_counts.get(k, 0)
        P(f"  {k} (出現{cnt:>3}回, {len(missing)}/{total}読み未学習): {readings_disp}")
    P("")

    # --- 未カバー用例（一部抜粋） ---
    P(f"【未カバー用例（学習データに無い熟語）】 {len(uncovered)} 件")
    P("  抜粋（先頭30件）:")
    for ex in uncovered[:30]:
        P(f"    {ex}")
    if len(uncovered) > 30:
        P(f"    ... 残り {len(uncovered) - 30} 件")
    P("")

    # --- 付表（熟字訓） ---
    P("【常用漢字付表（熟字訓）カバレッジ】")
    total_app   = len(appendix)
    cov_app_cnt = len(appendix_covered)
    if total_app == 0:
        P("  (付表ファイルが指定されていないためスキップ)")
    else:
        P(f"  付表エントリ数: {total_app}")
        P(f"  学習データに出現: {cov_app_cnt} ({cov_app_cnt/total_app*100:>5.1f}%)")
        P(f"  未出現: {len(appendix_uncovered)}")
        if appendix_uncovered:
            P("  未出現の付表語（抜粋）:")
            for u in appendix_uncovered[:30]:
                P(f"    {u}")
            if len(appendix_uncovered) > 30:
                P(f"    ... 残り {len(appendix_uncovered) - 30} 件")
    P("")
    P("=" * 70)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 未カバー熟語の優先度付きソート
# ---------------------------------------------------------------------------
def sort_uncovered_by_priority(
    uncovered: List[str],
    kanji_counts: Dict[str, int],
) -> List[Tuple[str, int, str]]:
    """
    未カバー熟語に優先度を付けてソートする。

    優先度の決め方:
      - 熟語に含まれる漢字のうち、学習データでの出現回数が「最少」の漢字
        の出現回数を熟語のスコアとする (min_count)。
      - min_count が小さい熟語ほど、補充効果が高い（弱い漢字が確実に増える）
        ので優先順位が高い。
      - 同点は「最弱漢字」を含む熟語をまとめて扱いたいので、
        (min_count, weakest_kanji, 用例文字列) でソートする。

    Returns:
        [(熟語, min_count, weakest_kanji), ...] を優先度順に並べたもの
    """
    out: List[Tuple[str, int, str]] = []
    for ex in uncovered:
        # ex に含まれる漢字を取り出す（送り仮名や記号は除外）
        kanji_chars = [c for c in ex if KANJI_RE.match(c)]
        if not kanji_chars:
            # 漢字を含まない用例は優先度を最低（=巨大数）にしておく
            out.append((ex, 10**9, ''))
            continue
        # 含まれる漢字のうち、出現回数が最小のものを「最弱漢字」とする
        weakest = min(kanji_chars, key=lambda c: (kanji_counts.get(c, 0), c))
        min_cnt = kanji_counts.get(weakest, 0)
        out.append((ex, min_cnt, weakest))

    out.sort(key=lambda t: (t[1], t[2], t[0]))
    return out


def write_uncovered(
    uncovered: List[str],
    kanji_counts: Dict[str, int],
    path: Path,
    top_n: int | None = None,
    annotate: bool = False,
) -> None:
    """
    未カバー熟語をファイルに書き出す。

    annotate=True のときはコメント形式で「最弱漢字とその出現回数」も付ける。
    例:
        亜流          # 亜:0回
        縦走          # 縦:1回
    """
    sorted_items = sort_uncovered_by_priority(uncovered, kanji_counts)
    if top_n is not None:
        sorted_items = sorted_items[:top_n]

    with path.open('w', encoding='utf-8') as f:
        if annotate:
            # 簡単なヘッダ
            f.write("# 学習データに未出現の常用漢字熟語\n")
            f.write("# 形式: <熟語>\\t# <最弱漢字>:<出現回数>回\n")
            f.write(f"# 件数: {len(sorted_items)}\n")
            f.write("\n")
            for ex, cnt, weakest in sorted_items:
                if weakest:
                    f.write(f"{ex}\t# {weakest}:{cnt}回\n")
                else:
                    f.write(f"{ex}\n")
        else:
            for ex, _, _ in sorted_items:
                f.write(ex + "\n")


# ---------------------------------------------------------------------------
# 付表分析
# ---------------------------------------------------------------------------
def analyze_appendix(
    appendix: List[dict],
    sentences: List[str],
) -> Tuple[List[str], List[str]]:
    full_text = '\n'.join(sentences)
    covered, uncovered = [], []
    for entry in appendix:
        words = entry.get('語', [])
        # どれか1つの語が出現すれば cover とみなす
        hit = any(w in full_text for w in words)
        repr_word = words[0] if words else '?'
        if hit:
            covered.append(repr_word)
        else:
            uncovered.append(repr_word)
    return covered, uncovered


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="学習データの常用漢字カバレッジ分析")
    p.add_argument('--raw',      required=True, type=Path, help='basic_raw.tsv のパス')
    p.add_argument('--joyo',     required=True, type=Path, help='joyo_kanji.json のパス')
    p.add_argument('--appendix', type=Path,                help='joyo_kanji_appendix.json のパス（任意）')
    p.add_argument('--out',      type=Path,                help='レポート出力先（省略時は stdout）')
    # 未カバー熟語のリスト出力
    p.add_argument('--out-uncovered', type=Path, metavar='PATH',
                   help='未カバー熟語の全リストを書き出す（弱い漢字を含む順）')
    p.add_argument('--out-uncovered-top', type=int, metavar='N', default=None,
                   help='--out-uncovered と併用。上位 N 件のみを出す')
    p.add_argument('--annotate', action='store_true',
                   help='--out-uncovered 出力に「最弱漢字とその出現回数」のコメントを付与')
    args = p.parse_args()

    sentences = load_raw_tsv(args.raw)
    joyo      = load_joyo(args.joyo)
    appendix  = load_appendix(args.appendix) if args.appendix else []

    print(f"📥 学習データ: {len(sentences):,} 文 を読み込みました", file=sys.stderr)
    print(f"📥 常用漢字表: {len(joyo):,} 字 を読み込みました", file=sys.stderr)
    if appendix:
        print(f"📥 付表       : {len(appendix):,} 語 を読み込みました", file=sys.stderr)

    examples, kanji_to_readings = collect_all_examples(joyo)
    covered, uncovered           = analyze_example_coverage(examples, sentences)
    kanji_counts                 = analyze_kanji_coverage(kanji_to_readings, count_kanji(sentences))
    reading_variety              = analyze_reading_variety(kanji_to_readings, sentences)
    app_covered, app_uncovered   = analyze_appendix(appendix, sentences) if appendix else ([], [])

    report = format_report(
        sentences, examples, covered, uncovered,
        kanji_to_readings, kanji_counts, reading_variety,
        appendix, app_covered, app_uncovered,
    )

    if args.out:
        args.out.write_text(report, encoding='utf-8')
        print(f"📝 レポートを {args.out} に保存しました", file=sys.stderr)
    else:
        print(report)

    # --- 未カバー熟語の書き出し ---
    if args.out_uncovered:
        write_uncovered(
            uncovered=uncovered,
            kanji_counts=kanji_counts,
            path=args.out_uncovered,
            top_n=args.out_uncovered_top,
            annotate=args.annotate,
        )
        n_out = args.out_uncovered_top if args.out_uncovered_top else len(uncovered)
        print(f"📝 未カバー熟語 {n_out:,} 件を {args.out_uncovered} に保存しました", file=sys.stderr)


if __name__ == '__main__':
    main()
