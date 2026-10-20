"""読み分けの評価データ dataset/eval_context.tsv の作成と検査。

同じ語が文脈によって違う読みになる箇所（人気=ニンキ/ヒトケ、行った=イッタ/オコナッタ、
清水=シミズ/キヨミズ）だけを集めた評価用のデータ。モデルが「語の読みを知っているか」ではなく
「文脈で読みを選べるか」を測る。LLM の素の評価にも、エンコーダを作ってからの評価にも使う。

形式（タブ区切り・UTF-8・# で始まる行はコメント）:

    id       語-通し番号（一意）。例: 人気-1
    word     対象の語。text の中にそのままの文字列で現れる（漢字の列、または漢字＋送り仮名）
    expected word の読み。momo の表記（点字用カタカナ・長音は ー・促音は ッ）
    occ      text の中で word が複数回現れるとき、何番目か（空欄は 1）
    text     対象の文（1文）
    context  同じ文書の前の文（最大3文、空欄＝文脈なし）。読みを決める手がかりが前の文にあるものは必須
    kind     読みが分かれる理由（下の KINDS）
    cue      決め手がどこにあるか（下の CUES、空欄＝未判定）
    split    この文が学習に使われているか（下の SPLITS）
    note     なぜその読みになるか（一言）。迷ったときに後から見返すためのもの

同じ word の行が2つ以上の読み（expected）を持つこと。1つの語について読みごとに用例を複数そろえ、
語ごとに「全部の読みを当てられたか」を測る（いつも多数派の読みを選ぶだけでは全問正解にならない）。

KINDS（読みが分かれる理由）
    meaning     語義・用法の違い（人気、生物、上手、行った）
    name        人名・地名・固有名（清水、源）
    counter     数字に続く助数詞・単位（日、月、分、年）
    sound       複合語の中での音の変化だけ（連濁・促音・半濁音: 菓子/ガシ、日本/ニッポン）
    historical  歴史的・古典的な読み
    idiom       慣用句・成句の中の読み（一枚上手=ウワテ）

CUES（決め手の所在）
    sentence    同じ文の中にある
    context     前の文にある（context 列が必要）
    world       固有名や事実の知識がないと決まらない
    none        文脈では決まらない（点訳者が調べる）。解けない問題として別に数える

SPLITS（学習データとの関係）
    dev    basic_raw.tsv の dev（原文ハッシュ mod 10 == 0）の文。実験用モデルの学習に入っていない
    train  basic_raw.tsv の dev 以外の文。momo の学習に使われている（LLM の素の評価にだけ使える）
    new    評価のために新しく書いた文。basic_raw.tsv にも学習データにも入れない
           （評価に使い終えて学習に回すときは、行を消して basic_raw.tsv に移す）

使い方:

    uv run python 09_eval_context.py bootstrap   # 今の basic_raw.tsv から初期データを作る（上書きしない）
    uv run python 09_eval_context.py check       # 形式と整合を検査し、件数をまとめる

bootstrap は一度だけ使う。人が手を入れた後に作り直さないよう、既存のファイルがあれば止まる。
"""

from __future__ import annotations

import argparse
import collections
import importlib
import os
import re
import sys
import unicodedata

from _common import DATASET, RAW_PATH, parse_gold, read_raw_numbered
from momo_py.name_dict import parse_name_marks

EVAL_PATH = os.path.join(DATASET, "eval_context.tsv")
COLUMNS = [
    "id",
    "word",
    "expected",
    "occ",
    "text",
    "context",
    "kind",
    "cue",
    "split",
    "note",
]
KINDS = ("meaning", "name", "counter", "sound", "historical", "idiom")
CUES = ("sentence", "context", "world", "none")
SPLITS = ("dev", "train", "new")

PER_READING = 3  # bootstrap: 語×読みごとに取る用例の上限
MAX_CONTEXT = 3  # 前の文を何文まで添えるか
OKURI = 2  # 漢字1字の語に添える後ろのひらがなの最大文字数（送り仮名・助詞）
NUMERIC = frozenset({"NUM", "JAPANESE_NUMERIC"})
SENTENCE_END = re.compile(r"[。！？!?]+[」』）)]*")
READING_OK = re.compile(r"^[ァ-ヶーA-Za-z0-9ー]+$")  # 全角カタカナ・長音・英数字


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in SENTENCE_END.finditer(text):
        spans.append((start, m.end()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def strip_sound(reading: str) -> str:
    """濁点・半濁点と促音・長音を除く。これで同じになる読みの差は音の変化だけとみなす。"""
    s = "".join(
        c
        for c in unicodedata.normalize("NFD", reading)
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"[ッー]", "", s)


def candidates(line_no: int, line: str, is_dev):
    """1行から (語, 読み, 文, 前の文, 種別, 分割, occ) の候補を取り出す。"""
    g = parse_gold(line, line_no)
    raw_part = line.split("\t")[0]
    clean, name_spans = parse_name_marks(raw_part)
    # 名前スパンを g.text（[...] 記法も除いた座標）へ写す。clean と g.text の差は [] だけ
    drop = [m.start() for m in re.finditer(r"[\[\]]", clean)]
    name_in_text = [
        (a - sum(d < a for d in drop), b - sum(d < b for d in drop))
        for a, b in name_spans
    ]
    sents = sentence_spans(g.text)
    dev = is_dev(g.text)
    slots = g.slots
    out = []
    i = 0
    while i < len(slots):
        if slots[i].ctype != "KANJI":
            i += 1
            continue
        j = i
        while (
            j < len(slots)
            and slots[j].ctype == "KANJI"
            and slots[j].start
            == slots[i].start + sum(len(s.surface) for s in slots[i:j])
        ):
            j += 1
        run_end = j
        # 直前の数字（一日・3日）も語に含める。含めないと「一人の」の「人の」が別の読みの
        # 同じ語に見えてしまう
        head = i
        while (
            head > 0
            and slots[head - 1].ctype in NUMERIC
            and slots[head - 1].end == slots[head].start
        ):
            head -= 1
        # 漢字1字は送り仮名・助詞まで含めて1語とみなす（行った・入って）
        if j - i == 1:
            extra = 0
            while (
                j < len(slots)
                and slots[j].ctype == "HIRAGANA"
                and extra + len(slots[j].surface) <= OKURI
                and slots[j].start == slots[j - 1].end
            ):
                extra += len(slots[j].surface)
                j += 1
        span = slots[head:j]
        a, b = span[0].start, span[-1].end
        word = g.text[a:b]
        reading = "".join(s.gold for s in span)
        # 読みを持たない継続スロットだけの語や、全角カタカナ以外が混じる読みは対象外
        if READING_OK.match(reading):
            si = next(k for k, (x, y) in enumerate(sents) if x <= a < y)
            sa, sb = sents[si]
            before = g.text[sa:a]
            occ = before.count(word) + 1
            if any(x < b and a < y for x, y in name_in_text):
                kind = "name"
            elif head < i:
                kind = "counter"
            else:
                kind = "meaning"
            context = "".join(
                g.text[x:y] for x, y in sents[max(0, si - MAX_CONTEXT) : si]
            )
            out.append(
                (
                    word,
                    reading,
                    g.text[sa:sb],
                    context,
                    kind,
                    "dev" if dev else "train",
                    occ,
                )
            )
        i = run_end if run_end > i else i + 1
    return out


def bootstrap() -> None:
    if os.path.exists(EVAL_PATH):
        sys.exit(
            f"{EVAL_PATH} がすでにある。手を入れたデータを壊さないよう、作り直さない。"
        )
    is_dev = importlib.import_module("01_split").is_dev
    groups: dict[str, dict[str, list]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for line_no, line in read_raw_numbered(RAW_PATH):
        try:
            found = candidates(line_no, line, lambda t: is_dev(t, 10))
        except ValueError:
            continue
        for word, reading, text, context, kind, split, occ in found:
            groups[word][reading].append((line_no, text, context, kind, split, occ))

    rows = []
    for word, readings in groups.items():
        if len(readings) < 2:
            continue
        # 多数派の読みと音の変化だけの差なら sound（複合語の中の連濁・促音）
        major = max(readings, key=lambda r: len(readings[r]))
        picked = {}
        for reading, items in readings.items():
            seen, chosen = set(), []
            # dev を先に、同じなら短い文を先に（読みの手がかりが少ない文は決め手がはっきりしている）
            for it in sorted(items, key=lambda t: (t[4] != "dev", len(t[1]), t[0])):
                if (it[1], it[5]) in seen:
                    continue
                seen.add((it[1], it[5]))
                chosen.append(it)
                if len(chosen) == PER_READING:
                    break
            picked[reading] = chosen
        n = 0
        for reading in sorted(picked, key=lambda r: -len(readings[r])):
            for _, text, context, kind, split, occ in picked[reading]:
                n += 1
                rows.append(
                    [
                        f"{word}-{n}",
                        word,
                        reading,
                        "" if occ == 1 else str(occ),
                        text,
                        context,
                        "sound"
                        if kind == "meaning"
                        and reading != major
                        and strip_sound(reading) == strip_sound(major)
                        else kind,
                        "",
                        split,
                        "",
                    ]
                )
    rows.sort(key=lambda r: (r[1], r[0]))
    with open(EVAL_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(
            "# 読み分けの評価データ。形式は tools/llm_reading/09_eval_context.py の冒頭を参照。\n"
        )
        f.write(
            "# bootstrap で basic_raw.tsv から取り出した初期データ。kind は自動判定なので見直す。cue は未判定（空欄）。\n"
        )
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"{len(rows)} 行を {EVAL_PATH} に書いた")


def load_rows() -> list[dict[str, str]]:
    with open(EVAL_PATH, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#")]
    header = lines[0].split("\t")
    if header != COLUMNS:
        sys.exit(f"列が違う: {header}")
    rows = []
    for n, ln in enumerate(lines[1:], 1):
        cells = ln.split("\t")
        if len(cells) != len(COLUMNS):
            sys.exit(
                f"{n} 行目（データ部）: 列数が {len(cells)}（{len(COLUMNS)} のはず）: {ln[:40]}"
            )
        rows.append(dict(zip(COLUMNS, cells)))
    return rows


def check() -> None:
    rows = load_rows()
    raw_texts = [
        line.split("\t")[0].translate(str.maketrans("", "", "{}[]"))
        for _, line in read_raw_numbered(RAW_PATH)
    ]
    problems = []
    seen_ids = set()
    by_word = collections.defaultdict(set)
    for r in rows:
        rid = r["id"]
        if rid in seen_ids:
            problems.append(f"{rid}: id が重複")
        seen_ids.add(rid)
        occ = int(r["occ"] or 1)
        if r["text"].count(r["word"]) < occ:
            problems.append(f"{rid}: word「{r['word']}」が text に {occ} 回現れない")
        if not READING_OK.match(r["expected"]):
            problems.append(
                f"{rid}: expected「{r['expected']}」が全角カタカナ・ー・英数字でない"
            )
        if r["kind"] not in KINDS:
            problems.append(f"{rid}: kind「{r['kind']}」")
        if r["cue"] and r["cue"] not in CUES:
            problems.append(f"{rid}: cue「{r['cue']}」")
        if r["cue"] == "context" and not r["context"]:
            problems.append(f"{rid}: cue が context なのに context が空")
        if r["split"] not in SPLITS:
            problems.append(f"{rid}: split「{r['split']}」")
        in_raw = any(r["text"] in t for t in raw_texts)
        if r["split"] == "new" and in_raw:
            problems.append(
                f"{rid}: split が new なのに text が basic_raw.tsv にある（評価と学習が混ざる）"
            )
        if r["split"] in ("dev", "train") and not in_raw:
            problems.append(
                f"{rid}: split が {r['split']} なのに text が basic_raw.tsv にない"
            )
        by_word[r["word"]].add(r["expected"])
    for word, readings in sorted(by_word.items()):
        if len(readings) < 2:
            problems.append(f"{word}: 読みが1つしかない（読み分けにならない）")

    print(f"{len(rows)} 行・{len(by_word)} 語")
    for col in ("kind", "split", "cue"):
        c = collections.Counter(r[col] or "（空）" for r in rows)
        print(f"  {col}: " + "・".join(f"{k} {v}" for k, v in c.most_common()))
    if problems:
        print(f"\n問題 {len(problems)} 件:")
        for p in problems[:50]:
            print("  " + p)
        sys.exit(1)
    print("問題なし")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["bootstrap", "check"])
    args = ap.parse_args()
    {"bootstrap": bootstrap, "check": check}[args.command]()


if __name__ == "__main__":
    main()
