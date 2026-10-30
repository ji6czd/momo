"""basic_raw.tsv の注釈を、注釈どうしの食い違いから点検する（注釈の点検用）。

学習したモデルは学習データの誤りも覚えるので、モデルと比べるだけでは誤りを見落とす
（step11_audit が出すのは、モデルが覚えきれなかった箇所だけ）。ここでは、同じ字・同じ並び・
同じ前後の文字に付いた注釈どうしを比べて、少数派を出す。多数派が正しいとは限らないし、
正当な読み分け・切れ続けも混じるので、人が判断する。

    uv run audit_raw                                   # 注釈どうしの比較だけ
    uv run audit_raw --w7 ../dataset/basic_data_7.mbm  # w7 のマスあけとの食い違いも出す
    uv run --with sudachipy --with sudachidict-full audit_raw --sudachi   # 品詞の並びでも比べる

出力: work/audit/raw/
  reading_rare.tsv       全体で 1 回しか出ない読みと、よく出る字のめったに出ない読み
  reading_notation.tsv   長音の書き方の揺れ（コウ／コー、セイ／セー、ホオ／ホー）と、
                         「ー」や小さい字で始まる読み（フ/ービ のような区切りのずれ）
  compound_variants.tsv  隣り合う 2 つの漢字に付いた読みとマスあけの揺れ（学校 ガッ/コー と ガク/コー）
  space_variation.tsv    前後 2 文字が同じ切れ目なのに、マスあけの有無が少数派のもの
  space_vs_w7.tsv        w7 のマスあけと注釈の食い違い（--w7 のとき）。w7 は全データで学習した
                         もの（学習済みの文）を渡す。合わせきれなかった＝ほかの注釈と食い違う候補
  space_pos.tsv          Sudachi の品詞の並び（を | 動詞 など）で強い多数派から外れた切れ目（--sudachi）
  space_pos_pairs.tsv    品詞の並びごとのマスあけの割合。多数派そのものの誤り（決まりの取り違え）を見る用
行番号は basic_raw.tsv の行番号（1 始まり）。読みの列は basic_raw.tsv と同じ「/」区切り。
"""

from __future__ import annotations

import argparse
import collections
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .common import WORK
from .export_data import RAW_PATH
from .gold import parse_gold

OUT = WORK / "audit" / "raw"
SPACE_CONTEXTS = (2, 3, 4)  # 切れ目の前後に見る文字数
SPACE_K = 2  # 少数派の判定に使う前後の文字数（3 だと同じ並びがほとんど無い）
SPACE_MIN = 3  # 前後が同じ切れ目がこれだけ無ければ判定しない
POS_MIN = 10  # 品詞の並びがこれだけ無ければ判定しない
POS_MINORITY = 0.1  # 品詞の並びで、少数派がこの割合以下なら出す

O_DAN = set("オコソトノホモヨロゴゾドボポョォヲ")
U_DAN = set("ウクスツヌフムユルグズヅブプュゥ")
E_DAN = set("エケセテネヘメレゲゼデベペェ")
# 漢字の読みの頭に来ない字。来ていれば「/」の位置がずれている（フ/ービ）
BAD_START = set("ーァィゥェォャュョヮッ")


def is_kanji(ch: str) -> bool:
    # step10_infer.is_kanji と同じ（あちらを読むと torch まで読み込むので）
    return ch in "々〇" or "CJK UNIFIED" in unicodedata.name(ch, "")


@dataclass
class Unit:
    """読みの単位。[今日] のような継続はまとめて 1 単位にする。"""

    surface: str
    reading: str
    ctype: str
    start: int
    end: int
    space: bool  # 後ろにマスあけ

    @property
    def kanji(self) -> bool:
        return any(is_kanji(c) for c in self.surface) and self.reading != self.surface


@dataclass
class Line:
    no: int  # basic_raw.tsv の行番号
    text: str
    units: list[Unit]


def load(path: Path) -> list[Line]:
    lines = []
    with open(path, encoding="utf-8") as f:
        for no, raw in enumerate(f, 1):
            raw = raw.rstrip("\n")
            if not raw.strip() or raw.startswith("#"):
                continue
            try:
                g = parse_gold(raw, no)
            except ValueError as e:
                print(f"  読めない行: {e}")
                continue
            units: list[Unit] = []
            for s in g.slots:
                if s.gold == "" and units:  # 継続
                    u = units[-1]
                    u.surface += s.surface
                    u.end, u.space = s.end, s.space
                else:
                    units.append(
                        Unit(s.surface, s.gold, s.ctype, s.start, s.end, s.space)
                    )
            lines.append(Line(no, g.text, units))
    return lines


def context(line: Line, start: int, end: int, mark: str = "", width: int = 10) -> str:
    """原文の前後つき。mark を渡すと start の位置（切れ目）に挟む。"""
    t = line.text
    if mark:
        return t[max(0, start - width) : start] + mark + t[start : start + width]
    return f"{t[max(0, start - width) : start]}【{t[start:end]}】{t[end : end + width]}"


def readings(units: list[Unit], lo: int, hi: int) -> str:
    """units[lo:hi] の読みを basic_raw.tsv と同じ「/」区切りで（マスあけは「/ /」）。"""
    out = []
    for u in units[max(0, lo) : hi]:
        out.append(u.reading)
        if u.space:
            out.append(" ")
    return "/".join(out)


def counts_text(c: collections.Counter, sep: str = " ") -> str:
    return sep.join(f"{k}×{n}" for k, n in c.most_common())


def write_tsv(name: str, header: list[str], rows: list[list]) -> None:
    with open(OUT / name, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"  {name}: {len(rows)} 行")


def reading_rare(lines: list[Line]) -> None:
    pair, surf, label = (
        collections.Counter(),
        collections.Counter(),
        collections.Counter(),
    )
    by_surf = collections.defaultdict(collections.Counter)
    for line in lines:
        for u in line.units:
            if u.kanji:
                pair[u.surface, u.reading] += 1
                surf[u.surface] += 1
                label[u.reading] += 1
                by_surf[u.surface][u.reading] += 1
    rows = []
    for line in lines:
        for i, u in enumerate(line.units):
            if not u.kanji:
                continue
            n, total = pair[u.surface, u.reading], surf[u.surface]
            rare = total >= 5 and n <= 2 and n / total <= 0.1
            once = label[u.reading] == 1
            if not (rare or once):
                continue
            kind = "まれな読み" if rare else "1回だけの読み"
            rows.append(
                [
                    kind,
                    line.no,
                    u.surface,
                    u.reading,
                    n,
                    total,
                    counts_text(by_surf[u.surface]),
                    context(line, u.start, u.end),
                    readings(line.units, i - 3, i + 4),
                ]
            )
    # 1 回だけの読みは短いので先に。まれな読みは、よく出る字ほど上に
    rows.sort(key=lambda r: (r[0] != "1回だけの読み", -r[5], r[2], r[1]))
    write_tsv(
        "reading_rare.tsv",
        [
            "種類",
            "行",
            "字",
            "読み",
            "この読み",
            "字の総数",
            "字の読み",
            "原文",
            "読みの列",
        ],
        rows,
    )


def notation_key(r: str) -> str:
    """長音の書き方を寄せる（オ段・ウ段＋ウ→ー、エ段＋イ→ー、オ段＋オ→ー）。

    オオ（多・遠・凍）は仮名遣いのとおりで正しいが、同じ字に「ホー」と「ホオ」があれば揺れとして出す。
    """
    out: list[str] = []
    for ch in r:
        prev = out[-1] if out else ""
        if (ch == "ウ" and prev in O_DAN | U_DAN) or (ch == "イ" and prev in E_DAN):
            out.append("ー")
        elif ch == "オ" and prev in O_DAN:
            out.append("ー")
        else:
            out.append(ch)
    return "".join(out)


def has_odd_notation(r: str) -> bool:
    """1 単位の読みの中に、長音を「ー」で書いていない並び（コウ・スウ）か「セー」がある。"""
    for a, b in zip(r, r[1:]):
        if b == "ウ" and a in O_DAN | U_DAN:
            return True
        if b == "ー" and a in E_DAN:
            return True
    return False


def reading_notation(lines: list[Line]) -> None:
    groups = collections.defaultdict(collections.Counter)
    for line in lines:
        for u in line.units:
            if u.kanji:
                groups[u.surface, notation_key(u.reading)][u.reading] += 1
    rows = []
    for line in lines:
        for i, u in enumerate(line.units):
            if not u.kanji:
                continue
            g = groups[u.surface, notation_key(u.reading)]
            top = g.most_common(1)[0][0]
            if len(g) > 1 and u.reading != top:
                kind = "揺れ"
            elif u.reading[:1] in BAD_START:
                kind = "区切り"
            elif has_odd_notation(u.reading):
                kind = "書き方"
            else:
                continue
            rows.append(
                [
                    kind,
                    line.no,
                    u.surface,
                    u.reading,
                    counts_text(g),
                    context(line, u.start, u.end),
                    readings(line.units, i - 3, i + 4),
                ]
            )
    order = {"揺れ": 0, "区切り": 1, "書き方": 2}
    rows.sort(key=lambda r: (order[r[0]], r[2], r[1]))
    write_tsv(
        "reading_notation.tsv",
        ["種類", "行", "字", "読み", "同じ字の書き方", "原文", "読みの列"],
        rows,
    )


def compound_variants(lines: list[Line]) -> None:
    """隣り合う 2 つの漢字の単位ごとに、読みとマスあけの組み合わせを数える。"""
    seen = collections.defaultdict(collections.Counter)
    where = collections.defaultdict(list)
    for line in lines:
        us = line.units
        for i in range(len(us) - 1):
            a, b = us[i], us[i + 1]
            if not (a.kanji and b.kanji and a.end == b.start):
                continue
            key = a.surface + b.surface
            val = readings(us, i, i + 1) + "/" + b.reading
            seen[key][val] += 1
            where[key, val].append(line.no)
    rows = []
    for key, c in seen.items():
        if len(c) < 2:
            continue
        (top, top_n), *rest = c.most_common()
        minority = sum(n for _, n in rest)
        for val, n in rest:
            lines_at = where[key, val]
            rows.append(
                [
                    key,
                    val,
                    n,
                    top,
                    top_n,
                    sum(c.values()),
                    "均衡" if minority / sum(c.values()) > 0.2 else "",
                    " ".join(map(str, lines_at[:10])),
                ]
            )
    # 多数派が強いのに外れているものほど上に（均衡しているものは読み分けのことが多い）
    rows.sort(key=lambda r: (r[6] != "", -r[4], r[0]))
    write_tsv(
        "compound_variants.tsv",
        ["字", "読み", "件数", "多数派", "多数派の件数", "総数", "均衡", "行"],
        rows,
    )


def boundaries(line: Line):
    """マスあけを判定する切れ目 (i, 位置)。units[i] と units[i+1] の間。原文の空白の前後は除く。"""
    us = line.units
    for i in range(len(us) - 1):
        a, b = us[i], us[i + 1]
        if a.surface.isspace() or b.surface.isspace() or a.end != b.start:
            continue
        yield i, a.end


def space_key(line: Line, pos: int, k: int) -> str:
    pad = "^" * k + line.text + "$" * k
    return pad[pos : pos + k] + "|" + pad[pos + k : pos + 2 * k]


def space_stats(lines: list[Line]) -> dict[int, dict[str, list[int]]]:
    """前後 k 文字ごとの [マスあけなし, あり] の件数。"""
    stats = {k: collections.defaultdict(lambda: [0, 0]) for k in SPACE_CONTEXTS}
    for line in lines:
        for i, pos in boundaries(line):
            for k in SPACE_CONTEXTS:
                stats[k][space_key(line, pos, k)][int(line.units[i].space)] += 1
    return stats


def stat_text(s: list[int]) -> str:
    return f"なし{s[0]}:あり{s[1]}"


def space_variation(lines: list[Line], stats) -> None:
    rows = []
    for line in lines:
        for i, pos in boundaries(line):
            sp = int(line.units[i].space)
            s = stats[SPACE_K][space_key(line, pos, SPACE_K)]
            if sum(s) < SPACE_MIN or s[sp] >= s[1 - sp]:
                continue
            rows.append(
                [
                    line.no,
                    "あり" if sp else "なし",
                    space_key(line, pos, SPACE_K),
                    s[1 - sp],
                    *(
                        stat_text(stats[k][space_key(line, pos, k)])
                        for k in SPACE_CONTEXTS
                    ),
                    context(line, pos, pos, "▲" if sp else "・"),
                    readings(line.units, i - 3, i + 4),
                ]
            )
    rows.sort(key=lambda r: (-r[3], r[2], r[0]))
    write_tsv(
        "space_variation.tsv",
        [
            "行",
            "注釈",
            f"前後{SPACE_K}字",
            "多数派の件数",
            *(f"{k}字" for k in SPACE_CONTEXTS),
            "原文（▲あり・なし）",
            "読みの列",
        ],
        rows,
    )


def pos_class(m) -> str:
    """Sudachi の形態素を、マスあけの判定に使う粗い品詞にする。

    付属語・接辞・非自立の語（いる・こと・ない など）は語そのものが切れ続けを決めるので、
    品詞に終止形を添える。それ以外は品詞の上 2 段だけ。
    """
    pos = m.part_of_speech()
    if pos[0] == "補助記号":
        return f"記号:{m.surface()}"
    if pos[0] in ("助詞", "助動詞", "接頭辞", "接尾辞") or pos[1] == "非自立可能":
        return f"{pos[0]}:{m.dictionary_form()}"
    # 英字の語（AI を）や外来語の複合（スーパー マーケット）は切れ続けの決まりが違うので分ける
    s = m.surface()
    if all(c.isascii() and c.isalnum() for c in s):
        return f"{pos[0]}-{pos[1]}(英数字)"
    if all("ァ" <= c <= "ヺ" or c == "ー" for c in s):
        return f"{pos[0]}-{pos[1]}(カタカナ)"
    return f"{pos[0]}-{pos[1]}"


def space_pos(lines: list[Line]) -> None:
    """Sudachi の品詞の並びごとにマスあけの有無を数え、強い多数派から外れた切れ目を出す。

    注釈とも momo のモデルとも別の物差しなので、前後の文字が同じ並びが無い切れ目でも比べられる。
    """
    from sudachipy import Dictionary, SplitMode

    tok = Dictionary(dict="full").create(SplitMode.A)
    found = []  # (行, i, 位置, 並び)
    stats = collections.defaultdict(lambda: [0, 0])
    for line in lines:
        ms = list(tok.tokenize(line.text))
        ends = {m.end(): k for k, m in enumerate(ms)}
        inside = {p: k for k, m in enumerate(ms) for p in range(m.begin() + 1, m.end())}
        for i, pos in boundaries(line):
            if pos in ends and ends[pos] + 1 < len(ms):
                k = ends[pos]
                key = f"{pos_class(ms[k])} | {pos_class(ms[k + 1])}"
            elif pos in inside:
                key = f"語中 | {pos_class(ms[inside[pos]])}"
            else:
                continue
            stats[key][int(line.units[i].space)] += 1
            found.append((line, i, pos, key))
    rows = []
    for line, i, pos, key in found:
        sp = int(line.units[i].space)
        s = stats[key]
        if sum(s) < POS_MIN or s[sp] > POS_MINORITY * sum(s):
            continue
        rows.append(
            [
                key,
                line.no,
                "あり" if sp else "なし",
                stat_text(s),
                context(line, pos, pos, "▲" if sp else "・"),
                readings(line.units, i - 3, i + 4),
            ]
        )
    # 並びごとにまとめて、多数派が強い並びから
    rows.sort(key=lambda r: (-max(stats[r[0]]), r[0], r[1]))
    write_tsv(
        "space_pos.tsv",
        ["品詞の並び", "行", "注釈", "並び全体", "原文（▲あり・なし）", "読みの列"],
        rows,
    )
    pairs = sorted(stats.items(), key=lambda kv: -sum(kv[1]))
    write_tsv(
        "space_pos_pairs.tsv",
        ["品詞の並び", "件数", "なし", "あり", "ありの割合"],
        [[k, sum(s), s[0], s[1], f"{s[1] / sum(s):.3f}"] for k, s in pairs],
    )


def space_vs_w7(lines: list[Line], stats, w7_path: Path) -> None:
    import momors_py

    w7 = momors_py.Predictor(str(w7_path))
    rows = []
    for line in lines:
        res = w7.predict(line.text)
        owner = {p: i for i, u in enumerate(line.units) for p in range(u.start, u.end)}
        pred = set()
        for ch, src in zip(res.kana, res.kana_to_source):
            # 原文の空白に当たるかなの空白は、マスあけではない
            if ch == " " and src < len(line.text) and not line.text[src].isspace():
                if src in owner:
                    pred.add(owner[src])
        for i, pos in boundaries(line):
            gold = line.units[i].space
            if gold == (i in pred):
                continue
            s = stats[SPACE_K][space_key(line, pos, SPACE_K)]
            # 注釈が前後の文字の多数派からも外れていれば、食い違いの裏付けになる
            minority = sum(s) >= SPACE_MIN and s[int(gold)] < s[1 - int(gold)]
            rows.append(
                [
                    line.no,
                    "注釈あり・w7なし" if gold else "注釈なし・w7あり",
                    "少数派" if minority else "",
                    stat_text(s),
                    context(line, pos, pos, "▲" if gold else "・"),
                    readings(line.units, i - 3, i + 4),
                ]
            )
    rows.sort(key=lambda r: (r[2] != "少数派", r[0]))
    write_tsv(
        "space_vs_w7.tsv",
        [
            "行",
            "食い違い",
            f"前後{SPACE_K}字で",
            f"{SPACE_K}字",
            "原文（▲注釈あり・注釈なし）",
            "読みの列",
        ],
        rows,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=RAW_PATH, help="点検する basic_raw.tsv")
    ap.add_argument(
        "--w7", type=Path, help="全データで学習した w7（.mbm）。マスあけを比べる"
    )
    ap.add_argument(
        "--sudachi",
        action="store_true",
        help="Sudachi の品詞の並びでマスあけを比べる（sudachipy と sudachidict-full が要る）",
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    lines = load(args.raw)
    print(f"{args.raw}: {len(lines)} 行 → {OUT}")
    reading_rare(lines)
    reading_notation(lines)
    compound_variants(lines)
    stats = space_stats(lines)
    space_variation(lines, stats)
    if args.w7:
        space_vs_w7(lines, stats, args.w7)
    if args.sudachi:
        space_pos(lines)


if __name__ == "__main__":
    main()
