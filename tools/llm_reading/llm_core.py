"""LLM に、文字ごとの候補から漢字の読みを選ばせる本体（torch と transformers だけで動く）。

05_llm_select.py（手元の実験）と 07_gpu_eval.py（クラウドの GPU で複数モデルを回す）が使う。
momo_py・momors_py には依存しない（クラウドの Linux でもそのまま動かすため）。スロットは
surface / gold / ctype / start / end を持つものなら何でもよい（_common.Slot か、ここの BSlot）。

生成はさせない。候補は1文字ごとの読みなので、どれを選んでも書き換えや欠落は構造的に
起きず、かなと原文の対応も保たれる。LLM が選ぶのは漢字の読みだけで、かな・記号・漢数字は
w7 の出力で固定する。

方式:
- RubySelector（既定）: 漢字ランの直後に「（よみ）」を書く括弧書きの形で問う。日本語の文章に多い
  形なので事前学習の知識が効き、読みの直前に漢字があるので対応そのものが問われる。
- YomiSelector: ひらがなの続け書きで問う。短い読み・ありふれた語に引っ張られる（比較用）。
- InterleaveSelector: `右=ミギ|側=ガワ|` と1文字ずつ問う。語が見えず弱い（比較用）。

文脈: 同じ文書の前の文を「原文」の行の先頭に、その読み（w7 で振ったもの）を「ふりがな」の
行の先頭に置く（select の context / context_ruby）。1冊を点訳するとき前の文が処理済みである
状況と同じ形。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BEAM = 8  # ruby: 漢字ランの読みを1文字ずつ伸ばすときに残す本数
LOOKAHEAD = 3  # 漢字ランの後ろに含めて採点する文字数（送り仮名・助詞）
MAX_COMBOS = 120  # yomi: 漢字ラン1つあたりに採点する組み合わせの上限
LABEL_CONTINUE = "---"  # momo_py.features と同じ
LABEL_SKIP = "_"
KANJI_TYPES = frozenset({"KANJI", "JAPANESE_NUMERIC"})


@dataclass
class BSlot:
    """持ち込みデータ（バンドル）のスロット。_common.Slot と同じ属性を持つ。"""

    surface: str
    gold: str
    ctype: str
    start: int
    end: int


# ---------------------------------------------------------------- かなの変換

# 長音「ー」を、直前の拍の段に応じた普通のかなに戻す（オ段は「う」: ヨー→よう）
_LONG_VOWEL = {
    **dict.fromkeys("アカサタナハマヤラワガザダバパァャヮ", "あ"),
    **dict.fromkeys("イキシチニヒミリギジヂビピィ", "い"),
    **dict.fromkeys("ウクスツヌフムユルグズヅブプゥュヴ", "う"),
    **dict.fromkeys("エケセテネヘメレゲゼデベペェ", "え"),
    **dict.fromkeys("オコソトノホモヨロヲゴゾドボポォョ", "う"),
}


def to_katakana(s: str) -> str:
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s)


def braille_kana_to_hiragana(reading: str) -> str:
    """点字用の表音カタカナ（キョー）を、普通のかな遣いのひらがな（きょう）に戻す（採点用）。"""
    out = []
    for c in reading:
        if c == "ー" and out:
            c = _LONG_VOWEL.get(to_katakana(out[-1]), "ー")
        out.append(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c)
    return "".join(out)


# ---------------------------------------------------------------- スロットと辞書


def load_dict_ordered(path: str) -> dict[str, list[str]]:
    """single_character_dic.tsv を 文字 → 読み候補（頻度順）で読む。

    継続（---）は "" として残す。継続を許すのは、学習データで実際に継続になったことの
    ある文字（今日の日など）だけにするため。どの漢字にも許すと、2文字目以降を空にした
    短い読み（人気=ナ、上手=カミ）が尤度で勝ってしまう。_ は除く。
    """
    dic = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                dic[parts[0]] = [
                    "" if r == LABEL_CONTINUE else r
                    for r in parts[1:]
                    if r != LABEL_SKIP
                ]
    return dic


def kanji_runs(slots) -> list[list[int]]:
    """連続する漢字スロット（漢数字を除く）の番号のまとまり。"""
    runs, cur = [], []
    for i, s in enumerate(slots):
        if s.ctype == "KANJI" and (not cur or slots[cur[-1]].end == s.start):
            cur.append(i)
            continue
        if cur:
            runs.append(cur)
        cur = [i] if s.ctype == "KANJI" else []
    if cur:
        runs.append(cur)
    return runs


def ruby(slots, readings: list[str]) -> str:
    """漢字ランの直後に「（よみ）」を添えた文字列（ひらがな・普通のかな遣い）。"""
    out = ""
    runs = {r[0]: r for r in kanji_runs(slots)}
    i = 0
    while i < len(slots):
        run = runs.get(i)
        if run is None:
            out += slots[i].surface
            i += 1
            continue
        surface = "".join(slots[j].surface for j in run)
        reading = "".join(braille_kana_to_hiragana(readings[j]) for j in run)
        out += f"{surface}（{reading}）"
        i = run[-1] + 1
    return out


def fmt_reading(r: str) -> str:
    return r if r else "---"


def fmt_sentence(slots, readings: list[str]) -> str:
    return "".join(f"{s.surface}={fmt_reading(r)}|" for s, r in zip(slots, readings))


# ---------------------------------------------------------------- LLM


class LM:
    """KV キャッシュを使って、接頭辞のあとに続く文字列の対数尤度を測る。"""

    ANCHOR = "\n"

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        dtype: str = "float32",
        batch: int = 16,
    ):
        self.device, self.batch = device, batch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        kwargs = {"dtype": getattr(torch, dtype)}
        if device != "cpu":
            kwargs["device_map"] = device  # 大きいモデルを CPU の RAM を経ずに載せる
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self.pad_id = next(
            (
                t
                for t in (self.tok.pad_token_id, self.tok.eos_token_id)
                if t is not None
            ),
            0,
        )
        self._anchor = self.tok(self.ANCHOR, add_special_tokens=False)["input_ids"]

    def ids(self, s: str) -> list[int]:
        """文の途中に続く文字列としてトークン化する。

        SentencePiece 系（LLM-jp など）は断片ごとに先頭へ「▁」を付けるので、そのまま
        区切ってつなぐと実際の文章にない「▁（」などが混ざる。目印（改行）を前に付けて
        区切り、目印の分を取り除くと、文章をまとめて区切ったときと同じになる。
        """
        ids = self.tok(self.ANCHOR + s, add_special_tokens=False)["input_ids"]
        if ids[: len(self._anchor)] != self._anchor:
            raise ValueError(f"目印のトークンが前に来ない: {s!r}")
        return ids[len(self._anchor) :]

    def ids_start(self, s: str) -> list[int]:
        """文書の先頭（BOS の直後）としてトークン化する。"""
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def _tensor(self, rows: list[list[int]]) -> torch.Tensor:
        return torch.tensor(rows, device=self.model.device)

    @torch.no_grad()
    def feed(self, ids: list[int], cache):
        """ids を読ませ、(キャッシュ, 最後の位置の logits) を返す。"""
        if not ids:
            raise ValueError("空の入力")
        out = self.model(
            input_ids=self._tensor([ids]), past_key_values=cache, use_cache=True
        )
        return out.past_key_values, out.logits[0, -1].float()

    @torch.no_grad()
    def score(self, ids: list[int], cache, last_logits) -> float:
        """接頭辞（cache, last_logits）のあとに ids が続く対数尤度。キャッシュは元に戻す。"""
        lp = torch.log_softmax(last_logits, -1)[ids[0]].item()
        if len(ids) > 1:
            base = cache.get_seq_length()
            out = self.model(
                input_ids=self._tensor([ids[:-1]]),
                past_key_values=cache,
                use_cache=True,
            )
            lps = torch.log_softmax(out.logits[0].float(), -1)
            lp += sum(lps[j, ids[j + 1]].item() for j in range(len(ids) - 1))
            cache.crop(-(cache.get_seq_length() - base))
        return lp

    @torch.no_grad()
    def score_batch(self, seqs: list[list[int]], cache, last_logits) -> list[float]:
        """score を複数の ids についてまとめて測る（同じ接頭辞を共有する候補の一括採点）。

        接頭辞のキャッシュを複製して行数ぶんに増やし、候補を右詰めの埋め草でそろえて
        1回で流す。因果的な注意なので、埋め草は自分より後ろの位置にしか影響しない。
        """
        lp0 = torch.log_softmax(last_logits, -1)
        scores = [lp0[s[0]].item() for s in seqs]
        multi = [(i, s) for i, s in enumerate(seqs) if len(s) > 1]
        k = 0
        while k < len(multi):
            part = multi[k : k + self.batch]
            try:
                self._score_part(part, cache, scores)
            except torch.OutOfMemoryError:
                # GPU のメモリが足りなければ、一括の行数を半分にしてやり直す
                if self.batch == 1:
                    raise
                self.batch = max(1, self.batch // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    f"  （GPU メモリ不足のため一括採点を {self.batch} 行に減らす）",
                    flush=True,
                )
                continue
            k += len(part)
        return scores

    def _score_part(self, part, cache, scores: list[float]) -> None:
        width = max(len(s) for _, s in part) - 1
        rows = [s[:-1] + [self.pad_id] * (width - len(s) + 1) for _, s in part]
        c = copy.deepcopy(cache)
        c.batch_repeat_interleave(len(part))
        out = self.model(
            input_ids=self._tensor(rows), past_key_values=c, use_cache=True
        )
        lps = torch.log_softmax(out.logits.float(), -1)
        for b, (i, s) in enumerate(part):
            pos = torch.arange(len(s) - 1, device=lps.device)
            tgt = torch.tensor(s[1:], device=lps.device)
            scores[i] += lps[b, pos, tgt].sum().item()
        del c, out, lps


# ---------------------------------------------------------------- 選び方


class YomiSelector:
    """漢字ランごとに候補の組み合わせを、ひらがなの続け書きの尤度で選ぶ（比較用）。"""

    INSTRUCTION = "以下は日本語の文と、その読みをひらがなで書いたものです。\n\n"

    def __init__(self, lm: LM, fewshot, dic: dict[str, list[str]]):
        self.lm, self.dic = lm, dic
        prompt = self.INSTRUCTION + "".join(
            f"原文: {g.text}\n読み: {self.yomi(g.slots, [s.gold for s in g.slots])}\n\n"
            for g in fewshot
        )
        self.base_cache, _ = lm.feed([lm.tok.bos_token_id] + lm.ids_start(prompt), None)

    @staticmethod
    def piece(s, reading: str) -> str:
        """採点用の読みの一片。漢字は候補の読みをひらがなに、それ以外は原文の文字のまま。"""
        return braille_kana_to_hiragana(reading) if s.ctype == "KANJI" else s.surface

    def yomi(self, slots, readings: list[str]) -> str:
        return "".join(self.piece(s, r) for s, r in zip(slots, readings))

    def cands(self, slots, i: int, w7: str, first: bool, k: int | None) -> list[str]:
        ordered = self.dic.get(slots[i].surface, [])[:k]
        cands = list(dict.fromkeys(ordered + [w7]))
        return [c for c in cands if c] if first else cands

    def _lookahead(self, slots, start: int, chosen: list[str]) -> str:
        out = ""
        for j in range(start, len(slots)):
            if slots[j].ctype == "KANJI" or len(out) >= LOOKAHEAD:
                break
            out += self.piece(slots[j], chosen[j])
        return out[:LOOKAHEAD]

    @torch.no_grad()
    def select(
        self, text: str, slots, w7: list[str], context: str = "", context_ruby: str = ""
    ):
        import itertools

        cache = copy.deepcopy(self.base_cache)
        pending = (f"文脈: {context}\n" if context else "") + f"原文: {text}\n読み: "
        chosen, report = list(w7), []
        run_at = {run[0]: run for run in kanji_runs(slots)}
        i = 0
        while i < len(slots):
            run = run_at.get(i)
            if run is None:
                pending += self.piece(slots[i], chosen[i])
                i += 1
                continue
            if pending:
                cache, last = self.lm.feed(self.lm.ids(pending), cache)
                pending = ""
            after = run[-1] + 1
            look = self._lookahead(slots, after, chosen)
            k = max(2, int(MAX_COMBOS ** (1 / len(run))))
            lists = [self.cands(slots, j, w7[j], n == 0, k) for n, j in enumerate(run)]
            table = []
            for combo in itertools.product(*lists):
                piece = "".join(braille_kana_to_hiragana(r) for r in combo) + look
                if piece:
                    table.append(
                        (combo, self.lm.score(self.lm.ids(piece), cache, last))
                    )
            table.sort(key=lambda t: -t[1])
            best = table[0][0]
            for j, r in zip(run, best):
                chosen[j] = r
            report.append((run, table[:4]))
            best_text = "".join(braille_kana_to_hiragana(r) for r in best)
            if best_text:
                cache, last = self.lm.feed(self.lm.ids(best_text), cache)
            i = after
        return chosen, report


class RubySelector(YomiSelector):
    """漢字ランの直後に「（よみ）」を書く括弧書きの形で、候補の組み合わせを選ぶ。

    続け書き（yomi）では、かな列の自然さだけで選ぶため短い読みやありふれた語に引っ張られる
    （右側=う+継続、入る=いる）。括弧書きなら読みの直前に漢字があり、閉じ括弧まで採点するので、
    漢字と読みの対応そのものが問われる。日本語の文章に多い形なので、事前学習の知識も効く。
    """

    INSTRUCTION = "以下は日本語の文に、漢字の読みを括弧書きで添えたものです。\n\n"
    MASK = "＿"  # 中立な文脈で漢字を伏せる文字

    def __init__(self, lm: LM, fewshot, dic: dict[str, list[str]], calib: float = 0.0):
        """calib > 0 なら PMI 較正: 漢字を伏せた中立な文脈での尤度を calib 倍して差し引く。

        素の尤度は、読みの文字列そのものの出やすさ（トークン数・頻度）に引っ張られる
        （「にんき」=3トークン より「なく」=1トークン が高く、人気=ナク になる）。
        中立な文脈での尤度を差し引くと、その漢字がその読みをどれだけ支持するかが残る、はず。

        2026-10-19 の実験では効かなかった。伏せ字だけの中立文脈（ここの実装）はモデルの
        振る舞いが不安定で、文の中で漢字だけ伏せる形も試したが、直る例と崩れる例が半々で、
        金星=キンボシ・生物=ナマモノのような知識の不足は補えなかった。既定は 0（無効）。
        """
        self.lm, self.dic, self.calib = lm, dic, calib
        prompt = self.INSTRUCTION + "".join(
            f"原文: {g.text}\nふりがな: {ruby(g.slots, [s.gold for s in g.slots])}\n\n"
            for g in fewshot
        )
        self.base_cache, _ = lm.feed([lm.tok.bos_token_id] + lm.ids_start(prompt), None)
        self._neutral_prefix: dict[int, tuple] = {}  # 伏せ字の長さ → (cache, last)
        self._neutral_score: dict[tuple, float] = {}  # (伏せ字の長さ, ids) → 尤度

    def neutral(self, n_chars: int, ids: list[int]) -> float:
        """漢字 n_chars 文字を伏せた中立な文脈で、ids が続く尤度（文字列ごとに覚えておく）。"""
        key = (n_chars, tuple(ids))
        if key not in self._neutral_score:
            if n_chars not in self._neutral_prefix:
                mask = self.MASK * n_chars
                self._neutral_prefix[n_chars] = self.lm.feed(
                    self.lm.ids(f"原文: {mask}\nふりがな: {mask}"),
                    copy.deepcopy(self.base_cache),
                )
            cache, last = self._neutral_prefix[n_chars]
            self._neutral_score[key] = self.lm.score(ids, cache, last)
        return self._neutral_score[key]

    def _beam(
        self, slots, run, w7, look: str, cache, last
    ) -> list[tuple[tuple, float]]:
        """漢字ランの読みを1文字ずつ伸ばし、途中の尤度で上位 BEAM 本だけ残す。

        候補を頻度順に上位で切ってから全組み合わせを採点すると、3文字以上のランで
        正解が切り落とされる（枚上手 の 上=ウワ は5番目で落ちた）。途中まで（閉じ括弧の前）で
        採点しながら残すので、候補を切らずに済む。最後の1文字だけ「）」と送り仮名まで採点する。
        同じ位置の候補は lm.score_batch でまとめて採点する。
        """
        hyps: list[tuple[tuple, float]] = [((), 0.0)]
        for n, j in enumerate(run):
            final = n == len(run) - 1
            combos, seqs = [], []
            for combo, _ in hyps:
                for c in self.cands(slots, j, w7[j], n == 0, None):
                    nc = combo + (c,)
                    reading = "".join(braille_kana_to_hiragana(r) for r in nc)
                    combos.append(nc)
                    seqs.append(
                        self.lm.ids(f"（{reading}）{look}" if final else f"（{reading}")
                    )
            scores = self.lm.score_batch(seqs, cache, last)
            if final and self.calib:
                scores = [
                    lp - self.calib * self.neutral(len(run), ids)
                    for lp, ids in zip(scores, seqs)
                ]
            hyps = sorted(zip(combos, scores), key=lambda t: -t[1])[:BEAM]
        return hyps

    @torch.no_grad()
    def select(
        self, text: str, slots, w7: list[str], context: str = "", context_ruby: str = ""
    ):
        """漢字ランごとに読みを選ぶ。(読み, 漢字ランごとの上位候補) を返す。

        context / context_ruby を渡すと、同じ文書の前の文とその読みを先頭に置く。
        """
        cache = copy.deepcopy(self.base_cache)
        pending = f"原文: {context}{text}\nふりがな: {context_ruby}"
        chosen, report = list(w7), []
        run_at = {run[0]: run for run in kanji_runs(slots)}
        i = 0
        while i < len(slots):
            run = run_at.get(i)
            if run is None:
                pending += slots[i].surface
                i += 1
                continue
            pending += "".join(slots[j].surface for j in run)
            cache, last = self.lm.feed(self.lm.ids(pending), cache)
            pending = ""
            after = run[-1] + 1
            table = self._beam(
                slots, run, w7, self._lookahead(slots, after, chosen), cache, last
            )
            best = table[0][0]
            for j, r in zip(run, best):
                chosen[j] = r
            report.append((run, table[:4]))
            best_reading = "".join(braille_kana_to_hiragana(r) for r in best)
            cache, last = self.lm.feed(self.lm.ids(f"（{best_reading}）"), cache)
            i = after
        return chosen, report


class InterleaveSelector:
    """`右=ミギ|側=ガワ|…` と1文字ずつ候補を選ぶ（比較用。素のモデルでは語が見えず弱い）。"""

    INSTRUCTION = (
        "以下は日本語の文の各文字に、点字用のカタカナの読みを付けた例です。"
        "直前の文字とまとめて読む文字には --- と書きます。\n\n"
    )
    KANA_VARIANTS = {"は": ("ワ",), "へ": ("エ",), "う": ("ー",)}

    def __init__(self, lm: LM, fewshot, dic: dict[str, list[str]]):
        self.lm, self.dic = lm, dic
        prompt = self.INSTRUCTION + "".join(
            f"原文: {g.text}\n読み: {fmt_sentence(g.slots, [s.gold for s in g.slots])}\n\n"
            for g in fewshot
        )
        self.base_cache, _ = lm.feed([lm.tok.bos_token_id] + lm.ids_start(prompt), None)

    def cands(self, slots, i: int, w7: str) -> list[str]:
        s = slots[i]
        if s.ctype == "KANJI":
            c = set(self.dic.get(s.surface, ())) | {w7}
            if (
                i > 0
                and slots[i - 1].ctype in KANJI_TYPES
                and slots[i - 1].end == s.start
            ):
                c.add("")
        elif s.ctype in ("HIRAGANA", "KATAKANA"):
            c = {to_katakana(s.surface), w7, *self.KANA_VARIANTS.get(s.surface, ())}
        else:
            c = {w7}
        return sorted(c)

    @torch.no_grad()
    def select(
        self, text: str, slots, w7: list[str], context: str = "", context_ruby: str = ""
    ):
        cache = copy.deepcopy(self.base_cache)
        header = (f"文脈: {context}\n" if context else "") + f"原文: {text}\n読み: "
        cache, last = self.lm.feed(self.lm.ids(header), cache)
        chosen, report = [], []
        for i, s in enumerate(slots):
            cache, last = self.lm.feed(self.lm.ids(f"{s.surface}="), cache)
            cands = self.cands(slots, i, w7[i])
            seqs = [self.lm.ids(fmt_reading(c) + "|") for c in cands]
            table = sorted(
                (
                    ((c,), lp)
                    for c, lp in zip(cands, self.lm.score_batch(seqs, cache, last))
                ),
                key=lambda t: -t[1],
            )
            best = table[0][0][0]
            chosen.append(best)
            if s.ctype == "KANJI":
                report.append(([i], table[:4]))
            cache, last = self.lm.feed(self.lm.ids(fmt_reading(best) + "|"), cache)
        return chosen, report


SELECTORS = {
    "ruby": RubySelector,
    "yomi": YomiSelector,
    "interleave": InterleaveSelector,
}


# ---------------------------------------------------------------- 評価の集計

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def gating_table(rows) -> dict:
    """漢字スロットの (正解, LLM, w7, w7 の自信度) から、w7 単独・LLM 単独と、
    w7 の自信度がしきい値未満のスロットだけ LLM に置き換えた場合の一致率をまとめる。"""
    n = len(rows)
    if not n:
        return {"slots": 0}
    res = {
        "slots": n,
        "w7": sum(w == g for g, _, w, _ in rows) / n,
        "llm": sum(a == g for g, a, _, _ in rows) / n,
    }
    for t in THRESHOLDS:
        mixed = [(g, a if c < t else w, w) for g, a, w, c in rows]
        res[f"gated_{t}"] = sum(m == g for g, m, _ in mixed) / n
        res[f"fixed_{t}"] = sum(m == g and w != g for g, m, w in mixed)
        res[f"broke_{t}"] = sum(w == g and m != g for g, m, w in mixed)
    return res
