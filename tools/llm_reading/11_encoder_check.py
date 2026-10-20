"""エンコーダ実験の整合性の検査（学習の前に手元で）。

- スロットと原文の位置が合っている（往復）。語彙にない文字（[UNK] になる漢字）の数。
- 対象スロットのトークン位置が、その漢字のトークンを指している（文脈を付けても）。
- 正解ラベルをそのまま logits に見立てて復号すると、候補に正解がある全スロットで 100% 一致する。
- 学習データから作った候補表が、work/frac_100/single_character_dic.tsv（makedic の辞書）と同じ。
- 下限の目安: 各文字の頻度1位の読みを選ぶだけの精度。

    uv run --group llm python 11_encoder_check.py [--bundle work/gpu_kit/bundle]
"""

import argparse
import collections
import os

import torch

from encoder_core import (
    Candidates,
    CharVocab,
    Labels,
    collate,
    is_target,
    load_jsonl,
    make_example,
    masked_logits,
    verify_records,
)
from llm_core import load_dict_ordered

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "tohoku-nlp/bert-base-japanese-char-v3"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", default=os.path.join(HERE, "work", "gpu_kit", "bundle"))
    ap.add_argument(
        "--dic",
        default=os.path.join(HERE, "work", "frac_100", "single_character_dic.tsv"),
    )
    args = ap.parse_args()

    names = ("train_full", "dev_full", "probe", "eval_context")
    data = {n: load_jsonl(os.path.join(args.bundle, f"{n}.jsonl")) for n in names}
    for n, recs in data.items():
        verify_records(recs)
        print(f"往復 OK: {n} {len(recs)} 件")
    train, dev = data["train_full"], data["dev_full"]
    vocab = CharVocab.from_model(MODEL)
    labels = Labels(train)
    cands = Candidates(train, labels)
    print(f"語彙 {vocab.size}・ラベル {len(labels)}・辞書の文字 {len(cands.chars)}")

    # 語彙にない文字
    for n, recs in (("train", train), ("dev", dev)):
        unk = collections.Counter(
            s.surface
            for r in recs
            for s in r.slots
            if is_target(s) and vocab.token_id(s.surface) == vocab.unk
        )
        total = sum(1 for r in recs for s in r.slots if is_target(s))
        print(
            f"[UNK] の対象スロット({n}): {sum(unk.values())}/{total}・{len(unk)} 種 {dict(unk)}"
        )
    max_text = max(len(r.text) for r in train + dev)
    print(f"最長 {max_text} 字（上限 {254}）")

    # トークン位置・復号
    for n, recs in (("train", train), ("dev", dev)):
        bad = n_slots = in_cands = decoded = 0
        for r in recs:
            ctx = "前の文です。" if r.id % 2 else ""
            ex = make_example(r, vocab, labels, cands, ctx)
            for p, si in zip(ex.pos, ex.slot):
                n_slots += 1
                bad += ex.ids[p] != vocab.token_id(r.slots[si].surface)
            if len(ex.pos) != sum(is_target(s) for s in r.slots):
                bad += 1
            ids, mask, pos_b, pos_t, rows, gold = collate([ex], vocab.pad, "cpu")
            if not len(gold):
                continue
            ok = gold >= 0
            logits = torch.zeros(len(gold), len(labels))
            logits[ok, gold[ok]] = 10.0
            pred = masked_logits(logits, cands.table, rows).argmax(-1)
            reachable = ok & cands.table[rows, gold.clamp(min=0)]
            in_cands += int(reachable.sum())
            decoded += int((pred[reachable] == gold[reachable]).sum())
        print(
            f"{n}: 位置のずれ {bad}・候補に正解あり {in_cands}/{n_slots}・"
            f"そのうち復号一致 {decoded}（{decoded / max(in_cands, 1):.4%}）"
        )

    # 辞書との一致
    dic = load_dict_ordered(args.dic)
    diff = []
    for ch, row in cands.chars.items():
        mine = {labels.names[i] for i in cands.table[row].nonzero().flatten().tolist()}
        theirs = set(dic.get(ch, ()))
        if mine != theirs:
            diff.append((ch, sorted(mine - theirs), sorted(theirs - mine)))
    only_dic = [
        c for c in dic if c not in cands.chars
    ]
    print(
        f"辞書との差: 文字 {len(diff)} 件（上位 5: {diff[:5]}）・辞書だけにある文字 {len(only_dic)} 件"
    )

    # 下限: 頻度1位
    hit = tot = unseen = 0
    for r in dev:
        for s in r.slots:
            if is_target(s):
                tot += 1
                if s.surface in cands.top:
                    hit += labels.names[cands.top[s.surface]] == s.gold
                else:
                    unseen += 1
    print(
        f"dev の漢字スロット {tot}・頻度1位を選ぶだけの一致率 {hit / tot:.4f}・辞書にない文字 {unseen}"
    )



if __name__ == "__main__":
    main()
