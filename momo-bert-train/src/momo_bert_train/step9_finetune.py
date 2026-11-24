"""BERT を微調整して、読みの分類モデルを保存する（クラウドの GPU 用。手元の CPU では --smoke のみ）。

tools/llm_reading の 10_encoder_train.py と同じ手順（8エポック・lr 5e-5・バッチ 32・漢字スロットを
辞書候補に絞った softmax で選ぶ）で学習し、結果を1つのディレクトリに保存する。保存したものは
step10_infer.py が手元の CPU で読み込める。

    uv run python -m momo_bert_train.step9_finetune --smoke                  # 動作確認（極小モデル・数十行）
    uv run python -m momo_bert_train.step9_finetune                          # train で学習し、dev で評価して保存
    uv run python -m momo_bert_train.step9_finetune --all-data               # train+dev の全データで学習して保存
    uv run python -m momo_bert_train.step9_finetune --boundary               # 読みと境界（マスあけ）を同時に学習
    uv run python -m momo_bert_train.step9_finetune --boundary --fragments 1 # 境界の学習に行の断片も足す

出力: work/finetuned/<名前>/
  model.pt      ReadingModel の重み（float32。--boundary では境界の出力層も入る）
  encoder.json  BERT 本体の構成（学習済みの重みを取り直さずに読み込める）
  labels.json   読みのラベル表
  cands.json    文字 → 読み候補（ラベル番号）の辞書
  meta.json     学習の設定と dev の一致率（--boundary では境界の誤りの数と F1 も。dev の行全体と、
                dev から決まった作り方で切り出した断片の両方）、学習データの meta（basic_raw.tsv のハッシュ）

名前の既定は train / full。--boundary を付けると train_joint / full_joint、さらに --fragments を
付けると train_joint_frag_unit / full_joint_frag_unit（--fragment-cut free では train_joint_frag / full_joint_frag）。
--all-data では dev も学習に使うので、meta の dev 一致率は学習済みの文での値（評価にならない）。
境界の dev の数字は、切れ目ごと（行末と原文の空白をはさむ切れ目を除く）に、確率 0.5 以上をマスあけとして数える。

--fragments N は、境界の学習に、行の途中から途中までを切り出した断片（--fragment-min〜--fragment-max 字）を
1 行あたり平均 N 個足す。断片はエポックごとに切り直す。読みの学習には使わない（読みは行全体だけで学ぶ）。
行全体だけで学習すると、見出し・表の欄のような短い入力でマスあけを外す。
--fragment-cut は切り出し方。既定の unit はマスあけの単位の頭から終わりまでを 2 単位以上（断片の終わりは
必ず妥当な境界）。free はスロットの切れ目ならどこでも切る（語の途中で切れたスクリーンリーダーの切れ端に
近い。優先しないと決めたので、比べるために残す）。
--fragment-last-gap（free のときだけ）を付けると、断片の最後の切れ目（[SEP] の直前の字の後ろ）も元の行の
注釈で学習する（名前は _frag_last）。推論では行末の切れ目を問わないので、使うときの動きは変わらない。
unit の断片の最後の切れ目は元の行で必ずマスあけなので、学習しても何も教えない。

各エポックの終わりに、dev の読み・境界に加えて、最終層の隠れ状態が [SEP] と同じに潰れた字の割合
（dev の行全体、dev の断片の最後の字とそれ以外）と、[SEP] の位置のマスあけの確率を出す。
潰れた字どうしは境界の出力層から区別できない。エポックごとの値は history.jsonl と meta.json の history に残る。
dev の断片は unit と free の両方で測る（history の fragment_* と collapse_fragment_* は unit、free_fragment_* と
collapse_free_fragment_* は free。2026-11-22 より前のモデルの fragment_* と dev_boundary_fragments は free のもの）。

学習の途中の状態を 2 分ごとと各エポックの終わりに work/finetuned/<名前>/checkpoint.pt に保存する
（CPU に写しを取ってから別のスレッドで書くので、学習はほとんど止まらない）。
VM がプリエンプトされたら、同じコマンドを打ち直せば続きから学習する（設定かデータが違えば最初から）。
保存し終えたら消える。meta の秒数は、再開したときは再開してからの時間。

--model で元のモデルを替えられる（既定は char-v3。BERT で、語彙が文字単位の vocab.txt のもの）。
--no-weights は重み（model.pt）を保存しない。dev の数字だけを比べる取り直し（学習率・シード）で、
VM のディスクと持ち帰る量を減らす（large の重みは 1 本 1.2GB 余り）。重みの無いモデルは
step10_infer・export_onnx では読めない。
"""

import argparse
import hashlib
import json
import random
import time

import torch

from .common import (
    DATA,
    DEFAULT_MODEL,
    FRAGMENT_CUTS,
    WORK,
    Candidates,
    CharVocab,
    Labels,
    boundary_examples,
    build_model,
    collapse_stats,
    dev_fragments,
    fragments,
    load_jsonl,
    make_examples,
    predict,
    predict_boundary,
    train,
    verify_records,
)


def dev_accuracy(model, recs, exs, cands, labels, vocab, device) -> float:
    preds = predict(model, exs, cands, vocab, device)
    hit = n = 0
    for r, ex, pr in zip(recs, exs, preds):
        for si, (lab, _) in zip(ex.slot, pr):
            hit += labels.names[lab] == r.slots[si].gold
            n += 1
    return hit / n if n else 0.0


def dev_boundary(model, exs, vocab, device) -> dict:
    """dev の切れ目ごとの境界の当たり。missed＝注釈はマスあけなのに続けた（くっつけすぎ）、
    extra＝注釈は続けるのにあけた（分けすぎ）。"""
    probs = predict_boundary(model, exs, vocab, device)
    tp = fp = fn = tn = 0
    for ex, pr in zip(exs, probs):
        for gold, p in zip(ex.bgold, pr):
            split = p >= 0.5
            tp += split and gold == 1
            fp += split and gold == 0
            fn += not split and gold == 1
            tn += not split and gold == 0
    gaps = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "gaps": gaps,
        "spaces": tp + fn,
        "errors": fp + fn,
        "missed": fn,
        "extra": fp,
        "accuracy": (tp + tn) / gaps if gaps else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
    }


def boundary_summary(b: dict) -> str:
    return (
        f"境界 誤り {b['errors']}/{b['gaps']}（くっつけすぎ {b['missed']}・分けすぎ {b['extra']}）"
        f" F1={b['f1']:.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--name",
        default="",
        help="保存先の名前（既定: train か full。--boundary では train_joint か full_joint）",
    )
    ap.add_argument("--all-data", action="store_true", help="dev も学習に使う")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true", help="極小モデルで数十行だけ通す")
    ap.add_argument(
        "--boundary",
        action="store_true",
        help="境界（マスあけ）の出力層を足し、読みと同時に学習する",
    )
    ap.add_argument(
        "--boundary-weight",
        type=float,
        default=1.0,
        help="同時学習で、読みの損失に足す境界の損失の重み",
    )
    ap.add_argument(
        "--fragments",
        type=float,
        default=0.0,
        help="境界の学習に足す、行を切り出した断片の 1 行あたりの平均の数（--boundary のとき。0 で足さない）",
    )
    ap.add_argument("--fragment-min", type=int, default=2, help="断片の最小の字数")
    ap.add_argument("--fragment-max", type=int, default=20, help="断片の最大の字数")
    ap.add_argument(
        "--fragment-cut",
        choices=FRAGMENT_CUTS,
        default="unit",
        help="断片の切り出し方（unit: マスあけの単位の頭から終わりまで 2 単位以上・free: スロットの切れ目ならどこでも）",
    )
    ap.add_argument(
        "--fragment-last-gap",
        action="store_true",
        help="断片の最後の切れ目も、元の行の注釈で境界の学習に使う（--fragment-cut free のとき）",
    )
    ap.add_argument(
        "--checkpoint-minutes",
        type=float,
        default=2.0,
        help="途中の状態を保存する間隔（分）",
    )
    ap.add_argument(
        "--no-weights",
        action="store_true",
        help="重み（model.pt）を保存しない（dev の数字だけを比べる取り直し用）",
    )
    args = ap.parse_args()

    train_recs = load_jsonl(str(DATA / "train_full.jsonl"))
    dev_recs = load_jsonl(str(DATA / "dev_full.jsonl"))
    if args.smoke:
        args.model, args.epochs = "tiny", 1
        train_recs, dev_recs = train_recs[:100], dev_recs[:30]
    verify_records(train_recs + dev_recs)
    if args.boundary and not any(s.space for r in train_recs for s in r.slots):
        raise SystemExit(
            "学習データにマスあけ（+S）が入っていない（古いデータ）。"
            "手元で make_kit を実行してキットを作り直す"
        )
    if args.fragments and not args.boundary:
        raise SystemExit(
            "--fragments は境界の学習に使うので、--boundary と一緒に指定する"
        )
    if args.fragment_last_gap and not args.fragments:
        raise SystemExit("--fragment-last-gap は --fragments と一緒に指定する")
    if args.fragment_last_gap and args.fragment_cut != "free":
        raise SystemExit(
            "--fragment-last-gap は --fragment-cut free と一緒に指定する"
            "（unit の断片の最後の切れ目は必ずマスあけで、学習しても何も教えない）"
        )

    fit = train_recs + dev_recs if args.all_data else train_recs
    name = args.name or (
        ("smoke" if args.smoke else "full" if args.all_data else "train")
        + ("_joint" if args.boundary else "")
        + ("_frag" if args.fragments else "")
        + ("_unit" if args.fragments and args.fragment_cut == "unit" else "")
        + ("_last" if args.fragment_last_gap else "")
    )
    out_dir = WORK / "finetuned" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 語彙は元のモデルのもの（tiny は char-v3 のものを借りる）
    vocab = CharVocab.from_model(DEFAULT_MODEL if args.model == "tiny" else args.model)
    labels = Labels(fit)
    cands = Candidates(fit, labels)
    print(
        f"device={args.device} model={args.model} 学習 {len(fit)} 行 / dev {len(dev_recs)} 行"
        f" / ラベル {len(labels)}"
        + (
            f" / 読みと境界の同時学習（境界の重み {args.boundary_weight}）"
            if args.boundary
            else ""
        )
        + (
            f" / 境界に断片を 1 行平均 {args.fragments} 個"
            f"（{args.fragment_min}〜{args.fragment_max} 字"
            + (
                "・マスあけの単位の頭から終わりまで 2 単位以上"
                if args.fragment_cut == "unit"
                else "・スロットの切れ目ならどこでも"
            )
            + ("・最後の切れ目も学習" if args.fragment_last_gap else "")
            + "）"
            if args.fragments
            else ""
        ),
        flush=True,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(args.model, len(labels), vocab, boundary=args.boundary)
    fit_exs = make_examples(fit, vocab, labels, cands)
    dev_exs = make_examples(dev_recs, vocab, labels, cands)
    # dev の断片は決まった作り方（モデルどうしで同じもの）。最後の切れ目は測らない（推論でも問わない）。
    # 学習の切り出し方にかかわらず、unit（本命）と free（語の途中の切れ端に近い）の両方で測る
    dev_frag_exs = make_examples(dev_fragments(dev_recs, "unit"), vocab, labels, cands)
    dev_free_exs = make_examples(dev_fragments(dev_recs, "free"), vocab, labels, cands)
    # 学習の断片はエポックごとに切り直す（再開しても同じものになるよう、乱数の種を決めておく）
    extra = (
        {
            epoch: boundary_examples(
                fragments(
                    fit,
                    random.Random(f"fragments-{args.seed}-{epoch}"),
                    args.fragments,
                    args.fragment_min,
                    args.fragment_max,
                    cut=args.fragment_cut,
                ),
                vocab,
                labels,
                cands,
                last_gap=args.fragment_last_gap,
            )
            for epoch in range(1, args.epochs + 1)
        }
        if args.fragments
        else None
    )

    # エポックごとの診断の記録（再開しても前のエポックの分が残るよう、ファイルに足していく）
    history_path = out_dir / "history.jsonl"
    if not (out_dir / "checkpoint.pt").exists():
        history_path.unlink(missing_ok=True)

    def diag(epoch: int) -> str:
        acc = dev_accuracy(model, dev_recs, dev_exs, cands, labels, vocab, args.device)
        rec: dict = {"epoch": epoch, "dev_acc": acc}
        msg = f"dev={acc:.4f}"
        if args.boundary:
            b = dev_boundary(model, dev_exs, vocab, args.device)
            f = dev_boundary(model, dev_frag_exs, vocab, args.device)
            g = dev_boundary(model, dev_free_exs, vocab, args.device)
            rec |= {
                "boundary_errors": b["errors"],
                "fragment_errors": f["errors"],
                "free_fragment_errors": g["errors"],
            }
            msg += (
                f" 境界F1={b['f1']:.4f}（誤り {b['errors']}）"
                f" 断片の境界F1={f['f1']:.4f}（誤り {f['errors']}）"
                f" free の断片の境界F1={g['f1']:.4f}（誤り {g['errors']}）"
            )
        cl = collapse_stats(model, dev_exs, vocab, args.device)
        cf = collapse_stats(model, dev_frag_exs, vocab, args.device)
        cg = collapse_stats(model, dev_free_exs, vocab, args.device)
        rec |= {
            "collapse_lines": cl["all"],
            "collapse_fragment_last": cf["last"],
            "collapse_fragment_inner": cf["inner"],
            "collapse_free_fragment_last": cg["last"],
            "collapse_free_fragment_inner": cg["inner"],
        }
        msg += (
            f" 潰れ: 行 {cl['all']:.1%}・断片の最後 {cf['last']:.1%}・断片のほか {cf['inner']:.1%}"
            f"・free の断片の最後 {cg['last']:.1%}・free の断片のほか {cg['inner']:.1%}"
        )
        if "sep_prob" in cl:
            rec["sep_prob"] = cl["sep_prob"]
            msg += f"・[SEP] の確率 {cl['sep_prob']:.2f}"
        with open(history_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return msg

    # 続きから再開してよいかの照合に使う（保存先の名前・デバイス・保存間隔・重みを残すかは学習の中身に関わらない）
    digest = hashlib.sha256()
    for f in ("train_full.jsonl", "dev_full.jsonl"):
        digest.update((DATA / f).read_bytes())
    run_key = {
        k: v
        for k, v in vars(args).items()
        if k not in ("name", "device", "checkpoint_minutes", "no_weights")
    } | {"data": digest.hexdigest()}
    checkpoint = out_dir / "checkpoint.pt"

    t0 = time.time()
    train(
        model,
        fit_exs,
        cands,
        vocab,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        on_epoch=diag,
        checkpoint=checkpoint,
        run_key=run_key,
        checkpoint_every=args.checkpoint_minutes * 60,
        boundary_weight=args.boundary_weight,
        extra=extra,
    )
    acc = dev_accuracy(model, dev_recs, dev_exs, cands, labels, vocab, args.device)
    print(f"dev 漢字スロット一致率 {acc:.4f}（{time.time() - t0:.0f} 秒）", flush=True)
    bnd = dev_boundary(model, dev_exs, vocab, args.device) if args.boundary else None
    bnd_frag = (
        dev_boundary(model, dev_frag_exs, vocab, args.device) if args.boundary else None
    )
    bnd_free = (
        dev_boundary(model, dev_free_exs, vocab, args.device) if args.boundary else None
    )
    if bnd:
        print(f"dev {boundary_summary(bnd)}", flush=True)
        print(f"dev の断片 {boundary_summary(bnd_frag)}", flush=True)
        print(f"dev の free の断片 {boundary_summary(bnd_free)}", flush=True)

    model.cpu()
    if args.no_weights:
        # 前に同じ名前で保存した重みが残っていると、この meta と食い違うので消す
        (out_dir / "model.pt").unlink(missing_ok=True)
    else:
        torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "encoder.json").write_text(
        model.encoder.config.to_json_string(), encoding="utf-8"
    )
    (out_dir / "labels.json").write_text(
        json.dumps(labels.names, ensure_ascii=False), encoding="utf-8"
    )
    chars = {
        c: [i for i in range(len(labels)) if cands.table[row, i]]
        for c, row in cands.chars.items()
    }
    (out_dir / "cands.json").write_text(
        json.dumps({"chars": chars}, ensure_ascii=False), encoding="utf-8"
    )
    meta = {
        "base_model": args.model,
        "all_data": args.all_data,
        "train_lines": len(fit),
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "dev_acc": acc,
        "dev_is_heldout": not args.all_data,
        "seconds": round(time.time() - t0),
        "boundary": args.boundary,
        "weights_saved": not args.no_weights,
    }
    if bnd:
        meta["boundary_weight"] = args.boundary_weight
        meta["fragments"] = args.fragments
        if args.fragments:
            meta["fragment_chars"] = [args.fragment_min, args.fragment_max]
            meta["fragment_cut"] = args.fragment_cut
            meta["fragment_last_gap"] = args.fragment_last_gap
        meta["dev_boundary"] = bnd
        # dev_fragments_cut が無い古い meta の dev_boundary_fragments は free の断片のもの
        meta["dev_fragments_cut"] = "unit"
        meta["dev_boundary_fragments"] = bnd_frag
        meta["dev_boundary_fragments_free"] = bnd_free
    # エポックごとの診断（再開で同じエポックが 2 度書かれていれば、後のものを使う）
    if history_path.exists():
        by_epoch = {}
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                h = json.loads(line)
                by_epoch[h["epoch"]] = h
        meta["history"] = [by_epoch[e] for e in sorted(by_epoch)]
    # どの basic_raw.tsv から作ったかを残す（export_data が書いた meta。古いキットには無い）
    data_meta = DATA / "meta.json"
    if data_meta.exists():
        meta["data"] = json.loads(data_meta.read_text(encoding="utf-8"))
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checkpoint.unlink(missing_ok=True)
    print(f"保存先: {out_dir}")


if __name__ == "__main__":
    main()
