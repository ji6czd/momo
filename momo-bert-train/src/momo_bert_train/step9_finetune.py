"""BERT を微調整して、読みの分類モデルを保存する（クラウドの GPU 用。手元の CPU では --smoke のみ）。

tools/llm_reading の 10_encoder_train.py と同じ手順（8エポック・lr 5e-5・バッチ 32・漢字スロットを
辞書候補に絞った softmax で選ぶ）で学習し、結果を1つのディレクトリに保存する。保存したものは
step10_infer.py が手元の CPU で読み込める。

    uv run python -m momo_bert_train.step9_finetune --smoke                  # 動作確認（極小モデル・数十行）
    uv run python -m momo_bert_train.step9_finetune                          # train で学習し、dev で評価して保存
    uv run python -m momo_bert_train.step9_finetune --all-data --name full   # train+dev の全データで学習して保存
    uv run python -m momo_bert_train.step9_finetune --boundary               # 読みと境界（マスあけ）を同時に学習

出力: work/finetuned/<名前>/
  model.pt      ReadingModel の重み（float32。--boundary では境界の出力層も入る）
  encoder.json  BERT 本体の構成（学習済みの重みを取り直さずに読み込める）
  labels.json   読みのラベル表
  cands.json    文字 → 読み候補（ラベル番号）の辞書
  meta.json     学習の設定と dev の一致率（--boundary では境界の誤りの数と F1 も）、学習データの meta
                （basic_raw.tsv のハッシュ）

名前の既定は train / full。--boundary を付けると train_joint / full_joint。
--all-data では dev も学習に使うので、meta の dev 一致率は学習済みの文での値（評価にならない）。
境界の dev の数字は、切れ目ごと（行末と原文の空白をはさむ切れ目を除く）に、確率 0.5 以上をマスあけとして数える。

学習の途中の状態を 5 分ごとと各エポックの終わりに work/finetuned/<名前>/checkpoint.pt に保存する。
VM がプリエンプトされたら、同じコマンドを打ち直せば続きから学習する（設定かデータが違えば最初から）。
保存し終えたら消える。meta の秒数は、再開したときは再開してからの時間。
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
    WORK,
    Candidates,
    CharVocab,
    Labels,
    build_model,
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
        "--checkpoint-minutes",
        type=float,
        default=5.0,
        help="途中の状態を保存する間隔（分）",
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

    fit = train_recs + dev_recs if args.all_data else train_recs
    name = args.name or (
        ("smoke" if args.smoke else "full" if args.all_data else "train")
        + ("_joint" if args.boundary else "")
    )
    out_dir = WORK / "finetuned" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 語彙は char-v3 のもの（tiny はこれを借りる）
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
        ),
        flush=True,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(args.model, len(labels), vocab, boundary=args.boundary)
    fit_exs = make_examples(fit, vocab, labels, cands)
    dev_exs = make_examples(dev_recs, vocab, labels, cands)

    def diag(_epoch: int) -> str:
        acc = dev_accuracy(model, dev_recs, dev_exs, cands, labels, vocab, args.device)
        msg = f"dev={acc:.4f}"
        if args.boundary:
            b = dev_boundary(model, dev_exs, vocab, args.device)
            msg += f" 境界F1={b['f1']:.4f}（誤り {b['errors']}）"
        return msg

    # 続きから再開してよいかの照合に使う（保存先の名前・デバイス・保存間隔は学習の中身に関わらない）
    digest = hashlib.sha256()
    for f in ("train_full.jsonl", "dev_full.jsonl"):
        digest.update((DATA / f).read_bytes())
    run_key = {
        k: v
        for k, v in vars(args).items()
        if k not in ("name", "device", "checkpoint_minutes")
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
    )
    acc = dev_accuracy(model, dev_recs, dev_exs, cands, labels, vocab, args.device)
    print(f"dev 漢字スロット一致率 {acc:.4f}（{time.time() - t0:.0f} 秒）", flush=True)
    bnd = dev_boundary(model, dev_exs, vocab, args.device) if args.boundary else None
    if bnd:
        print(f"dev {boundary_summary(bnd)}", flush=True)

    model.cpu()
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
    }
    if bnd:
        meta["boundary_weight"] = args.boundary_weight
        meta["dev_boundary"] = bnd
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
