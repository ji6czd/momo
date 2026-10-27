"""BERT を微調整して、読みの分類モデルを保存する（クラウドの GPU 用。手元の CPU では --smoke のみ）。

tools/llm_reading の 10_encoder_train.py と同じ手順（8エポック・lr 5e-5・バッチ 32・漢字スロットを
辞書候補に絞った softmax で選ぶ）で学習し、結果を1つのディレクトリに保存する。保存したものは
step10_infer.py が手元の CPU で読み込める。

    uv run python -m momo_bert_train.step9_finetune --smoke                  # 動作確認（極小モデル・数十行）
    uv run python -m momo_bert_train.step9_finetune                          # train で学習し、dev で評価して保存
    uv run python -m momo_bert_train.step9_finetune --all-data --name full   # train+dev の全データで学習して保存

出力: work/finetuned/<名前>/
  model.pt      ReadingModel の重み（float32）
  encoder.json  BERT 本体の構成（学習済みの重みを取り直さずに読み込める）
  labels.json   読みのラベル表
  cands.json    文字 → 読み候補（ラベル番号）の辞書
  meta.json     学習の設定と dev の一致率、学習データの meta（basic_raw.tsv のハッシュ）

--all-data では dev も学習に使うので、meta の dev 一致率は学習済みの文での値（評価にならない）。
"""

import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--name", default="", help="保存先の名前（既定: train か full）")
    ap.add_argument("--all-data", action="store_true", help="dev も学習に使う")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true", help="極小モデルで数十行だけ通す")
    args = ap.parse_args()

    train_recs = load_jsonl(str(DATA / "train_full.jsonl"))
    dev_recs = load_jsonl(str(DATA / "dev_full.jsonl"))
    if args.smoke:
        args.model, args.epochs = "tiny", 1
        train_recs, dev_recs = train_recs[:100], dev_recs[:30]
    verify_records(train_recs + dev_recs)

    fit = train_recs + dev_recs if args.all_data else train_recs
    name = args.name or (
        "smoke" if args.smoke else "full" if args.all_data else "train"
    )
    out_dir = WORK / "finetuned" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 語彙は char-v3 のもの（tiny はこれを借りる）
    vocab = CharVocab.from_model(DEFAULT_MODEL if args.model == "tiny" else args.model)
    labels = Labels(fit)
    cands = Candidates(fit, labels)
    print(
        f"device={args.device} model={args.model} 学習 {len(fit)} 行 / dev {len(dev_recs)} 行"
        f" / ラベル {len(labels)}",
        flush=True,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(args.model, len(labels), vocab)
    fit_exs = make_examples(fit, vocab, labels, cands)
    dev_exs = make_examples(dev_recs, vocab, labels, cands)

    def diag(_epoch: int) -> str:
        acc = dev_accuracy(model, dev_recs, dev_exs, cands, labels, vocab, args.device)
        return f"dev={acc:.4f}"

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
    )
    acc = dev_accuracy(model, dev_recs, dev_exs, cands, labels, vocab, args.device)
    print(f"dev 漢字スロット一致率 {acc:.4f}（{time.time() - t0:.0f} 秒）", flush=True)

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
    }
    # どの basic_raw.tsv から作ったかを残す（export_data が書いた meta。古いキットには無い）
    data_meta = DATA / "meta.json"
    if data_meta.exists():
        meta["data"] = json.loads(data_meta.read_text(encoding="utf-8"))
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"保存先: {out_dir}")


if __name__ == "__main__":
    main()
