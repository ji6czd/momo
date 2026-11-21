"""step9_finetune で保存した微調整モデルを、Rust（momors-bert）が読む ONNX に書き出す。

    uv run export_onnx                    # work/finetuned/train → work/onnx/train
    uv run export_onnx --name full        # 実用（全データで学習したもの）
    uv run export_onnx --no-eval          # dev での突き合わせを省く
    uv run export_onnx --name full --no-eval --zip-f32   # 配布用 zip に float32 版も入れる

グラフは BERT 本体と読みの出力層だけで、入力 input_ids / attention_mask [B, T]（int64）、
出力 logits [B, T, ラベル数]（float32）。候補への絞り込みと softmax はグラフの外（Rust 側）で行う。
境界も学習したモデル（step9_finetune --boundary）は、出力 boundary_logits [B, T]（float32。その文字の
後ろでマスをあける logit）も持つ。Rust はこの出力があればマスあけも BERT で決める。
int8 版は onnxruntime の動的量子化（重みだけ int8、活性は実行時に量子化）。出力層と、最後の 2 層の
FFN 出力（3072→768）は float32 のまま残す。全部を int8 にすると dev が 97.78% → 97.43% に落ちるが、
これらを残すと 97.77% に戻る（GELU の後の外れ値に弱いのは後ろの層。前の層を残しても戻らない）。

出力: work/onnx/<名前>/
  reading_bert.onnx        float32
  reading_bert.int8.onnx   int8 動的量子化
  labels.json              読みのラベル表（継続は ""）
  cands.json               文字 → 読み候補（ラベル番号）
  vocab.txt                文字の語彙（1行1トークン）
  reference.jsonl          突き合わせ用。数文分の入力 id と、漢字位置の logit（float32 版の値）。
                           境界の出力があれば、全文字の境界の logit も
  meta.json                書き出し元、学習に使った basic_raw.tsv のハッシュ、dev の一致率
                           （境界の出力があれば dev の境界の誤りと F1 も）
work/onnx/momo-bert-<名前>.zip   配布用（GitHub のリリースに `gh release upload <tag>` で上げる）
  momo-bert/ の下に reading_bert.int8.onnx・labels.json・cands.json・vocab.txt・meta.json・NOTICE.txt・
  LICENSE.txt（--zip-f32 で reading_bert.onnx も入る）。展開した momo-bert/ を `momo --bert` に渡す。
  元のモデルは Apache-2.0 なので、再配布にはライセンスの本文（LICENSE.txt）と、変更したことの
  表示（NOTICE.txt）を添える。

meta.json の basic_raw_sha256 は、モデルを学習したときの basic_raw.tsv の SHA-256（改行コードは
無視）。momo 本体の w7 と同じ basic_raw.tsv から作ったかを確かめるのに使う。書き出すときに
今の dataset/basic_raw.tsv と比べ、違えば注意を出す。
"""

import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common import DATA, DEFAULT_MODEL, WORK, load_jsonl
from .export_data import raw_digest
from .step10_infer import FineTunedReader, is_kanji

OPSET = 18
# 突き合わせ用の文。dev の先頭からも足す
REFERENCE_TEXTS = [
    "駅前の市場で魚を買った。",
    "株式市場が反発した。",
    "今日のテレビで、「今日の中東情勢」という番組が放送されていた。",
    "人々は堂々と歩いた。",
]
N_REFERENCE_DEV = 4

ZIP_DIR = "momo-bert"  # 配布用 zip の中のディレクトリ名
ZIP_FILES = [
    "reading_bert.int8.onnx",
    "labels.json",
    "cands.json",
    "vocab.txt",
    "meta.json",
]
NOTICE = f"""momo BERT reading model

This model is a fine-tuned derivative of {DEFAULT_MODEL}
(https://huggingface.co/{DEFAULT_MODEL}),
licensed under the Apache License, Version 2.0
(https://www.apache.org/licenses/LICENSE-2.0). A copy of the license is in LICENSE.txt.

Modifications: fine-tuned to predict the readings (kana) of kanji for the momo
Japanese braille translation project, and converted to ONNX (float32 / int8).
See meta.json for the training data and the evaluation.
"""
# Apache License 2.0 の本文（https://www.apache.org/licenses/LICENSE-2.0.txt のまま）
LICENSE = Path(__file__).with_name("LICENSE-Apache-2.0.txt")


class LogitsGraph(nn.Module):
    """ReadingModel を「全トークンの logit」を返す形にしたもの（位置の選び出しは Rust 側）。

    境界の出力層があれば (読みの logit, 境界の logit) を返す。"""

    def __init__(self, reading_model):
        super().__init__()
        self.encoder = reading_model.encoder
        self.head = reading_model.head
        self.boundary_head = reading_model.boundary_head

    def forward(self, input_ids, attention_mask):
        h = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        if self.boundary_head is None:
            return self.head(h)
        return self.head(h), self.boundary_head(h).squeeze(-1)


def output_names(graph: LogitsGraph) -> list[str]:
    return ["logits"] + (["boundary_logits"] if graph.boundary_head is not None else [])


def export_float(graph: LogitsGraph, path: Path) -> None:
    ids = torch.tensor([[2, 100, 200, 300, 3]], dtype=torch.long)
    mask = torch.ones_like(ids)
    names = output_names(graph)
    torch.onnx.export(
        graph,
        (ids, mask),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=names,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            **{n: {0: "batch", 1: "seq"} for n in names},
        },
        opset_version=OPSET,
        dynamo=False,
    )


def int8_excluded(num_layers: int, boundary: bool) -> list[str]:
    """int8 にしない MatMul のノード名（出力層と、最後の 2 層の FFN 出力）。"""
    heads = ["/head/MatMul"] + (["/boundary_head/MatMul"] if boundary else [])
    return heads + [
        f"/encoder/encoder/layer.{i}/output/dense/MatMul"
        for i in range(num_layers - 2, num_layers)
    ]


def export_int8(src: Path, dst: Path, num_layers: int, boundary: bool) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = dst.with_suffix(".pre.onnx")
    # BERT の注意マスクの形は記号的な形推論が通らないので、ONNX 標準の形推論と最適化だけ掛ける
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    quantize_dynamic(
        str(pre),
        str(dst),
        weight_type=QuantType.QInt8,
        per_channel=True,
        nodes_to_exclude=int8_excluded(num_layers, boundary),
    )
    pre.unlink()


def write_zip(out: Path, name: str, with_f32: bool) -> Path:
    """配布用の zip（中は momo-bert/ の下）を work/onnx/ に作る。"""
    path = out.parent / f"momo-bert-{name}.zip"
    files = ZIP_FILES + (["reading_bert.onnx"] if with_f32 else [])
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(out / f, f"{ZIP_DIR}/{f}")
        z.writestr(f"{ZIP_DIR}/NOTICE.txt", NOTICE)
        z.write(LICENSE, f"{ZIP_DIR}/LICENSE.txt")
    return path


def data_provenance(src: Path) -> dict:
    """微調整モデルの meta と、学習に使った basic_raw.tsv のハッシュ。今のファイルと違えば注意を出す。"""
    meta_path = src / "meta.json"
    finetune = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    trained = finetune.get("data", {}).get("basic_raw_sha256")
    current = raw_digest()
    if trained is None:
        print(
            "注意: 学習に使った basic_raw.tsv のハッシュがモデルに記録されていない"
            "（記録を始める前のキットで学習したモデル）"
        )
    elif trained != current:
        print(
            "注意: 学習に使った basic_raw.tsv と、今の dataset/basic_raw.tsv が違う"
            f"（学習 {trained[:12]} / 今 {current[:12]}）"
        )
    return {
        "basic_raw_sha256": trained,
        "basic_raw_sha256_at_export": current,
        "finetune": finetune,
    }


def session(path: Path, threads: int):
    import onnxruntime as ort

    opt = ort.SessionOptions()
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), opt, providers=["CPUExecutionProvider"])


def encode(reader: FineTunedReader, text: str) -> list[int]:
    v = reader.vocab
    return [v.cls, *v.encode(text[: reader.max_chars]), v.sep]


def pick(logits: np.ndarray, allowed: list[int] | None) -> int:
    if not allowed:
        return int(logits.argmax())
    return allowed[int(logits[allowed].argmax())]


def gaps(rec, max_chars: int):
    """境界の学習と同じ切れ目（encoder_core.make_example）の (直前の字の位置, 注釈のマスあけ)。"""
    for a, b in zip(rec.slots, rec.slots[1:]):
        if rec.text[a.end : b.start] or a.surface.isspace() or b.surface.isspace():
            continue
        if b.start >= max_chars:
            break
        yield a.end - 1, a.space


def dev_eval(reader: FineTunedReader, run, recs) -> dict:
    """dev の漢字スロット一致率・境界の誤り・1行あたりの平均時間（ms）。

    run(ids) は ([T, L] の読みの logit, [T] の境界の logit か None) を返す。"""
    label_id = {g: i for i, g in enumerate(reader.labels)}
    hit = n = 0
    tp = fp = fn = 0
    t0 = time.perf_counter()
    for r in recs:
        logits, boundary = run(encode(reader, r.text))
        for s in r.slots:
            if (
                s.ctype not in ("KANJI", "JAPANESE_NUMERIC")
                or s.start >= reader.max_chars
            ):
                continue
            best = pick(logits[1 + s.start], reader.cands.get(s.surface))
            hit += best == label_id.get(s.gold, -1)
            n += 1
        if boundary is not None:
            for p, gold in gaps(r, reader.max_chars):
                got = bool(boundary[1 + p] >= 0)
                tp += got and gold
                fp += got and not gold
                fn += gold and not got
    out = {
        "acc": hit / n if n else 0.0,
        "ms": (time.perf_counter() - t0) * 1000 / max(len(recs), 1),
    }
    if reader.model.boundary_head is not None:
        out["boundary_errors"] = fp + fn
        out["boundary_f1"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="train", help="work/finetuned/ 下の名前")
    ap.add_argument("--threads", type=int, default=0, help="評価のスレッド数（0=既定）")
    ap.add_argument("--no-eval", action="store_true", help="dev での突き合わせを省く")
    ap.add_argument("--no-zip", action="store_true", help="配布用 zip を作らない")
    ap.add_argument(
        "--zip-f32", action="store_true", help="配布用 zip に float32 版も入れる"
    )
    args = ap.parse_args()

    src = WORK / "finetuned" / args.name
    out = WORK / "onnx" / args.name
    out.mkdir(parents=True, exist_ok=True)
    reader = FineTunedReader(args.name)
    graph = LogitsGraph(reader.model).eval()
    boundary = graph.boundary_head is not None
    if boundary:
        print("境界の出力層あり: boundary_logits も書き出す")

    f32, i8 = out / "reading_bert.onnx", out / "reading_bert.int8.onnx"
    t0 = time.time()
    export_float(graph, f32)
    print(
        f"float32: {f32}（{f32.stat().st_size / 2**20:.0f} MiB、{time.time() - t0:.0f} 秒）"
    )
    t0 = time.time()
    export_int8(f32, i8, reader.model.encoder.config.num_hidden_layers, boundary)
    print(
        f"int8:    {i8}（{i8.stat().st_size / 2**20:.0f} MiB、{time.time() - t0:.0f} 秒）"
    )

    for name in ("labels.json", "cands.json"):
        shutil.copyfile(src / name, out / name)
    from huggingface_hub import hf_hub_download

    shutil.copyfile(hf_hub_download(DEFAULT_MODEL, "vocab.txt"), out / "vocab.txt")

    sess_f32 = session(f32, args.threads)
    sess_i8 = session(i8, args.threads)

    names = output_names(graph)

    def run_onnx(sess):
        def run(ids):
            a = np.array([ids], dtype=np.int64)
            res = sess.run(names, {"input_ids": a, "attention_mask": np.ones_like(a)})
            return res[0][0], (res[1][0] if boundary else None)

        return run

    @torch.no_grad()
    def run_torch(ids):
        a = torch.tensor([ids])
        res = graph(a, torch.ones_like(a))
        if not boundary:
            return res[0].numpy(), None
        return res[0][0].numpy(), res[1][0].numpy()

    dev = load_jsonl(str(DATA / "dev_full.jsonl"))
    texts = REFERENCE_TEXTS + [r.text for r in dev[:N_REFERENCE_DEV]]
    max_diff = 0.0
    with open(out / "reference.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for text in texts:
            text = text[: reader.max_chars]
            ids = encode(reader, text)
            ref, ref_b = run_onnx(sess_f32)(ids)
            got, got_b = run_torch(ids)
            max_diff = max(max_diff, float(np.abs(ref - got).max()))
            pos = [i for i, c in enumerate(text) if is_kanji(c)]
            row = {
                "text": text,
                "ids": ids,
                "pos": pos,
                "logits": [ref[1 + p].tolist() for p in pos],
            }
            if boundary:
                max_diff = max(max_diff, float(np.abs(ref_b - got_b).max()))
                # 文字ごと（[CLS] と [SEP] を除く）
                row["boundary"] = ref_b[1 : 1 + len(text)].tolist()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"PyTorch と ONNX float32 の logit の最大差: {max_diff:.2e}")

    meta = {
        "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(src),
        **data_provenance(src),
        "opset": OPSET,
        "labels": len(reader.labels),
        "max_chars": reader.max_chars,
        "boundary": boundary,
        "torch_vs_onnx_max_diff": max_diff,
    }
    if not args.no_eval:
        for key, run in (
            ("torch", run_torch),
            ("onnx_f32", run_onnx(sess_f32)),
            ("onnx_int8", run_onnx(sess_i8)),
        ):
            ev = dev_eval(reader, run, dev)
            meta[f"dev_acc_{key}"] = ev["acc"]
            meta[f"dev_ms_per_line_{key}"] = round(ev["ms"], 2)
            line = f"dev 漢字スロット一致率 {key:9s} {ev['acc']:.4f}"
            if boundary:
                meta[f"dev_boundary_errors_{key}"] = ev["boundary_errors"]
                meta[f"dev_boundary_f1_{key}"] = ev["boundary_f1"]
                line += f"・境界の誤り {ev['boundary_errors']}（F1={ev['boundary_f1']:.4f}）"
            print(f"{line}（1行 {ev['ms']:.1f} ms）", flush=True)
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"保存先: {out}")
    if not args.no_zip:
        z = write_zip(out, args.name, args.zip_f32)
        print(f"配布用: {z}（{z.stat().st_size / 2**20:.0f} MiB）")


if __name__ == "__main__":
    main()
