"""step9_finetune で保存した微調整モデルを、Rust（momors-bert）が読む ONNX に書き出す。

    uv run export_onnx                    # work/finetuned/train → work/onnx/train
    uv run export_onnx --name full        # 実用（全データで学習したもの）
    uv run export_onnx --no-eval          # dev での突き合わせを省く
    uv run export_onnx --name full --no-eval --zip-f32   # 配布用 zip に float32 版も入れる

グラフは BERT 本体と読みの出力層だけで、入力 input_ids / attention_mask [B, T]（int64）、
出力 logits [B, T, ラベル数]（float32）。候補への絞り込みと softmax はグラフの外（Rust 側）で行う。
int8 版は onnxruntime の動的量子化（重みだけ int8、活性は実行時に量子化）。出力層と、最後の 2 層の
FFN 出力（3072→768）は float32 のまま残す。全部を int8 にすると dev が 97.78% → 97.43% に落ちるが、
これらを残すと 97.77% に戻る（GELU の後の外れ値に弱いのは後ろの層。前の層を残しても戻らない）。

出力: work/onnx/<名前>/
  reading_bert.onnx        float32
  reading_bert.int8.onnx   int8 動的量子化
  labels.json              読みのラベル表（継続は ""）
  cands.json               文字 → 読み候補（ラベル番号）
  vocab.txt                文字の語彙（1行1トークン）
  reference.jsonl          突き合わせ用。数文分の入力 id と、漢字位置の logit（float32 版の値）
  meta.json                書き出し元、学習に使った basic_raw.tsv のハッシュ、dev の一致率
work/onnx/momo-bert-<名前>.zip   配布用（GitHub のリリースに `gh release upload <tag>` で上げる）
  momo-bert/ の下に reading_bert.int8.onnx・labels.json・cands.json・vocab.txt・meta.json・NOTICE.txt
  （--zip-f32 で reading_bert.onnx も入る）。展開した momo-bert/ を `momo --bert` に渡す。

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
ZIP_FILES = ["reading_bert.int8.onnx", "labels.json", "cands.json", "vocab.txt", "meta.json"]
NOTICE = f"""momo BERT reading model

This model is a fine-tuned derivative of {DEFAULT_MODEL}
(https://huggingface.co/{DEFAULT_MODEL}),
licensed under the Apache License, Version 2.0
(https://www.apache.org/licenses/LICENSE-2.0).

Modifications: fine-tuned to predict the readings (kana) of kanji for the momo
Japanese braille translation project, and converted to ONNX (float32 / int8).
See meta.json for the training data and the evaluation.
"""


class LogitsGraph(nn.Module):
    """ReadingModel を「全トークンの logit」を返す形にしたもの（位置の選び出しは Rust 側）。"""

    def __init__(self, reading_model):
        super().__init__()
        self.encoder = reading_model.encoder
        self.head = reading_model.head

    def forward(self, input_ids, attention_mask):
        h = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return self.head(h)


def export_float(graph: nn.Module, path: Path) -> None:
    ids = torch.tensor([[2, 100, 200, 300, 3]], dtype=torch.long)
    mask = torch.ones_like(ids)
    torch.onnx.export(
        graph,
        (ids, mask),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=OPSET,
        dynamo=False,
    )


def int8_excluded(num_layers: int) -> list[str]:
    """int8 にしない MatMul のノード名（出力層と、最後の 2 層の FFN 出力）。"""
    return ["/head/MatMul"] + [
        f"/encoder/encoder/layer.{i}/output/dense/MatMul"
        for i in range(num_layers - 2, num_layers)
    ]


def export_int8(src: Path, dst: Path, num_layers: int) -> None:
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
        nodes_to_exclude=int8_excluded(num_layers),
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
    return path


def data_provenance(src: Path) -> dict:
    """微調整モデルの meta と、学習に使った basic_raw.tsv のハッシュ。今のファイルと違えば注意を出す。"""
    meta_path = src / "meta.json"
    finetune = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
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


def dev_accuracy(reader: FineTunedReader, run, recs) -> tuple[float, float]:
    """dev の漢字スロット一致率と、1行あたりの平均時間（ms）。run(ids) は [T, L] の logit を返す。"""
    label_id = {g: i for i, g in enumerate(reader.labels)}
    hit = n = 0
    t0 = time.perf_counter()
    for r in recs:
        logits = run(encode(reader, r.text))
        for s in r.slots:
            if (
                s.ctype not in ("KANJI", "JAPANESE_NUMERIC")
                or s.start >= reader.max_chars
            ):
                continue
            best = pick(logits[1 + s.start], reader.cands.get(s.surface))
            hit += best == label_id.get(s.gold, -1)
            n += 1
    ms = (time.perf_counter() - t0) * 1000 / max(len(recs), 1)
    return (hit / n if n else 0.0), ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="train", help="work/finetuned/ 下の名前")
    ap.add_argument("--threads", type=int, default=0, help="評価のスレッド数（0=既定）")
    ap.add_argument("--no-eval", action="store_true", help="dev での突き合わせを省く")
    ap.add_argument("--no-zip", action="store_true", help="配布用 zip を作らない")
    ap.add_argument("--zip-f32", action="store_true", help="配布用 zip に float32 版も入れる")
    args = ap.parse_args()

    src = WORK / "finetuned" / args.name
    out = WORK / "onnx" / args.name
    out.mkdir(parents=True, exist_ok=True)
    reader = FineTunedReader(args.name)
    graph = LogitsGraph(reader.model).eval()

    f32, i8 = out / "reading_bert.onnx", out / "reading_bert.int8.onnx"
    t0 = time.time()
    export_float(graph, f32)
    print(
        f"float32: {f32}（{f32.stat().st_size / 2**20:.0f} MiB、{time.time() - t0:.0f} 秒）"
    )
    t0 = time.time()
    export_int8(f32, i8, reader.model.encoder.config.num_hidden_layers)
    print(
        f"int8:    {i8}（{i8.stat().st_size / 2**20:.0f} MiB、{time.time() - t0:.0f} 秒）"
    )

    for name in ("labels.json", "cands.json"):
        shutil.copyfile(src / name, out / name)
    from huggingface_hub import hf_hub_download

    shutil.copyfile(hf_hub_download(DEFAULT_MODEL, "vocab.txt"), out / "vocab.txt")

    sess_f32 = session(f32, args.threads)
    sess_i8 = session(i8, args.threads)

    def run_onnx(sess):
        def run(ids):
            a = np.array([ids], dtype=np.int64)
            return sess.run(
                ["logits"], {"input_ids": a, "attention_mask": np.ones_like(a)}
            )[0][0]

        return run

    @torch.no_grad()
    def run_torch(ids):
        a = torch.tensor([ids])
        return graph(a, torch.ones_like(a))[0].numpy()

    dev = load_jsonl(str(DATA / "dev_full.jsonl"))
    texts = REFERENCE_TEXTS + [r.text for r in dev[:N_REFERENCE_DEV]]
    max_diff = 0.0
    with open(out / "reference.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for text in texts:
            text = text[: reader.max_chars]
            ids = encode(reader, text)
            ref = run_onnx(sess_f32)(ids)
            max_diff = max(max_diff, float(np.abs(ref - run_torch(ids)).max()))
            pos = [i for i, c in enumerate(text) if is_kanji(c)]
            row = {
                "text": text,
                "ids": ids,
                "pos": pos,
                "logits": [ref[1 + p].tolist() for p in pos],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"PyTorch と ONNX float32 の logit の最大差: {max_diff:.2e}")

    meta = {
        "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(src),
        **data_provenance(src),
        "opset": OPSET,
        "labels": len(reader.labels),
        "max_chars": reader.max_chars,
        "torch_vs_onnx_max_diff": max_diff,
    }
    if not args.no_eval:
        for key, run in (
            ("torch", run_torch),
            ("onnx_f32", run_onnx(sess_f32)),
            ("onnx_int8", run_onnx(sess_i8)),
        ):
            acc, ms = dev_accuracy(reader, run, dev)
            meta[f"dev_acc_{key}"] = acc
            meta[f"dev_ms_per_line_{key}"] = round(ms, 2)
            print(
                f"dev 漢字スロット一致率 {key:9s} {acc:.4f}（1行 {ms:.1f} ms）",
                flush=True,
            )
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"保存先: {out}")
    if not args.no_zip:
        z = write_zip(out, args.name, args.zip_f32)
        print(f"配布用: {z}（{z.stat().st_size / 2**20:.0f} MiB）")


if __name__ == "__main__":
    main()
