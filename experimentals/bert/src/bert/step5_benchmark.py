import statistics
import time

import torch

from step4_char_vectors import CharEncoder, encode_with_alignment

TEXT = """このプロジェクトは、一人プロジェクトのSeikenラボが開発している、にほんご→点字変換プログラムです。きわめて個人的な理由から開発は始められましたが、案外使えるようになったこと、また、この分野において実用的に動作するオープンソースの実装が存在しないことから、公開することにしました。
さて、さっそく昔話。時は1991年。その時代はWindowsは一応あったけど、私の周りでは使ってる人いなかったなあ。
MS-DOS Shell なんてものもあるにはあったけど、まあ、みんなコマンドプロンプトで作業してたよなあ。"""

WARMUP = 3
RUNS = 30


def measure(fn, runs: int = RUNS) -> list[float]:
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def report(label: str, times: list[float]) -> None:
    print(
        f"  {label:<28} 平均 {statistics.mean(times):8.2f} ms"
        f"   中央値 {statistics.median(times):8.2f} ms"
        f"   最小 {min(times):8.2f} ms   最大 {max(times):8.2f} ms"
    )


def main() -> None:
    t0 = time.perf_counter()
    enc = CharEncoder()
    print(f"モデル読み込み: {time.perf_counter() - t0:.2f} s")
    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}  device=cpu")

    ids, _ = encode_with_alignment(TEXT, enc.vocab)
    print(f"元の文字数: {len(TEXT)}  トークン数([CLS][SEP]含む): {len(ids)}")
    print(f"計測: ウォームアップ {WARMUP} 回 + 本番 {RUNS} 回")
    print()

    print("== 全文を1回で処理 ==")
    report(
        "前処理(正規化+語彙引き)",
        measure(lambda: encode_with_alignment(TEXT, enc.vocab)),
    )
    tensor = torch.tensor([ids])

    def forward():
        with torch.no_grad():
            enc.model(tensor)

    report("モデル推論のみ", measure(forward))
    total = measure(lambda: enc.embed(TEXT))
    report("embed() 全体", total)
    print(
        f"  -> 1秒あたり約 {1000 / statistics.mean(total):.1f} 回 / "
        f"約 {len(TEXT) * 1000 / statistics.mean(total):.0f} 文字"
    )
    print()

    print("== 行ごとに3回に分けて処理 ==")
    lines = TEXT.split("\n")
    report("3行を順に embed()", measure(lambda: [enc.embed(line) for line in lines]))


if __name__ == "__main__":
    main()
