/**
 * MOMO (momors-wasm) を使ったサンプル。
 *
 * 日本語テキストをかなに変換し、原文の 1 文字ずつがどのかなに読まれたかを表にして表示する。
 *
 *     pnpm start               # 既定の文字列を変換
 *     pnpm start 吾輩は猫である  # 引数の文字列を変換
 */

import { readFileSync } from "node:fs";
import { initSync, Predictor } from "momors-wasm";

// 引数を省略したときに変換する文字列。
const DEFAULT_TEXT = "今日、東京でＴシャツを2枚買った。";

// 読みモデル (.mbm)。既定は `task sdk:wasm` の出力先。環境変数 MOMO_MODEL で差し替えられる。
const MODEL_PATH =
  process.env.MOMO_MODEL ||
  new URL("../../../dist/sdk/wasm/data/basic_data_7.mbm", import.meta.url);

// 端末で全角（2 桁）として表示される文字のおおまかな範囲。
const WIDE =
  /[ᄀ-ᅟ⺀-꓏가-힣豈-﫿︰-﹏＀-｠￠-￦\u{20000}-\u{3fffd}]/u;

function displayWidth(s: string): number {
  let width = 0;
  for (const ch of s) {
    width += WIDE.test(ch) ? 2 : 1;
  }
  return width;
}

// 空白はそのままでは見えないので記号に置き換える。
function visible(ch: string): string {
  return /\s/u.test(ch) ? "␣" : ch;
}

// 列ごとに表示幅をそろえて表を組む。最終列は後ろに何も続かないので詰めない。
function formatTable(rows: string[][]): string {
  const widths = rows[0].map((_, c) =>
    Math.max(...rows.map((row) => displayWidth(row[c]))),
  );
  return rows
    .map((row) =>
      row
        .map((cell, c) =>
          c === row.length - 1
            ? cell
            : cell + " ".repeat(widths[c] - displayWidth(cell)),
        )
        .join("  "),
    )
    .join("\n");
}

// wasm-pack の web ターゲットは既定で fetch() を使って .wasm を読むが、Node の fetch() は
// file: URL を扱えない。そこでバイト列を自分で読み、同期で初期化する。
const wasmUrl = new URL(import.meta.resolve("momors-wasm/momors_wasm_bg.wasm"));
initSync({ module: readFileSync(wasmUrl) });

const text = process.argv.slice(2).join(" ") || DEFAULT_TEXT;

// wasm 側のメモリは GC では解放されないので、using でスコープを抜けるときに解放する。
using predictor = new Predictor(readFileSync(MODEL_PATH));
using result = predictor.predict(text);

// sourceToKana のインデックスはコードポイント単位。JS の文字列は UTF-16 なので、
// サロゲートペア（𠮷 など）でずれないよう Array.from でコードポイントの配列にしておく。
const source = Array.from(result.source);
const kana = Array.from(result.kana);
// 原文の i 文字目 → その文字から読まれたかなのインデックス列（Uint32Array）。
// 熟語の 2 文字目以降など、前の文字とまとめて読まれた文字は空になる。
const sourceToKana: Uint32Array[] = result.sourceToKana;

const rows = [["位置", "原文", "かな位置", "かな"]];
source.forEach((ch, i) => {
  const indices = Array.from(sourceToKana[i]);
  rows.push([
    String(i),
    visible(ch),
    indices.join(","),
    indices.length > 0
      ? indices.map((k) => visible(kana[k])).join("")
      : "（前の字に含む）",
  ]);
});

console.log(`原文: ${result.source}`);
console.log(`かな: ${result.kana}`);
console.log();
console.log(formatTable(rows));
console.log();
console.log("␣ は空白（かなの側ではマスあけ）");
