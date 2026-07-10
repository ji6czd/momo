# UEB Grade 2 仕様書 — Phase 1「文脈非依存コア」

MOMO の英語点字（UEB grade2）対応の**第1段**の仕様。単一セルで位置にほぼ依存しない縮約だけを扱う。
設計した「単一ルール形＋語境界プリミティブ」を最小コストで検証するためのスライス。

- 出典: The Rules of Unified English Braille, 3rd Edition 2024 (ICEB) を規範とする。
  本書はそこから該当部分を自分で書き起こした要約であり、原文の再配布ではない。
- 対象読者は EBAE を読める前提。ここに挙げた縮約は EBAE と UEB で共通（UEB が削除・変更したのは別の箇所）。

---

## 1. スコープ

**含む（Phase 1）:**
- アルファベット wordsign（23）
- strong contraction（5）… wordsign 兼 groupsign
- strong groupsign（12）
- strong wordsign（6, セルは strong groupsign と共有）

**含まない（Phase 2 以降）:** lower signs（be/con/dis/en/in, bb/cc/ff/gg/ea… の位置制約）、
頭字符・尾字符（⠐⠨⠸＋字, ⠰⠆等＋字）、shortforms（~76。例: `your`→`yr`=⠽⠗。
Phase 1 は shortform 非適用で `⠽⠳⠗`）、および「いつ縮約しないか」の usage 規則。

---

## 2. 記法と適用カテゴリ

- 点の番号は標準の 6 点セル：
  ```
  1 4
  2 5
  3 6
  ```
- 各エントリは `綴り → セル(点番号)` で書く。
- **適用カテゴリ:**
  - **A（always）** … その綴りが語中のどこに現れても使う（後述の例外を除く）。
  - **W（wordsign / standing-alone）** … その綴りが**単独で1語**をなすとき（前後が空白または約物）だけ使う。

---

## 3. ルール形（設計との対応）

各エントリは、設計中の単一ルール形に落ちる：

```
左文脈 [ 綴り ] 右文脈  =>  セル
```

- **A エントリ**: 文脈条件なし（`[ 綴り ]`）。最長一致で適用。
- **W エントリ**: 両側が語境界（`WB [ 綴り ] WB`）。`WB` = 空白・約物・行頭・行末。
- **優先**: 同じトークンが W（語全体）と A（部品）の両方に一致しうるとき、**W が勝つ**
  （語境界という深い文脈を伴うぶん特異度が高い）。例: `this` 単独 → `⠹`(this) であって `⠹⠊⠎`(th+i+s) ではない。

セルを共有する例（`⠹`）が wordsign/groupsign 両義を持つのはこの仕組みで解決する：
`this`（単独）=`⠹`、`thistle`（語中）= `th+i+st+l+e` の `th` として `⠹`。

---

## 4. Table A — アルファベット wordsign（W: 単独時のみ）

単独で1語をなすときだけ、その1文字が語を表す。groupsign ではない（語中では普通の文字）。

| 綴り(語) | 字 | セル(点) |
|---|---|---|
| but | b | ⠃ (12) |
| can | c | ⠉ (14) |
| do | d | ⠙ (145) |
| every | e | ⠑ (15) |
| from | f | ⠋ (124) |
| go | g | ⠛ (1245) |
| have | h | ⠓ (125) |
| just | j | ⠚ (245) |
| knowledge | k | ⠅ (13) |
| like | l | ⠇ (123) |
| more | m | ⠍ (134) |
| not | n | ⠝ (1345) |
| people | p | ⠏ (1234) |
| quite | q | ⠟ (12345) |
| rather | r | ⠗ (1235) |
| so | s | ⠎ (234) |
| that | t | ⠞ (2345) |
| us | u | ⠥ (136) |
| very | v | ⠧ (1236) |
| will | w | ⠺ (2456) |
| it | x | ⠭ (1346) |
| you | y | ⠽ (13456) |
| as | z | ⠵ (1356) |

（`a` `i` `o` はそれ自身が1語なので wordsign を持たない。）

---

## 5. Table B — strong contraction（A: 常に。wordsign 兼 groupsign）

単独でも語中でも、位置に関係なく使う。

| 綴り | セル(点) | 例 |
|---|---|---|
| and | ⠯ (12346) | and, sand(s+and), band |
| for | ⠿ (123456) | for, before(be+for+e) |
| of | ⠷ (12356) | of, profit(pr+of+it) ← 境界跨ぎでも使う（確定） |
| the | ⠮ (2346) | the, mother(m+o+the+r) |
| with | ⠾ (23456) | with, within(with+in) |

---

## 6. Table C — strong groupsign（A: 常に、語中のどこでも）

その綴りが現れたら使う。**位置例外は `ing`（語頭では使わない、例: `ingot`）の1つだけ（確定）**。
`gh`(ghost)、`ed`/`er` などは語頭でも通常どおり使う。

| 綴り | セル(点) | 例 |
|---|---|---|
| ch | ⠡ (16) | rich(r+i+ch) |
| gh | ⠣ (126) | high(h+i+gh) |
| sh | ⠩ (146) | wish(w+i+sh) |
| th | ⠹ (1456) | with→別／path(p+a+th) |
| wh | ⠱ (156) | why(wh+y) |
| ed | ⠫ (1246) | landed(l+and+ed), used(u+s+ed) |
| er | ⠻ (12456) | her(h+er), over(o+v+er) |
| ou | ⠳ (1256) | loud(l+ou+d) |
| ow | ⠪ (246) | now(n+ow) |
| ar | ⠜ (345) | car(c+ar) |
| ing | ⠬ (346) | sing(s+ing) ／語頭は不可 |
| st | ⠌ (34) | fast(f+a+st) |

---

## 7. Table D — strong wordsign（W: 単独時のみ。セルは Table C と共有）

Table C と同じセルが、単独で1語をなすとき下記の語を表す。

| 綴り(語) | 共有セル(点) | 由来 groupsign |
|---|---|---|
| child | ⠡ (16) | ch |
| shall | ⠩ (146) | sh |
| this | ⠹ (1456) | th |
| which | ⠱ (156) | wh |
| out | ⠳ (1256) | ou |
| still | ⠌ (34) | st |

（`gh, ed, er, ow, ar, ing` に wordsign はない。）

---

## 8. Phase 1 の既知の限界（Phase 2 で対応）

Phase 1 は「最長一致＋語境界」だけの素朴な適用。以下は**あえて未対応**で、そのぶん過剰縮約が出る：

- **形態素・音節境界を跨ぐ縮約の抑制**。UEB は語構成境界を跨ぐ強縮約を使わない
  （例: `mishap` = mis+hap の `sh` は非縮約、`hophead` の `ph`/`he` 等）。Phase 1 は素朴に縮約してしまう。
- **縮約どうしの競合時の優先規則**（どれを優先するか）。
- **wordsign を使わない条件**（アポストロフィ+s、語中大文字、特定の約物隣接など）。
- **`WB`（語境界）に含める約物の正確な定義**。当面は「空白・行頭行末・基本約物」で近似。

---

## 9. 確定事項（2026-07-10 ユーザ確認済み）

1. **Table C の位置例外は `ing`（語頭不可）だけ**。`gh`/`ed`/`er` などは語頭でも通常どおり使う。
2. **wordsign の「単独」判定に含める約物は、当面「空白・行頭行末・基本約物」で確定**。
   必要な約物（ハイフン・アポストロフィ等）は後から追加していく。
3. **strong contraction（`of`/`for` 等）は境界跨ぎでも無制限に使う**（`profit` = pr+of+it は有効）。

→ Phase 1 は §4〜7 の表＋§3 のルール形＋`ing` 語頭例外で**実装確定**。
（形態素境界跨ぎの抑制＝§8 は Phase 2。Phase 1 は素朴に縮約する割り切りのまま。）
