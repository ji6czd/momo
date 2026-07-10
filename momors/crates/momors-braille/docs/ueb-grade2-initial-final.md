# UEB Grade 2 仕様書 — Phase 3「頭字符・尾字符」

Phase 1（`ueb-grade2-spec.md`）・Phase 2a（`ueb-grade2-lower-signs.md`）に続く第3段。
**2セル**の縮約＝**頭字符**（initial-letter contraction, 33）と**尾字符**（final-letter groupsign, 12）。

- 出典: The Rules of UEB 2024（ICEB）／APH Braille Brain Unit 9・12・17／UEB Contractions Summary。
- **セルは要最終確認**（体系の内部整合で検算済み。後述 §5）。

---

## 1. 重要な発見: **機構をまったく増やさない**

Phase 3 は **データ追加だけ**で通る（`english.rs` はゼロ変更の見込み）：

- **2セル出力**は既存の `cell: String` がそのまま対応（`"⠐⠙"`）。
- **頭字符 = `positions = ["always"]`**（語全体でも語中でも使う。`today` = to+`day`、`money` = m+`one`+y）。
- **尾字符 = `positions = ["noninitial"]`**。規範文「語の一部としてのみ使い、単独で語になってはならない。
  語頭では使えない」＝ `i > word_start` そのもの。単独一致は `i == word_start` なので同時に排除される。

「拡張は**データ**に流し込み、**機構**は増やさない」——[[braille-design]] で立てた liblouis の轍を避ける方針が、
ここで実際に効いていることの検証になる。

---

## 2. Table G — 頭字符（initial-letter contraction, `always`）

2セル目は**その語の先頭文字（または先頭の縮約セル）**。

### dot 5（⠐）

| 語 | セル | 語 | セル | 語 | セル |
|---|---|---|---|---|---|
| day | ⠐⠙ | ever | ⠐⠑ | father | ⠐⠋ |
| here | ⠐⠓ | know | ⠐⠅ | lord | ⠐⠇ |
| mother | ⠐⠍ | name | ⠐⠝ | one | ⠐⠕ |
| part | ⠐⠏ | question | ⠐⠟ | right | ⠐⠗ |
| some | ⠐⠎ | time | ⠐⠞ | under | ⠐⠥ |
| work | ⠐⠺ | young | ⠐⠽ | | |

先頭が縮約セルのもの: there ⠐⠮(the) / character ⠐⠡(ch) / through ⠐⠹(th) / where ⠐⠱(wh) / ought ⠐⠳(ou)

### dots 45（⠘）

upon ⠘⠥ / word ⠘⠺ / these ⠘⠮(the) / those ⠘⠹(th) / whose ⠘⠱(wh)

### dots 456（⠸）

cannot ⠸⠉ / had ⠸⠓ / many ⠸⠍ / spirit ⠸⠎ / world ⠸⠺ / their ⠸⠮(the)

---

## 3. Table H — 尾字符（final-letter groupsign, `noninitial`）

2セル目は**表す綴りの末尾文字**。

| dots 46（⠨） | セル | dots 56（⠰） | セル |
|---|---|---|---|
| ound | ⠨⠙ | ence | ⠰⠑ |
| ance | ⠨⠑ | ong | ⠰⠛ |
| sion | ⠨⠝ | ful | ⠰⠇ |
| less | ⠨⠎ | tion | ⠰⠝ |
| ount | ⠨⠞ | ness | ⠰⠎ |
| | | ment | ⠰⠞ |
| | | ity | ⠰⠽ |

例: sound=⠎⠨⠙ / count=⠉⠨⠞ / vision=⠧⠊⠨⠝ / nation=⠝⠁⠰⠝ / city=⠉⠰⠽ /
kindness=⠅⠔⠙⠰⠎（Phase 2a の `in` と併用）/ careful=⠉⠜⠑⠰⠇

**語頭不可の効き方**: `mention` の `ment` は語頭なので使わず、`m`+`en`+`tion` = ⠍⠢⠰⠝ になる。

---

## 4. 既存テストへの影響（Phase 3 で正しくなる）

- **`mother`**: Phase 1 は m+o+the+r = `⠍⠕⠮⠗`（4セル）だったが、頭字符で **`⠐⠍`（2セル）** が正しい。
  既存テスト `strong_contraction_the_within_word` を更新し、`the` の語中 groupsign 検証は
  頭字符を持たない語（例 `gather` = g+a+the+r = ⠛⠁⠮⠗）に差し替える。
- `father` も単独では `⠐⠋`（従来 f+a+the+r）。

---

## 5. 内部整合の検算（セルの正しさの傍証）

- 頭字符の2セル目 = 語の先頭文字/先頭縮約。尾字符の2セル目 = 綴りの末尾文字。全項目でこの規則が成立。
- 三つ組が一致: **work ⠐⠺ / word ⠘⠺ / world ⠸⠺**、**there ⠐⠮ / these ⠘⠮ / their ⠸⠮**、
  **through ⠐⠹ / those ⠘⠹**、**where ⠐⠱ / whose ⠘⠱**、**here ⠐⠓ / had ⠸⠓**、
  **some ⠐⠎ / spirit ⠸⠎**、**mother ⠐⠍ / many ⠸⠍**、**character ⠐⠡ / cannot ⠸⠉**。
- 指示符: dot5=⠐ / dots45=⠘ / dots456=⠸ / dots46=⠨ / dots56=⠰（全て相異なる）。

---

## 6. 既知の割り切り（後続フェーズ）

- **「最も字数を節約する縮約を優先」という規範の優先規則**は未実装。現状は**綴りの最長一致**で近似する
  （`ound`(4字→2セル) が `ou`+n+d(3セル) に勝つ等、実用上ほぼ一致）。厳密化は Phase 5（usage 規則）。
- 形態素境界を跨ぐ縮約の抑制は引き続き未対応。
- shortforms（Phase 4）・lower sign rule 厳密版（Phase 2b）は別。

---

## 7. 要確認（EBAE 知識で照合したい）

1. **Table G / H の項目とセル**（特に dot5 リストに `work` を含めた点、`ought`=⠐⠳、`lord`=⠐⠇）。
2. **頭字符は位置無制限（`always`）**でよいか（語中でも使う: `today`, `money`, `somewhere`）。
3. **尾字符は `noninitial`（語頭不可・単独不可）**だけで足りるか。他の位置制約はないか。
4. `mother` / `father` が単独で頭字符（⠐⠍ / ⠐⠋）になる、で正しいか（§4 のテスト更新を伴う）。

1〜4 が確定すれば、`ueb_english.toml` に45エントリ追加＋既存テスト更新のみで Phase 3 完了。
