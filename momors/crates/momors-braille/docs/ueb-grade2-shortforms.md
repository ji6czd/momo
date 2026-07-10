# UEB Grade 2 仕様書 — Phase 4a「shortforms（略字）」

Phase 1〜3 に続く第4段。**shortform** = 語をその略記綴りで書く縮約（`about`→`ab`）。UEB は **75個**
（EBAE の76個から `o'clock` が削除された）。

- 出典: The Rules of UEB 2024 Appendix 1「Shortforms List」／NFB Lesson 11 (§10.9)／UEB Contractions Summary。
- **点字セルは綴り略記から機械的に導ける**（略記を grade 2 で書いたもの）。例: `shd` → `sh`縮約+`d` = ⠩⠙。
  これが強い自己検算になっている（§4）。

---

## 1. スコープ

**含む（Phase 4a）:** 75 の shortform を **単独で1語のときだけ**適用（`positions = ["wordsign"]`）。

**含まない（Phase 4b）:** 「より長い語の一部としての shortform」。規範では
「shortform は、その**長い語自体が Appendix 1 のリストに載っている**場合に限り、長い語の一部として使える」。
つまり該当する長い語も**それぞれ1語として**扱えるので、Phase 4b も**データ追加だけ**で済む見込み。

**機構ゼロ変更**: `positions = ["wordsign"]` と複数セル `cell: String` は既存。
`max_contraction_len` はデータから自動算出（`yourselves` 等で 10 になる）。

---

## 2. Table I — shortforms（75件、`wordsign`）

| 語 | 略記 | セル | 語 | 略記 | セル |
|---|---|---|---|---|---|
| about | ab | ⠁⠃ | little | ll | ⠇⠇ |
| above | abv | ⠁⠃⠧ | much | mch | ⠍⠡ |
| according | ac | ⠁⠉ | must | mst | ⠍⠌ |
| across | acr | ⠁⠉⠗ | myself | myf | ⠍⠽⠋ |
| after | af | ⠁⠋ | necessary | nec | ⠝⠑⠉ |
| afternoon | afn | ⠁⠋⠝ | neither | nei | ⠝⠑⠊ |
| afterward | afw | ⠁⠋⠺ | oneself | onef | ⠐⠕⠋ |
| again | ag | ⠁⠛ | ourselves | ourvs | ⠳⠗⠧⠎ |
| against | agst | ⠁⠛⠌ | paid | pd | ⠏⠙ |
| almost | alm | ⠁⠇⠍ | perceive | percv | ⠏⠻⠉⠧ |
| already | alr | ⠁⠇⠗ | perceiving | percvg | ⠏⠻⠉⠧⠛ |
| also | al | ⠁⠇ | perhaps | perh | ⠏⠻⠓ |
| although | alth | ⠁⠇⠹ | quick | qk | ⠟⠅ |
| altogether | alt | ⠁⠇⠞ | receive | rcv | ⠗⠉⠧ |
| always | alw | ⠁⠇⠺ | receiving | rcvg | ⠗⠉⠧⠛ |
| because | bec | ⠆⠉ | rejoice | rjc | ⠗⠚⠉ |
| before | bef | ⠆⠋ | rejoicing | rjcg | ⠗⠚⠉⠛ |
| behind | beh | ⠆⠓ | said | sd | ⠎⠙ |
| below | bel | ⠆⠇ | should | shd | ⠩⠙ |
| beneath | ben | ⠆⠝ | such | sch | ⠎⠡ |
| beside | bes | ⠆⠎ | themselves | themvs | ⠮⠍⠧⠎ |
| between | bet | ⠆⠞ | thyself | thyf | ⠹⠽⠋ |
| beyond | bey | ⠆⠽ | today | td | ⠞⠙ |
| blind | bl | ⠃⠇ | together | tgr | ⠞⠛⠗ |
| braille | brl | ⠃⠗⠇ | tomorrow | tm | ⠞⠍ |
| children | chn | ⠡⠝ | tonight | tn | ⠞⠝ |
| conceive | concv | ⠒⠉⠧ | would | wd | ⠺⠙ |
| conceiving | concvg | ⠒⠉⠧⠛ | your | yr | ⠽⠗ |
| could | cd | ⠉⠙ | yourself | yrf | ⠽⠗⠋ |
| deceive | dcv | ⠙⠉⠧ | yourselves | yrvs | ⠽⠗⠧⠎ |
| deceiving | dcvg | ⠙⠉⠧⠛ | | | |
| declare | dcl | ⠙⠉⠇ | | | |
| declaring | dclg | ⠙⠉⠇⠛ | | | |
| either | ei | ⠑⠊ | | | |
| first | fst | ⠋⠌ | | | |
| friend | fr | ⠋⠗ | | | |
| good | gd | ⠛⠙ | | | |
| great | grt | ⠛⠗⠞ | | | |
| herself | herf | ⠓⠻⠋ | | | |
| him | hm | ⠓⠍ | | | |
| himself | hmf | ⠓⠍⠋ | | | |
| immediate | imm | ⠊⠍⠍ | | | |
| its | xs | ⠭⠎ | | | |
| itself | xf | ⠭⠋ | | | |
| letter | lr | ⠇⠗ | | | |

（計 75）

---

## 3. 略記に埋め込まれた縮約

セルは略記を grade 2 で書いたもの。縮約が埋まっているのは：

| 略記 | 使う縮約 | 結果 |
|---|---|---|
| bec/bef/beh/bel/ben/bes/bet/bey | `be`(語頭 lower) ⠆ | ⠆＋字 |
| concv, concvg | `con`(語頭 lower) ⠒ | ⠒⠉⠧… |
| chn | `ch` ⠡ | ⠡⠝ |
| shd | `sh` ⠩ | ⠩⠙ |
| mch, sch | `ch` ⠡ | ⠍⠡ / ⠎⠡ |
| mst, fst, agst | `st` ⠌ | ⠍⠌ / ⠋⠌ / ⠁⠛⠌ |
| alth | `th` ⠹ | ⠁⠇⠹ |
| thyf | `th` ⠹ | ⠹⠽⠋ |
| themvs | `the` ⠮ | ⠮⠍⠧⠎ |
| ourvs | `ou` ⠳ | ⠳⠗⠧⠎ |
| herf, perh, percv(g) | `er` ⠻ | ⠓⠻⠋ / ⠏⠻⠓ / ⠏⠻⠉⠧ |
| onef | `one`(頭字符) ⠐⠕ | ⠐⠕⠋ |

`its`=`xs` / `itself`=`xf` は、`x` が「it」のアルファベット wordsign であることに由来（セルは素の x=⠭）。

---

## 4. 既存テストへの影響

- **`today`**: Phase 3 テストは `today` = t+o+`day` = `⠞⠕⠐⠙` を検証していたが、shortform `td` = **`⠞⠙`** が正しい。
  頭字符の語中適用の検証は shortform でない語（`sunday` = s+u+n+`day` = `⠎⠥⠝⠐⠙`）に差し替える。
- **`your`**: Phase 1 から「`⠽⠳⠗`（暫定）／Phase 4 で `yr`=`⠽⠗` に更新」と注記していた宿題を回収する。
  `you` wordsign が単独でないことの検証は `youth` = y+`ou`+`th` = `⠽⠳⠹` に差し替える。

---

## 4b. Phase 4b — 長い語の中での shortform（実装済み）

規範（Rules of UEB §10.9.3 / Appendix 1）: **shortform は、その長い語が Appendix 1 の Shortforms List に
載っている場合に限り、長い語の一部として使える。**

- **データ**: `dataset/ueb_shortform_words.txt`（584語）。ICEB Rules of UEB, Appendix 1 の PDF から抽出。
- **機構**: 縮約に `shortform = true` フラグ（75件）。`wordsign` の判定に
  「語全体がリストにあるなら語中のどこでも可」を追加。アルファベット/強/下方 wordsign には効かない
  （`butter` に `but` を使わない、を保つため）。
- **`s` / `'s` 規則**: リストの語に `s` を付けた語も shortform を使う（`friends`→`⠋⠗⠎`）。
  `'s` は語の切れ目になるので自動的に扱える。**例外は3語のみ**:
  `abouts`→`⠁⠃⠳⠞⠎` / `almosts`→`⠁⠇⠍⠕⠌⠎` / `hims`→`⠓⠊⠍⠎`。
- **リストに無い語では使わない**（規則4・5がそう定める）。出典の例がそのまま検算になった:

| 語 | 出典（ASCII点字） | 我々の出力 |
|---|---|---|
| blinded | `bl9d$` | `⠃⠇⠔⠙⠫` |
| blinding | `bl9d+` | `⠃⠇⠔⠙⠬` |
| aftereffect | `aft]e6ect` | `⠁⠋⠞⠻⠑⠖⠑⠉⠞` |
| misconceived | `misconceiv$` | `⠍⠊⠎⠉⠕⠝⠉⠑⠊⠧⠫` |
| inbetween | `9betwe5` | `⠔⠃⠑⠞⠺⠑⠢` |
| hereinbefore | `"h9be=e` | `⠐⠓⠔⠃⠑⠿⠑` |

---

## 5. 既知の割り切り（後続フェーズ）

- 「最も字数を節約する縮約を優先」規範は綴りの最長一致で近似（Phase 5）。
- 形態素境界跨ぎの抑制は未対応（Phase 5）。
- lower sign rule は Phase 2b で対応済み（厳密実装は不要と判明）。

---

## 6. 要確認（EBAE 知識で照合したい）

1. **Table I の 75 項目と略記**（特に `altogether`=`alt` と `although`=`alth` の対、`its`=`xs` / `itself`=`xf`、
   `such`=`sch`、`braille`=`brl`、`immediate`=`imm`）。
2. **shortform は単独時のみ（Phase 4a）**でよいか。長い語への適用（4b）を後回しにする方針で合意できるか。
3. §3 の「略記に埋め込まれた縮約」の解釈（特に `onef` が頭字符 `one`=⠐⠕ を使う点、`themvs` が `the`=⠮ を使う点）。
4. §4 のテスト差し替え（`today`→shortform、`your`→`yr`）で正しいか。
