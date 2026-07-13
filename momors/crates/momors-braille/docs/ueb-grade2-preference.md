# UEB Grade 2 仕様書 — Phase 8「縮約の優先順位と、読み手のための grade 1」

Phase 7 で「UEB は完成」と思ったが、`braille` の shortform を語の途中で使うのかという問いから
**3つの穴**が見つかった。Phase 8 でそれを塞ぐ。

出典: The Rules of UEB 2024 §10.9.3（リストに無い語の中の shortform）／§10.9.8・§10.9.9
（誤読を防ぐ grade 1）／§10.10（優先順位）。

---

## 1. §10.9.3 — リストに無い語の中でも使える10の shortform

Phase 4b は「shortform は Appendix 1 のリストにある語の中でだけ使える」としていた。だが規範は
**10語だけ別扱い**していた。リストに無い語の中でも使う（語が standing alone であること）。

| shortform | 範囲 | 母音・`y` が続くとき |
|---|---|---|
| braille, great | 語のどこでも | **使う**（制限なし） |
| children | 語のどこでも | 使わない |
| blind, first, friend, good, letter, little, quick | **語頭のみ** | 使わない |

```text
Braillette  ⠠⠃⠗⠇⠞⠞⠑     rebrailled  ⠗⠑⠃⠗⠇⠙     Greatorex  ⠠⠛⠗⠞⠕⠗⠑⠭（母音でも使う）
Blindcraft  ⠠⠃⠇⠉⠗⠁⠋⠞    Firstchoice ⠠⠋⠌⠡⠕⠊⠉⠑   Quicksburg ⠠⠟⠅⠎⠃⠥⠗⠛
But: Blindoc ⠠⠃⠇⠔⠙⠕⠉    Goodacre ⠠⠛⠕⠕⠙⠁⠉⠗⠑    Littlearm ⠠⠇⠊⠞⠞⠇⠑⠜⠍（母音で止まる）
```

**データで表す**: 縮約に `free = "anywhere" | "initial"` と `not_before_vowel` を足しただけ。
機構は `applicable` に判定が1つ増えるのみ。

---

## 2. §10.10 — セル数が同じときの優先順位

Phase 5 の DP は「セル数最小、同数なら綴りの長い方」で決めていた。だが規範は**明示的な順位**を
持っていた。§10.10.2（セル数）が第1基準なのは合っていたが、**同数のときの決着法が違った**。

| 順位 | 種類 | 規範 |
|---|---|---|
| 1 | 語頭の下方 groupsign（be / con / dis。第一音節のとき） | §10.10.4 |
| 2 | strong groupsign / strong contraction | §10.10.5（strong > lower） |
| 3 | その他の下方 groupsign（en / in / ea / bb / cc / ff / gg） | |
| 8 | 頭字符・尾字符・shortform（2セル） | §10.10.7（strong/lower > 2セル） |

2セルのものを **8** にしてあるのは、strong 2つ（2+2=4）より重くするため。`adhered` は
`⠁⠙⠐⠓⠙`（頭字符 `here`）ではなく `⠁⠙⠓⠻⠫`（`er` + `ed`）が正しい。

```text
bear ⠃⠑⠜（⠃⠂⠗ ではない。ar > ea）   bacchanal ⠃⠁⠉⠡⠁⠝⠁⠇（ch > cc）
beatitude ⠆⠁⠞⠊⠞⠥⠙⠑（be > ea）      onerous ⠕⠝⠻⠳⠎（er+ou > one）
timer ⠐⠞⠗（セル数が減るなら頭字符を使う）
```

これは **`bear` `heart` `nearly` のような普通の語に効く**バグだった。DP の第2キーに順位の
合計を入れて解いた（`preference`）。

**データも足りていなかった**: `be`/`con`/`dis` の `initial_stems` に規範の用例語（beatitude,
bedraggle, benight, benign, berate, congee, congru, dishonest…）を追加。形態素境界には
`egg|head`（`gh` を跨がせない）と `captain|ess`（`ness` を跨がせない）を追加した。
どちらも規範が "But:" で挙げていた例で、**データを足すだけで規範どおりになった**。

---

## 3. §10.9.8 / §10.9.9 — 読み手のための grade 1

§10.9.3 で `brl` が「語のどこにあっても braille と読まれる」ようになった以上、**そう読まれては
困る語**が出てくる。規範はそこに grade 1 を置けと言う。

```text
Dobrljin  ⠰⠰⠠⠙⠕⠃⠗⠇⠚⠊⠝   語中の brl → grade 1 語符で語全体を無縮約に（§10.9.9）
ozbrl     ⠰⠰⠕⠵⠃⠗⠇
Blvd      ⠰⠠⠃⠇⠧⠙          語頭の bl → grade 1 記号符で足りる（§10.9.8）
```

grade 1 語符 `⠰⠰` はその並び全体を無縮約にする（「他の縮約も一切使ってはならない」）。
空白で効力が切れるので**終止符は要らない**（終止符が要るのは grade 1 句符 `⠰⠰⠰` で、これは未対応）。

**誤検出を避けるのが要点**。`table` の `bl` は語頭でないので `blind` とは読まれない。
`blackboard` の `bl` は母音が続くので読まれない。だから指示符は要らない。判定は**読み手の規則
（§10.9.3 の範囲と母音制限）をそのまま逆向きに使う**ことで自然に出る（`misread_as_shortform`）。

### 【Phase 9 で解消】10語以外の shortform も誤読される

当初は「10語だけ」を検出対象にし、規範の `UNSD` → `⠰⠰⠠⠠⠥⠝⠎⠙` は根拠が読めないとして
割り切っていた。しかしこれは誤りだった。**我々自身が `unsaid` を `⠥⠝⠎⠙` と書く**以上、
`UNSD` の `⠠⠠⠥⠝⠎⠙` は「UNSAID」と同じ点字であり、実際に曖昧である。

正しい判定は「**セル列を shortform として読み戻した語が Appendix 1 のリストにあるか**」。
`UNSD` の `sd` は読み戻すと `unsaid` でリストにある → 誤読される → `⠰⠰`。
`wisdom` の `sd` は読み戻すと `wisaidom` でリストに無い → 読み手は適用しない → 指示符は不要。
読み手の規則をそのまま逆に回すだけで、誤検出なしに規範と一致した（`corpus.md` §2 参照）。

---

## 4. 検証

規範の用例をそのまま回帰テストにした（§10.9.3 が13件、§10.10 が20件、§10.9.8/9 が5件）。
momors-braille のテストは 226 件。Phase 8 で規範に合うようになった語:

`bear` `heart` `nearly` `nuclear` `bacchanal` `egghead` `beatitude` `bedraggled` `congee`
`congruity` `dishonesty` `adhered` `heredity` `Parthian` `captainess` `Littlearm`
`Braillette` `Greatford` `Blindcraft` `Firstchoice` `Quicksburg` `Friendly` `rebrailled`
`Dobrljin` `ozbrl` `Blvd`
