# UEB Grade 2 仕様書 — Phase 2a「lower signs（下方の縮約）」

Phase 1（`ueb-grade2-spec.md`）に続く第2段。**lower signs**＝下方の点（2,3,5,6 のみ・dot 1/4 を含まない）で
作る縮約。これらは下方約物（コンマ・セミコロン等）とセルが衝突するため、**位置規則**が本質。
Phase 2a の主眼は「ルール形に**位置制約**を導入する」こと（設計した単一ルール形の自然な拡張）。

- 出典: The Rules of UEB 2024（ICEB）／APH Braille Brain Unit 14・15。
- **セルの点パターンは要最終確認**（取得ツールが一部誤値を返したため、下方シフト導出＋
  約物との既知の一致で再構成した。EBAE 読者の目で照合してほしい）。

---

## 1. スコープ

**含む（Phase 2a）:** lower groupsign 10（be, con, dis, en, in, ea, bb, cc, ff, gg）＋
lower wordsign 6（be, his, was, were, in, enough）。be と in は groupsign 役と wordsign 役を兼ねる。

**含まない（Phase 2b 以降）:** 「lower sign rule」（下方記号の連なりは dot 1/4 を持つ記号との接触で
アンカーされねばならない）の完全実装、および lower wordsign の**下方約物との接触制限**の精密化。
Phase 2a は位置カテゴリ（initial/medial/always/wordsign）で近似する。

---

## 2. ルール形の拡張＝位置カテゴリ

Phase 1 の `category`（always / wordsign）を、**位置の集合** `positions`（複数可・いずれか一致で適用）に一般化する。
語 `[start, end)`（連続英字）の中の位置 `i`・長さ `len` に対して：

| position | 条件 | 使う縮約 |
|---|---|---|
| `always` | 常に（語中どこでも） | strong 系（Phase 1）, en, in(groupsign) |
| `noninitial` | `i > start`（語頭でない） | ing（Phase 1 の例外をここに統合） |
| `initial` | `i == start && i+len < end`（語頭＋後続が文字） | be, con, dis |
| `medial` | `start < i && i+len < end`（両側が文字＝語中） | ea, bb, cc, ff, gg |
| `wordsign` | `i == start && i+len == end`（単独で1語） | 全 wordsign |

- **最長一致は据え置き**。各位置で最長の「positions のいずれかを満たす」縮約を選ぶ。
- 1つの綴りが複数役を持つ場合は `positions` に併記（例: `be` = `["wordsign","initial"]`、
  `in` = `["wordsign","always"]`）。セル共有は綴りが別なので衝突しない（`be`≠`bb`）。

---

## 3. Table E — lower groupsign

| 綴り | セル(点) | positions | 例 |
|---|---|---|---|
| be | ⠆ (23) | initial（＋wordsign, Table F） | become = be+c+o+m+e |
| con | ⠒ (25) | initial | concern = con+c+ern? → con+c+er+n |
| dis | ⠲ (256) | initial | disagree = dis+... |
| en | ⠢ (26) | always | end = en+d ／ children…（語中 en） |
| in | ⠔ (35) | always（＋wordsign, Table F） | into = in+t+o |
| ea | ⠂ (2) | medial | eat? →語頭不可。meat = m+ea+t |
| bb | ⠆ (23) | medial | rubber = r+u+bb+er |
| cc | ⠒ (25) | medial | accent = a+cc+en+t |
| ff | ⠖ (235) | medial | affair = a+ff+air? → a+ff+ar? |
| gg | ⠶ (2356) | medial | egg? →語末不可。bigger = b+i+gg+er |

**セル共有（位置で弁別）:** ⠆ = be(initial) / bb(medial) / be(wordsign)。
⠒ = con(initial) / cc(medial)。⠶ = gg(medial) / were(wordsign)。
⠢ = en(always) / enough(wordsign)。⠔ = in(always/wordsign)。⠲ = dis(initial)。

---

## 4. Table F — lower wordsign（単独で1語）

| 綴り(語) | セル(点) | positions | 注 |
|---|---|---|---|
| be | ⠆ (23) | wordsign | 下方約物と接触時は使用不可（Phase 2b） |
| his | ⠦ (236) | wordsign | 同上 |
| was | ⠴ (356) | wordsign | 同上 |
| were | ⠶ (2356) | wordsign | 同上 |
| in | ⠔ (35) | wordsign | 上方の点と接触する連なりで可（個別規則） |
| enough | ⠢ (26) | wordsign | ソリダス（/）の隣で使用不可（要単独） |

---

## 5. lower sign rule（Phase 2b で精密化）

出典の規範文（要約）: 「下方 groupsign と下方約物は、いずれかが **dot 1 か dot 4 を含む記号と接触**して
いれば、空白なしで連続してよい。大文字符はこの判定で無視する。」

- **意味**: 純粋な下方記号だけが連なると約物と区別できないので、語中の文字（dot 1/4 を持つ）で
  アンカーされていることを要求する規則。
- **Phase 2a の近似**: 位置カテゴリでほぼ捕まえる——be/con/dis は語頭＋後続文字がアンカー、
  ea/bb/cc/ff/gg は両隣の文字がアンカー、en/in は語中で文字に隣接。通常語はこれで正しい。
- **Phase 2b で対応**: 下方 wordsign が下方約物に接触する場合（例 `be,` `was.`）に縮約を抑制する、
  および下方記号列のアンカー判定の厳密化。

---

## 5b. 【重要】be / con / dis は第一音節でなければ使えない（Phase 2b で対応済み）

UEB の規範: **be / con / dis の groupsign は「語の第一音節であるとき」だけ使える。**
音節は綴りから決まらない（＝UEB は音韻依存を捨てきっていない）。

- `congenial` → con が第一音節 → `⠒⠛⠢⠊⠁⠇`（出典 "3g5ial"）
- `benzene` → ben|zene で be は音節でない → 縮約せず `⠃⠢⠵⠢⠑`（出典 "b5z5e"）
- lower sign rule（dot 1/4 アンカー）では救えない（`be`(⠆) の次が `n`(⠝) で anchored）＝**音節規則は既約**
- 併せて: be/con/dis が2つ連続するときは**最初のものだけ**縮約
  （`disbelief`→"4belief"、`disconnect`→"4connect"）。`initial`＝語頭限定の実装で自然に成立。

**実装（(b)→(a) 方針）**: `initial` を**既定で不許可**にし、TOML の `initial_stems`（語幹リスト）で
始まる語だけ解禁する。誤らない側に倒し、知識をデータで足す。リストを増やすほど縮約率が上がるだけで、
正しさは壊れない。→ liblouis が語リストを抱えるのは**必然的複雑さ**だった（記法の問題ではない）。

---

## 6. 要確認（EBAE 知識／Rules 2024 で照合したい）

1. **Table E/F のセル（点パターン）が正しいか**（取得ツールの誤値を再構成したため最重要）。
   特に con=⠒(25), ff=⠖(235), gg=⠶(2356), were=⠶(2356), his=⠦(236), was=⠴(356), enough=⠢(26)。
2. **be/con/dis は本当に「語頭のみ」**か（語中では bb/cc や普通の綴りになる、で正しいか）。
3. **en は本当に位置無制限（always）**か。語頭・語末での可否。
4. lower wordsign の下方約物接触制限を Phase 2a で近似（下方約物隣接時は縮約しない）してよいか、
   それとも Phase 2b まで一切実装しない（素朴に縮約して既知の過剰縮約とする）か。

Phase 2a 実装は、1〜3 が確定すれば §2 の位置カテゴリ＋ §3/§4 の表で進められる。
4 は「近似する / しない」を選んでもらえれば確定。
