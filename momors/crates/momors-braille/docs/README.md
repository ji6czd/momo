# momors-braille のドキュメント

英語点字（UEB grade 2）の仕様書。**MOMO がどの規則をどう実装したか**を、規範の条文番号つきで
記録してある。日本語点字については `src/japanese_translator.rs` と `tables/japanese_*.toml` を参照。

実装は [`src/english_translator.rs`](../src/english_translator.rs)、データは
`tables/english_ueb_grade2.toml`（grade 1 は `english_ueb_grade1.toml`）。

## 規範（出典）

- **The Rules of Unified English Braille, Third Edition 2024 (ICEB)** — 408ページ。
  <https://iceb.org/wp-content/uploads/2025/10/Rules-of-Unified-English-Braille-2024.pdf>
  ICEB が Creative Commons ライセンスで公開している。仕様書中の §番号はすべてこれを指す。
- **UEB List of Punctuation**（Section 7 の一覧）
  <https://uebot.niu.edu/courses/pluginfile.php/96/block_html/content/UEB%20List%20of%20Punctuation.pdf>

PDF からテキストを取り出すときは `uv run --with pypdf python`（この環境に poppler は無い）。
語も点字も行で折り返すので、継続行を繋ぐこと（`ueb-grade2-corpus.md` 参照）。

## 仕様書（Phase 順に読むとよい）

| ファイル | Phase | 内容 |
|---|---|---|
| [ueb-grade2-spec.md](ueb-grade2-spec.md) | 1 | 文脈非依存コア。アルファベット wordsign・strong contraction / groupsign・単一ルール形と語境界 |
| [ueb-grade2-lower-signs.md](ueb-grade2-lower-signs.md) | 2a/2b | lower signs（be/con/dis/en/in/ea/bb/cc/ff/gg）。位置制約の導入と、be/con/dis の第一音節規則（`initial_stems`） |
| [ueb-grade2-initial-final.md](ueb-grade2-initial-final.md) | 3 | 頭字符（33）と尾字符（12）。2セルの縮約 |
| [ueb-grade2-shortforms.md](ueb-grade2-shortforms.md) | 4a/4b | shortform 75語と、Appendix 1 のリストに載る長い語の中での使用 |
| [ueb-grade2-usage.md](ueb-grade2-usage.md) | 5 | usage 規則。セル数最小化（DP）・語の区切り（`[divisions]`）・standing alone・grade 1 記号符・語中の大文字 |
| [ueb-grade2-punctuation.md](ueb-grade2-punctuation.md) | 6 | 約物48種・記号・引用符（`[quotes]`）。standing alone（§2.6）と数値モード（§6）の規範化 |
| [ueb-grade2-capitals.md](ueb-grade2-capitals.md) | 7 | 大文字の3モード（`⠠` / `⠠⠠` / `⠠⠠⠠`）と大文字終止符 `⠠⠄` |
| [ueb-grade2-preference.md](ueb-grade2-preference.md) | 8 | 縮約の優先順位（§10.10）と、読み手のための grade 1（§10.9.8/9）。リストに無い語でも使える10の shortform（§10.9.3） |
| [ueb-grade2-corpus.md](ueb-grade2-corpus.md) | 9 | 規範の用例1,029語を回帰コーパスにして測る。**なぜ liblouis ではなく原典を採ったか** |

Phase 1〜4 は出典が手に入る前の再構成なので、後の Phase（特に 6 と 9）で訂正した箇所がある。
食い違ったときは**新しい Phase の記述が正**。

## 回帰コーパス（重要）

`dataset/ueb_appendix2_words.tsv` は、規範の **Appendix 2「Word List」1,029語**
（原文 / 期待する点字 / 適用規則番号）。[`tests/ueb_appendix2.rs`](../tests/ueb_appendix2.rs) が
これを流し、未対応語は `KNOWN_FAIL` に理由つきで明示してある（現在16語・一致率 98.4%）。
**増えたら退行**としてテストが落ちる。

各語に**規則番号が付いている**のが効く。不一致が出たら、どの規則を実装できていないかが即座に分かる。
実際 `bear`（strong > lower の優先順位）や `in-depth`（下方 wordsign の接触制限）のような、
仕様書を読んだだけでは気づけない誤りをこのコーパスが暴いた。**仕様を足すときは、まず出典の用例を
テストにすること。**

## 拡張のしかた

精度改善は原則 **TOML のデータ追加**で行う（機構を増やさない）。

- `[divisions] boundaries` … 縮約を跨がせない区切り（`mis|hap`。形態素・音節の両方）
- `initial_stems` … be/con/dis を第一音節として解禁する語幹
- `[shortforms] words` … shortform を語中で使ってよい語（Appendix 1）

いずれも「増やすほど縮約が正しくなり、正しさは壊れない」設計。ただし **`[divisions]` は要注意**——
「その語が直る」だけでは足りない。区切りで表せないものを区切りで表すと**別の語が壊れる**
（`wh|ere` は `where'er` を直すが、普通の `where` を壊す）。コーパスは安全網として不完全なので、
普通の語で直接確かめること。詳細は `ueb-grade2-corpus.md` §2。

## liblouis について

liblouis の UEB テーブル（`en-ueb-g2.ctb`）にも同種の語リストがあるが、**LGPL-2.1+ なので
取り込まない**（MOMO は BSD-3-Clause）。突き合わせ相手として手元で使うのは問題ない。
原典の Appendix 2 のほうが規範的で、規則番号も付いている。
