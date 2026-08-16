# ESP32（P4/S3）で momo を動かす ― 実現性メモ

> ステータス: **調査・未着手**。安価な点字ディスプレイ（Android SoC を積まない）を狙う
> 布石。今月の一般リリース後、別ブランチで扱う。実機計測は未実施＝以下は構造からの推定。
> 関連: [`zerocopy-model-plan.md`](zerocopy-model-plan.md)（このハードが欲しがるフォーマット）、
> [`mbm-format.html`](mbm-format.html)（現行 v0x07 レイアウト）。

## 動機

点字ディスプレイのコストはピエゾセルが支配的だが、compute を Android SoC ではなく
数ドルの MCU に落とせれば電子部品の BOM を実際に下げられる。momo（Rust コア）が
ESP32 クラスで動けば、**Android を積まない安価な点字ディスプレイ**が視野に入る。

## チップの見立て（P4 が本命、S3 は苦しい）

| | ESP32-S3 | ESP32-P4 |
|---|---|---|
| コア | Xtensa LX7 ×2 @240MHz | RISC-V ×2 @400MHz |
| 内蔵SRAM | 512KB | ~768KB(L2)（要確認） |
| PSRAM上限 | **~8MB** | **32MB** |
| SIMD | int8(ESP-NN) | AI/SIMD拡張 |
| Rust | Xtensa=espupフォーク | **RISC-V=mainline** |

- 分かれ目は **PSRAM 上限**。w=4 モデルは 11.4MB なので **S3 の 8MB には入らない**。
  P4 の 32MB なら w=4/w=5 とも余裕。
- 「P4 と同じ RISC-V コアの新しい石」があるならそれも RISC-V 系（型番は未確認）。
- **S3/C3 は非対象**という判断でよい。C3 は非力すぎ、S3 は PSRAM が足りない。

## ロジックは載る／モデルが問題

- 推論の可変作業領域は **十数〜数十KB**（スコア配列 `n_classes`1591×f32=6.4KB＋int版6.4KB＋
  特徴量バッファ＋木走査スタック）。コード(.text)も数十〜百KB台。**SRAM に収まる**。
- 常に問題なのはモデル本体（語彙＋重み）だけ。置き場所は **PSRAM** か **flash（XIP）**。

## 速度の見立て ― バーは低い

点字ディスプレイは人間の読速度で「読んでいる行を訳す」用途。必要スループットは
**毎秒せいぜい数十文字**。PSRAM/flash ミス込みで 1〜5ms/文字でも 200〜1000文字/秒出るので
読速度は楽に超える。**指標は per-line のもたつき（数百msストール）を避けられるか**であって、
生スループットではない。バッチ（本一冊）の話とは別。

## 速度レバー（狙う「ホット部分」は小さい）

- 真にランダムなのは **語彙二分探索の“下の方”だけ**。上位10数レベルはどの検索でも同じ
  中央値を叩くのでキャッシュ常駐。→ **上位ピボット索引を SRAM に**（2^11=2048 個＝数十KB）
  or **Eytzinger(BFS順)レイアウト**で bsearch 全体をキャッシュ局所化（2〜4倍既知）。
- スコア配列(6.4KB)はもともとホットで SRAM 向き。CSC 列読み出しは**シーケンシャル**
  （プリフェッチ有効）。→ 実際にキャッシュ敵対的なのは vocab bsearch の末端のみ。
- **計算量ワイルドカードは GBDT 境界（木300本・毎文字・約1MB）**。SRAM には収まらないが
  毎文字ホットなのでキャッシュに上位ノードが乗る。MCU ティアでは**木を浅く/本数を減らした
  軽い境界モデル**にして精度を少し譲る判断がありうる（語彙刈りとは別軸）。

## flash-XIP vs PSRAM

- **S3**: 8MB PSRAM に入らない → モデルを **flash に置いてキャッシュ経由で直接読む**
  （XIP/データmmap、`esp_partition_mmap`→`const u8*`）。PSRAM を空けたまま動く。
  ただし ESP の **flash mmap 窓サイズ上限** に 11MB が収まるか要検証（S3 は苦しい）。
- **P4**: 32MB PSRAM が直接保持。窓もキャッシュも大きく有利。
- どちらも「モデルを借用」できれば PSRAM ヒープに 11MB を確保せずに済む
  → [`zerocopy-model-plan.md`](zerocopy-model-plan.md) と同じものを欲しがる。

## Rust ツールチェーンの現実（「Rust が無い」は誤解）

- **Espressif は Rust を公式サポート**（専任チーム・`esp-rs`・`espup`・`esp-hal`）。
- **RISC-V の ESP（C3/C6/H2/P4）は mainline Rust をそのまま使える**（フォーク不要）。
  → **P4 が本命なのはツールチェーン観点でも正しい**。
- **Xtensa（S3/S2/初代）は espup のフォークが要る**（動くがフォーク）。P4 なら回避。
- 2つのランタイム: **no_std（esp-hal、リーン・将来線）** / **std（esp-idf-hal、std＋スレッド/fs、
  移植が楽）**。

## momors の移植コスト（限定的・書き直しではない）

現状 momors-core は `std`（`#![no_std]` なし）。ただし:

- **ロードは `load_from_bytes(&[u8])` に集約**され、推論経路に fs は焼き込まれていない。
  **momors-wasm が `Predictor::new(&[u8])`→`from_model_bytes` で wasm32 で動いている**ので、
  「fs を切り離しバイト列から読む」形は**すでに実証済み**。
- 数値計算(`Vec`/スライス/`binary_search`/`sort`)と GBDT の木(`Box`)は core/alloc の範囲。
  **外部Cライブラリ依存ゼロ**（LightGBM は木データとして書き出し boundary.rs で純Rust再実装）。

no_std 移植の具体項目:
1. `#![no_std]` ＋ `extern crate alloc`（`Vec`/`String`/`Box`→alloc）。
2. `Read`/`Cursor`/`File` を `&[u8]` スライス読みに置換。**依存の勘所2つ**:
   - `byteorder`: `ReadBytesExt`（`reader.read_u32::<LE>()`、std::io::Read 依存）ではなく
     **`ByteOrder` トレイト（`LittleEndian::read_u32(&buf)`、no_std）** に置換。
   - `thiserror`: 1.x は std 専用。**no_std 対応版(2.x)か手書きの Error enum** に置換。
3. `std::path`/`std::io::Error` のエラー文脈をスタブ化。
4. グローバルアロケータ（esp-hal/esp-idf の PSRAM アロケータ）。ただし flash 借用に倒せば
   巨大 PSRAM 確保自体を回避できる。

## 進め方（二段構え）

1. **プロトタイプ: `std` on ESP-IDF（P4）**。momors-core をほぼそのままコンパイルし、
   flash パーティションを mmap→`&[u8]`→`from_model_bytes`、**1文字の推論時間を実測**。
   「載る/速い」を no_std 投資の前に確定。
2. **本番: no_std（esp-hal）** で footprint を絞る。
3. materialize コストが問題なら **ゼロコピー＋flash借用**へ（別ブランチの本丸）。

**最初のマイルストーン**: *w=4 を flash に焼いて mmap し、P4 で1文字の推論時間を測る*。
これで載る/載らない・速い/遅いが一発で決まる。

## 未確定・要実機

- flash mmap 窓のサイズ上限（S3 で 11MB が単一窓に収まるか）。
- PSRAM アロケータ（esp-hal 側）の実運用成熟度。
- per-文字レイテンシの実数（devkit 計測必須）。
- 「P4 と同じコアの新系列」の正確な型番と入手性（国内流通は未確認）。

## メモ

電子工作は誤って壊すと悲しいので、モジュールは予備込みで数枚買っておく方針。
