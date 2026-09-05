# ESP32-P4 で momors を動かす計画（シリアルコンソール REPL）

> 作成 2026-09-04。対象ボード: ESP32-P4（16MB フラッシュ / 32MB PSRAM）。
> ゴール: シリアルコンソールから日本語テキストを1行受け取り、かな・分かち書き・点字を
> シリアルに書き戻す。

## 0. 前提の実測（PC 上、2026-09-04）

| 項目 | 値 | 出典 |
|---|---|---|
| window=4 モデル `.mbm` のサイズ | **11.8 MB**（w5=24.9MB, w7=38.0MB） | `dataset/basic_data_4.mbm` |
| w4 ロード後の常駐メモリ（Vec 全展開） | **19.6 MB**（ワークロード中ピーク 24.3MB） | `mmap_experiment::mmap_bench_memory_vec` |
| momors-core の外部依存 | thiserror, byteorder のみ | `Cargo.toml` |
| momors-braille の外部依存 | thiserror, serde, toml（起動時に埋め込み TOML を解析） | 同上 |
| std 依存箇所 | `std::fs`（loader/辞書）, `HashMap`, `LazyLock`, `std::io::Read` | grep |
| 既存の「バイト列からロード」API | `Predictor::from_model_bytes(&[u8])`（WASM 用） | prediction.rs:513 |
| 手元のツール | espflash 4.5.0 導入済み、`riscv32imafc-unknown-none-elf` ターゲット導入済み | rustup |

結論:
- **w4 モデル一択**。フラッシュ 16MB に 11.8MB のモデル＋アプリ（〜2-3MB）が収まり、
  PSRAM 32MB に常駐 20MB が収まる。w5/w7 はどちらも収まらない。
- ボードのメモリ余裕は「収まるが潤沢ではない」。ロード時に **モデルを一度 PSRAM に
  丸ごとコピーしてから展開する（12MB+20MB=32MB）方式は不可**。フラッシュから
  直接読みながら展開する必要がある（後述）。

## 1. 方針の選択: std（ESP-IDF）で行く

| | A: std / ESP-IDF（`riscv32imafc-esp-espidf`） | B: no_std / esp-hal（`riscv32imafc-unknown-none-elf`） |
|---|---|---|
| momors-core/braille の改修 | **ほぼ不要**（std がそのまま使える） | 大（HashMap→hashbrown, thiserror v2, toml→事前展開, LazyLock, alloc 化） |
| 標準入出力 | ESP-IDF VFS 経由で `std::io::stdin/stdout` がそのまま動く | UART/USB-JTAG ドライバを直接叩く |
| ヒープを PSRAM に置く | sdkconfig 一行（`CONFIG_SPIRAM_USE_MALLOC`） | `esp-alloc` で PSRAM 領域を登録 |
| フラッシュ上モデルへのアクセス | `esp_partition_mmap` で `&[u8]` が得られる | 自前で XIP 領域を指す |
| ツールチェーン | nightly + `-Zbuild-std` + ESP-IDF（embuild が自動取得）+ `ldproxy` | stable + esp-hal |
| 将来の XIP ゼロコピー | 可（mmap した `&[u8]` をそのまま参照） | 可 |

**A を採用**。理由: 今回の目的は「まず動かす」であり、コード改修が最小で済むうえ、
Rust の RISC-V ESP-IDF ターゲットは std が（プロセス・シグナル以外）完全実装で
tier 2 昇格が進んでいる。B（no_std）は `docs/zerocopy-model-plan.md` の
XIP/ゼロコピー化と一緒に扱うべき別テーマとして後回しにする。

## 2. 成果物の構成

```
momors/crates/momors-esp32/          # 新規クレート（default-members から除外。wasm/pyo3 と同じ扱い）
├── Cargo.toml                       # momors-core, momors-braille, esp-idf-svc, esp-idf-sys
├── .cargo/config.toml               # target=riscv32imafc-esp-espidf, build-std, MCU=esp32p4, ESP_IDF_VERSION
├── build.rs                         # embuild::espidf::sysenv::output()
├── sdkconfig.defaults               # PSRAM/コンソール/スタック/パーティション設定
├── partitions.csv                   # nvs, phy_init, factory(4MB), model(12MB, data/0x40)
├── rust-toolchain.toml              # nightly 固定
└── src/main.rs                      # シリアル REPL
```

`Taskfile.yml` に `esp:build` / `esp:flash` / `esp:flash-model` / `esp:monitor` を追加。

## 3. 設計の要点

### 3.1 モデルの配置とロード（mmap 方式を主線にする）
- モデルは **専用パーティション**（`model`, 12MB）に `espflash write-bin` で書き込む。
  アプリの再フラッシュでモデルを毎回書き直さずに済む。
- 起動時に `esp_partition_find` → `esp_partition_mmap` でパーティション全域を
  データ領域にマップし `&[u8]` を得て、`Predictor::from_model_bytes()` に渡す。
  現行の loader は `Cursor<&[u8]>` から `Read` で逐次展開するので、
  **フラッシュ→PSRAM への 12MB コピーは発生しない**（展開後の 20MB のみ）。
- mmap のサイズ上限に当たった場合のフォールバック: `esp_partition_read` を
  チャンクで呼ぶ `std::io::Read` アダプタを書き、momors-core に
  `Predictor::from_model_reader(impl Read)` を **1 関数だけ追加**する
  （`load_from_reader` は既に `R: Read` 総称なので中身は既存関数の公開だけ）。

### 3.2 辞書（単漢字辞書・人名辞書）
- どちらも **`.mbm` 内のセクション**（loader.rs の `read_name_dict` / `read_single_char_dict`）に
  入っており、`from_model_bytes()` はモデル側の辞書をそのまま使う
  （`PredictorConfig` のパス指定は PC で上書きしたい時の口）。
  → **端末側で辞書ファイルを別途持つ必要はない**。カスタム辞書（フレーズ辞書）だけは
  外部 TSV なので、必要になった段階で埋め込みか VFS で扱う。

### 3.3 sdkconfig の要点
- `CONFIG_SPIRAM=y`, `CONFIG_SPIRAM_USE_MALLOC=y`, `CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL=4096`
  → 大きなアロケーション（モデルの Vec）は自動的に PSRAM に行く。内部 SRAM は ESP-IDF 用に温存。
- `CONFIG_ESP_MAIN_TASK_STACK_SIZE=65536` 以上（toml 解析・serde の再帰と推論バッファ）。
  足りなければ推論スレッドを `std::thread::Builder::stack_size` で別途立てる。
- コンソール: `CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y`（P4 開発ボードの USB ポート想定。
  UART0 経由のボードなら切替）。
- `CONFIG_ESPTOOLPY_FLASHSIZE_16MB=y`, `CONFIG_PARTITION_TABLE_CUSTOM=y`。
- タスク WDT は main タスクで無効化（1 文の推論が秒単位になりうるため）。

### 3.4 シリアルプロトコル（行指向・UTF-8）
```
> 吾輩は猫である            ← 入力 1 行（改行で確定）
kana:   ワガハイワ ネコデアル
seg:    吾輩は/猫で/ある
braille: ⠄⠡⠓⠁⠊⠄⠀⠇⠪⠐⠟⠁⠗
time:   123 ms   heap: free 9.8MB (psram 9.6MB)
```
- `:` 始まりはコマンド（`:stat` ヒープ/モデル情報、`:mode kana|braille|all`、`:help`）。
- 出力は PC 版 `momo` CLI と同じ変換段（predict → BrailleTranslator）を踏み、
  PC 側スクリプトからシリアルに流して **PC の momo 出力と突き合わせられる** ようにする。

## 4. マイルストーン

| # | 内容 | 完了条件 |
|---|---|---|
| M0 | ツールチェーン整備（nightly, ldproxy, esp-idf-template で Hello）。PSRAM 認識確認 | シリアルに `hello` と `psram free: ~32MB` が出る。コンソールのエコーが動く |
| M1 | momors-braille のみ載せる（モデル不要）。かな→点字を REPL で | ワークスペースのクレートがターゲットでビルドできる（toml/serde/LazyLock の確認）。スタック量の見積り |
| M2 | モデルパーティション＋`esp_partition_mmap`＋`from_model_bytes` | `吾輩は猫である` が PC と同じかなになる。ロード時間・ロード後ヒープを記録 |
| M3 | 完全 REPL（テキスト→かな→点字、時間・ヒープ表示）。PC 側から `eval_regression.tsv` を流して突き合わせ | 回帰セットで PC 版と一致。1 文あたりの所要時間を記録 |
| M4（次期） | メモリ節約: 語彙 bsearch を mmap 領域から直接引く（v0x08 のキー順ソートを活用）→ 常駐 20MB を数 MB へ。no_std 化の要否判断 | `docs/zerocopy-model-plan.md` へ実測を追記 |

## 5. リスクと対処

- **Windows 上の ESP-IDF ビルド**: embuild が ESP-IDF を clone するパスが長くなると
  失敗しがち。`ESP_IDF_TOOLS_INSTALL_DIR=global` と短いワークスペースパスで回避。
  行き詰まったら WSL2 + usbipd に逃げる（espflash は WSL からも使える）。
- **ESP-IDF バージョン**: P4 は v5.3 以降。v5.4/5.5 系を固定する。
- **`esp_partition_mmap` の上限**: P4 の MMU は大きな領域をマップできるはずだが、
  12MB 一括が通らなければ 3.1 のフォールバックへ。
- **推論速度**: 語彙 31.8 万件の二分探索が PSRAM 上のランダムアクセスになる。
  PC 比で 1〜2 桁遅くなる見込み。まず実測し、M4 の課題にする。
- **バイナリサイズ**: std + ESP-IDF で 1.5〜2.5MB。factory 4MB で余裕あり。

## 6. mmap 方式の位置づけと no_std の損得（2026-09-04 追記）

### 6.1 mmap は 2 段階ある

ESP32-P4 の `esp_partition_mmap` は OS のデマンドページングではなく、フラッシュ MMU
＋キャッシュによる XIP（読み出しは自動でキャッシュに引き込まれ、追い出しも自動）。
効き方は Linux の mmap と同じで「触った所だけが SRAM/キャッシュに載る」。

| 段階 | やること | 常駐 (PSRAM) | core の改修 |
|---|---|---|---|
| ① 展開ロード | mmap した `&[u8]` を `from_model_bytes()` に渡す | ≈20MB（Vec 全展開） | **ゼロ** |
| ② 借用ロード（ゼロコピー） | 語彙表・CSC 配列を mmap 領域の `&[u8]` から **借用したまま** 二分探索する（v0x08 のキー順ソートが前提。境界 GBDT 木は約 1MB なので従来どおり展開） | **数 MB**（GBDT 木＋クラス表＋人名辞書索引） | loader を「読み込んで Vec を作る」から「スライスを切り出して持つ」へ。exporter 側で各セクションを 4 バイト整列（u32 配列を直接キャストするため）。`MomoModel<'a>` に寿命が付く（端末側は `'static` に leak すればよい） |

②が `docs/zerocopy-model-plan.md` で「未着手」となっていた本体で、このボードがその
実装を駆動する自然な場になる。ただし ②はメモリを減らす代わりに **速度で払う**:
二分探索 18 プローブ×数百キー/文字 がフラッシュ読み（PSRAM より遅い）に落ちる。
このボードは PSRAM 32MB で ①が収まるので、**①で動かして実測 → ②で常駐と速度を
比べる** の順にする。②は「もっと小さいボード」「w5 を載せたい」時に効く保険。

### 6.2 no_std の損得

このボード・この目的（シリアル REPL）では **no_std の利得はほぼ無い**:

| 観点 | std (ESP-IDF) | no_std (esp-hal) | 効くか |
|---|---|---|---|
| バイナリ | 1.5〜2.5MB | 数百 KB | 16MB フラッシュでは無関係 |
| 内部 SRAM | ESP-IDF が 100〜200KB 消費 | ほぼゼロ | P4 内部 768KB で余裕。モデルは PSRAM |
| 起動時間 | +100ms 程度 | 短い | モデル展開（秒単位）に埋もれる |
| ツールチェーン | nightly + build-std + ESP-IDF 取得 | stable のみ | CI・再現性で多少効く |
| リアルタイム性 | FreeRTOS | ベアメタル | 翻訳には不要 |
| **移植性** | Espressif 限定 | 他社 MCU、**他社ファームウェアへのリンク** | **ここだけ本質的** |

本質的な利得は最後の 1 行だけ。点字ディスプレイのメーカー製ファームウェア
（多くは C の RTOS 環境で ESP-IDF ではない）に「liblouis の代替」として
`momors-core` を静的リンクさせたい場合、`no_std + alloc + C ABI` が要件になる。
これは [project-liblouis-integration] の長期戦略の話であり、今回のゴールとは独立。

改修コストの見積り（機械的な置換が中心）:
- momors-core: `HashMap`→`hashbrown`（互換）、`thiserror` 1→2（no_std 対応）、
  `std::io::Read` ベースの loader → スライスカーソル（②の借用ロードと同時にやると自然）、
  `std::fs`/`Path` を受ける API を feature `std` の後ろへ。
- momors-braille: 起動時 TOML 解析（`toml` crate は std 必須）→ ビルド時にテーブルを
  コード生成 or 事前直列化、`LazyLock`→`once_cell`/`spin`、`from_file` を feature `std` へ。

結論: **今回は std で進める**。ただし②の借用ロードを実装する際、loader を
「`Read` 経由」でなく「`&[u8]` スライス経由」で書くと、それがそのまま no_std 化の
半分を済ませたことになるので、その形で書く。no_std 本体は M3 後に、他社ファームへの
組み込みが具体化した時点で判断する。

## 7. 進捗記録

### 2026-09-04: M0〜M3 まで到達（ボード上で PC と同一の点字出力を確認）

構成: `momors/crates/momors-esp32`（`task esp:build` / `esp:flash` / `esp:flash-model` / `esp:monitor`）。

| 項目 | 実測 |
|---|---|
| アプリサイズ | 1.07MB（factory 4MB の 26%） |
| 起動時ヒープ | 内部 500KB / PSRAM 32.0MB |
| 点字テーブル（埋め込み TOML 解析） | 189 ms、PSRAM 0.16MB |
| モデル展開（`esp_partition_mmap` → `from_model_bytes`） | **2.05 秒**、PSRAM 使用 **14.5MB**（残 17.3MB）、内部 SRAM 消費ほぼゼロ |
| 推論（1 文） | 吾輩は猫である 22 ms／今日は良い天気ですね。34 ms／東京都渋谷区で3人の学生が本を読んだ。60 ms |
| PC（同じ w4 モデル） | 0.30 / 0.31 / 0.56 ms → **端末は約 100 倍遅い**（CPU 360MHz + PSRAM 常駐） |
| 出力一致 | 3 文とも PC と点字が完全一致（`渋/谷区` の分割も w4 モデル自体の挙動で PC と同じ） |

躓いた点（再現時の手引き）:

1. **パス長**: esp-idf-sys は出力ディレクトリが 88 文字を超えると拒否する。`task esp:build` が Windows では `CARGO_TARGET_DIR` をドライブ直下の `<ドライブ>:/mo-esp` に設定する（環境変数 `ESP_TARGET_DIR` で変更可）。`.cargo/config.toml` の `CARGO_WORKSPACE_DIR` はそのために必要。
2. **esp-idf-hal 0.46.2 が ESP-IDF v5.5.5 と不整合**（`spi_transaction_t` が不透明型）。必要なのは生 API だけなので `esp-idf-svc` を外し `esp-idf-sys` 直接依存にした。
3. **VFS 関数名**: esp-idf-sys 0.37 の bindings.h は v5 系で旧名ヘッダを取り込む → `esp_vfs_usb_serial_jtag_use_driver` / `esp_vfs_dev_usb_serial_jtag_set_*_line_endings`。
4. **パーティション表**: `CONFIG_PARTITION_TABLE_CUSTOM` は CMake プロジェクトが out dir に置かれるため使えず、espflash の `--partition-table` に渡す。独自パーティションは `type=0x40, subtype=0x00`（data 型に独自サブタイプは不可）。
5. **ブートローダ**: 当初 espflash 同梱の汎用ブートローダで `overlaps bootloader stack` が出て起動せず、ESP-IDF ビルドの `bootloader.bin` を `--bootloader` で渡していた。しかしこれは次項のチップ改版不整合（v3.01 向けの RAM 配置でビルドしていた）が原因で、改版設定を直した後は同梱ブートローダで起動する（2026-09-05 確認）。`--bootloader` 指定は撤去。
6. **チップ改版**: ESP-IDF v5.5 の既定は P4 v3.01 以上向け。手元は **v1.3** なので `CONFIG_ESP32P4_SELECTS_REV_LESS_V3=y` + `CONFIG_ESP32P4_REV_MIN_100=y` が必須（違うとブートローダのエントリで Illegal instruction）。
7. **内部 SRAM 枯渇**: `CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL=4096`（既定）だとモデルの大量の小オブジェクト（String・GBDT の Box ノード・辞書）が内部 380KB を使い切り、直後の pthread mutex 生成で abort。**`=0`（常に PSRAM 優先）** にして解決。内部は `RESERVE_INTERNAL=131072` で ESP-IDF 用に確保。
8. クラッシュループ中は USB-Serial-JTAG が再列挙を繰り返し `espflash` の接続が不安定になる。数回リトライすれば通る。再列挙で **COM 番号が変わることがある**（COM9 → COM10）。`task esp:*` はポート省略時に espflash の自動検出に任せる。
9. 配布は `task esp:image`（`espflash save-image --merge --skip-padding`）の結合イメージを `espflash write-bin 0x0` で書く。`--skip-padding` を付けないと 16MB にパディングされ、モデル領域まで 0xFF で潰す。

次の候補:
- 回帰セット（`dataset/eval_regression.tsv`）を PC 側からシリアルへ流して全文突き合わせ（M3 の完了条件）。

### 2026-09-05: フラッシュ QIO 化と、M4（借用ロード）の GO/NO-GO 判定

M4 の借用ロードは **見送り**。代わりに、判定のために測ったことから
**フラッシュ QIO 化**という別の当たりが出た。

#### 1. 測定手段: `:xip`

loader を書き換える前に「フラッシュ読みは PSRAM より何倍遅いか」だけを切り出す
REPL コマンドを追加した（`src/main.rs` の `bench_xip`）。model パーティションの
mmap 領域の先頭 4MB と、それを PSRAM にコピーしたバッファに対し、**同じ関数・同じ
乱数種**で 2 パターンを回す:

- **bsearch probe**: 固定幅レコード配列を log2(N) 回、区間を狭めながら触る
  （キー比較はしない。触るアドレスの系列だけ本物の二分探索と同じにする）
- **csc column**: ランダム位置から 96B 連続読み（CSC の 1 列相当）

#### 2. フラッシュ QIO 化（`CONFIG_ESPTOOLPY_FLASHMODE_QIO` + espflash `--flash-mode qio`）

最初の `:xip` が「フラッシュのプローブ 8.8µs」という異常な値を出したので設定を見ると、
フラッシュが **既定の DIO**（80MHz でも 2bit/clk = 20MB/s）だった。L2 ラインは 128B に
してあるので 1 ミスの充填に 6.4µs かかる計算で、実測と一致する。QIO（4bit/clk = 40MB/s）
に変えた結果:

| 項目 | DIO@80MHz | QIO@80MHz | |
|---|---|---|---|
| モデル展開 | 1,911 ms | **1,011 ms** | 47% 減 |
| 推論 total | 511.4 µs/字 | **436.8 µs/字** | 15% 減 |
| ├ normalize | 9.8 | **3.1** | |
| ├ source | 16.8 | **6.3** | |
| ├ featurize | 29.9 | 23.4 | |
| ├ resolve | 81.3 | 72.2 | |
| ├ read | 189.4 | 185.0 | ほぼ不変 |
| ├ boundary | 105.2 | 100.4 | ほぼ不変 |
| └ finalize | 11.6 | **6.3** | |
| `:xip` bsearch probe (flash/psram) | 8,840 / 379 ns = 23.3x | **2,420 / 443 ns = 5.46x** | |
| `:xip` csc column (flash/psram) | 21,556 / 3,021 ns = 7.1x | **6,440 / 3,026 ns = 2.13x** | |

**効き方の内訳が理屈通り**なのが重要: 縮んだのは normalize / source / finalize / featurize
のような **コード主体（アプリはフラッシュから XIP 実行している）** の段で、read /
boundary のような **PSRAM 上のデータ主体**の段はほぼ動かない。つまり QIO 化は
命令フェッチのミスコストを半減させた。**借用ロードの是非とは無関係に効く、設定 2 行の改善**。

出力は 3 文とも QIO 化前と完全一致（`渋/谷区` の分割も含め PC と同じ）。1 文あたり
4 / 5 / 8 ms（2026-09-04 の 22 / 34 / 60 ms は c946639・6d0eb31 の推論高速化より前の値）。

#### 3. 常駐 14.5MB の内訳（`:probe` の CSC 統計 + 構造体サイズから）

| 領域 | サイズ | 算出 |
|---|---|---|
| 語彙表 `Vec<VocabEntry>` | **10.2MB** | 318,971 件 × 32B（`FeatureKey` 20B + feature_id/cat_column/cat_code 12B） |
| CSC `colptr`/`rowind`/`data` | **4.2MB** | 318,972×4B + 985,482×2B + 985,482×1B |
| 計 | 14.4MB | 実測の PSRAM 使用 14.5MB とほぼ一致（GBDT 木・クラス表・辞書は残りの端数） |

借用ロードで消せるのはこの 14.4MB。逆に言えば**それ以外に消せるものは無い**。

#### 4. 判定: 見送り

QIO 化で比が 23.3x → 5.46x（語彙）/ 7.1x → 2.13x（CSC）まで改善したが、**このボードでは
やらない**。理由:

- **払う先が悪い**。`:probe` の実測は 1 文字あたり lookup 73µs / read 181µs / boundary 147µs で、
  境界 GBDT の高速化（反転索引）が効いた今、**重心は読みモデル側（語彙引き + CSC 読み）に
  移っている**。借用ロードはまさにそこを遅くする。ラフな外挿で **437 → 850µs/字 程度（約 2 倍）**。
- **買うものが要らない**。PSRAM は 32MB 中 17.25MB が空いている。ロード 1 秒も、
  常時電源のシリアル REPL では困らない。

ただし **DIO 時代の「10 倍遅くなる」から「2 倍」に落ちた**ので、選択肢としては生きている。
効くのは「PSRAM が無い / 8MB しかないボード」「w5 を載せたい」ときで、その際は
**語彙表から先に借用する**のが良い（下記）。

#### 5. 再挑戦するときのための気づき

- **語彙のほうが割が良い**。合成ベンチの比は語彙 5.46x・CSC 2.13x で CSC のほうが「安全」に
  見えるが、**MB あたりで見ると逆**: 語彙は 10.2MB 消せて触るのは 1 文字あたり
  31.2 keys × 18 プローブ ≈ 数十ライン、CSC は 4.2MB しか消せないのに 1 文字あたり
  nnz 11,433 × 3B ≈ 34KB（128B ライン換算で 268 ライン）を舐める。**消えるメモリは語彙が 2.4 倍、
  触るバイト数は CSC が桁違いに多い**。「借用するなら語彙から、CSC は最後」。
- **`:xip` は実際よりやや楽観的**。合成バッファは 4MB・16B レコードだが、本物の語彙表は
  10.2MB・32B レコードなので、二分探索の上位段がキャッシュに残る割合はもっと低い。
- **実アクセスは合成より局所性が良い**。`:probe` の lookup 73µs / (31.2 keys × 18 プローブ)
  = 130ns/プローブで、`:xip` の PSRAM 443ns/プローブより 3.4 倍速い。同じ特徴量キーが
  文字をまたいで繰り返し出るため。合成ベンチは両側に同じだけ効くので比としては妥当だが、
  絶対値の外挿には使わないこと。
- **固定幅化するとファイルが膨らむ**。語彙は `.mbm` 上ではもっと詰めて格納されている
  （ファイル全体で 11.8MB）。32B 固定幅で並べると語彙だけで 10.2MB になり、
  **model パーティション 12MB がきつい**。借用ロードをやるならパーティション拡張が要る。
- 速度: 100 倍差の内訳を計測（語彙 bsearch の PSRAM アクセス vs GBDT 木の Box 追跡）。小オブジェクトだけ内部 SRAM に置く/木のフラット化の再検討。
- M4: 借用ロードで常駐 14.5MB → 数 MB（`docs/zerocopy-model-plan.md`）。

### 2026-09-04（続き）: 速度の内訳と設定チューニング

目標: 1 文字 1 ms（2 ms でも許容）。`:bench`（38 文字×5 回の `predict`）と `:probe`
（momors-core の feature `diagnostics`、`Predictor::char_latency`）で計測。

| 手当て | predict µs/文字 | 語彙引き | 読み | 境界 | 備考 |
|---|---|---|---|---|---|
| 初期（L2 128KB） | 2,997 | 160 | 1,000 | 1,112 | |
| **L2 キャッシュ 512KB / 128B ライン** | 1,643 | 77 | 699 | 358 | 内部 SRAM は 118KB に減る（`RESERVE_INTERNAL=32KB` に） |
| 読みループの境界チェック除去（model.rs） | **1,512** | 76 | 549 | 369 | PC テスト 246 件通過。PC でも同一結果 |
| 400MHz（`FORCE_400MHZ_ON_REV_LESS_V3`） | ― | | | | **v1.3 では `esp_clk_init` の assert で起動不能**。360MHz 固定 |

内訳の理解:
- 語彙: 1 文字 31 キー（ヒット 14.6）。二分探索 1 回 ≈ 2.5 µs。
- 読み: 1 文字あたり **nnz ≈ 11,400 要素**を散布加算。モデル全体では密な列（非ゼロがクラス数の
  半分以上）は **67 本（nnz の 9%）** しかないが、文字種など「常に出る」特徴なので
  1 文字あたりの nnz の 96% を占める。≈48 ns/要素。
- 境界: 語彙引きをもう一度（≈76 µs）＋ GBDT 200 本を `Box` ノードで辿る。
- 特徴量生成: 43 µs/文字。
- 残り ≈500 µs/文字は `predict` の周辺処理（argmax・ラベル出力・インデックス構築・
  PSRAM 上の小さな確保など）。PC でも同じ構造で `predict` 29.5 µs に対し内訳合計 12.9 µs と
  2.3 倍の差があるので PSRAM 固有ではなく、プロファイルで詰める余地がある。

次の候補（結果を変えない最適化、効果の見込み順）:
1. **密な列の組み合わせ和のメモ化**: 1 文字がヒットする密な列は約 7 本で、その組み合わせは
   文字種パターンで決まり種類が少ない。組み合わせをキーに 1,591 要素の和ベクトルを LRU で
   持てば、読みは 540 → 100 µs 前後（疎な残り ≈400 要素＋ベクトル加算）。
2. **境界での語彙引き再利用**: 読み用に引いた feature_id/VocabRef を境界にも渡す（−76 µs）。
3. **GBDT 木のフラット化**: PC では遅くなった（docs/zerocopy-model-plan.md）が、PSRAM では
   キャッシュミス削減が効く可能性が高い。端末で再評価。
4. `predict` 周辺処理のプロファイル（PC 上で可能）。

### 2026-09-04（続き 2）: 密な列のメモ化と語彙引きの一本化 → **1,055 µs/文字**

momors-core 本体の変更（結果は完全不変: 2 万行相当 8,463 行の出力が w4/w7 とも一致、
単体テスト 246 件通過。回帰テストの不一致 2 件 `8日`/`1人` は変更前から存在する既存のずれ）:

1. **密な列の組み合わせ和キャッシュ**（`model.rs` `DenseSumCache`）: ヒットした密な列
   （非ゼロがクラス数の半分以上）の id 集合をキーに int32 和ベクトルを 64 スロット LRU で保持。
   ヒット時は疎な列（≈400 要素）だけを足す。`MomoModel` に `Mutex` で持つので呼び出しをまたいで効く。
2. **語彙引きの一本化**（`prediction.rs` `resolve_all`）: 各キーを `resolve` で 1 回だけ引き、
   読み用 feature_id と境界用 `VocabRef` を同時に作る。`boundary_has_split` は `VocabRef` 列を受ける
   （`Boundary::compute_score_resolved`）。

| | predict µs/文字 | 語彙引き | 読み | 境界 |
|---|---|---|---|---|
| 前（L2 512KB + 境界チェック除去） | 1,512 | 76 | 549 | 369 |
| **後** | **1,055** | 78 | **200** | 340（probe はキー版のまま。predict 内では二重引きなし） |

PC への効果: 2 万行コーパスで **w4 9.8 → 6.9 s（−29%）、w7 21.3 → 16.7 s（−22%）**。

残り: `predict` 全体 1,055 に対し内訳合計 ≈666（特徴量 48 + 語彙 78 + 読み 200 + 境界 340）。
差 ≈390 µs は 1 文字ごとの小さな確保（ラベル `String`、3 本のインデックス `Vec` の伸長、
`src_to_kana_index` の原文バイト数ぶんの `Vec`）と argmax。次の候補は
GBDT 木のフラット化（端末で再評価）と、この確保の削減。

### 2026-09-04（続き 3）: 段階別プロファイル → 制約付き argmax の線形走査を排除 → **779 µs/文字**

`predict` に段階別タイマー（`momors-core/src/phase.rs`、feature `diagnostics` 無効時は no-op）を
仕込み、端末 `:phases` / PC `profile_predict_phases` テストで内訳を出した。

端末（µs/文字、変更前）: total 1,045 = normalize 10 + source 18 + featurize 32 + resolve 81
+ loop 855（read **474** / boundary 313 / その他 70）+ finalize 14。

`:probe` の読みスコア計算は 200 µs なのに `read_argmax` は 474 µs → 差は **単一文字辞書
制約付き argmax** が漢字 1 文字ごとに 1,591 クラスのラベル文字列を線形走査して辞書の読みと
比較していたため（PSRAM 上の String 比較）。ロード時に読み→クラス id へ解決
（`resolve_single_char_dict`）し、候補 id のスコア比較だけにした（`constrained_argmax_cls`。
同点はクラス id 昇順で最初、という元の規則を維持）。

| | predict µs/文字（端末） | read | boundary | PC µs/文字 | PC 2 万行 w4 / w7 |
|---|---|---|---|---|---|
| 前 | 1,045 | 474 | 313 | 15.1 | 6.9 s / 16.7 s |
| **後** | **779** | **222** | 295 | **8.4** | **3.3 s / 9.7 s**（初期比 w4 9.8→3.3、w7 21.3→9.7） |

結果は不変（8,463 行の出力が w4/w7 とも一致、テスト 246 件通過）。
残りの内訳（端末）: boundary 295（GBDT 200 本の木を Box で辿る）、resolve 84、loop_other 65、
featurize 32。次は GBDT 木のレイアウト（フラット化）で boundary を削る。

### 2026-09-04（続き 4）: GBDT 木を「2 ワードノード + 反転索引」に → **514 µs/文字**

まず 2026-08-16 に撤回した形に近い「前順の圧縮配列（cats はノードに隣接）」を試したが、
端末 boundary 295 → 276 µs、PC は w4 がやや遅く w7 がやや速い、と効果は小さかった。
木の統計（`:probe`/PC テストの `tree:` 行）を取ると: 木 100 本・split 2,200・葉 2,300・
split あたり cats 平均 29（最大 32）・最大深さ 22・列 21 本、列のコード空間は文字 identity
系（列 0〜2）が 5 万、文字種系は 1〜300。1 ノード訪問 ≈110 ns は cats の二分探索
（分岐）が支配的で、レイアウトでは削れない。

そこで **膜構造を反転**した（`boundary.rs` `TreeEnsemble`、ファイル形式は不変）:
- ロード時に「(列, コード) → そのコードを cats に含む split id」の反転索引を作る
  （列ごとにコード順の `codes`/`splits` 配列）。
- ノードは cats を持たず 2 ワード（header: 列・default_left・split id / 右の子の位置）。
  全木 36KB で L1/L2 に収まる。
- 推論時は 1 文字につき列ごとに 1 回だけ索引を引き、「左へ進む split」のビット集合
  （2,200 ビット、スタック上）を作る。木を辿るときはビットを見るだけ。
  元の `cats.binary_search(&code)` と厳密に同じ判定。

| | predict µs/文字（端末） | boundary | PC µs/文字 | PC 2 万行 w4 / w7 |
|---|---|---|---|---|
| 前（argmax 事前解決まで） | 779 | 295 | 8.4 | 3.3 s / 9.7 s |
| 圧縮配列（撤回案の再評価） | 736 | 276 | 9.7 | 3.8 s / 9.3 s |
| **反転索引** | **514** | **103** | **6.7** | **2.7 s / 8.1 s** |

結果は不変（8,463 行一致、テスト 246 件通過）。PSRAM 常駐は 14.5MB → 14.6MB（索引 0.5MB 増、
Box ノード 0.4MB 減）。

残りの内訳（端末 µs/文字）: read 189 > resolve 80 > loop_other 41 > featurize 31 > finalize 12。
今日の初期状態 2,997 から **5.8 倍**。
