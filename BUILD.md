# Momoビルド手順

## Prerequisites

以下の環境を想定しています。私は可能な限り新しものに更新する方針なので、開発環境はさらに新しいものになっている可能性があります。

### ハードウェア

現在の私の環境です。これよりメモリを搭載した環境を推奨します。

- CPU: Intel Core i9-13900H
- RAM: 32GB
- SSD: 1TB NVMe

### ソフトウェア

以下のソフトウェアが必要です。

- Python 3.13 以上
- uv 0.11.19 以上(Pythonのビルドツール)
- Rust 1.94 以上
- Cargo (Rustのビルドツール)
- Node.js 26.3 以上
- pnpm (Node.jsのパッケージマネージャー)
- C++コンパイラ
- Hugo
- Wix Toolset (Windowsの場合)

C++コンパイラは、Rustのビルドに必要です。Windowsの場合はVisual StudioのC++ビルドツールをインストールしてください。LinuxやmacOSの場合は、GCCやClangなどの一般的なC++コンパイラが必要です。

## ビルド手順

ビルドは主に二つのステップに分かれます。一つは、モデルのトレーニングと、そのモデルを使ったパッケージのビルドです。モデルのトレーニングには３０分程度かかると思います。

プロジェクトルートで以下を実行します。

```bash
task train
```

これでモデルのトレーニングが行われ、datasetディレクトリにモデルが保存されます。トレーニングが完了したら、次のコマンドでパッケージのビルドを行います。

```bash
task
```

それぞれのサブプロジェクトでビルドが行われ、成果物が生成されます。ビルドが成功したら、各サブプロジェクトのREADME.mdを参照して、生成された成果物の使用方法を確認してください。

## Windows用インストーラ

Rust版のWindowsパッケージを作成できます。wingetなどでインストールしてください。

下記のコマンドでWindows用のインストーラを生成できます。

```bash
task dist:msi
```

## ESP32-P4 ファームウェア（momors-esp32）

`momors/crates/momors-esp32` は ESP32-P4（16MB フラッシュ・32MB PSRAM）向けのシリアルコンソール
REPL です。ESP-IDF（std）上で momors-core / momors-braille をそのまま動かします。
設計と計測の記録は `docs/esp32p4-plan.md` を参照してください。

### 動かすだけの場合

ビルド環境は不要です。`espflash`（`cargo install espflash`、または GitHub Releases のバイナリ）だけ
用意し、配布された結合イメージとモデルを書き込みます。

```bash
espflash write-bin 0x0 momors-esp32-esp32p4.bin       # ブートローダ + パーティション表 + アプリ
espflash write-bin 0x400000 basic_data_4.mbm          # window=4 モデル（約 12MB、初回のみ）
espflash monitor                                      # 115200 bps、1 行入力すると変換結果が返る
```

### ビルドする場合

通常のビルドに加えて次が必要です。

- Rust nightly と `rust-src`（`rustup toolchain install nightly --component rust-src`。
  クレート内の `rust-toolchain.toml` が自動で選択します）
- `cargo install ldproxy espflash`
- Python 3、git、CMake、Ninja が PATH にあること（ESP-IDF のビルドに使います）
- 初回ビルド時に ESP-IDF v5.5.5 とツールチェーンが `~/.espressif` に自動取得されます
  （数 GB・時間がかかります。プロキシ環境では git と pip の設定が必要です）

```bash
task esp:build        # ビルド
task esp:flash        # 書き込み（ESP-IDF が生成したブートローダを自動選択）
task esp:flash-model  # dataset/basic_data_4.mbm を 0x400000 に書き込む（初回のみ）
task esp:monitor      # シリアルモニタ
task esp:image        # 配布用の結合イメージを dist/ に保存
```

注意点:

- **Windows のパス長**: esp-idf-sys は出力ディレクトリが 88 文字を超えるとビルドを拒否します。
  `task esp:build` は `CARGO_TARGET_DIR` をリポジトリのあるドライブ直下の `<ドライブ>:/mo-esp` に
  設定します。変えたいときは環境変数 `ESP_TARGET_DIR` で指定してください。
- **チップ改版**: `sdkconfig.defaults` は v1.x のチップ（`CONFIG_ESP32P4_REV_MIN_100`）向けです。
  v3.x のボードでは `CONFIG_ESP32P4_SELECTS_REV_LESS_V3` と `REV_MIN_100` の行を外してください
  （合わないとブートローダが起動しません）。
- **ブートローダ**: espflash 同梱の汎用ブートローダではこのアプリは起動しません。`task esp:flash`
  と `task esp:image` は ESP-IDF ビルドが生成した `bootloader.bin` を使います。
- **シリアルポート**: 省略時は espflash が自動検出します。複数ある場合は `PORT=COM9 task esp:flash`
  のように指定してください。

## バージョン番号の更新

以下のコマンドでバージョン番号を更新します。

```bash
task version:bump V=0.9.0
```

## さらに

Taskfile.ymlには、他にも便利なタスクが定義されています。例えば、ビルド成果物のクリーンアップや、sdkの作成などがあります。
