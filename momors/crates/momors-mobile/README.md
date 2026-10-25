# momors-mobile

Momo の Android / iOS バインディング。[UniFFI](https://mozilla.github.io/uniffi-rs/) で Kotlin と Swift のコードを生成する。
API は momors-wasm / momors-pyo3 と同じ形（`Predictor` / `BrailleTranslator` / `BrailleBackTranslator`）。

モデル（`.mbm`）はバイト列で渡せるので、Android の assets や iOS の bundle から読んだ内容をそのまま使える。
配布物には w4 / w5 / w7 の3モデルを同梱する。使う側がサイズと精度で選ぶ（`basic_data_4.mbm` は約 3.3MB。実機でのメモリ・ロード時間は未測定）。

## 使い方

```kotlin
// Kotlin
val predictor = Predictor(context.assets.open("basic_data_4.mbm").readBytes())
val translator = BrailleTranslator()
val kana = predictor.predict("吾輩は猫である").kana
val braille = translator.translateJapanese(kana).braille
```

```swift
// Swift
let model = try Data(contentsOf: Bundle.main.url(forResource: "basic_data_4", withExtension: "mbm")!)
let predictor = try Predictor(mbmData: model)
let translator = try BrailleTranslator()
let kana = try predictor.predict(text: "吾輩は猫である").kana
let braille = try translator.translateJapanese(kana: kana).braille
```

`predict` の `kana` にはマスあけ（空白）が入る。読みだけが欲しいときは空白を除く。

## ビルド

```bash
# ホスト向け（型検査・テスト）
cargo test -p momors-mobile

# Android（cargo-ndk と Android NDK が必要）
rustup target add aarch64-linux-android x86_64-linux-android
task sdk:android      # dist/sdk/android/{jniLibs,kotlin,assets}

# iOS（macOS + Xcode が必要）
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
task sdk:ios          # dist/sdk/ios/{MomorsMobile.xcframework,swift,resources}
```

Kotlin / Swift の生成だけなら、ホストでビルドしたライブラリから `uniffi-bindgen` を呼ぶ。

```bash
cargo run -p momors-mobile --bin uniffi-bindgen -- generate \
  --library target/debug/momors_mobile.dll --language kotlin --out-dir out/kotlin
```

Kotlin 版は JNA（`net.java.dev.jna:jna`、AAR 版）に依存する。

## 範囲外

`BrailleDocument` 系（`momo_doc_*`）は含めない。必要になったら追加する。
