# 自動点訳エンジン Momo

できてきた。3月の花、果物ということで、"Momo"と名付けよう。

## 準備

``` bash
pip install sudachipy
pip install sudachidict_core
pip install protobuf
```

## 変換してみる

``` bash
./momo
```

標準入力からデータを読み込みます。現在変換できそうな文字種については、sample_dataディレクトリを確認してください。

## 今後

* Rustで書き直したい、高速化に貢献すると思われる。
* Windows用のDLLを作りたい
* 点訳結果とソーステキストの間のマッピングテーブルの出力
* 英語Grade2への対応
* 日本語情報処理点字への対応
* 他にもあるかな
