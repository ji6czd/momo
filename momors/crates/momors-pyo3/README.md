# momors-py

日本語テキスト→点字変換エンジン [MOMO](https://github.com/ji6czd/momo) の Python バインディング（Rust製）です。

## インストール

```bash
pip install momors-py
```

## 使い方

```python
from momors_py import Predictor, BrailleTranslator

predictor = Predictor.from_bundled()
result = predictor.predict("吾輩は猫である")
print(result.kana)  # ワガハイワ ネコデ アル

translator = BrailleTranslator()
braille = translator.translate(result.kana)
print(braille.braille)
```

`Predictor` は同梱の学習済みモデルからカナ変換を行い、`BrailleTranslator` はカナ（または英文）を点字に変換します。
