import sys
import sklearn_crfsuite


def char2features(sentence: str, i: int):
    char = sentence[i][0]
    ctype = sentence[i][2]

    features = {
        'bias': 1.0,
        'char': char,
        'type': ctype,
    }
    
    # 前の文字の情報
    if i > 0:
        features.update({
            '-1:char': sentence[i-1][0],
            '-1:type': sentence[i-1][2],
        })
    else:
        features['BOS'] = True # 文頭

    # 次の文字の情報
    if i < len(sentence) - 1:
        features.update({
            '+1:char': sentence[i+1][0],
            '+1:type': sentence[i+1][2],
        })
    else:
        features['EOS'] = True # 文末

    return features


def load_tsv(filename):
    sentences = []
    current_sentence = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                current_sentence.append(parts)
    return sentences

def train_and_test(tsv_file, test_text):
    # 1. データの読み込み
    data = load_tsv(tsv_file)
    X = [[char2features(s, i) for i in range(len(s))] for s in data]
    y = [[s[i][1] for i in range(len(s))] for s in data]

    # 2. モデルの訓練（CRF）
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100)
    crf.fit(X, y)
    print("🤖 学習が完了しました！")

    # 3. 未知の文章の予測（テスト）
    # テスト用に簡易的な文字種判定を適用
    from create_data import get_char_type # 前に作った関数を再利用
    test_sentence = [[c, "", get_char_type(c)] for c in test_text]
    X_test = [char2features(test_sentence, i) for i in range(len(test_sentence))]
    
    y_pred = crf.predict_single(X_test)

    # 4. 結果の表示
    print(f"\n【推論結果】 原文: {test_text}")
    result_kana = []
    for char, pred in zip(test_text, y_pred):
        if pred == "_": continue # スキップ記号
        if pred == "---": continue # ブロック継続記号
        result_kana.append(pred)
    
    print(f"予測される読み: {''.join(result_kana)}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        train_and_test(sys.argv[1], sys.argv[2])
    else:
        print(f"使用法: python {sys.argv[0]} <TSVファイル> <テストしたい文字列>")
