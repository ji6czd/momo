from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

# 既存の features.py からインポート
# from features import compute_source_features, get_units

def train_lr_model(train_data_list):
    """
    train_data_list: [(source_text, reading_segmented), ...] のリスト
    """
    pass

def main():
    """コマンドラインで指定された学習用。.tsvファイルを読み込んで学習する。
点	テン	CharType.KANJI	S	0
字	ジ	CharType.KANJI	S	1
は	ワ+S	CharType.HIRAGANA	S	2
フ	フ	CharType.KATAKANA	S	3
ラ	ラ	CharType.KATAKANA	S	4
ン	ン	CharType.KATAKANA	S	5
ス	ス	CharType.KATAKANA	S	6
人	ジン	CharType.KANJI	S	7
の	ノ+S	CharType.HIRAGANA	S	8
    """

    train_data_list: list[str] = []
    with open("train_data.txt", "r", encoding="utf-8") as f :
        for line in f:
            line = line.strip()
            if not line:
                continue
            train_data_list.append(line)
    model_data = train_lr_model(train_data_list)
    joblib.dump(model_data, "lr_model_data.pkl")

if __name__ == "__main__":
    main()
    