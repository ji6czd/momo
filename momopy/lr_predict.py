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
    X_features = []
    y_labels = []
    kanji_candidates = {} # C++移植に必須の「漢字->読み候補」辞書

    for text, reading_str in train_data_list:
        # 1. 前処理: 読みから '+S' を除去してラベル化
        # 例: "ワ+S" -> "ワ"
        labels = [r.replace("+S", "") for r in reading_str.split("/")]
        
        # 2. 特徴量抽出 (既存のロジックをそのまま使用)
        units = get_units(text)
        # source_seq の形に整形
        source_seq = [(char, i, ctype) for i, (char, _, ctype) in enumerate(units)]
        feats = compute_source_features(source_seq)
        
        X_features.extend(feats)
        y_labels.extend(labels)

        # 3. 候補辞書の構築 (推論時の Masking 用)
        for i, (char, _, ctype) in enumerate(source_seq):
            if ctype == 'KANJI':
                if char not in kanji_candidates:
                    kanji_candidates[char] = set()
                kanji_candidates[char].add(labels[i])

    # ベクトル化
    vec = DictVectorizer(sparse=True)
    X_vec = vec.fit_transform(X_features)
    
    label_enc = LabelEncoder()
    y_vec = label_enc.fit_transform(y_labels)

    # 学習 (C++移植を考え solver='liblinear' / One-vs-Rest)
    model = LogisticRegression(solver='liblinear', multi_class='ovr', C=1.0)
    model.fit(X_vec, y_vec)

    # C++移植用のメタデータ保存
    model_data = {
        'weights': model.coef_,           # (ラベル数, 特徴量数)
        'intercept': model.intercept_,     # (ラベル数,)
        'feature_names': vec.get_feature_names_out().tolist(),
        'label_names': label_enc.classes_.tolist(),
        'kanji_candidates': {k: list(v) for k, v in kanji_candidates.items()}
    }
    return model_data

def main():
    # コマンドラインのファイルを読み込んでリストに保存
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
    