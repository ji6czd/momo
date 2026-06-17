// 典型的な呼び出し手順 (UTF-8 の場合)
MomoPredictor p = momo_predictor_new("basic_data_7.mbm");
MomoPrediction r = momo_predict(p, "東京都民のために");

// ① かなテキスト取得（2フェーズ）
int32_t n = momo_prediction_kana(r, NULL, 0);   // 必要サイズ確認
char* buf = malloc(n);
momo_prediction_kana(r, buf, n);                 // buf に書き込み

// ①' 点字テキスト取得（かなを日本語点字に変換したもの。同じ2フェーズ）
int32_t bn = momo_prediction_braille(r, NULL, 0);
char* bbuf = malloc(bn);
momo_prediction_braille(r, bbuf, bn);

// ② インデックス取得
int32_t kc = momo_prediction_kana_char_count(r);
int32_t sc = momo_prediction_src_char_count(r);

int32_t* k2s     = malloc(kc * sizeof(int32_t));        // kana→src
int32_t* row_ptr = malloc((sc+1) * sizeof(int32_t));    // CSR行ポインタ
int32_t* col_idx = malloc(kc * sizeof(int32_t));        // CSR列インデックス

momo_prediction_kana_to_src(r, k2s);
momo_prediction_src_to_kana(r, row_ptr, col_idx);

// src[i] に対応するかな文字インデックス範囲: row_ptr[i] .. row_ptr[i+1]

momo_prediction_free(r);
momo_predictor_free(p);
