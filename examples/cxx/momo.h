#ifndef MOMO_H
#define MOMO_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct BackTransHandle BackTransHandle;

typedef struct BackTranslatorHandle BackTranslatorHandle;

typedef struct BrailleDocBuilder BrailleDocBuilder;

typedef struct BrailleDocHandle BrailleDocHandle;

typedef struct ByteBuffer ByteBuffer;

typedef struct FormattedDocHandle FormattedDocHandle;

typedef struct PredictionHandle PredictionHandle;

typedef struct PredictorHandle PredictorHandle;

typedef struct WrapLinesHandle WrapLinesHandle;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
 * UTF-8 パスからモデルを読み込み予測器を作成する。
 * 失敗時は NULL を返す。
 */
PredictorHandle* momo_predictor_new(const char* model_path);

/**
 * UTF-16 パスからモデルを読み込み予測器を作成する。
 * 失敗時は NULL を返す。
 */
PredictorHandle* momo_predictor_new_w(const uint16_t* model_path);

/**
 * 予測器を解放する。NULL は無視する。
 */
void momo_predictor_free(PredictorHandle* handle);

/**
 * UTF-8 テキストを予測する。失敗時は NULL を返す。
 */
PredictionHandle* momo_predict(const PredictorHandle* handle, const char* src_text);

/**
 * UTF-16 テキストを予測する。失敗時は NULL を返す。
 */
PredictionHandle* momo_predict_w(const PredictorHandle* handle, const uint16_t* src_text);

/**
 * 予測結果を解放する。NULL は無視する。
 */
void momo_prediction_free(PredictionHandle* handle);

/**
 * かなテキスト (UTF-8, null 終端) を buf に書き込む。
 *
 * 戻り値: 必要なバイト数 (null 終端含む)。
 * buf_len が不足または buf が NULL の場合は書き込まず、必要サイズだけ返す。
 * handle が NULL なら -1 を返す。
 */
int momo_prediction_kana(const PredictionHandle* handle, char* buf, int buf_len);

/**
 * かなテキスト (UTF-16, null 終端) を buf に書き込む。
 *
 * 戻り値: 必要な u16 要素数 (null 終端含む)。
 * buf_len が不足または buf が NULL の場合は書き込まず、必要サイズだけ返す。
 * handle が NULL なら -1 を返す。
 */
int momo_prediction_kana_w(const PredictionHandle* handle, uint16_t* buf, int buf_len);

/**
 * 点字テキスト (UTF-8, null 終端) を buf に書き込む。
 *
 * 戻り値: 必要なバイト数 (null 終端含む)。
 * buf_len が不足または buf が NULL の場合は書き込まず、必要サイズだけ返す。
 * handle が NULL なら -1 を返す。
 */
int momo_prediction_braille(const PredictionHandle* handle, char* buf, int buf_len);

/**
 * 点字テキスト (UTF-16, null 終端) を buf に書き込む。
 *
 * 戻り値: 必要な u16 要素数 (null 終端含む)。
 * buf_len が不足または buf が NULL の場合は書き込まず、必要サイズだけ返す。
 * handle が NULL なら -1 を返す。
 */
int momo_prediction_braille_w(const PredictionHandle* handle, uint16_t* buf, int buf_len);

/**
 * かなテキストのコードポイント数を返す。handle が NULL なら -1。
 */
int momo_prediction_kana_char_count(const PredictionHandle* handle);

/**
 * 原文のコードポイント数を返す。handle が NULL なら -1。
 */
int momo_prediction_src_char_count(const PredictionHandle* handle);

/**
 * かな→原文 コードポイントインデックス配列を out に書き込む。
 *
 * out は kana_char_count 要素以上の領域を確保しておくこと。
 * handle または out が NULL なら何もしない。
 */
void momo_prediction_kana_to_src(const PredictionHandle* handle, int32_t* out);

/**
 * src→かな インデックスを CSR 形式で書き込む。
 *
 * - row_ptr: src_char_count+1 要素以上の領域を確保すること。
 *   row_ptr[i]..row_ptr[i+1] が原文文字 i に対応するかな文字インデックスの範囲。
 * - col_idx: kana_char_count 要素以上の領域を確保すること。
 *
 * handle が NULL、または両ポインタが NULL なら何もしない。
 * 片方のみ NULL の場合は非 NULL 側だけ書く。
 */
void momo_prediction_src_to_kana(const PredictionHandle* handle, int32_t* row_ptr, int32_t* col_idx);

/**
 * 点字テキストのコードポイント数を返す。handle が NULL なら -1。
 */
int momo_prediction_braille_char_count(const PredictionHandle* handle);

/**
 * src→点字 インデックスを CSR 形式で書き込む。
 *
 * - row_ptr: src_char_count+1 要素以上の領域を確保すること。
 *   row_ptr[i]..row_ptr[i+1] が原文文字 i に対応する点字文字インデックスの範囲。
 * - col_idx: braille_char_count 要素以上の領域を確保すること。
 *   複合音（キャ など）は重複が除去されるため、実際の要素数は
 *   row_ptr[src_char_count]（末尾値）で確認する。
 *
 * handle が NULL、または両ポインタが NULL なら何もしない。
 * 片方のみ NULL の場合は非 NULL 側だけ書く。
 */
void momo_prediction_src_to_braille(const PredictionHandle* handle, int32_t* row_ptr, int32_t* col_idx);

/**
 * バイト列を点字ドキュメントへ読み込む。format: 0=MBR, 1=BES, 2=BET。
 * 失敗時（破損・不正 UTF-8・NULL）は NULL を返す。
 */
BrailleDocHandle* momo_doc_read(const uint8_t* bytes, int len, int format);

/**
 * ドキュメントハンドルを解放する。NULL は無視する。
 */
void momo_doc_free(BrailleDocHandle* handle);

int momo_doc_line_width(const BrailleDocHandle* h);

int momo_doc_lines_per_page(const BrailleDocHandle* h);

bool momo_doc_page_header(const BrailleDocHandle* h);

int momo_doc_number_start(const BrailleDocHandle* h);

/**
 * タイトル（UTF-16, null 終端）を buf に書く。タイトル無しなら 0、handle NULL なら -1。
 */
int momo_doc_title_w(const BrailleDocHandle* h, uint16_t* buf, int buf_len);

int momo_doc_paragraph_count(const BrailleDocHandle* h);

int momo_doc_line_count(const BrailleDocHandle* h, int para);

/**
 * 物理行のテキスト（UTF-16, null 終端）。無効な引数なら -1。
 */
int momo_doc_line_w(const BrailleDocHandle* h, int para, int line, uint16_t* buf, int buf_len);

/**
 * 物理行が論理行末尾か。無効な引数なら false。
 */
bool momo_doc_line_logical_end(const BrailleDocHandle* h, int para, int line);

/**
 * 物理行の改ページマーカー（==== で始まる行）を UTF-16 で buf に書く。
 * 改ページ無しなら 0、無効な引数なら -1。
 */
int momo_doc_line_page_break_w(const BrailleDocHandle* h, int para, int line, uint16_t* buf, int buf_len);

/**
 * ビルダーを作る。number_start: 開始ページ番号。title は NULL/空で無し。
 */
BrailleDocBuilder* momo_doc_builder_new(int line_width, int lines_per_page, bool page_header, int number_start,
                                        const uint16_t* title);

/**
 * 物理行を1行追加する。logical_end が true ならその行で段落を確定する。
 * page_break は ==== マーカー文字列（NULL/空で改ページ無し）。
 */
void momo_doc_builder_add_line(BrailleDocBuilder* b, const uint16_t* content, bool logical_end,
                               const uint16_t* page_break);

/**
 * ビルダーからドキュメントを確定して返す。ビルダーは解放される（再利用・再解放不可）。
 * b が NULL なら NULL。
 */
BrailleDocHandle* momo_doc_builder_build(BrailleDocBuilder* b);

/**
 * ビルダーを解放する（build を呼ばずに破棄する場合）。NULL は無視する。
 */
void momo_doc_builder_free(BrailleDocBuilder* b);

/**
 * ドキュメントを指定形式のバイト列へ書き出す。
 * format: 0=MBR, 1=BES, 3=BASE(.bse), 4=BrailleText(.brf)。無効/NULL なら NULL。
 */
ByteBuffer* momo_doc_write(const BrailleDocHandle* h, int format);

/**
 * バイト列の長さ。NULL なら -1。
 */
int momo_bytes_len(const ByteBuffer* b);

/**
 * バイト列を out へコピーする。out は momo_bytes_len バイト以上確保すること。
 * b または out が NULL なら何もしない。
 */
void momo_bytes_copy(const ByteBuffer* b, uint8_t* out);

/**
 * バイト列を解放する。NULL は無視する。
 */
void momo_bytes_free(ByteBuffer* b);

/**
 * 正本ドキュメントを印刷イメージへ描画してハンドルを返す。NULL なら NULL。
 */
FormattedDocHandle* momo_doc_render(const BrailleDocHandle* h);

/**
 * 印刷イメージハンドルを解放する。NULL は無視する。
 */
void momo_formatted_free(FormattedDocHandle* h);

int momo_formatted_page_count(const FormattedDocHandle* h);

int momo_formatted_line_count(const FormattedDocHandle* h, int page);

/**
 * 印刷イメージの物理行テキスト（UTF-16, null 終端）。無効な引数なら -1。
 */
int momo_formatted_line_w(const FormattedDocHandle* h, int page, int line, uint16_t* buf, int buf_len);

/**
 * 物理行がページヘッダ行か。無効な引数なら false。
 */
bool momo_formatted_line_is_header(const FormattedDocHandle* h, int page, int line);

/**
 * 物理行が論理行末尾か。無効な引数なら false。
 */
bool momo_formatted_line_logical_end(const FormattedDocHandle* h, int page, int line);

/**
 * 物理行の元セグメント通し番号（ヘッダ行や無効な引数なら -1）。
 * エディタが表示行 ↔ 論理位置（セグメント+オフセット）を対応づけるのに使う。
 */
int momo_formatted_line_segment_index(const FormattedDocHandle* h, int page, int line);

/**
 * 1論理行（UTF-16 点字文字列）を line_width マスで折返して物理行リストを返す。
 * 空文字列は count=0 のハンドル（NULL ではない）。text が NULL なら NULL。
 */
WrapLinesHandle* momo_wrap_line_w(const uint16_t* text, int line_width);

/**
 * 論理行のサフィックスを折返す。first_line_remaining は現在行の残りマス数
 * （0 で先頭に空行を出し以降通常幅）。空文字列は count=0。text が NULL なら NULL。
 */
WrapLinesHandle* momo_wrap_suffix_w(const uint16_t* text, int line_width, int first_line_remaining);

/**
 * 折返しハンドルを解放する。NULL は無視する。
 */
void momo_wrap_lines_free(WrapLinesHandle* h);

int momo_wrap_lines_count(const WrapLinesHandle* h);

/**
 * 物理行テキスト（UTF-16, null 終端）。無効な引数なら -1。
 */
int momo_wrap_lines_get_w(const WrapLinesHandle* h, int index, uint16_t* buf, int buf_len);

/**
 * 物理行が論理行末尾か。無効な引数なら false。
 */
bool momo_wrap_lines_logical_end(const WrapLinesHandle* h, int index);

/**
 * 段落 `para` の論理テキスト（物理行を連結したもの）を UTF-16 で buf に書く。
 * 無効な引数なら -1。
 */
int momo_doc_logical_text_w(const BrailleDocHandle* h, int para, uint16_t* buf, int buf_len);

/**
 * 論理段落（UTF-16, `\n` 区切り）を折返し・ページ分割して印刷イメージを返す。
 * paragraphs が NULL なら NULL。
 */
FormattedDocHandle* momo_doc_render_from_paragraphs(const uint16_t* paragraphs, int line_width, int lines_per_page,
                                                    bool page_header, int number_start, const uint16_t* title);

/**
 * 論理段落（UTF-16, `\n` 区切り）を折返した上で指定形式のバイト列へ書き出す。
 * `format`: 0=MBR, 1=BES, 3=BASE, 4=BrailleText。paragraphs が NULL／format 不正なら NULL。
 */
ByteBuffer* momo_doc_write_from_paragraphs(const uint16_t* paragraphs, int line_width, int lines_per_page,
                                           bool page_header, int number_start, const uint16_t* title, int format);

/**
 * 組み込みテーブルで逆変換器を作る。失敗時は NULL。
 */
BackTranslatorHandle* momo_back_translator_new(void);

/**
 * `japanese_braille.toml`（UTF-16 パス）から逆変換器を作る。失敗時は NULL。
 */
BackTranslatorHandle* momo_back_translator_new_from_file_w(const uint16_t* toml_path);

/**
 * 逆変換器を解放する。NULL は無視する。
 */
void momo_back_translator_free(BackTranslatorHandle* handle);

/**
 * 点字文字列（UTF-16）を逆変換し、全文とセグメントを持つハンドルを返す。
 * handle または braille が NULL なら NULL。
 */
BackTransHandle* momo_back_translate_w(const BackTranslatorHandle* handle, const uint16_t* braille);

/**
 * 逆変換結果を解放する。NULL は無視する。
 */
void momo_back_trans_free(BackTransHandle* handle);

/**
 * 復元された全文（UTF-16, null 終端）を buf に書く。
 * 戻り値は必要な u16 要素数（null 含む）。buf 不足／NULL なら書かずにサイズだけ返す。
 * handle が NULL なら -1。
 */
int momo_back_trans_text_w(const BackTransHandle* handle, uint16_t* buf, int buf_len);

/**
 * セグメント数を返す。handle が NULL なら -1。
 */
int momo_back_trans_segment_count(const BackTransHandle* handle);

/**
 * 各セグメントの点字セル範囲を out_start / out_end へまとめて書く。
 *
 * 各配列は momo_back_trans_segment_count 要素以上を確保すること。
 * 範囲は半開区間 `[start, end)`、インデックスは入力点字の UTF-16 位置に対応する。
 * handle が NULL なら何もしない。片方のみ NULL なら非 NULL 側だけ書く。
 */
void momo_back_trans_cell_bounds(const BackTransHandle* handle, int32_t* out_start, int32_t* out_end);

/**
 * セグメント idx のテキスト（UTF-16, null 終端）を buf に書く。
 * 戻り値は必要な u16 要素数（null 含む）。buf 不足／NULL なら書かずにサイズだけ返す。
 * handle が NULL、または idx が範囲外なら -1。
 */
int momo_back_trans_segment_text_w(const BackTransHandle* handle, int idx, uint16_t* buf, int buf_len);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif /* MOMO_H */
