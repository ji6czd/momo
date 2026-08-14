#pragma once
#include <memory>
#include "momo.h"

// momo.h（cbindgen 自動生成）の C++ RAII ラッパー。
// momo.h に新しいハンドル型や free 関数が追加された場合はこのファイルも更新すること。
//
// C FFI ハンドルを自動解放する RAII ラッパー。
// std::unique_ptr のカスタムデリータとして各 free 関数を渡す。

namespace momo {

namespace detail {
struct PredictorDel  { void operator()(PredictorHandle*      p) const { momo_predictor_free(p); } };
struct TableDel      { void operator()(TableHandle*          p) const { momo_table_free(p); } };
struct JapaneseTranslatorDel { void operator()(JapaneseTranslatorHandle* p) const { momo_japanese_translator_free(p); } };
struct EnglishTranslatorDel  { void operator()(EnglishTranslatorHandle*  p) const { momo_english_translator_free(p); } };
struct BrailleTranslatorDel  { void operator()(BrailleTranslatorHandle*  p) const { momo_braille_translator_free(p); } };
struct BrailleResultDel      { void operator()(BrailleResultHandle*      p) const { momo_braille_result_free(p); } };
struct DocDel        { void operator()(BrailleDocHandle*     p) const { momo_doc_free(p); } };
struct FormattedDel  { void operator()(FormattedDocHandle*   p) const { momo_formatted_free(p); } };
struct BytesDel      { void operator()(ByteBuffer*           p) const { momo_bytes_free(p); } };
struct WrapLinesDel  { void operator()(WrapLinesHandle*      p) const { momo_wrap_lines_free(p); } };
struct BackTransDel  { void operator()(BackTransHandle*      p) const { momo_back_trans_free(p); } };
struct BackTranslatorDel { void operator()(BackTranslatorHandle* p) const { momo_back_translator_free(p); } };
}  // namespace detail

using PredictorPtr      = std::unique_ptr<PredictorHandle,      detail::PredictorDel>;
using TablePtr          = std::unique_ptr<TableHandle,          detail::TableDel>;
// 変換器の new/braille_translator_new はハンドルを消費するので、その場合は .release() で
// 所有権を放棄してから渡すこと（放棄しないと二重解放になる）。
using JapaneseTranslatorPtr = std::unique_ptr<JapaneseTranslatorHandle, detail::JapaneseTranslatorDel>;
using EnglishTranslatorPtr  = std::unique_ptr<EnglishTranslatorHandle,  detail::EnglishTranslatorDel>;
using BrailleTranslatorPtr  = std::unique_ptr<BrailleTranslatorHandle,  detail::BrailleTranslatorDel>;
using BrailleResultPtr      = std::unique_ptr<BrailleResultHandle,      detail::BrailleResultDel>;
using DocPtr            = std::unique_ptr<BrailleDocHandle,     detail::DocDel>;
using FormattedPtr      = std::unique_ptr<FormattedDocHandle,   detail::FormattedDel>;
using BytesPtr          = std::unique_ptr<ByteBuffer,           detail::BytesDel>;
using WrapLinesPtr      = std::unique_ptr<WrapLinesHandle,      detail::WrapLinesDel>;
using BackTransPtr      = std::unique_ptr<BackTransHandle,      detail::BackTransDel>;
using BackTranslatorPtr = std::unique_ptr<BackTranslatorHandle, detail::BackTranslatorDel>;

// BrailleDocBuilder は build() が所有権を奪う特殊ケースのため専用クラス。
class DocBuilder {
public:
    explicit DocBuilder(BrailleDocBuilder* b) : b_(b) {}
    ~DocBuilder() { if (b_) momo_doc_builder_free(b_); }

    DocBuilder(const DocBuilder&) = delete;
    DocBuilder& operator=(const DocBuilder&) = delete;

    void add_line(const uint16_t* content, bool logical_end) {
        momo_doc_builder_add_line(b_, content, logical_end);
    }

    // marker: "====" マーカー文字列（NULL/空でプレーンな改ページ）。
    // テキストを一切持たない、独立した改ページエントリを追加する。
    void add_page_break(const uint16_t* marker = nullptr) {
        momo_doc_builder_add_page_break(b_, marker);
    }

    // build() を呼ぶと内部ポインタを放棄し、返された DocPtr がライフタイムを管理する。
    DocPtr build() {
        auto* raw = momo_doc_builder_build(b_);
        b_ = nullptr;  // build() が解放済み
        return DocPtr(raw);
    }

    bool valid() const { return b_ != nullptr; }

private:
    BrailleDocBuilder* b_;
};

// number_style: 0=Standard, 1=Alternative（既定 0）。
inline DocBuilder make_doc_builder(int line_width, int lines_per_page, bool page_header,
                                   int number_start, const uint16_t* title = nullptr,
                                   int number_style = 0) {
    return DocBuilder(momo_doc_builder_new(line_width, lines_per_page, page_header, number_start,
                                            number_style, title));
}

}  // namespace momo
