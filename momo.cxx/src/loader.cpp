#include "loader.hpp"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

// ============================================================
// バイナリ読み込みユーティリティ
// ============================================================

static void read_exact(std::FILE* fp, void* buf, std::size_t n, const char* label) {
    if (std::fread(buf, 1, n, fp) != n) {
        throw std::runtime_error(std::string("読み込み失敗: ") + label);
    }
}

static uint8_t  read_u8 (std::FILE* fp) { uint8_t  v; read_exact(fp, &v, 1, "u8");  return v; }
static uint32_t read_u32(std::FILE* fp) { uint32_t v; read_exact(fp, &v, 4, "u32"); return v; }
static float    read_f32(std::FILE* fp) { float    v; read_exact(fp, &v, 4, "f32"); return v; }

// ============================================================
// CSR → CSC 変換
// ============================================================

static void csr_to_csc(
    uint32_t                     n_classes,
    uint32_t                     n_features,
    const std::vector<uint32_t>& csr_indptr,
    const std::vector<uint32_t>& csr_indices,
    const std::vector<int8_t>&   csr_data,
    std::vector<uint32_t>&       csc_colptr,
    std::vector<uint32_t>&       csc_rowind,
    std::vector<int8_t>&         csc_data)
{
    const uint32_t n_nonzero = static_cast<uint32_t>(csr_data.size());

    csc_colptr.assign(n_features + 1, 0);
    for (uint32_t idx = 0; idx < n_nonzero; ++idx) {
        ++csc_colptr[csr_indices[idx] + 1];
    }
    for (uint32_t f = 1; f <= n_features; ++f) {
        csc_colptr[f] += csc_colptr[f - 1];
    }

    csc_rowind.resize(n_nonzero);
    csc_data.resize(n_nonzero);

    std::vector<uint32_t> pos(csc_colptr.begin(), csc_colptr.end());

    for (uint32_t cls = 0; cls < n_classes; ++cls) {
        for (uint32_t j = csr_indptr[cls]; j < csr_indptr[cls + 1]; ++j) {
            const uint32_t feat_id = csr_indices[j];
            const uint32_t dest    = pos[feat_id]++;
            csc_rowind[dest] = cls;
            csc_data[dest]   = csr_data[j];
        }
    }
}

// ============================================================
// FeatureKey の復元
// ============================================================

static FeatureKey read_feature_key(std::FILE* fp) {
    FeatureKey key;
    key.type  = static_cast<FeatureType>(read_u8(fp));
    key.u8val = 0;

    const int  n_ct  = feature_chartype_count(key.type);
    const int  n_cp  = feature_char32_count(key.type);
    const bool is_u8 = feature_is_uint8_payload(key.type);

    for (int i = 0; i < n_ct; ++i) {
        key.ct[i] = static_cast<CharType>(read_u8(fp));
    }
    for (int i = 0; i < n_cp; ++i) {
        key.cp[i] = static_cast<char32_t>(read_u32(fp));
    }
    if (is_u8) {
        key.u8val = read_u8(fp);
    }
    return key;
}

// ============================================================
// .mbm ファイルの読み込み
// ============================================================

MomoModel load_model(const std::string& path) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("モデルファイルを開けません: " + path);
    }

    MomoModel model;

    // --- ファイルヘッダ ---
    char magic[4];
    read_exact(fp, magic, 4, "magic");
    if (std::memcmp(magic, "MOMO", 4) != 0) {
        std::fclose(fp);
        throw std::runtime_error("不正なマジックバイト: " + path);
    }
    const uint8_t version = read_u8(fp);
    if (version != 0x01) {
        std::fclose(fp);
        throw std::runtime_error("未対応のバージョン: " + std::to_string(version));
    }
    read_u8(fp); read_u8(fp); read_u8(fp);  // reserved

    model.n_classes  = read_u32(fp);
    model.n_features = read_u32(fp);

    // --- 語彙テーブル（ソート済み配列として構築）---
    model.vocab.resize(model.n_features);
    for (uint32_t i = 0; i < model.n_features; ++i) {
        model.vocab[i].first  = read_feature_key(fp);
        model.vocab[i].second = read_u32(fp);
    }
    // FeatureKey の < 演算子でソート
    std::sort(model.vocab.begin(), model.vocab.end(),
              [](const VocabEntry& a, const VocabEntry& b) {
                  return a.first < b.first;
              });

    // --- 読みラベルテーブル ---
    model.read_classes.resize(model.n_classes);
    for (uint32_t i = 0; i < model.n_classes; ++i) {
        const uint8_t len = read_u8(fp);
        std::string label(len, '\0');
        read_exact(fp, label.data(), len, "label");
        model.read_classes[i] = std::move(label);
    }

    // --- 読みモデル重み（CSR で読み込んで CSC に変換）---
    model.read_scale = read_f32(fp);
    const uint32_t n_nonzero = read_u32(fp);

    std::vector<uint32_t> csr_indptr(model.n_classes + 1);
    read_exact(fp, csr_indptr.data(),
               (model.n_classes + 1) * sizeof(uint32_t), "csr_indptr");

    std::vector<uint32_t> csr_indices(n_nonzero);
    read_exact(fp, csr_indices.data(),
               n_nonzero * sizeof(uint32_t), "csr_indices");

    std::vector<int8_t> csr_data(n_nonzero);
    read_exact(fp, csr_data.data(),
               n_nonzero * sizeof(int8_t), "csr_data");

    csr_to_csc(model.n_classes, model.n_features,
               csr_indptr, csr_indices, csr_data,
               model.csc_colptr, model.csc_rowind, model.csc_data);

    // --- 読みモデル intercept ---
    model.intercept_read.resize(model.n_classes);
    read_exact(fp, model.intercept_read.data(),
               model.n_classes * sizeof(float), "intercept_read");

    // --- 境界モデル重み ---
    model.boundary_scale = read_f32(fp);
    model.boundary_data.resize(model.n_features);
    read_exact(fp, model.boundary_data.data(),
               model.n_features * sizeof(int8_t), "boundary_data");
    model.boundary_intercept[0] = read_f32(fp);
    model.boundary_intercept[1] = read_f32(fp);

    std::fclose(fp);
    return model;
}
