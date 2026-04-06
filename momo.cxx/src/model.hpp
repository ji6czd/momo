#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include "feature_type.hpp"
#include "char_type.hpp"

// ============================================================
// 語彙キー — 特徴量のルックアップキー
// ============================================================

struct FeatureKey {
    FeatureType type   = FeatureType::BIAS;
    uint8_t     u8val  = 0;
    CharType    ct[3]  = {};
    char32_t    cp[3]  = {};

    bool operator==(const FeatureKey& o) const noexcept {
        if (type  != o.type)  return false;
        if (u8val != o.u8val) return false;
        for (int i = 0; i < 3; ++i) {
            if (ct[i] != o.ct[i]) return false;
            if (cp[i] != o.cp[i]) return false;
        }
        return true;
    }
};

struct FeatureKeyHash {
    std::size_t operator()(const FeatureKey& k) const noexcept {
        std::size_t h = 2166136261u;
        auto mix = [&](std::size_t v) {
            h ^= v;
            h *= 16777619u;
        };
        mix(static_cast<uint8_t>(k.type));
        mix(k.u8val);
        for (int i = 0; i < 3; ++i) {
            mix(static_cast<uint8_t>(k.ct[i]));
            mix(static_cast<std::size_t>(k.cp[i]));
        }
        return h;
    }
};

using VocabMap = std::unordered_map<FeatureKey, uint32_t, FeatureKeyHash>;

// ============================================================
// モデルデータ
// ============================================================

struct MomoModel {
    // --- 語彙テーブル ---
    VocabMap vocab;

    // --- 読みラベルテーブル ---
    std::vector<std::string> read_classes;  // [class_id] = ラベル文字列（UTF-8）

    // --- 読みモデル重み（CSC・int8量子化）---
    // CSCは列優先: feature_id をインデックスとして対応クラスと重みを直接引ける
    // score[cls] += csc_data[j] * read_scale  (j は feature_id 列の非ゼロ要素)
    float                read_scale = 1.0f;
    std::vector<uint32_t> csc_colptr;   // size: n_features + 1
    std::vector<uint32_t> csc_rowind;   // size: n_nonzero  （クラスID）
    std::vector<int8_t>   csc_data;     // size: n_nonzero

    // --- 読みモデル intercept ---
    std::vector<float> intercept_read;  // size: n_classes

    // --- 境界モデル重み（int8量子化）---
    float               boundary_scale = 1.0f;
    std::vector<int8_t> boundary_data;  // size: n_features
    float               boundary_intercept[2] = {0.0f, 0.0f};

    // --- サイズ情報 ---
    uint32_t n_classes  = 0;
    uint32_t n_features = 0;
};
