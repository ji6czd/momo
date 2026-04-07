#pragma once
#include <cstdint>
#include <vector>
#include <string>
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

    bool operator<(const FeatureKey& o) const noexcept {
        if (type  != o.type)  return type  < o.type;
        if (u8val != o.u8val) return u8val < o.u8val;
        for (int i = 0; i < 3; ++i) {
            if (ct[i] != o.ct[i]) return ct[i] < o.ct[i];
            if (cp[i] != o.cp[i]) return cp[i] < o.cp[i];
        }
        return false;  // 等しい
    }
};

// ソート済み配列によるルックアップ
// unordered_map の代わりに使用することでメモリ使用量を削減する
using VocabEntry = std::pair<FeatureKey, uint32_t>;
using VocabVec   = std::vector<VocabEntry>;

// 語彙テーブルから feature_id を検索する（バイナリサーチ）
// 見つからない場合は n_features を返す
inline uint32_t vocab_find(const VocabVec& vocab, const FeatureKey& key) {
    const VocabEntry target{key, 0};
    auto it = std::lower_bound(
        vocab.begin(), vocab.end(), target,
        [](const VocabEntry& a, const VocabEntry& b) {
            return a.first < b.first;
        });
    if (it != vocab.end() && it->first == key) {
        return it->second;
    }
    return UINT32_MAX;  // 見つからない
}

// ============================================================
// モデルデータ
// ============================================================

struct MomoModel {
    // --- 語彙テーブル（ソート済み配列）---
    VocabVec vocab;

    // --- 読みラベルテーブル ---
    std::vector<std::string> read_classes;  // [class_id] = ラベル文字列（UTF-8）

    // --- 読みモデル重み（CSC・int8量子化）---
    float                 read_scale = 1.0f;
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
