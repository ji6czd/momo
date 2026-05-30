//! 特徴量の型定義モジュール。
//!
//! C++ 版の `feature_type.hpp` と `model.hpp` の `FeatureKey` 部分に対応。
//! Python 版 `momo_py/exporter.py` の `FT` クラスとも同じ値を共有する。
//!
//! ## バイナリエンコーディング
//!
//! [`FeatureType`] の `u8` 値はビットフィールドになっている:
//!
//! - bit7-6 : ペイロード種別
//!   - `0b00` (`0x0_`)      = なし
//!   - `0b01` (`0x4_–0x7_`) = `CharType × N`
//!   - `0b10` (`0x8_–0xB_`) = `char32_t × N`
//!   - `0b11` (`0xC_`)      = `u8 × 1`
//! - bit5-4 : 個数
//!   - `0b00` = 0 個（または `u8×1` 固定）
//!   - `0b01` = 1 個
//!   - `0b10` = 2 個
//!   - `0b11` = 3 個
//!
//! これにより `u8` 値だけからペイロードの読み方が決定できる。

use crate::char_type::CharType;

// ============================================================
// FeatureType
// ============================================================

/// 特徴量の種別。
///
/// 値は C++ 版 `feature_type.hpp` の `enum class FeatureType` と一致する。
/// モデルファイル (.mbm) のバイト互換性を保つために値を変更してはいけない。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(u8)]
pub enum FeatureType {
    // --- ペイロードなし (0b00'00'xxxx) ---
    #[default]
    Bias = 0x00,
    KanjiPosFirst = 0x01,

    // --- CharType×1 (0b01'01'xxxx) ---
    TypeSelf = 0x50,
    TypePrev1 = 0x51,
    TypePrev2 = 0x52,
    TypeNext1 = 0x53,
    TypeNext2 = 0x54,
    TypePrev3 = 0x55,
    TypeNext3 = 0x56,

    // --- CharType×2 (0b01'10'xxxx) ---
    TypeTransition = 0x60,

    // --- CharType×3 (0b01'11'xxxx) ---
    TypeTriPrev2Prev1Self = 0x70,
    TypeTriPrev1SelfNext1 = 0x71,
    TypeTriSelfNext1Next2 = 0x72,
    TypeTriPrev3Prev2Prev1 = 0x73,
    TypeTriNext1Next2Next3 = 0x74,

    // --- char32_t×1 (0b10'01'xxxx) ---
    CharSelf = 0x90,
    CharPrev1 = 0x91,
    CharPrev2 = 0x92,
    CharNext1 = 0x93,
    CharNext2 = 0x94,
    CharPrev3 = 0x95,
    CharNext3 = 0x96,

    // --- char32_t×2 (0b10'10'xxxx) ---
    BigramPrev1Self = 0xA0,
    BigramPrev2Prev1 = 0xA1,
    BigramSelfNext1 = 0xA2,
    BigramNext1Next2 = 0xA3,
    BigramPrev3Prev2 = 0xA4,
    BigramNext2Next3 = 0xA5,

    // --- char32_t×3 (0b10'11'xxxx) ---
    TrigramPrev2Prev1Self = 0xB0,
    TrigramPrev1SelfNext1 = 0xB1,
    TrigramSelfNext1Next2 = 0xB2,
    TrigramPrev3Prev2Prev1 = 0xB3,
    TrigramNext1Next2Next3 = 0xB4,

    // --- u8×1 (0b11'00'xxxx) ---
    KanjiRunLen = 0xC0,
    JapaneseNumericRunLen = 0xC1,
    PrevJapaneseNumericRunLen = 0xC2,
}

impl FeatureType {
    /// ペイロードに含まれる [`CharType`] の個数。
    ///
    /// CharType ペイロード系 (`0x4_`–`0x7_`) のときのみ非ゼロ。
    #[inline]
    pub fn chartype_count(self) -> usize {
        let v = self as u8;
        if (v & 0xC0) != 0x40 {
            0
        } else {
            ((v >> 4) & 0x03) as usize
        }
    }

    /// ペイロードに含まれる 32bit コードポイント (`u32`) の個数。
    ///
    /// char32 ペイロード系 (`0x8_`–`0xB_`) のときのみ非ゼロ。
    #[inline]
    pub fn char32_count(self) -> usize {
        let v = self as u8;
        if (v & 0xC0) != 0x80 {
            0
        } else {
            ((v >> 4) & 0x03) as usize
        }
    }

    /// ペイロードが `u8 × 1` かどうか。
    ///
    /// run_len 系 (`0xC_`) のときに `true`。
    #[inline]
    pub fn is_uint8_payload(self) -> bool {
        (self as u8) & 0xC0 == 0xC0
    }

    /// `u8` 値から [`FeatureType`] を構築する。
    ///
    /// 主にモデルファイル読み込み時に使用する。
    /// 未定義の値は `None` を返す。
    pub fn from_u8(v: u8) -> Option<Self> {
        // 32 個の variant すべてを列挙する。
        // match の網羅性チェックでバリアント追加を検出できる。
        use FeatureType::*;
        Some(match v {
            0x00 => Bias,
            0x01 => KanjiPosFirst,

            0x50 => TypeSelf,
            0x51 => TypePrev1,
            0x52 => TypePrev2,
            0x53 => TypeNext1,
            0x54 => TypeNext2,
            0x55 => TypePrev3,
            0x56 => TypeNext3,

            0x60 => TypeTransition,

            0x70 => TypeTriPrev2Prev1Self,
            0x71 => TypeTriPrev1SelfNext1,
            0x72 => TypeTriSelfNext1Next2,
            0x73 => TypeTriPrev3Prev2Prev1,
            0x74 => TypeTriNext1Next2Next3,

            0x90 => CharSelf,
            0x91 => CharPrev1,
            0x92 => CharPrev2,
            0x93 => CharNext1,
            0x94 => CharNext2,
            0x95 => CharPrev3,
            0x96 => CharNext3,

            0xA0 => BigramPrev1Self,
            0xA1 => BigramPrev2Prev1,
            0xA2 => BigramSelfNext1,
            0xA3 => BigramNext1Next2,
            0xA4 => BigramPrev3Prev2,
            0xA5 => BigramNext2Next3,

            0xB0 => TrigramPrev2Prev1Self,
            0xB1 => TrigramPrev1SelfNext1,
            0xB2 => TrigramSelfNext1Next2,
            0xB3 => TrigramPrev3Prev2Prev1,
            0xB4 => TrigramNext1Next2Next3,

            0xC0 => KanjiRunLen,
            0xC1 => JapaneseNumericRunLen,
            0xC2 => PrevJapaneseNumericRunLen,

            _ => return None,
        })
    }
}

// ============================================================
// FeatureKey
// ============================================================

/// 特徴量のルックアップキー。
///
/// 語彙テーブルの検索用 (`Vec<(FeatureKey, u32)>` のソート + `binary_search`)。
/// 未使用フィールドはゼロ初期化されることを前提とする。
///
/// `derive(PartialOrd, Ord)` は **フィールド宣言順** で辞書順比較するため、
/// C++ 版の `operator<` の順序 (`type → u8val → ct[0..3] → cp[0..3]`)
/// とは異なる。これはモデルファイル側のソート順とも異なるため、loader 側で
/// 読み込み後に Rust の `Ord` で再ソートする必要がある。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct FeatureKey {
    pub feature_type: FeatureType,
    pub u8val: u8,
    pub ct: [CharType; 3],
    pub cp: [u32; 3],
}

impl FeatureKey {
    /// ペイロードなしの特徴量キー (Bias, KanjiPosFirst) を作る。
    pub fn no_payload(feature_type: FeatureType) -> Self {
        debug_assert_eq!(feature_type.chartype_count(), 0);
        debug_assert_eq!(feature_type.char32_count(), 0);
        debug_assert!(!feature_type.is_uint8_payload());
        Self {
            feature_type,
            ..Self::default()
        }
    }

    /// `CharType × 1` の特徴量キーを作る (`TypeSelf` 等)。
    pub fn type_1(feature_type: FeatureType, ct: CharType) -> Self {
        debug_assert_eq!(feature_type.chartype_count(), 1);
        Self {
            feature_type,
            ct: [ct, CharType::default(), CharType::default()],
            ..Self::default()
        }
    }

    /// `CharType × 2` の特徴量キーを作る (`TypeTransition`)。
    pub fn type_2(feature_type: FeatureType, ct0: CharType, ct1: CharType) -> Self {
        debug_assert_eq!(feature_type.chartype_count(), 2);
        Self {
            feature_type,
            ct: [ct0, ct1, CharType::default()],
            ..Self::default()
        }
    }

    /// `CharType × 3` の特徴量キーを作る (`TypeTri*`)。
    pub fn type_3(
        feature_type: FeatureType,
        ct0: CharType,
        ct1: CharType,
        ct2: CharType,
    ) -> Self {
        debug_assert_eq!(feature_type.chartype_count(), 3);
        Self {
            feature_type,
            ct: [ct0, ct1, ct2],
            ..Self::default()
        }
    }

    /// `char32 × 1` の特徴量キーを作る (`CharSelf` 等)。
    pub fn char_1(feature_type: FeatureType, cp: u32) -> Self {
        debug_assert_eq!(feature_type.char32_count(), 1);
        Self {
            feature_type,
            cp: [cp, 0, 0],
            ..Self::default()
        }
    }

    /// `char32 × 2` の特徴量キー (Bigram 系) を作る。
    pub fn char_2(feature_type: FeatureType, cp0: u32, cp1: u32) -> Self {
        debug_assert_eq!(feature_type.char32_count(), 2);
        Self {
            feature_type,
            cp: [cp0, cp1, 0],
            ..Self::default()
        }
    }

    /// `char32 × 3` の特徴量キー (Trigram 系) を作る。
    pub fn char_3(feature_type: FeatureType, cp0: u32, cp1: u32, cp2: u32) -> Self {
        debug_assert_eq!(feature_type.char32_count(), 3);
        Self {
            feature_type,
            cp: [cp0, cp1, cp2],
            ..Self::default()
        }
    }

    /// `u8 × 1` の特徴量キー (run_len 系) を作る。
    pub fn u8_payload(feature_type: FeatureType, val: u8) -> Self {
        debug_assert!(feature_type.is_uint8_payload());
        Self {
            feature_type,
            u8val: val,
            ..Self::default()
        }
    }
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_type_repr_values() {
        // 主要な値が C++ 版と一致することを確認
        assert_eq!(FeatureType::Bias as u8, 0x00);
        assert_eq!(FeatureType::KanjiPosFirst as u8, 0x01);
        assert_eq!(FeatureType::TypeSelf as u8, 0x50);
        assert_eq!(FeatureType::TypeTransition as u8, 0x60);
        assert_eq!(FeatureType::TypeTriPrev2Prev1Self as u8, 0x70);
        assert_eq!(FeatureType::CharSelf as u8, 0x90);
        assert_eq!(FeatureType::BigramPrev1Self as u8, 0xA0);
        assert_eq!(FeatureType::TrigramPrev2Prev1Self as u8, 0xB0);
        assert_eq!(FeatureType::KanjiRunLen as u8, 0xC0);
        assert_eq!(FeatureType::PrevJapaneseNumericRunLen as u8, 0xC2);
    }

    #[test]
    fn from_u8_known_values() {
        assert_eq!(FeatureType::from_u8(0x00), Some(FeatureType::Bias));
        assert_eq!(FeatureType::from_u8(0x50), Some(FeatureType::TypeSelf));
        assert_eq!(FeatureType::from_u8(0xC2), Some(FeatureType::PrevJapaneseNumericRunLen));
    }

    #[test]
    fn from_u8_unknown_values() {
        assert_eq!(FeatureType::from_u8(0x02), None);
        assert_eq!(FeatureType::from_u8(0x57), None);
        assert_eq!(FeatureType::from_u8(0xFF), None);
        assert_eq!(FeatureType::from_u8(0xC3), None);
    }

    #[test]
    fn payload_counts() {
        // CharType 系
        assert_eq!(FeatureType::TypeSelf.chartype_count(), 1);
        assert_eq!(FeatureType::TypeTransition.chartype_count(), 2);
        assert_eq!(FeatureType::TypeTriPrev2Prev1Self.chartype_count(), 3);

        // char32 系
        assert_eq!(FeatureType::CharSelf.char32_count(), 1);
        assert_eq!(FeatureType::BigramPrev1Self.char32_count(), 2);
        assert_eq!(FeatureType::TrigramPrev2Prev1Self.char32_count(), 3);

        // u8 ペイロード系
        assert!(FeatureType::KanjiRunLen.is_uint8_payload());
        assert!(FeatureType::JapaneseNumericRunLen.is_uint8_payload());
        assert!(!FeatureType::Bias.is_uint8_payload());

        // ペイロードなし系
        assert_eq!(FeatureType::Bias.chartype_count(), 0);
        assert_eq!(FeatureType::Bias.char32_count(), 0);
        assert!(!FeatureType::Bias.is_uint8_payload());

        // 排他性: CharType 系は char32_count=0, char32 系は chartype_count=0
        assert_eq!(FeatureType::TypeSelf.char32_count(), 0);
        assert_eq!(FeatureType::CharSelf.chartype_count(), 0);
    }

    #[test]
    fn feature_key_default_is_bias_zero() {
        // C++ の `FeatureKey{}` 相当のゼロ初期化
        let k = FeatureKey::default();
        assert_eq!(k.feature_type, FeatureType::Bias);
        assert_eq!(k.u8val, 0);
        assert_eq!(k.ct, [CharType::Space, CharType::Space, CharType::Space]);
        assert_eq!(k.cp, [0, 0, 0]);
    }

    #[test]
    fn feature_key_constructors() {
        let k = FeatureKey::no_payload(FeatureType::Bias);
        assert_eq!(k.feature_type, FeatureType::Bias);

        let k = FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji);
        assert_eq!(k.ct[0], CharType::Kanji);
        assert_eq!(k.ct[1], CharType::Space); // ゼロ埋め

        let k = FeatureKey::char_1(FeatureType::CharSelf, 0x4E00);
        assert_eq!(k.cp[0], 0x4E00);

        let k = FeatureKey::char_2(FeatureType::BigramPrev1Self, 0x4E00, 0x4E01);
        assert_eq!(k.cp, [0x4E00, 0x4E01, 0]);

        let k = FeatureKey::u8_payload(FeatureType::KanjiRunLen, 3);
        assert_eq!(k.u8val, 3);
    }

    #[test]
    fn feature_key_equality() {
        let a = FeatureKey::char_1(FeatureType::CharSelf, 0x4E00);
        let b = FeatureKey::char_1(FeatureType::CharSelf, 0x4E00);
        let c = FeatureKey::char_1(FeatureType::CharSelf, 0x4E01);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn feature_key_ordering() {
        // FeatureType が違えば順序は FeatureType で決まる
        let a = FeatureKey::no_payload(FeatureType::Bias); // 0x00
        let b = FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji); // 0x50
        assert!(a < b);

        // 同じ FeatureType なら u8val → ct → cp の順で比較
        let a = FeatureKey::char_1(FeatureType::CharSelf, 0x4E00);
        let b = FeatureKey::char_1(FeatureType::CharSelf, 0x4E01);
        assert!(a < b);
    }
}
