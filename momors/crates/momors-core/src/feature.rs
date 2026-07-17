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
    /// 2文字複合ユニット (拗音・漢字語) の char_s 特徴量。
    /// C++ 版 `CHAR_SELF_COMPOUND_2`、Python 版 `FT.CHAR_SELF_COMPOUND_2` と対応。
    CharSelfCompound2 = 0xA6,

    // --- char32_t×3 (0b10'11'xxxx) ---
    TrigramPrev2Prev1Self = 0xB0,
    TrigramPrev1SelfNext1 = 0xB1,
    TrigramSelfNext1Next2 = 0xB2,
    TrigramPrev3Prev2Prev1 = 0xB3,
    TrigramNext1Next2Next3 = 0xB4,
    /// 3文字複合ユニットの char_s 特徴量。
    /// C++ 版 `CHAR_SELF_COMPOUND_3`、Python 版 `FT.CHAR_SELF_COMPOUND_3` と対応。
    CharSelfCompound3 = 0xB5,

    // --- u8×1 (0b11'00'xxxx) ---
    KanjiRunLen = 0xC0,
    JapaneseNumericRunLen = 0xC1,
    PrevJapaneseNumericRunLen = 0xC2,
    /// 人名辞書マッチフラグ（自分）。ペイロード: 1=B (スパン先頭), 2=I (継続)。
    /// Python 版 `name_s=B/I`、exporter の `FT.NAME_FLAG_SELF` と対応。
    NameFlagSelf = 0xC3,
    /// 人名辞書マッチフラグ（前1文字）。Python 版 `name_p1=B/I`。
    NameFlagPrev1 = 0xC4,
    /// 人名辞書マッチフラグ（後1文字）。Python 版 `name_n1=B/I`。
    NameFlagNext1 = 0xC5,
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
            0xA6 => CharSelfCompound2,

            0xB0 => TrigramPrev2Prev1Self,
            0xB1 => TrigramPrev1SelfNext1,
            0xB2 => TrigramSelfNext1Next2,
            0xB3 => TrigramPrev3Prev2Prev1,
            0xB4 => TrigramNext1Next2Next3,
            0xB5 => CharSelfCompound3,

            0xC0 => KanjiRunLen,
            0xC1 => JapaneseNumericRunLen,
            0xC2 => PrevJapaneseNumericRunLen,
            0xC3 => NameFlagSelf,
            0xC4 => NameFlagPrev1,
            0xC5 => NameFlagNext1,

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
    pub fn type_3(feature_type: FeatureType, ct0: CharType, ct1: CharType, ct2: CharType) -> Self {
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
// 人間可読な特徴量名 ⇄ FeatureKey
// ============================================================
//
// 診断（explain / label スキャナ）用に、Python 版 `momo_py/features.py` の
// 特徴量名（`char_s=金` / `type_trans_p1_s=KANJI->KANJI` / `kanji_run=5+` 等）と
// FeatureKey を相互変換する。ペイロード符号化（run_len の "5+"→5、name の B/I→1/2、
// CharSelf の複合ユニット 1〜3 文字の振り分け）は `momo_py/exporter.py` の
// `parse_feature_key` と厳密に一致させること（vocab_find が一致しなくなる）。

/// run_len ペイロード（u8）を表示キーへ。5 以上は "5+"（Python の run_key と同じ）。
fn run_key_to_string(v: u8) -> String {
    if v >= 5 {
        "5+".to_string()
    } else {
        v.to_string()
    }
}

/// run_key 文字列 → u8（"1".."4"→1..4, "5+"→5）。
fn parse_run_key(s: &str) -> Option<u8> {
    match s {
        "1" => Some(1),
        "2" => Some(2),
        "3" => Some(3),
        "4" => Some(4),
        "5+" => Some(5),
        _ => None,
    }
}

/// 人名フラグ u8 → "B"/"I"。
fn name_flag_to_str(v: u8) -> &'static str {
    if v == 2 {
        "I"
    } else {
        "B"
    }
}

fn cp_char(cp: u32) -> char {
    char::from_u32(cp).unwrap_or('\u{FFFD}')
}

impl std::fmt::Display for FeatureKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use FeatureType::*;
        let ct = self.ct;
        let cp = self.cp;
        match self.feature_type {
            Bias => write!(f, "bias"),
            KanjiPosFirst => write!(f, "kanji_pos_first"),

            TypeSelf => write!(f, "type_s={}", ct[0].as_str()),
            TypePrev1 => write!(f, "type_p1={}", ct[0].as_str()),
            TypePrev2 => write!(f, "type_p2={}", ct[0].as_str()),
            TypePrev3 => write!(f, "type_p3={}", ct[0].as_str()),
            TypeNext1 => write!(f, "type_n1={}", ct[0].as_str()),
            TypeNext2 => write!(f, "type_n2={}", ct[0].as_str()),
            TypeNext3 => write!(f, "type_n3={}", ct[0].as_str()),

            TypeTransition => {
                write!(f, "type_trans_p1_s={}->{}", ct[0].as_str(), ct[1].as_str())
            }

            TypeTriPrev3Prev2Prev1 => write!(
                f,
                "type_tri_p3_p2_p1={}-{}-{}",
                ct[0].as_str(),
                ct[1].as_str(),
                ct[2].as_str()
            ),
            TypeTriPrev2Prev1Self => write!(
                f,
                "type_tri_p2_p1_s={}-{}-{}",
                ct[0].as_str(),
                ct[1].as_str(),
                ct[2].as_str()
            ),
            TypeTriPrev1SelfNext1 => write!(
                f,
                "type_tri_p1_s_n1={}-{}-{}",
                ct[0].as_str(),
                ct[1].as_str(),
                ct[2].as_str()
            ),
            TypeTriSelfNext1Next2 => write!(
                f,
                "type_tri_s_n1_n2={}-{}-{}",
                ct[0].as_str(),
                ct[1].as_str(),
                ct[2].as_str()
            ),
            TypeTriNext1Next2Next3 => write!(
                f,
                "type_tri_n1_n2_n3={}-{}-{}",
                ct[0].as_str(),
                ct[1].as_str(),
                ct[2].as_str()
            ),

            CharSelf => write!(f, "char_s={}", cp_char(cp[0])),
            CharPrev1 => write!(f, "char_p1={}", cp_char(cp[0])),
            CharPrev2 => write!(f, "char_p2={}", cp_char(cp[0])),
            CharPrev3 => write!(f, "char_p3={}", cp_char(cp[0])),
            CharNext1 => write!(f, "char_n1={}", cp_char(cp[0])),
            CharNext2 => write!(f, "char_n2={}", cp_char(cp[0])),
            CharNext3 => write!(f, "char_n3={}", cp_char(cp[0])),

            BigramPrev1Self => write!(f, "bi_p1_s={}{}", cp_char(cp[0]), cp_char(cp[1])),
            BigramPrev2Prev1 => write!(f, "bi_p2_p1={}{}", cp_char(cp[0]), cp_char(cp[1])),
            BigramPrev3Prev2 => write!(f, "bi_p3_p2={}{}", cp_char(cp[0]), cp_char(cp[1])),
            BigramSelfNext1 => write!(f, "bi_s_n1={}{}", cp_char(cp[0]), cp_char(cp[1])),
            BigramNext1Next2 => write!(f, "bi_n1_n2={}{}", cp_char(cp[0]), cp_char(cp[1])),
            BigramNext2Next3 => write!(f, "bi_n2_n3={}{}", cp_char(cp[0]), cp_char(cp[1])),
            CharSelfCompound2 => write!(f, "char_s={}{}", cp_char(cp[0]), cp_char(cp[1])),

            TrigramPrev2Prev1Self => write!(
                f,
                "tri_p2_p1_s={}{}{}",
                cp_char(cp[0]),
                cp_char(cp[1]),
                cp_char(cp[2])
            ),
            TrigramPrev1SelfNext1 => write!(
                f,
                "tri_p1_s_n1={}{}{}",
                cp_char(cp[0]),
                cp_char(cp[1]),
                cp_char(cp[2])
            ),
            TrigramSelfNext1Next2 => write!(
                f,
                "tri_s_n1_n2={}{}{}",
                cp_char(cp[0]),
                cp_char(cp[1]),
                cp_char(cp[2])
            ),
            TrigramPrev3Prev2Prev1 => write!(
                f,
                "tri_p3_p2_p1={}{}{}",
                cp_char(cp[0]),
                cp_char(cp[1]),
                cp_char(cp[2])
            ),
            TrigramNext1Next2Next3 => write!(
                f,
                "tri_n1_n2_n3={}{}{}",
                cp_char(cp[0]),
                cp_char(cp[1]),
                cp_char(cp[2])
            ),
            CharSelfCompound3 => write!(
                f,
                "char_s={}{}{}",
                cp_char(cp[0]),
                cp_char(cp[1]),
                cp_char(cp[2])
            ),

            KanjiRunLen => write!(f, "kanji_run={}", run_key_to_string(self.u8val)),
            JapaneseNumericRunLen => write!(f, "jnum_run={}", run_key_to_string(self.u8val)),
            PrevJapaneseNumericRunLen => {
                write!(f, "jnum_run_p1={}", run_key_to_string(self.u8val))
            }
            NameFlagSelf => write!(f, "name_s={}", name_flag_to_str(self.u8val)),
            NameFlagPrev1 => write!(f, "name_p1={}", name_flag_to_str(self.u8val)),
            NameFlagNext1 => write!(f, "name_n1={}", name_flag_to_str(self.u8val)),
        }
    }
}

impl FeatureKey {
    /// 人間可読な特徴量名（[`Display`](std::fmt::Display) と同じ文字列）を
    /// FeatureKey へ解析する。未知・不正な名前は `None`。
    ///
    /// `momo_py/exporter.py` の `parse_feature_key` と符号化を一致させる。
    pub fn parse_name(s: &str) -> Option<Self> {
        use FeatureType::*;

        // --- ペイロードなし系 ---
        if s == "bias" {
            return Some(FeatureKey::no_payload(Bias));
        }
        if s == "kanji_pos_first" {
            return Some(FeatureKey::no_payload(KanjiPosFirst));
        }

        let (prefix, value) = s.split_once('=')?;

        // 型ペイロード（CharType）を n 個に分解して解析する。
        let parse_types = |sep: &str, n: usize| -> Option<Vec<CharType>> {
            let parts: Vec<&str> = value.split(sep).collect();
            if parts.len() != n {
                return None;
            }
            parts.iter().map(|p| CharType::from_name(p)).collect()
        };
        // 文字ペイロード（char32）をちょうど n 文字として解析する。
        let chars_exact = |n: usize| -> Option<Vec<u32>> {
            let cs: Vec<char> = value.chars().collect();
            if cs.len() != n {
                return None;
            }
            Some(cs.iter().map(|&c| c as u32).collect())
        };

        match prefix {
            // --- run 系（u8 ペイロード）---
            "kanji_run" => Some(FeatureKey::u8_payload(KanjiRunLen, parse_run_key(value)?)),
            "jnum_run" => Some(FeatureKey::u8_payload(
                JapaneseNumericRunLen,
                parse_run_key(value)?,
            )),
            "jnum_run_p1" => Some(FeatureKey::u8_payload(
                PrevJapaneseNumericRunLen,
                parse_run_key(value)?,
            )),

            // --- 人名フラグ系（u8 ペイロード: B=1, I=2）---
            "name_s" => Some(FeatureKey::u8_payload(
                NameFlagSelf,
                parse_name_flag(value)?,
            )),
            "name_p1" => Some(FeatureKey::u8_payload(
                NameFlagPrev1,
                parse_name_flag(value)?,
            )),
            "name_n1" => Some(FeatureKey::u8_payload(
                NameFlagNext1,
                parse_name_flag(value)?,
            )),

            // --- type_tri（CharType×3）---
            "type_tri_p3_p2_p1" => {
                let c = parse_types("-", 3)?;
                Some(FeatureKey::type_3(TypeTriPrev3Prev2Prev1, c[0], c[1], c[2]))
            }
            "type_tri_p2_p1_s" => {
                let c = parse_types("-", 3)?;
                Some(FeatureKey::type_3(TypeTriPrev2Prev1Self, c[0], c[1], c[2]))
            }
            "type_tri_p1_s_n1" => {
                let c = parse_types("-", 3)?;
                Some(FeatureKey::type_3(TypeTriPrev1SelfNext1, c[0], c[1], c[2]))
            }
            "type_tri_s_n1_n2" => {
                let c = parse_types("-", 3)?;
                Some(FeatureKey::type_3(TypeTriSelfNext1Next2, c[0], c[1], c[2]))
            }
            "type_tri_n1_n2_n3" => {
                let c = parse_types("-", 3)?;
                Some(FeatureKey::type_3(TypeTriNext1Next2Next3, c[0], c[1], c[2]))
            }

            // --- type_trans（CharType×2）---
            "type_trans_p1_s" => {
                let c = parse_types("->", 2)?;
                Some(FeatureKey::type_2(TypeTransition, c[0], c[1]))
            }

            // --- type 系（CharType×1）---
            "type_s" => Some(FeatureKey::type_1(TypeSelf, CharType::from_name(value)?)),
            "type_p1" => Some(FeatureKey::type_1(TypePrev1, CharType::from_name(value)?)),
            "type_p2" => Some(FeatureKey::type_1(TypePrev2, CharType::from_name(value)?)),
            "type_p3" => Some(FeatureKey::type_1(TypePrev3, CharType::from_name(value)?)),
            "type_n1" => Some(FeatureKey::type_1(TypeNext1, CharType::from_name(value)?)),
            "type_n2" => Some(FeatureKey::type_1(TypeNext2, CharType::from_name(value)?)),
            "type_n3" => Some(FeatureKey::type_1(TypeNext3, CharType::from_name(value)?)),

            // --- trigram（char32×3）---
            "tri_p3_p2_p1" => {
                let c = chars_exact(3)?;
                Some(FeatureKey::char_3(TrigramPrev3Prev2Prev1, c[0], c[1], c[2]))
            }
            "tri_p2_p1_s" => {
                let c = chars_exact(3)?;
                Some(FeatureKey::char_3(TrigramPrev2Prev1Self, c[0], c[1], c[2]))
            }
            "tri_p1_s_n1" => {
                let c = chars_exact(3)?;
                Some(FeatureKey::char_3(TrigramPrev1SelfNext1, c[0], c[1], c[2]))
            }
            "tri_s_n1_n2" => {
                let c = chars_exact(3)?;
                Some(FeatureKey::char_3(TrigramSelfNext1Next2, c[0], c[1], c[2]))
            }
            "tri_n1_n2_n3" => {
                let c = chars_exact(3)?;
                Some(FeatureKey::char_3(TrigramNext1Next2Next3, c[0], c[1], c[2]))
            }

            // --- bigram（char32×2）---
            "bi_p1_s" => {
                let c = chars_exact(2)?;
                Some(FeatureKey::char_2(BigramPrev1Self, c[0], c[1]))
            }
            "bi_p2_p1" => {
                let c = chars_exact(2)?;
                Some(FeatureKey::char_2(BigramPrev2Prev1, c[0], c[1]))
            }
            "bi_p3_p2" => {
                let c = chars_exact(2)?;
                Some(FeatureKey::char_2(BigramPrev3Prev2, c[0], c[1]))
            }
            "bi_s_n1" => {
                let c = chars_exact(2)?;
                Some(FeatureKey::char_2(BigramSelfNext1, c[0], c[1]))
            }
            "bi_n1_n2" => {
                let c = chars_exact(2)?;
                Some(FeatureKey::char_2(BigramNext1Next2, c[0], c[1]))
            }
            "bi_n2_n3" => {
                let c = chars_exact(2)?;
                Some(FeatureKey::char_2(BigramNext2Next3, c[0], c[1]))
            }

            // --- char_s（複合ユニット対応: 1〜3 文字）---
            "char_s" => {
                let cs: Vec<char> = value.chars().collect();
                match cs.len() {
                    1 => Some(FeatureKey::char_1(CharSelf, cs[0] as u32)),
                    2 => Some(FeatureKey::char_2(
                        CharSelfCompound2,
                        cs[0] as u32,
                        cs[1] as u32,
                    )),
                    3 => Some(FeatureKey::char_3(
                        CharSelfCompound3,
                        cs[0] as u32,
                        cs[1] as u32,
                        cs[2] as u32,
                    )),
                    _ => None,
                }
            }

            // --- char 系（char32×1）---
            "char_p1" => Some(FeatureKey::char_1(CharPrev1, chars_exact(1)?[0])),
            "char_p2" => Some(FeatureKey::char_1(CharPrev2, chars_exact(1)?[0])),
            "char_p3" => Some(FeatureKey::char_1(CharPrev3, chars_exact(1)?[0])),
            "char_n1" => Some(FeatureKey::char_1(CharNext1, chars_exact(1)?[0])),
            "char_n2" => Some(FeatureKey::char_1(CharNext2, chars_exact(1)?[0])),
            "char_n3" => Some(FeatureKey::char_1(CharNext3, chars_exact(1)?[0])),

            _ => None,
        }
    }
}

/// 人名フラグ文字列 → u8（"B"=1, "I"=2）。
fn parse_name_flag(s: &str) -> Option<u8> {
    match s {
        "B" => Some(1),
        "I" => Some(2),
        _ => None,
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
        assert_eq!(
            FeatureType::from_u8(0xC2),
            Some(FeatureType::PrevJapaneseNumericRunLen)
        );
    }

    #[test]
    fn from_u8_compound_values() {
        assert_eq!(
            FeatureType::from_u8(0xA6),
            Some(FeatureType::CharSelfCompound2)
        );
        assert_eq!(
            FeatureType::from_u8(0xB5),
            Some(FeatureType::CharSelfCompound3)
        );
    }

    #[test]
    fn from_u8_unknown_values() {
        assert_eq!(FeatureType::from_u8(0x02), None);
        assert_eq!(FeatureType::from_u8(0x57), None);
        assert_eq!(FeatureType::from_u8(0xFF), None);
        assert_eq!(FeatureType::from_u8(0xC6), None);
        assert_eq!(FeatureType::from_u8(0xB6), None);
    }

    #[test]
    fn from_u8_name_flag_values() {
        assert_eq!(FeatureType::from_u8(0xC3), Some(FeatureType::NameFlagSelf));
        assert_eq!(FeatureType::from_u8(0xC4), Some(FeatureType::NameFlagPrev1));
        assert_eq!(FeatureType::from_u8(0xC5), Some(FeatureType::NameFlagNext1));
        assert!(FeatureType::NameFlagSelf.is_uint8_payload());
        assert!(FeatureType::NameFlagPrev1.is_uint8_payload());
        assert!(FeatureType::NameFlagNext1.is_uint8_payload());
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
        assert_eq!(FeatureType::CharSelfCompound2.char32_count(), 2);
        assert_eq!(FeatureType::TrigramPrev2Prev1Self.char32_count(), 3);
        assert_eq!(FeatureType::CharSelfCompound3.char32_count(), 3);

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

    // ------------------------------------------------------------
    // 特徴量名 ⇄ FeatureKey
    // ------------------------------------------------------------

    #[test]
    fn display_matches_python_names() {
        let kin = '金' as u32;
        assert_eq!(
            FeatureKey::no_payload(FeatureType::Bias).to_string(),
            "bias"
        );
        assert_eq!(
            FeatureKey::no_payload(FeatureType::KanjiPosFirst).to_string(),
            "kanji_pos_first"
        );
        assert_eq!(
            FeatureKey::char_1(FeatureType::CharSelf, kin).to_string(),
            "char_s=金"
        );
        assert_eq!(
            FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji).to_string(),
            "type_s=KANJI"
        );
        assert_eq!(
            FeatureKey::type_2(
                FeatureType::TypeTransition,
                CharType::Kanji,
                CharType::Hiragana
            )
            .to_string(),
            "type_trans_p1_s=KANJI->HIRAGANA"
        );
        assert_eq!(
            FeatureKey::type_3(
                FeatureType::TypeTriSelfNext1Next2,
                CharType::Kanji,
                CharType::Kanji,
                CharType::Kanji
            )
            .to_string(),
            "type_tri_s_n1_n2=KANJI-KANJI-KANJI"
        );
        // run_len: 5 以上は "5+"
        assert_eq!(
            FeatureKey::u8_payload(FeatureType::KanjiRunLen, 2).to_string(),
            "kanji_run=2"
        );
        assert_eq!(
            FeatureKey::u8_payload(FeatureType::KanjiRunLen, 5).to_string(),
            "kanji_run=5+"
        );
        // 人名フラグ
        assert_eq!(
            FeatureKey::u8_payload(FeatureType::NameFlagSelf, 1).to_string(),
            "name_s=B"
        );
        assert_eq!(
            FeatureKey::u8_payload(FeatureType::NameFlagNext1, 2).to_string(),
            "name_n1=I"
        );
    }

    #[test]
    fn char_self_compound_display() {
        // 複合ユニット（拗音）は char_s に全文字を出す
        let kya: Vec<u32> = "きゃ".chars().map(|c| c as u32).collect();
        assert_eq!(
            FeatureKey::char_2(FeatureType::CharSelfCompound2, kya[0], kya[1]).to_string(),
            "char_s=きゃ"
        );
    }

    #[test]
    fn parse_roundtrip_all_families() {
        // Display した文字列を parse し直して元の FeatureKey に戻ることを確認する。
        let samples = vec![
            FeatureKey::no_payload(FeatureType::Bias),
            FeatureKey::no_payload(FeatureType::KanjiPosFirst),
            FeatureKey::type_1(FeatureType::TypeSelf, CharType::Kanji),
            FeatureKey::type_1(FeatureType::TypeNext3, CharType::SymbolClose),
            FeatureKey::type_2(
                FeatureType::TypeTransition,
                CharType::Kanji,
                CharType::Hiragana,
            ),
            FeatureKey::type_3(
                FeatureType::TypeTriPrev2Prev1Self,
                CharType::Hiragana,
                CharType::Kanji,
                CharType::JapaneseNumeric,
            ),
            FeatureKey::char_1(FeatureType::CharSelf, '金' as u32),
            FeatureKey::char_1(FeatureType::CharNext1, 'の' as u32),
            FeatureKey::char_2(FeatureType::CharSelfCompound2, 'き' as u32, 'ゃ' as u32),
            FeatureKey::char_3(
                FeatureType::CharSelfCompound3,
                'ぐ' as u32,
                'ぁ' as u32,
                'い' as u32,
            ),
            FeatureKey::char_2(FeatureType::BigramSelfNext1, '本' as u32, 'を' as u32),
            FeatureKey::char_3(
                FeatureType::TrigramPrev1SelfNext1,
                'を' as u32,
                '読' as u32,
                'む' as u32,
            ),
            FeatureKey::u8_payload(FeatureType::KanjiRunLen, 1),
            FeatureKey::u8_payload(FeatureType::KanjiRunLen, 5),
            FeatureKey::u8_payload(FeatureType::JapaneseNumericRunLen, 3),
            FeatureKey::u8_payload(FeatureType::PrevJapaneseNumericRunLen, 4),
            FeatureKey::u8_payload(FeatureType::NameFlagSelf, 1),
            FeatureKey::u8_payload(FeatureType::NameFlagPrev1, 2),
        ];
        for key in samples {
            let name = key.to_string();
            let parsed = FeatureKey::parse_name(&name)
                .unwrap_or_else(|| panic!("parse failed for {name:?}"));
            assert_eq!(parsed, key, "roundtrip mismatch for {name:?}");
        }
    }

    #[test]
    fn parse_rejects_unknown() {
        assert_eq!(FeatureKey::parse_name("nope=1"), None);
        assert_eq!(FeatureKey::parse_name("type_s=NOPE"), None);
        assert_eq!(FeatureKey::parse_name("kanji_run=9"), None);
        assert_eq!(FeatureKey::parse_name("name_s=X"), None);
        // char_p1 はちょうど1文字
        assert_eq!(FeatureKey::parse_name("char_p1=あい"), None);
    }
}
