//! 統合語彙テーブル: 特徴量キー → 読みモデルの `feature_id` と GBDT のカテゴリカル情報。
//!
//! ## なぜ特徴量タイプごとに分けるか
//!
//! 以前は [`VocabEntry`](crate::model::VocabEntry) 32B（`FeatureKey` 20B + payload 12B）を
//! 1 本のソート済み配列に並べ、`FeatureKey` の `Ord` で二分探索していた。
//! 語彙は w4 で 31.9 万件・9.8MiB、w7 で 108 万件・33MiB あり、常駐メモリの最大項だった。
//!
//! ここでは `feature_type` ごとにセクションを分け、各セクションは**キーを u64 に詰めた
//! 昇順配列**を持つ。効果は 3 つ:
//!
//! 1. **メモリが 1/4**。エントリが持つのは詰めたキー 8B だけになる。`feature_id` と
//!    `cat_code` はセクション内の添字からの足し算、`cat_column` はセクションが 1 個
//!    持てば済む（version 0x09 で exporter がキー順に採番し直したため）。32B/件 → 8B/件。
//! 2. **探索深さが窓サイズに依らなくなる**。最大セクション（`TrigramPrev1SelfNext1`）は
//!    w4/w5/w7 のいずれでも 150,013 件で一定。全体を 1 本で引くと
//!    log2 が 18.3 / 19.4 / 20.0 と増えていくが、セクション内なら常に 17.2。
//! 3. **キー比較が u64 1 個の比較**になる（20B 構造体の辞書順比較ではなく）。
//!
//! ## キーの詰め方
//!
//! [`FeatureType`] のビットフィールド上、ペイロード種別（`CharType×N` / `char32×M` /
//! `u8×1` / なし）は排他なので、どのタイプでも 1 個の u64 で表せる。
//!
//! ```text
//! char32×M   : cp[0] を上位に 21bit ずつ（M ≤ 3 なので 63bit）
//! chartype×N : ct[0] を上位に 8bit ずつ（N ≤ 3）
//! u8×1       : u8val
//! ペイロードなし: 0
//! ```
//!
//! 上位から詰めるので、**詰めた u64 の昇順は `FeatureKey` の `Ord` の（同一タイプ内での）
//! 順序と一致する**。`.mbm` / `.mbmf` はどちらも version 0x09 でセクションごとに
//! キー昇順で格納される契約（exporter の `_build_vocab_bytes` が両方で共有）なので、
//! ロード時は一括読みするだけでよく、並べ替えは要らない。
//! 契約が破れたファイルは [`VocabBuilder`] が `CorruptModel` で弾く。
//!
//! ## 完全一致検索しかしない
//!
//! 語彙は `resolve` / `feature_id` の完全一致でしか引かれない（範囲検索・後続検索は
//! 1 箇所もない）。並び順は「同じキーを引ければよい」だけの内部都合なので、
//! 将来 B 木やハッシュへ替える余地がある。

use crate::boundary::{MAX_BOUNDARY_CAT_COLUMNS, VocabRef};
#[cfg(any(test, feature = "diagnostics"))]
use crate::char_type::CharType;
use crate::feature::{FeatureKey, FeatureType};
use crate::model::NO_CAT_COLUMN;
#[cfg(any(test, feature = "diagnostics"))]
use crate::model::VocabEntry;
use crate::{Error, Result};

/// コードポイント 1 個に割り当てるビット幅。Unicode の上限 0x10FFFF が収まる。
const CP_BITS: u32 = 21;
/// `slot_of_type` の「このタイプのセクションは無い」を表す番兵。
const NO_SLOT: u8 = u8::MAX;

/// 1 つの `feature_type` ぶんの語彙。
///
/// 持つのは詰めたキーの昇順配列だけ。`feature_id` と `cat_code` は
/// **セクション内の添字からの足し算**で求まる（version 0x09 で exporter が
/// キー順に採番し直したため）。
#[derive(Debug, Default)]
pub(crate) struct TypeSection {
    /// このセクションの特徴量タイプ。[`Vocab::iter`] がキーの復元に使う。
    feature_type: FeatureType,
    /// 詰めたキー。昇順・重複なし。
    keys: Vec<u64>,
    /// このタイプの GBDT カテゴリカル列。`NO_CAT_COLUMN` なら列なし。
    ///
    /// 列がタイプごとに 1 個に決まるのは偶然ではない。exporter 側
    /// (`momo_py/categorical.py` の `fit_categorical`) が「特徴量名ごとに 1 列」を作り、
    /// 特徴量名は `FeatureType` と 1 対 1 に対応するため。
    cat_column: u32,
    /// 先頭エントリのカテゴリカルコード。`cat_code = cat_code_base + 添字`。
    cat_code_base: u32,
    /// 先頭エントリの `feature_id`。`feature_id = feature_id_base + 添字`。
    feature_id_base: u32,
}

impl TypeSection {
    /// ヒープ占有バイト数（診断用）。
    #[cfg(feature = "diagnostics")]
    fn heap_bytes(&self) -> usize {
        self.keys.capacity() * 8
    }
}

/// 統合語彙テーブル。
#[derive(Debug)]
pub(crate) struct Vocab {
    /// `feature_type as u8` → `sections` の添字。[`NO_SLOT`] は「そのタイプは無い」。
    slot_of_type: [u8; 256],
    sections: Vec<TypeSection>,
}

impl Default for Vocab {
    fn default() -> Self {
        Self {
            slot_of_type: [NO_SLOT; 256],
            sections: Vec::new(),
        }
    }
}

impl Vocab {
    /// キーの属するセクションを引く。
    #[inline]
    fn section(&self, ft: FeatureType) -> Option<&TypeSection> {
        let slot = self.slot_of_type[ft as u8 as usize];
        if slot == NO_SLOT {
            return None;
        }
        // 不変条件: `VocabBuilder` が slot < sections.len() と、その枠のタイプ一致を保証する。
        let sec = self.sections.get(slot as usize)?;
        debug_assert_eq!(
            sec.feature_type, ft,
            "slot_of_type とセクションの feature_type が不整合です"
        );
        Some(sec)
    }

    /// セクション内でのキーの位置。
    #[inline]
    fn find(&self, key: &FeatureKey) -> Option<(&TypeSection, usize)> {
        let sec = self.section(key.feature_type)?;
        let packed = pack_key(key);
        let idx = sec.keys.binary_search(&packed).ok()?;
        Some((sec, idx))
    }

    /// 読みモデルの `feature_id` を引く。見つからなければ `None`。
    #[inline]
    pub(crate) fn feature_id(&self, key: &FeatureKey) -> Option<u32> {
        let (sec, i) = self.find(key)?;
        Some(sec.feature_id_base + i as u32)
    }

    /// 読みモデルの `feature_id` と GBDT のカテゴリカル `(column, code)` を
    /// 1 回の探索で返す。線形境界は `feature_id`、GBDT は `cat` を使う。
    #[inline]
    pub(crate) fn resolve(&self, key: &FeatureKey) -> Option<VocabRef> {
        let (sec, i) = self.find(key)?;
        let cat = (sec.cat_column != NO_CAT_COLUMN)
            .then(|| (sec.cat_column, sec.cat_code_base + i as u32));
        Some(VocabRef {
            feature_id: sec.feature_id_base + i as u32,
            cat,
        })
    }

    /// 全エントリ数（診断用）。
    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn len(&self) -> usize {
        self.sections.iter().map(|s| s.keys.len()).sum()
    }

    /// ヒープ占有バイト数（診断用）。
    #[cfg(feature = "diagnostics")]
    pub(crate) fn heap_bytes(&self) -> usize {
        self.sections
            .iter()
            .map(TypeSection::heap_bytes)
            .sum::<usize>()
            + self.sections.capacity() * std::mem::size_of::<TypeSection>()
    }

    /// 全エントリを [`VocabEntry`] として走査する。
    ///
    /// 診断（`diag.rs` の列コード空間・`mmap_experiment.rs`）専用。並びは
    /// タイプ順 → キー順（version 0x09 では `feature_id` 順と一致する）。
    /// キーを詰めた u64 から組み立て直すので、推論のホットパスでは使わない。
    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn iter(&self) -> impl Iterator<Item = VocabEntry> + '_ {
        self.sections.iter().flat_map(move |sec| {
            let ft = sec.feature_type;
            (0..sec.keys.len()).map(move |i| VocabEntry {
                key: unpack_key(ft, sec.keys[i]),
                feature_id: sec.feature_id_base + i as u32,
                cat_column: sec.cat_column,
                cat_code: if sec.cat_column == NO_CAT_COLUMN {
                    0
                } else {
                    sec.cat_code_base + i as u32
                },
            })
        })
    }
}

// ============================================================
// キーの詰め方・戻し方
// ============================================================

/// [`FeatureKey`] のペイロードを u64 に詰める。**同一タイプ内で単射かつ順序保存**。
///
/// 呼び出し側の前提: コードポイントが `2^21` 未満であること。推論時のキーは入力テキスト由来の
/// 実在するコードポイント（≤ 0x10FFFF）なので常に成り立つ。モデルファイル由来のキーは
/// exporter 側 (`_pack_vocab_key`) が範囲を検証する。
#[inline]
pub(crate) fn pack_key(key: &FeatureKey) -> u64 {
    let ft = key.feature_type;

    let m = ft.char32_count();
    if m > 0 {
        let mut v = 0u64;
        for i in 0..m {
            debug_assert!(
                key.cp[i] < (1 << CP_BITS),
                "コードポイントが 21bit を超えています"
            );
            v = (v << CP_BITS) | u64::from(key.cp[i]);
        }
        return v;
    }

    let n = ft.chartype_count();
    if n > 0 {
        let mut v = 0u64;
        for i in 0..n {
            v = (v << 8) | u64::from(key.ct[i] as u8);
        }
        return v;
    }

    if ft.is_uint8_payload() {
        return u64::from(key.u8val);
    }

    0
}

/// [`pack_key`] の逆。診断用の走査 ([`Vocab::iter`]) でだけ使う。
#[cfg(any(test, feature = "diagnostics"))]
fn unpack_key(ft: FeatureType, packed: u64) -> FeatureKey {
    let mut key = FeatureKey {
        feature_type: ft,
        ..FeatureKey::default()
    };

    let m = ft.char32_count();
    if m > 0 {
        for i in (0..m).rev() {
            key.cp[i] = ((packed >> (CP_BITS * (m - 1 - i) as u32)) & ((1 << CP_BITS) - 1)) as u32;
        }
        return key;
    }

    let n = ft.chartype_count();
    if n > 0 {
        for i in 0..n {
            let byte = ((packed >> (8 * (n - 1 - i) as u32)) & 0xFF) as u8;
            // 語彙に載っているのは読み込み時に検証済みの値だけ。
            key.ct[i] = CharType::from_u8(byte).unwrap_or_default();
        }
        return key;
    }

    if ft.is_uint8_payload() {
        key.u8val = packed as u8;
    }

    key
}

// ============================================================
// 組み立て
// ============================================================

/// [`Vocab`] をセクション単位で組み立てる。ローダー専用。
///
/// version 0x09 の語彙はセクションヘッダとキー配列だけなので、ビルダーも
/// 「セクションを開いて、キーを順に積む」だけでよい。検証はここに集約する:
///
/// - セクションは `feature_type` の狭義昇順（= キーの全体順序と一致する）
/// - 同じタイプのセクションは 1 つだけ
/// - セクション内のキーは狭義昇順（重複なし）
/// - 宣言した件数ぶんちょうど積まれたこと
///
/// 契約が破れたファイルを黙って通すと、`binary_search` が存在するキーを見失ったり
/// 別のエントリを返したりして静かに誤動作するため、すべて `CorruptModel` にする。
pub(crate) struct VocabBuilder {
    slot_of_type: [u8; 256],
    sections: Vec<TypeSection>,
    /// 開いているセクションが受け取るはずのキー数。
    expected: Vec<u32>,
    /// 直前に開いたセクションの `feature_type`（昇順の検証用）。
    prev_type: Option<u8>,
    /// 次のセクションに割り当てる `feature_id_base`。
    next_feature_id: u32,
}

impl VocabBuilder {
    pub(crate) fn new() -> Self {
        Self {
            slot_of_type: [NO_SLOT; 256],
            sections: Vec::new(),
            expected: Vec::new(),
            prev_type: None,
            next_feature_id: 0,
        }
    }

    /// セクションを開く。以降の [`Self::push_key`] はこのセクションに積まれる。
    pub(crate) fn begin_section(
        &mut self,
        feature_type: FeatureType,
        count: u32,
        cat_column: u32,
        cat_code_base: u32,
    ) -> Result<()> {
        let ft_u8 = feature_type as u8;

        if let Some(prev) = self.prev_type
            && ft_u8 <= prev
        {
            return Err(Error::CorruptModel {
                reason: format!(
                    "統合語彙のセクションが feature_type 昇順に並んでいません                      (0x{prev:02X} の後に 0x{ft_u8:02X})"
                ),
            });
        }
        self.prev_type = Some(ft_u8);

        if self.sections.len() >= NO_SLOT as usize {
            return Err(Error::CorruptModel {
                reason: format!("統合語彙のセクション数が上限 {NO_SLOT} を超えています"),
            });
        }

        if cat_column != NO_CAT_COLUMN && cat_column as usize >= MAX_BOUNDARY_CAT_COLUMNS {
            return Err(Error::CorruptModel {
                reason: format!(
                    "統合語彙のカテゴリカル列番号 {cat_column} が上限 {MAX_BOUNDARY_CAT_COLUMNS} 以上です"
                ),
            });
        }

        // feature_id / cat_code はセクション内の添字を足して作るので、末尾が
        // u32 を溢れないことを先に確かめる。
        if self.next_feature_id.checked_add(count).is_none()
            || (cat_column != NO_CAT_COLUMN && cat_code_base.checked_add(count).is_none())
        {
            return Err(Error::CorruptModel {
                reason: format!(
                    "統合語彙セクション 0x{ft_u8:02X} の件数 {count} で feature_id か                      cat_code が u32 を溢れます"
                ),
            });
        }

        self.slot_of_type[ft_u8 as usize] = self.sections.len() as u8;
        self.sections.push(TypeSection {
            feature_type,
            keys: Vec::with_capacity(count as usize),
            cat_column,
            cat_code_base,
            feature_id_base: self.next_feature_id,
        });
        self.expected.push(count);
        self.next_feature_id += count;
        Ok(())
    }

    /// 開いているセクションに詰めたキーを 1 個積む。狭義昇順であること。
    pub(crate) fn push_key(&mut self, packed: u64) -> Result<()> {
        let sec = self.sections.last_mut().ok_or_else(|| Error::CorruptModel {
            reason: "統合語彙のセクションを開く前にキーが現れました".to_string(),
        })?;
        if let Some(&last) = sec.keys.last()
            && packed <= last
        {
            return Err(Error::CorruptModel {
                reason: format!(
                    "統合語彙セクション 0x{:02X} のキーが昇順でないか重複しています                      (0x{last:016X} の後に 0x{packed:016X})",
                    sec.feature_type as u8
                ),
            });
        }
        sec.keys.push(packed);
        Ok(())
    }

    /// 積んだセクションから [`Vocab`] を作る。件数の一致を最後に確認する。
    pub(crate) fn finish(self, n_features: u32) -> Result<Vocab> {
        for (sec, &want) in self.sections.iter().zip(&self.expected) {
            if sec.keys.len() as u32 != want {
                return Err(Error::CorruptModel {
                    reason: format!(
                        "統合語彙セクション 0x{:02X} は {want} 件のはずが {} 件でした",
                        sec.feature_type as u8,
                        sec.keys.len()
                    ),
                });
            }
        }
        if self.next_feature_id != n_features {
            return Err(Error::CorruptModel {
                reason: format!(
                    "統合語彙の件数合計 {} と n_features={n_features} が一致しません",
                    self.next_feature_id
                ),
            });
        }
        Ok(Vocab {
            slot_of_type: self.slot_of_type,
            sections: self.sections,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cp_key(ft: FeatureType, cps: &[u32]) -> FeatureKey {
        let mut k = FeatureKey {
            feature_type: ft,
            ..FeatureKey::default()
        };
        for (i, &c) in cps.iter().enumerate() {
            k.cp[i] = c;
        }
        k
    }

    #[test]
    fn pack_is_order_preserving_for_trigram() {
        let a = cp_key(
            FeatureType::TrigramPrev1SelfNext1,
            &[0x3042, 0x3044, 0x3046],
        );
        let b = cp_key(
            FeatureType::TrigramPrev1SelfNext1,
            &[0x3042, 0x3044, 0x3047],
        );
        let c = cp_key(
            FeatureType::TrigramPrev1SelfNext1,
            &[0x3042, 0x3045, 0x3041],
        );
        assert!(pack_key(&a) < pack_key(&b));
        assert!(pack_key(&b) < pack_key(&c));
        // FeatureKey の Ord と同じ順序であること
        assert!(a < b && b < c);
    }

    #[test]
    fn pack_roundtrips() {
        for (ft, cps) in [
            (FeatureType::CharSelf, vec![0x20B9F]),
            (FeatureType::BigramPrev1Self, vec![0x3042, 0x10FFFF]),
            (
                FeatureType::TrigramSelfNext1Next2,
                vec![1, 0x10FFFF, 0x3046],
            ),
        ] {
            let k = cp_key(ft, &cps);
            assert_eq!(unpack_key(ft, pack_key(&k)), k, "{ft:?}");
        }

        let mut k = FeatureKey {
            feature_type: FeatureType::KanjiRunLen,
            u8val: 7,
            ..FeatureKey::default()
        };
        assert_eq!(unpack_key(k.feature_type, pack_key(&k)), k);

        k = FeatureKey {
            feature_type: FeatureType::TypeTransition,
            ct: [CharType::Kanji, CharType::Hiragana, CharType::default()],
            ..FeatureKey::default()
        };
        assert_eq!(unpack_key(k.feature_type, pack_key(&k)), k);

        let bias = FeatureKey::no_payload(FeatureType::Bias);
        assert_eq!(unpack_key(bias.feature_type, pack_key(&bias)), bias);
    }

    /// テスト用: セクションの並びから `Vocab` を組む。
    fn build(sections: &[(FeatureType, u32, u32, &[FeatureKey])]) -> Result<Vocab> {
        let mut b = VocabBuilder::new();
        let mut total = 0u32;
        for (ft, col, base, keys) in sections {
            b.begin_section(*ft, keys.len() as u32, *col, *base)?;
            for k in *keys {
                b.push_key(pack_key(k))?;
            }
            total += keys.len() as u32;
        }
        b.finish(total)
    }

    #[test]
    fn resolves_with_implicit_ids() {
        let bias = FeatureKey::no_payload(FeatureType::Bias);
        let a = cp_key(FeatureType::CharSelf, &[0x3042]);
        let i = cp_key(FeatureType::CharSelf, &[0x3044]);
        let u = cp_key(FeatureType::CharSelf, &[0x3046]);
        let v = build(&[
            (FeatureType::Bias, NO_CAT_COLUMN, 0, &[bias]),
            (FeatureType::CharSelf, 5, 10, &[a, i, u]),
        ])
        .unwrap();

        assert_eq!(v.len(), 4);

        // feature_id はセクション先頭からの通し番号
        let r = v.resolve(&bias).unwrap();
        assert_eq!(r.feature_id, 0);
        assert_eq!(r.cat, None);

        for (n, key) in [(0u32, a), (1, i), (2, u)] {
            let r = v.resolve(&key).unwrap();
            assert_eq!(r.feature_id, 1 + n, "feature_id");
            assert_eq!(r.cat, Some((5, 10 + n)), "cat_code");
            assert_eq!(v.feature_id(&key), Some(1 + n));
        }

        // 未登録のキーと、セクションごと存在しないタイプ
        assert!(v.resolve(&cp_key(FeatureType::CharSelf, &[0x3048])).is_none());
        assert!(v.resolve(&cp_key(FeatureType::CharNext1, &[0x3042])).is_none());
    }

    #[test]
    fn iter_roundtrips_entries() {
        let a = cp_key(FeatureType::CharSelf, &[0x3042]);
        let i = cp_key(FeatureType::CharSelf, &[0x3044]);
        let v = build(&[(FeatureType::CharSelf, 3, 7, &[a, i])]).unwrap();
        let entries: Vec<_> = v.iter().collect();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].key, a);
        assert_eq!(entries[0].feature_id, 0);
        assert_eq!((entries[0].cat_column, entries[0].cat_code), (3, 7));
        assert_eq!(entries[1].key, i);
        assert_eq!(entries[1].feature_id, 1);
        assert_eq!((entries[1].cat_column, entries[1].cat_code), (3, 8));
    }

    #[test]
    fn builder_rejects_descending_keys() {
        let err = build(&[(
            FeatureType::CharSelf,
            5,
            0,
            &[
                cp_key(FeatureType::CharSelf, &[0x3044]),
                cp_key(FeatureType::CharSelf, &[0x3042]),
            ],
        )])
        .unwrap_err();
        assert!(matches!(err, Error::CorruptModel { .. }), "{err:?}");
    }

    #[test]
    fn builder_rejects_duplicate_keys() {
        let k = cp_key(FeatureType::CharSelf, &[0x3042]);
        let err = build(&[(FeatureType::CharSelf, 5, 0, &[k, k])]).unwrap_err();
        assert!(matches!(err, Error::CorruptModel { .. }), "{err:?}");
    }

    #[test]
    fn builder_rejects_descending_section_types() {
        // セクションは feature_type 昇順でなければならない（キーの全体順序と一致させるため）
        let err = build(&[
            (
                FeatureType::CharSelf,
                NO_CAT_COLUMN,
                0,
                &[cp_key(FeatureType::CharSelf, &[0x3042])],
            ),
            (
                FeatureType::Bias,
                NO_CAT_COLUMN,
                0,
                &[FeatureKey::no_payload(FeatureType::Bias)],
            ),
        ])
        .unwrap_err();
        assert!(matches!(err, Error::CorruptModel { .. }), "{err:?}");
    }

    #[test]
    fn builder_rejects_count_mismatch() {
        let mut b = VocabBuilder::new();
        b.begin_section(FeatureType::CharSelf, 2, NO_CAT_COLUMN, 0)
            .unwrap();
        b.push_key(pack_key(&cp_key(FeatureType::CharSelf, &[0x3042])))
            .unwrap();
        let err = b.finish(2).unwrap_err();
        assert!(matches!(err, Error::CorruptModel { .. }), "{err:?}");
    }

    #[test]
    fn builder_rejects_total_mismatch() {
        let mut b = VocabBuilder::new();
        b.begin_section(FeatureType::CharSelf, 1, NO_CAT_COLUMN, 0)
            .unwrap();
        b.push_key(pack_key(&cp_key(FeatureType::CharSelf, &[0x3042])))
            .unwrap();
        let err = b.finish(99).unwrap_err();
        assert!(matches!(err, Error::CorruptModel { .. }), "{err:?}");
    }

    #[test]
    fn builder_rejects_out_of_range_cat_column() {
        let mut b = VocabBuilder::new();
        let err = b
            .begin_section(FeatureType::CharSelf, 1, MAX_BOUNDARY_CAT_COLUMNS as u32, 0)
            .unwrap_err();
        assert!(matches!(err, Error::CorruptModel { .. }), "{err:?}");
    }
}
