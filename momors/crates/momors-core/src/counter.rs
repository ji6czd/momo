//! 助数詞（「N日」「N人」…）の**多桁**の読みをルールで守る。
//!
//! ## 役割は「多桁の保護」だけ
//!
//! 一桁の読み（2日→ふつか・1人→ひとり・3つ→みっつ）は**モデル＝データが正本**。
//! ルールは触らない（読みが変ならデータを足す）。ルールの仕事は、**二桁以上の数が
//! 一桁の特殊読みで上書きされるのを防ぐ**こと：文字単位モデルは「21人→2ヒトリ」
//! 「22日→2ミッカ」のように下一桁へ特殊読みを漏らすので、多桁ランは数字＋接尾で
//! 確定させる（21人→21ニン・22日→22ニチ）。
//!
//! ## 多桁の例外（[`CounterSpec::exceptions`]）
//!
//! 数符（算用数字）で正しく読めない値だけを例外に持つ。点字の読み分けをそのまま表す：
//!
//! - `Spell`  … 数符では誤読するので訓読みを綴る。`数符10か`は「じゅうか」に読める
//!   ため 10日は **トオカ** と綴る（20日=ハツカ も同様）。
//! - `Suffix` … 数符で読めるので数字は保ち接尾だけ替える。`数符14か`は「じゅうよっか」
//!   と読めるため 14日は **14＋カ**（下一桁4＝よっか）。24日も同じ。
//! - 例外に無い多桁値 … 数字＋`default_suffix`（21日→21ニチ・21人→21ニン）。
//!
//! ## 発火ゲート（[`CounterSpec::gate_readings`]）
//!
//! 空でなければ、助数詞文字のモデル読みがこの集合のときだけ発火する。「2024日本」の
//! 日→ニ を誤って日付にしない安全弁（日のみ）。数字直後がまず助数詞になる 人 は空。
//!
//! ## 対象外
//!
//! - 一桁（モデル）。
//! - 漢数字ラン。後ろの助数詞も含めてモデルが読む（[`crate::numeric`] は自信度が
//!   低いときに漢数字を数字へ置き換えるだけで、助数詞の読みには触れない）。
//! - つ（多桁が実在しない＝ルール不関与。1〜9つ は全てモデル）。
//! - 匹・階・本・版・歩・分 等、連濁/促音の音韻規則で決まる助数詞はこの表
//!   （[`CounterSpec`]、値そのものをキーにする）ではなく、数を読んだときの
//!   最後の語だけをキーにする [`PhonoSpec`]/[`resolve_phono`] で別扱いする（下の節）。
//!   回のように点字上どの桁でも読みが変わらない助数詞は登録不要。
//!   世のように読みは一定でもモデルが外す助数詞は、例外の無い [`PhonoSpec`] で持つ。

use crate::char_type::CharType;
use crate::featurize::SourceEntry;

/// 多桁の例外読みの種類。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Multi {
    /// 数符では誤読するので訓読みを綴る（10→"トオカ"）。
    Spell(&'static str),
    /// 数字は保ち接尾だけ差し替える（14→ +"カ" ＝ 14カ）。
    Suffix(&'static str),
}

/// 助数詞パスの発火結果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CounterAction {
    /// 数字ラン＋助数詞を丸ごとこの訓読みで置き換える（10日→トオカ）。
    Spell(&'static str),
    /// 原文の数字はそのまま出し、助数詞をこの接尾で出す（21日→"2""1"＋"ニチ"）。
    DigitsPlus(&'static str),
}

/// 1つの助数詞のルール定義。
struct CounterSpec {
    /// 助数詞のコードポイント（`'日' as u32` 等）。
    counter_cp: u32,
    /// 例外に無い多桁値の接尾（日→"ニチ"・人→"ニン"）。
    default_suffix: &'static str,
    /// 発火ゲート：空でなければ助数詞文字のモデル読みがこの集合のときだけ発火（日本よけ）。
    gate_readings: &'static [&'static str],
    /// 多桁の例外（数符で正しく読めない値）。
    exceptions: &'static [(u32, Multi)],
}

impl CounterSpec {
    fn needs_gate(&self) -> bool {
        !self.gate_readings.is_empty()
    }

    fn gate_ok(&self, model_reading: &str) -> bool {
        self.gate_readings.is_empty() || self.gate_readings.contains(&model_reading)
    }

    /// 多桁値 `value` に対する動作。
    fn action(&self, value: u32) -> CounterAction {
        match self.exceptions.iter().find(|(v, _)| *v == value) {
            Some((_, Multi::Spell(k))) => CounterAction::Spell(k),
            Some((_, Multi::Suffix(s))) => CounterAction::DigitsPlus(s),
            None => CounterAction::DigitsPlus(self.default_suffix),
        }
    }
}

/// 助数詞テーブル。多桁の保護と例外を持つ助数詞だけを登録する。
const COUNTERS: &[CounterSpec] = &[
    // 日: 10/20 は綴る（数符で誤読）・14/24 は下一桁4＝カ・他は数字＋ニチ。
    CounterSpec {
        counter_cp: '日' as u32,
        default_suffix: "ニチ",
        gate_readings: &["ニチ", "カ", "タチ"],
        exceptions: &[
            (10, Multi::Spell("トオカ")),
            (14, Multi::Suffix("カ")),
            (20, Multi::Spell("ハツカ")),
            (24, Multi::Suffix("カ")),
        ],
    },
    // 人: 特殊な多桁読みは無い。多桁は一律 数字＋ニン（21人→21ニン で漏れを断つ）。
    CounterSpec {
        counter_cp: '人' as u32,
        default_suffix: "ニン",
        gate_readings: &[],
        exceptions: &[],
    },
];

fn find_counter(cp: u32) -> Option<&'static CounterSpec> {
    COUNTERS.iter().find(|c| c.counter_cp == cp)
}

/// コードポイント `counter_cp` の助数詞が発火ゲート（モデル読み確認）を要するか。
///
/// `false`（人）ならモデル読みを見ずに [`resolve_multi`] を呼んでよい。
pub(crate) fn counter_needs_gate(counter_cp: u32) -> bool {
    find_counter(counter_cp).is_some_and(CounterSpec::needs_gate)
}

/// **多桁**ラン（値 `value`）＋助数詞 `counter_cp` の動作を返す。一桁では呼ばない。
///
/// `model_reading` は助数詞文字のモデル読み（ゲート判定用。ゲートが空の助数詞では
/// 参照されない）。未登録の助数詞・ゲート不通過なら `None`（＝通常経路）。
pub(crate) fn resolve_multi(
    counter_cp: u32,
    value: u32,
    model_reading: &str,
) -> Option<CounterAction> {
    let spec = find_counter(counter_cp)?;
    if !spec.gate_ok(model_reading) {
        return None;
    }
    Some(spec.action(value))
}

// ============================================================
// 音韻助数詞（連濁・促音）
// ============================================================

/// 音韻規則（連濁・半濁音化）で読みが決まる助数詞（匹・階…）。
///
/// 日/人と違い、値そのものではなく**数を読んだときの最後の語**（[`last_numeral_word`]）
/// だけで読みが決まるため、桁数に関係なく同じ表がそのまま使える（多桁専用の例外カードは
/// 不要）。最後の語は、一の位が0でなければ一の位（一〜九）、0なら0でない最も下の位
/// （十・百・千・万・億）。同じ「末尾0」でも 10本=じゅっぽん・100本=ひゃっぽん（促音）と
/// 1000本=せんぼん・10000本=いちまんぼん（ンの後の連濁）で読みが分かれるため、位を区別する。
/// 点字では数字は数符で綴られ発音上の促音（っ）は表れないので、ここで守るのは
/// **助数詞側に残る子音の変化（半濁音化・連濁）だけ**でよい。
/// 促音のみで子音が変わらない助数詞（回 等）はそもそも表が不要（発火させない）。
/// 漢数字表記（三千匹 等）はここでは扱わない（助数詞も含めてモデルが読む）。
///
/// ## 一桁だけ意味が割れる助数詞（[`PhonoSpec::min_digits`]）
///
/// 分は「時間（フン）」と「歩合（ブ、割/分/厘の分）」で読みが違うが、歩合の分は
/// 割/分/厘という一桁ずつの位取り表記の一桁（百分率の小数第2位）なので、
/// **多桁の分は時間としか読めない**（曖昧さゼロ）。曖昧なのは一桁だけなので、
/// `min_digits: 2` にして一桁では発火させず（読みはモデル＝学習データに委ねる）、
/// 二桁以上だけこの表で確定させる。
///
/// ## 助数詞の文字が別の語の頭になる場合（[`PhonoSpec::not_before`]）
///
/// 世は「ルイ14世」ならセイだが、「3世代」「2世帯」「第3世界」では世=セの別の語に
/// なる。直後の文字で見分けられるので、その文字が続くときは発火させない。
struct PhonoSpec {
    /// 助数詞のコードポイント。
    counter_cp: u32,
    /// 表に無い最後の語（二・四・五・七・九…）での読み。
    default_reading: &'static str,
    /// (数の最後の語, その読み) の表。最後の語は [`last_numeral_word`] の漢数字1字
    /// （値そのものが 0 のときは最後の語が無いので既定読み・[`resolve_phono`] 側でガードする）。
    overrides: &'static [(char, &'static str)],
    /// 発火に必要な最小桁数。ほとんどは1（一桁からでも発火）。分だけ2
    /// （一桁は歩合と衝突するため発火させず、モデル任せにする）。
    min_digits: usize,
    /// 助数詞の直後にこの文字が続くときは発火させない（世代・世帯・世界の 代/帯/界）。
    not_before: &'static [char],
}

// 最後の語が 一・六・八・十・百（いっ・ろっ・はっ・じゅっ・ひゃっ＝促音）と
// 三・千・万（さん・せん・まん＝ン）で読みが変わる。四（よん）もンで終わるが連濁しない
// （よんほん・よんひき）。億で終わる数は実例を確かめられていないので、どの助数詞も
// 表に入れず既定読みにする（算用数字で 100000000匹 と書くことも実際にはまず無い）。
const PHONO_COUNTERS: &[PhonoSpec] = &[
    // 匹: は行なので半濁音化(ぴ)が生き残る。促音の後→ピキ・ンの後→ビキ・他ヒキ。
    // 実例: 千疋屋=せんびきや。
    PhonoSpec {
        counter_cp: '匹' as u32,
        default_reading: "ヒキ",
        overrides: &[
            ('一', "ピキ"),
            ('六', "ピキ"),
            ('八', "ピキ"),
            ('十', "ピキ"),
            ('百', "ピキ"),
            ('三', "ビキ"),
            ('千', "ビキ"),
            ('万', "ビキ"),
        ],
        min_digits: 1,
        not_before: &[],
    },
    // 階: か行に半濁音は無く促音も点字では消えるため、変化として残るのは3の連濁だけ。
    // （回は階と同じか行だが3も連濁しないため、点字上は常にカイ→登録不要。）
    // 三階の連濁（さんがい）は慣用で、千階・万階の実例は見つからないので既定読みのまま。
    PhonoSpec {
        counter_cp: '階' as u32,
        default_reading: "カイ",
        overrides: &[('三', "ガイ")],
        min_digits: 1,
        not_before: &[],
    },
    // 本: 匹と同型（は行）。促音の後→ポン・ンの後→ボン・他ホン。実例: 千本桜=せんぼんざくら。
    PhonoSpec {
        counter_cp: '本' as u32,
        default_reading: "ホン",
        overrides: &[
            ('一', "ポン"),
            ('六', "ポン"),
            ('八', "ポン"),
            ('十', "ポン"),
            ('百', "ポン"),
            ('三', "ボン"),
            ('千', "ボン"),
            ('万', "ボン"),
        ],
        min_digits: 1,
        not_before: &[],
    },
    // 分: 歩合（割/分/厘の分＝ブ）は定義上つねに一桁なので、一桁は曖昧（時間フンとの
    // 衝突）。多桁の分は時間としか読めないので min_digits: 2 にして一桁を除外する。
    // 一桁の読み分け（フン/ブ）は学習データ（隣接する「割」等の文脈）に委ねる。
    // 促音の後もンの後も（四=よんぷん も）プン。
    PhonoSpec {
        counter_cp: '分' as u32,
        default_reading: "フン",
        overrides: &[
            ('一', "プン"),
            ('三', "プン"),
            ('四', "プン"),
            ('六', "プン"),
            ('八', "プン"),
            ('十', "プン"),
            ('百', "プン"),
            ('千', "プン"),
            ('万', "プン"),
        ],
        min_digits: 2,
        not_before: &[],
    },
    // 版: は行だが匹/本と違い3も連濁せず半濁音のまま（さんぱん、さんばんではない）。
    // 実例: 第一版=だいいっぱん・第二版=だいにはん・第三版=だいさんぱん。
    // 千版・万版の実例は無いが、ンの後の型（三版）に揃える。
    PhonoSpec {
        counter_cp: '版' as u32,
        default_reading: "ハン",
        overrides: &[
            ('一', "パン"),
            ('三', "パン"),
            ('六', "パン"),
            ('八', "パン"),
            ('十', "パン"),
            ('百', "パン"),
            ('千', "パン"),
            ('万', "パン"),
        ],
        min_digits: 1,
        not_before: &[],
    },
    // 歩: 版と同型（3も半濁音のまま）。促音の後もンの後もポ・他ホ。
    // 実例: 一歩=いっぽ・三歩=さんぽ・四歩=よんほ・百歩=ひゃっぽ・万歩計=まんぽけい。
    // 兵/道が続くときはモデルに委ねる。歩兵（第3歩兵師団）は歩=ホの別の語。道は
    // 「第1歩道橋」（ホ）と「1歩道を外れる」（ポ）の両方があり、ルールで上書きすると
    // 学習データで読み分けを教えても反映されないため。
    PhonoSpec {
        counter_cp: '歩' as u32,
        default_reading: "ホ",
        overrides: &[
            ('一', "ポ"),
            ('三', "ポ"),
            ('六', "ポ"),
            ('八', "ポ"),
            ('十', "ポ"),
            ('百', "ポ"),
            ('千', "ポ"),
            ('万', "ポ"),
        ],
        min_digits: 1,
        not_before: &['兵', '道'],
    },
    // 世: 君主・移民の代（ルイ14世・エリザベス2世・日系3世）はどの桁でもセイ。音韻の変化は
    // 無いが、モデルは数字の後の世をセ（世界・世代から）やヨと読むので表で守る。
    // 世代・世帯・世界は世=セの別の語なので、代/帯/界が続くときはモデルに委ねる。
    PhonoSpec {
        counter_cp: '世' as u32,
        default_reading: "セイ",
        overrides: &[],
        min_digits: 1,
        not_before: &['代', '帯', '界'],
    },
];

/// 音韻助数詞 `counter_cp` の、数字ラン値 `value`（桁数 `digit_count`）に対する読みを返す。
///
/// `next_cp` は助数詞の直後の文字（無ければ `None`）。未登録の助数詞、`digit_count` が
/// [`PhonoSpec::min_digits`] に満たない（＝分の一桁のように意味が割れる）場合、
/// `next_cp` が [`PhonoSpec::not_before`] にある場合は `None`（＝通常経路・モデルに委ねる）。
pub(crate) fn resolve_phono(
    counter_cp: u32,
    value: u32,
    digit_count: usize,
    next_cp: Option<u32>,
) -> Option<&'static str> {
    let spec = PHONO_COUNTERS.iter().find(|s| s.counter_cp == counter_cp)?;
    if digit_count < spec.min_digits {
        return None;
    }
    if next_cp.is_some_and(|cp| spec.not_before.iter().any(|&c| c as u32 == cp)) {
        return None;
    }
    let Some(word) = last_numeral_word(value) else {
        // 値そのものが0（"0匹"）は最後の語が無いので既定読みへ。
        return Some(spec.default_reading);
    };
    Some(match spec.overrides.iter().find(|(w, _)| *w == word) {
        Some((_, reading)) => reading,
        None => spec.default_reading,
    })
}

/// 値 `value` を読んだときの最後の語を漢数字1字で返す。0 なら `None`。
///
/// 一の位が0でなければ一の位（一〜九）、0なら0でない最も下の位
/// （20→にじゅう＝十・300→さんびゃく＝百・1000→せん＝千・20000→にまん＝万・
/// 1000万→いっせんまん＝万・1億→いちおく＝億）。
fn last_numeral_word(value: u32) -> Option<char> {
    const DIGITS: [char; 10] = ['〇', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
    if value == 0 {
        return None;
    }
    let ones = value % 10;
    Some(if ones != 0 {
        DIGITS[ones as usize]
    } else if value % 100 != 0 {
        '十'
    } else if value % 1_000 != 0 {
        '百'
    } else if value % 10_000 != 0 {
        '千'
    } else if value % 100_000_000 != 0 {
        '万'
    } else {
        '億'
    })
}

/// 算用数字（半角/全角）1文字のコードポイントを 0..=9 に変換。数字でなければ `None`。
fn ascii_digit_value(cp: u32) -> Option<u32> {
    match cp {
        0x30..=0x39 => Some(cp - 0x30),       // 0-9
        0xFF10..=0xFF19 => Some(cp - 0xFF10), // ０-９（全角）
        _ => None,
    }
}

/// `seq[start]` から始まる算用数字ランの 10進値とラン終端 `end`（排他）を返す。
///
/// - `seq[start]` が算用数字でなければ `None`。
/// - 桁が非現実的に長く値が `u32` を溢れる場合は `None`。
///
/// 呼び出し側は `start` がランの先頭（直前が算用数字でない）であることを保証する。
/// 一桁か多桁かは `end - start` で判定する。
pub(crate) fn arabic_run(seq: &[SourceEntry], start: usize) -> Option<(u32, usize)> {
    ascii_digit_value(seq.get(start)?.cp)?;

    let mut end = start;
    let mut val: u64 = 0;
    while let Some(e) = seq.get(end) {
        match ascii_digit_value(e.cp) {
            Some(d) if e.ctype == CharType::Numeric => {
                val = val * 10 + u64::from(d);
                if val > u64::from(u32::MAX) {
                    return None;
                }
                end += 1;
            }
            _ => break,
        }
    }
    Some((val as u32, end))
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::featurize::to_source_seq;

    // --- 日 ---

    #[test]
    fn day_spell_exceptions() {
        // 数符で誤読する 10/20 は綴る
        assert_eq!(
            resolve_multi('日' as u32, 10, "カ"),
            Some(CounterAction::Spell("トオカ"))
        );
        assert_eq!(
            resolve_multi('日' as u32, 20, "カ"),
            Some(CounterAction::Spell("ハツカ"))
        );
    }

    #[test]
    fn day_suffix_yon() {
        // 下一桁4（14/24）は数字＋カ
        assert_eq!(
            resolve_multi('日' as u32, 14, "カ"),
            Some(CounterAction::DigitsPlus("カ"))
        );
        assert_eq!(
            resolve_multi('日' as u32, 24, "カ"),
            Some(CounterAction::DigitsPlus("カ"))
        );
    }

    #[test]
    fn day_default_protect() {
        // 例外以外の多桁は 数字＋ニチ（漏れ防止）
        for v in [11, 12, 13, 15, 19, 21, 22, 25, 30, 31, 100] {
            assert_eq!(
                resolve_multi('日' as u32, v, "ニチ"),
                Some(CounterAction::DigitsPlus("ニチ")),
                "v={v}"
            );
        }
    }

    #[test]
    fn day_gate_blocks_non_counter() {
        // 「2024日本」の 日→ニ はゲート外 → 発火しない
        assert_eq!(resolve_multi('日' as u32, 2024, "ニ"), None);
        assert!(counter_needs_gate('日' as u32));
    }

    // --- 人 ---

    #[test]
    fn nin_protect_no_gate() {
        // 多桁は一律 数字＋ニン・ゲート無し
        assert_eq!(
            resolve_multi('人' as u32, 10, ""),
            Some(CounterAction::DigitsPlus("ニン"))
        );
        assert_eq!(
            resolve_multi('人' as u32, 21, ""),
            Some(CounterAction::DigitsPlus("ニン"))
        );
        assert!(!counter_needs_gate('人' as u32));
    }

    #[test]
    fn unknown_counter() {
        // つ・匹 は未登録（多桁が無い/音韻系）→ 発火しない
        assert_eq!(resolve_multi('つ' as u32, 3, ""), None);
        assert_eq!(resolve_multi('匹' as u32, 3, ""), None);
    }

    // --- arabic_run（一桁/多桁の判定込み） ---

    #[test]
    fn arabic_run_len() {
        // 一桁: end-start==1
        let s = to_source_seq("5日");
        assert_eq!(arabic_run(&s, 0), Some((5, 1)));
        // 多桁: end-start==2
        let s = to_source_seq("21日");
        assert_eq!(arabic_run(&s, 0), Some((21, 2)));
        // 全角
        let s = to_source_seq("２０人");
        assert_eq!(arabic_run(&s, 0), Some((20, 2)));
    }

    #[test]
    fn arabic_run_stops_and_none() {
        let s = to_source_seq("3月5日");
        assert_eq!(arabic_run(&s, 0), Some((3, 1)));
        let s = to_source_seq("日");
        assert_eq!(arabic_run(&s, 0), None);
    }

    // --- resolve_phono（音韻助数詞） ---

    /// 値 `v` の桁数（テスト用: `resolve_phono` の `digit_count` 引数に渡す）。
    fn digits(v: u32) -> usize {
        v.to_string().len()
    }

    #[test]
    fn last_numeral_word_by_lowest_nonzero_place() {
        for (v, w) in [
            (1, '一'),
            (23, '三'),
            (10, '十'),
            (110, '十'),
            (100, '百'),
            (300, '百'),
            (1100, '百'),
            (1000, '千'),
            (3000, '千'),
            (21000, '千'),
            (10000, '万'),
            (20000, '万'),
            (10_000_000, '万'),
            (100_000_000, '億'),
        ] {
            assert_eq!(last_numeral_word(v), Some(w), "v={v}");
        }
        assert_eq!(last_numeral_word(0), None);
    }

    #[test]
    fn phono_hiki_classes() {
        // 半濁音化: 促音の後（一・六・八・十・百）
        for v in [1, 6, 8, 10, 16, 18, 20, 100, 300, 1100] {
            assert_eq!(
                resolve_phono('匹' as u32, v, digits(v), None),
                Some("ピキ"),
                "v={v}"
            );
        }
        // 連濁: ンの後（三・千・万）。千匹=せんびき・一万匹=いちまんびき。
        for v in [3, 13, 23, 1000, 3000, 10000, 20000] {
            assert_eq!(
                resolve_phono('匹' as u32, v, digits(v), None),
                Some("ビキ"),
                "v={v}"
            );
        }
        // 無変化
        for v in [2, 4, 5, 7, 9, 12, 24] {
            assert_eq!(
                resolve_phono('匹' as u32, v, digits(v), None),
                Some("ヒキ"),
                "v={v}"
            );
        }
    }

    #[test]
    fn phono_zero_is_not_juu() {
        // 値そのものが0（"0匹"）は最後の語が無いので既定読みへ。
        assert_eq!(resolve_phono('匹' as u32, 0, 1, None), Some("ヒキ"));
    }

    #[test]
    fn phono_oku_falls_back_to_default() {
        // 億で終わる数は実例未確認なので、どの助数詞も既定読み。
        assert_eq!(
            resolve_phono('本' as u32, 100_000_000, 9, None),
            Some("ホン")
        );
    }

    #[test]
    fn phono_kai_only_rendaku_at_three() {
        assert_eq!(resolve_phono('階' as u32, 3, 1, None), Some("ガイ"));
        assert_eq!(resolve_phono('階' as u32, 13, 2, None), Some("ガイ"));
        // 促音のみのクラスは点字では既定読みに潰れる。千階・万階も既定読み。
        for v in [1, 6, 8, 10, 100, 1000, 10000] {
            assert_eq!(
                resolve_phono('階' as u32, v, digits(v), None),
                Some("カイ"),
                "v={v}"
            );
        }
        for v in [2, 4, 5, 7, 9] {
            assert_eq!(
                resolve_phono('階' as u32, v, 1, None),
                Some("カイ"),
                "v={v}"
            );
        }
    }

    #[test]
    fn phono_unregistered_counter() {
        // 回は連濁も無いため未登録（常に既定読みでよい＝表を持たない）。
        assert_eq!(resolve_phono('回' as u32, 3, 1, None), None);
        assert_eq!(resolve_phono('日' as u32, 1, 1, None), None);
    }

    #[test]
    fn phono_hon_classes() {
        // 匹と同型: 半濁音化(一・六・八・十・百)と連濁(三・千・万)が両方ある。
        for v in [1, 6, 8, 10, 16, 18, 100, 1100] {
            assert_eq!(
                resolve_phono('本' as u32, v, digits(v), None),
                Some("ポン"),
                "v={v}"
            );
        }
        // 千本=せんぼん・一万本=いちまんぼん。
        for v in [3, 13, 23, 1000, 3000, 10000] {
            assert_eq!(
                resolve_phono('本' as u32, v, digits(v), None),
                Some("ボン"),
                "v={v}"
            );
        }
        for v in [2, 4, 5, 7, 9] {
            assert_eq!(
                resolve_phono('本' as u32, v, 1, None),
                Some("ホン"),
                "v={v}"
            );
        }
    }

    #[test]
    fn phono_han_three_stays_semivoiced() {
        // 版は本/匹と違い3も連濁しない（さんぱん、さんばんではない）。
        assert_eq!(resolve_phono('版' as u32, 3, 1, None), Some("パン"));
        for v in [1, 6, 8, 10, 100, 1000, 10000] {
            assert_eq!(
                resolve_phono('版' as u32, v, digits(v), None),
                Some("パン"),
                "v={v}"
            );
        }
        for v in [2, 4, 5, 7, 9] {
            assert_eq!(
                resolve_phono('版' as u32, v, 1, None),
                Some("ハン"),
                "v={v}"
            );
        }
    }

    #[test]
    fn phono_ho_like_han() {
        // 歩は版と同型（3も半濁音のまま、さんぽ）。百歩=ひゃっぽ・千歩=せんぽ・万歩=まんぽ。
        for v in [1, 3, 6, 8, 10, 13, 100, 1000, 10000] {
            assert_eq!(
                resolve_phono('歩' as u32, v, digits(v), None),
                Some("ポ"),
                "v={v}"
            );
        }
        for v in [2, 4, 5, 7, 9, 12, 25] {
            assert_eq!(
                resolve_phono('歩' as u32, v, digits(v), None),
                Some("ホ"),
                "v={v}"
            );
        }
    }

    #[test]
    fn phono_ho_not_before_other_words() {
        // 歩兵（第3歩兵師団）・歩道（第1歩道橋／1歩道を外れる）は発火しない（モデルに委ねる）。
        for next in ['兵', '道'] {
            assert_eq!(
                resolve_phono('歩' as u32, 1, 1, Some(next as u32)),
                None,
                "next={next}"
            );
        }
        // 「1歩歩く」のように歩が続くのは助数詞のまま。
        assert_eq!(
            resolve_phono('歩' as u32, 1, 1, Some('歩' as u32)),
            Some("ポ")
        );
    }

    #[test]
    fn phono_fun_single_digit_deferred_to_model() {
        // 分の一桁は歩合（ブ）と衝突するので発火しない（min_digits: 2）。
        for v in 0..=9 {
            assert_eq!(resolve_phono('分' as u32, v, 1, None), None, "v={v}");
        }
    }

    #[test]
    fn phono_fun_multi_digit_is_unambiguous() {
        // 歩合の分は定義上つねに一桁なので、多桁は時間（フン/プン）としか読めない。
        for v in [11, 13, 14, 16, 18, 20, 100, 1000, 10000] {
            assert_eq!(
                resolve_phono('分' as u32, v, digits(v), None),
                Some("プン"),
                "v={v}"
            );
        }
        for v in [12, 15, 17, 19, 22] {
            assert_eq!(
                resolve_phono('分' as u32, v, digits(v), None),
                Some("フン"),
                "v={v}"
            );
        }
    }

    #[test]
    fn phono_sei_fixed_reading() {
        // 世はどの桁でもセイ（ルイ14世・エリザベス2世・日系3世の）。
        for v in [1, 2, 3, 10, 14, 21] {
            assert_eq!(
                resolve_phono('世' as u32, v, digits(v), None),
                Some("セイ"),
                "v={v}"
            );
        }
        assert_eq!(
            resolve_phono('世' as u32, 3, 1, Some('の' as u32)),
            Some("セイ")
        );
        // 世紀もセイ（紀はモデルがキと読む）。
        assert_eq!(
            resolve_phono('世' as u32, 21, 2, Some('紀' as u32)),
            Some("セイ")
        );
    }

    #[test]
    fn phono_sei_not_before_other_words() {
        // 世代・世帯・世界は世=セの別の語なので発火しない（モデルに委ねる）。
        for next in ['代', '帯', '界'] {
            assert_eq!(
                resolve_phono('世' as u32, 3, 1, Some(next as u32)),
                None,
                "next={next}"
            );
        }
    }
}
