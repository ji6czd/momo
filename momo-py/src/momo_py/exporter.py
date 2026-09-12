"""
exporter.py  ―  momo モデルを C++ 推論エンジン向けバイナリ (.mbm) に変換する。

バイナリフォーマット概要
========================

[ファイルヘッダ]          16 bytes
  magic        : uint8[4]   "MOMO"
  version      : uint8      0x0A
  flags        : uint8      bit0 = 統合語彙が GBDT カテゴリカル(column,code)を持つ
                            bit1 = ソート済み配列を差分 + LEB128 varint で書く（下記）
  _reserved    : uint8[2]   0x00 x2
  n_classes    : uint32     読みラベル数
  n_features   : uint32     特徴量次元数（語彙サイズ）

[統合語彙テーブル]        version 0x09 で特徴量タイプ別セクションに変更
  n_sections   : uint32     セクション数（= 出現する feature_type の種類数）
  以下 n_sections 個（feature_type 昇順）:
    feature_type  : uint8
    _pad          : uint8[3] 0x00 x3
    count         : uint32   このセクションのエントリ数
    cat_column    : uint32   GBDT カテゴリカル列（0xFFFFFFFF = 列なし）
    cat_code_base : uint32   セクション先頭エントリのカテゴリカルコード
  続いて、各セクションのキー配列をセクションの順に連結:
    key           : uint64 × count   詰めたキー（昇順・重複なし）
                            flags bit1 のときはセクションごとに差分 varint 列
                            （「差分 + varint レイアウト」の節を参照）

  キーの詰め方（`_pack_vocab_key`。Rust 側 vocab.rs::pack_key と一致）:
    char32×M   : cp[0] を上位に 21bit ずつ（M <= 3 なので 63bit）
    chartype×N : ct[0] を上位に 8bit ずつ（N <= 3）
    uint8×1    : u8_val
    なし        : 0

  0x08 まではエントリごとに feature_id と (cat_column, cat_code) を書いており、
  可変長で 1 エントリ 13〜25 バイトあった。0x09 では 3 つとも採番し直して暗黙にする:

    feature_id = セクションの feature_id_base + セクション内の添字
                 （feature_id_base は先行セクションの count の累計。つまり
                   全体をキー順に並べたときの行位置がそのまま feature_id）
    cat_code   = セクションの cat_code_base + セクション内の添字
    cat_column = セクションが 1 個だけ持つ

  cat_column が feature_type ごとに 1 個に決まるのは、momo_py.categorical の
  fit_categorical() が「特徴量名ごとに 1 列」を作り、特徴量名が FeatureType と
  1 対 1 に対応するため。唯一 char_s 列だけは CharSelf(0x90) と
  CharSelfCompound2(0xA6) の 2 タイプが共有するが、コードは列内で通し番号なので
  各タイプのぶんは連続した区間になり、cat_code_base で表せる。

  この採番し直しに伴い、書き出し側で 3 つの並べ替えを行う:
    1. CSC 重み行列の列をキー順に並べ替える
    2. 線形境界の重みベクトルも同じ順に並べ替える
    3. GBDT の cats を新しいカテゴリコードへ書き換える

[読みラベルテーブル]      n_classes エントリ
  len          : uint8      UTF-8バイト長
  utf8         : uint8[len] ラベル文字列（UTF-8）

[読みモデル重み（CSC・int8量子化・クラスごとscale）]
  quant_scale  : float32 × n_classes   クラス(行)ごとの量子化スケール係数
  n_nonzero    : uint32     非ゼロ要素数
  col_len      : uint16 × n_features   列ごとの非ゼロ数。読み手が前置和して colptr にする
                            （0x08 までは uint32 × (n_features+1) の累積和だった。
                              1 列の非ゼロ数は高々 n_classes <= 65536 なので uint16 で足りる）
                            flags bit1 のときは値ごとに varint
  rowind       : uint16 × n_nonzero    行インデックス = クラスID（列内で昇順）
                            flags bit1 のときは列ごとに差分 varint 列
  data         : int8  × n_nonzero     （flags bit1 でも据え置き）

[読みモデル intercept]
  intercept    : float32 × n_classes

[境界モデル]              version 0x06 で algo_tag プレフィックス付きの可変レイアウトに変更
  algo_tag     : uint8      0x00 = 線形 (sgd)、0x01 = 木のアンサンブル (gbdt)

  --- algo_tag == 0x00 (線形。0x05 までと同一バイト列) ---
    quant_scale  : float32
    data         : int8 × n_features   （クラス1の重みベクトル）
    intercept    : float32 × 2         （クラス0, クラス1）

  --- algo_tag == 0x01 (木。.mbm/.mbmf で完全に同一バイト列、量子化しない) ---
    n_cat_columns : uint32   カテゴリカル列数（momo_py.categorical の列数）
    （version 0x07 で cat_vocab を統合語彙テーブルに移動。境界セクションは列数と木だけ。
      カテゴリカル (column, code) は統合語彙の各エントリ末尾が持つ。前提として
      カテゴリカルキーは読み語彙の部分集合であること。exporter が突合して保証する）
    [木のアンサンブル]
      n_trees : uint32
      以下 n_trees 本、各木は先行順（深さ優先）の再帰的ノード列（長さプレフィックス不要）:
        node_tag : uint8   0 = leaf, 1 = split
        --- leaf ---
          leaf_value : float32
        --- split ---
          split_feature : uint32   カテゴリカル列インデックス（column_index と同じ空間）
                                    flags bit1 のときは varint
          default_left  : uint8    0/1。この列に対応する特徴量がこの位置に存在しない
                                    （欠損）とき、左右どちらへ進むか
          n_cats        : uint32   flags bit1 のときは varint
          cats          : uint32 × n_cats   昇順ソート済み。この集合に列のコードが
                                    含まれれば左の子へ、含まれなければ右の子へ
                                    flags bit1 のときは差分 varint 列
          left_child    : 再帰的ノード
          right_child   : 再帰的ノード
      スコアリング: 全木の到達リーフ値の合計をそのまま生スコアとする（追加の
      intercept はない）。sigmoid + 0.5 閾値へ接続するのは線形モデルと同じ。

[人名辞書テーブル]        version 0x03 で追加、0x04 で読みを追加
  n_names      : uint32     人名エントリ数（辞書なしモデルは 0）
  以下 n_names エントリ:
    len        : uint8      UTF-8バイト長
    utf8       : uint8[len] 表層形（UTF-8）
    n_readings : uint8      ユニット別読みの個数（0=読みなし。
                            非0なら表層形のユニット数と一致する）
    以下 n_readings 個:
      len      : uint8      UTF-8バイト長
      utf8     : uint8[len] ユニット読み（カタカナ、UTF-8）
  推論側はこの辞書に対して get_units() 相当のユニット単位で最長一致を行い、
  文字ごとの人名フラグ（1=B: スパン先頭, 2=I: 継続）を NAME_FLAG_* 特徴量
  として発火させる（Python 側 name_dict.compute_name_matches と同一の手順）。
  読みが登録されているスパンでは、読みモデルが低自信度のとき辞書読みで
  置換するフォールバックに使う（人名の読みは文脈で変化しないため固定辞書）。

[単一漢字辞書テーブル]    version 0x04 の途中（アルファ期間）で追加
  n_entries    : uint32     エントリ数
  以下 n_entries エントリ:
    len        : uint8      UTF-8バイト長
    utf8       : uint8[len] 漢字（1文字、UTF-8）
    n_readings : uint8      既知の読みの個数
    以下 n_readings 個:
      len      : uint8      UTF-8バイト長
      utf8     : uint8[len] 読み（カタカナ、UTF-8）
  読みモデルの候補制約（辞書に載っている漢字は既知の読み+CONTINUE+SKIPのみを
  argmax 候補とする）に使う。モデル動作に必須のデータなので .mbm に同梱する。

バージョン履歴
  0x01: 初版
  0x02: 読みモデルの量子化scaleをクラスごと(n_classes個)に変更
  0x03: 人名辞書テーブルと NAME_FLAG_* 特徴量（0xC3-0xC5）を追加
  0x04: 人名辞書テーブルにユニット別読みを追加（低自信度フォールバック用）。
        アルファ期間中に単一漢字辞書テーブルも追加（バージョン番号は据え置き。
        追加前の 0x04 ファイルは読み込み時にエラーになるため再エクスポートすること）
  0x05: 読みモデル重みを CSR から CSC に変更し、行インデックスを uint16 に縮小。
        推論側は特徴量(列)で走査するため CSC が最終形であり、CSR で保存すると
        ローダーが変換のために CSR と CSC を同時に確保してピークメモリが跳ねる。
        CSC で保存すれば読んだ先がそのまま最終形になる。行インデックスは
        クラスID（< n_classes）なので uint16 に収まり、非ゼロあたり 4→2 バイト。
        ただし列ポインタが n_classes+1 個から n_features+1 個に増えるため、
        ファイルサイズの利得は nnz/n_features の比に依存する（2 が損益分岐）。
        同時に `.mbmf` のバージョン番号をこの `.mbm` の採番に合流させた
        （それまでは 0x01 から独立採番していた）。
  0x06: 境界モデルセクションに algo_tag プレフィックスを追加し、線形 (sgd) に加えて
        GBDT（LightGBM、カテゴリカル特徴量）の木のアンサンブルを書き出せるように
        変更。線形の0x05までのバイト列はalgo_tag=0x00として据え置き。
        0x05 ファイルは読み込み時にエラーになるため再エクスポートすること。
  0x07: 語彙を1枚に統合。0x06 までは GBDT の cat_vocab が読み語彙とほぼ同一のキー集合を
        丸ごと二重に格納していた（実測で語彙2枚がファイルの約6割）。読み語彙の各エントリに
        カテゴリカル (column, code) を吸収し（ヘッダ flags bit0）、feature_id を行番号で
        暗黙化して 4 バイト削減。GBDT 境界セクションは n_columns と木だけになった。
        実測でファイル・ロード時ピークともに約 1/3 削減（推論結果は不変）。前提は
        「カテゴリカルキーは読み語彙の部分集合」（exporter が突合して保証）。
        0x06 ファイルは読み込み時にエラーになるため再エクスポートすること。
  0x08: mmap 対応（docs/zerocopy-model-plan.md）の下ごしらえとして、統合語彙
        テーブルを feature_id 順ではなくキー順（Rust FeatureKey の Ord 順）で
        書くように変更。Rust 側ロード時の再ソートを廃止した（再ソートがある限り
        全 materialize が必須だった）。feature_id は行位置と一致しなくなるため
        明示フィールドに戻した（+4バイト/エントリ）。CSC 重み行列の列は今も
        feature_id 順のまま変更していない。推論結果は不変。
        （境界GBDT木のフラット配列化も同時に試したが、PC上の合成木ベンチマークでは
        高速化が見えたものの実モデルで検証したところ逆に40〜60%遅化したため見送った
        ―― 実モデルは1splitあたり平均約28カテゴリ持ち、共有catsプールへの
        アクセスがノードごとの専有Vecよりキャッシュに悪かったとみられる。木は
        引き続き0x07までと同じBoxポインタ+再帰ノード列のまま）。
        0x07 ファイルは読み込み時にエラーになるため再エクスポートすること。
  0x09: 統合語彙を特徴量タイプ別セクションに分け、エントリを詰めた uint64 キー
        8 バイトだけにした（feature_id / cat_code / cat_column を採番し直して暗黙化）。
        CSC の colptr(uint32 累積和) を col_len(uint16) に変更。ファイル 45〜50% 減。
        0x08 ファイルは読み込み時にエラーになるため再エクスポートすること。
  0x0A: ヘッダ flags bit1 を追加し、ソート済み配列を「隣との差分 + LEB128 varint」で
        書く圧縮レイアウトを選べるようにした（下記「差分 + varint レイアウト」）。
        0x09 ファイルは読み込み時にエラーになるため再エクスポートすること。

差分 + varint レイアウト（version 0x0A、flags bit1 = FLAG_DELTA_VARINT）
=======================================================================

ファイルの大半を占める配列（語彙キー・CSC rowind・col_len・GBDT cats）は
**どれもソート済み**なので、隣との差分は小さく、LEB128 varint（7bit/byte、
下位バイト先行、最上位ビット = 継続）で書くと汎用圧縮なしで半分になる
（実測 w4 6.3 → 3.4MiB、w7 18.4 → 9.5MiB）。減るのはフラッシュ上のサイズだけで、
ローダーは同じメモリ構造に展開するので RAM は変わらない。推論経路には触れない。

  ソート済み列 [a0, a1, a2, ...] → varint(a0), varint(a1 - a0), varint(a2 - a1), ...

  対象                        | 列の区切り
  ----------------------------|------------------------------------------
  語彙キー (uint64)            | セクションごと（セクション内は昇順・重複なし）
  CSC col_len (uint16)        | 差分なし。値ごとに varint（大半が 1 桁の小さい値）
  CSC rowind (uint16)         | 列ごと（col_len で区切る。列内はクラスID昇順）
  GBDT split_feature / n_cats | 差分なし。値ごとに varint
  GBDT cats (uint32)          | split ごと（昇順）

  それ以外（ヘッダ・セクションヘッダ・ラベル・quant_scale・CSC data・intercept・
  木の node_tag / default_left / leaf_value・辞書）は固定幅のまま。

差分符号化した配列は mmap でそのまま二分探索できないので、固定幅レイアウト
（flags bit1 = 0）も引き続き書ける（`--fixed-width`）。ローダーは両方読む。

.mbmf フォーマット（量子化前の float32 サイドカー）
====================================================

`.mbm` と量子化前の状態を比較するための補助フォーマット。セクション構成は
`.mbm` と完全に同一（ヘッダ flags・統合語彙の feature_id 暗黙化・GBDT カテゴリカルの
埋め込みも同じ）だが、以下の2セクションだけ量子化せず float32 のまま格納する
（`quant_scale` は書かない）:

  [読みモデル重み（CSC・float32・量子化なし）]
    n_nonzero    : uint32     非ゼロ要素数
    colptr       : uint32 × (n_features + 1)
    rowind       : uint16 × n_nonzero     行インデックス = クラスID
    data         : float32 × n_nonzero

  [境界モデル（algo_tag==0x00 線形のときのみ float32・量子化なし）]
    data         : float32 × n_features   （クラス1の重みベクトル）
    intercept    : float32 × 2            （クラス0, クラス1）

  algo_tag==0x01（木）のときは `.mbm` と完全に同一バイト列（元々量子化していない）。

ファイルヘッダの magic は `MBMF`。**version は `.mbm` と同じ番号を共有する**
（両者はセクション構成を共通に保つ設計なので、採番を分けると「どちらの 0x02 か」
を常に意識する羽目になる）。区別は magic だけで行う。語彙テーブル・読みラベル
テーブル・人名辞書テーブル・単一漢字辞書テーブルは `.mbm` と全く同じバイト列。

なお `.mbmf` は 0x01 → 0x02 と独立採番していた時期があるが、0x05 で `.mbm` に
合流した（0x03/0x04 の `.mbmf` は存在しない）。
"""

import re
import struct
import zipfile
import tempfile
import os
from typing import Any, Tuple

import joblib
import numpy as np
from scipy import sparse

from .name_dict import NAME_DICT_FILENAME, parse_name_dict_text
from .utils import parse_single_char_dict_tsv


# =====================================================================
# ファイルヘッダ定数
# =====================================================================
# `.mbm` と `.mbmf` はセクション構成を共通に保つ設計なので、バージョン番号も
# 共有する（採番を分けると「どちらの 0x02 か」を常に意識する羽目になる）。
# 区別は magic だけで行う。
VERSION = 0x0A

# 統合語彙のキーを詰める際、コードポイント 1 個に割り当てるビット幅。
# Unicode の上限 U+10FFFF が収まる。Rust 側 vocab.rs の CP_BITS と一致。
CP_BITS = 21

# 境界モデルセクションの algo_tag（version 0x06 で追加）
BOUNDARY_ALGO_LINEAR = 0x00
BOUNDARY_ALGO_TREE = 0x01

# ヘッダ flags バイト（reserved[0]、version 0x07 で追加）。Rust 側 loader.rs の
# FLAG_VOCAB_HAS_CAT と一致。bit0 = 統合語彙が GBDT カテゴリカル (column, code) を持つ。
FLAG_VOCAB_HAS_CAT = 0x01
# bit1 = ソート済み配列（語彙キー・CSC rowind/col_len・GBDT cats）を差分 + LEB128 varint で
# 書く（version 0x0A）。Rust 側 loader.rs の FLAG_DELTA_VARINT と一致。
FLAG_DELTA_VARINT = 0x02

# カテゴリカル列なしの番兵（Rust 側 loader.rs の NO_CAT_COLUMN と一致）。
NO_CAT_COLUMN = 0xFFFFFFFF

MAGIC_MBM = b"MOMO"
MAGIC_MBMF = b"MBMF"

# 行インデックス (rowind) を uint16 で書くため、n_classes はこれを超えられない。
# Rust 側 loader.rs の MAX_CLASSES と同じ値。
MAX_CLASSES = 0xFFFF + 1


# =====================================================================
# CharType 文字列 → uint8 対応表
# =====================================================================
CHARTYPE_TO_INT: dict[str, int] = {
    "SPACE": 0x00,
    "ALPHA": 0x10,
    "NUM": 0x11,  # Python 側は CharType.NUMERIC.value == 'NUM'
    "SYMBOL": 0x30,
    "SYMBOL_CLOSE": 0x31,
    "SYMBOL_OPEN": 0x32,
    "SYMBOL_STOP": 0x33,
    "SYMBOL_PAUSE": 0x34,
    "HIRAGANA": 0x40,
    "KATAKANA": 0x41,
    "KANJI": 0x42,
    "JAPANESE_NUMERIC": 0x43,
    "OTHER": 0xFF,
}


# =====================================================================
# FeatureType
#
# bit7-6 : ペイロード種別
#   0b00 (0x0_) = なし
#   0b01 (0x4_-0x7_) = CharType
#   0b10 (0x8_-0xB_) = char32_t
#   0b11 (0xC_) = uint8
#
# bit5-4 : 個数（CharType または char32_t の個数）
#   0b00 = 0 個（または uint8×1 固定）
#   0b01 = 1 個
#   0b10 = 2 個
#   0b11 = 3 個
# =====================================================================
class FT:
    BIAS = 0x00
    KANJI_POS_FIRST = 0x01

    TYPE_SELF = 0x50  # CharType×1
    TYPE_PREV1 = 0x51
    TYPE_PREV2 = 0x52
    TYPE_NEXT1 = 0x53
    TYPE_NEXT2 = 0x54
    TYPE_PREV3 = 0x55
    TYPE_NEXT3 = 0x56

    TYPE_TRANSITION = 0x60  # CharType×2

    TYPE_TRI_PREV2_PREV1_SELF = 0x70  # CharType×3  (前2-前1-対象)
    TYPE_TRI_PREV1_SELF_NEXT1 = 0x71  # CharType×3  (前1-対象-後1)
    TYPE_TRI_SELF_NEXT1_NEXT2 = 0x72  # CharType×3  (対象-後1-後2)
    TYPE_TRI_PREV3_PREV2_PREV1 = 0x73  # CharType×3  (前3-前2-前1)
    TYPE_TRI_NEXT1_NEXT2_NEXT3 = 0x74  # CharType×3  (後1-後2-後3)

    CHAR_SELF = 0x90  # char32_t×1
    CHAR_PREV1 = 0x91
    CHAR_PREV2 = 0x92
    CHAR_NEXT1 = 0x93
    CHAR_NEXT2 = 0x94
    CHAR_PREV3 = 0x95
    CHAR_NEXT3 = 0x96

    BIGRAM_PREV1_SELF = 0xA0  # char32_t×2
    BIGRAM_PREV2_PREV1 = 0xA1
    BIGRAM_SELF_NEXT1 = 0xA2
    BIGRAM_NEXT1_NEXT2 = 0xA3
    BIGRAM_PREV3_PREV2 = 0xA4
    BIGRAM_NEXT2_NEXT3 = 0xA5
    CHAR_SELF_COMPOUND_2 = 0xA6  # char32_t×2: 2文字複合ユニットの char_s

    TRIGRAM_PREV2_PREV1_SELF = 0xB0  # char32_t×3  (前2-前1-対象)
    TRIGRAM_PREV1_SELF_NEXT1 = 0xB1  # char32_t×3  (前1-対象-後1)
    TRIGRAM_SELF_NEXT1_NEXT2 = 0xB2  # char32_t×3  (対象-後1-後2)
    TRIGRAM_PREV3_PREV2_PREV1 = 0xB3  # char32_t×3  (前3-前2-前1)
    TRIGRAM_NEXT1_NEXT2_NEXT3 = 0xB4  # char32_t×3  (後1-後2-後3)
    CHAR_SELF_COMPOUND_3 = 0xB5  # char32_t×3: 3文字複合ユニットの char_s

    KANJI_RUN_LEN = 0xC0  # uint8
    JAPANESE_NUMERIC_RUN_LEN = 0xC1
    PREV_JAPANESE_NUMERIC_RUN_LEN = 0xC2
    NAME_FLAG_SELF = 0xC3  # uint8: 1=B, 2=I
    NAME_FLAG_PREV1 = 0xC4
    NAME_FLAG_NEXT1 = 0xC5


def chartype_count(ft: int) -> int:
    if (ft & 0xC0) != 0x40:
        return 0
    return (ft >> 4) & 0x03


def char32_count(ft: int) -> int:
    if (ft & 0xC0) != 0x80:
        return 0
    return (ft >> 4) & 0x03


def is_uint8_payload(ft: int) -> bool:
    return (ft & 0xC0) == 0xC0


# =====================================================================
# Python 特徴量キー文字列 → (FeatureType, chartype_list, char32_list, uint8_val)
# =====================================================================

# run_len の文字列値 → uint8
_RUN_LEN_MAP = {"1": 1, "2": 2, "3": 3, "4": 4, "5+": 5}

# 人名フラグの文字列値 → uint8
_NAME_FLAG_MAP = {"B": 1, "I": 2}


def parse_feature_key(key: str) -> Tuple[int, list, list, int | None]:
    """
    DictVectorizer の vocabulary_ キーを解析して
    (feature_type, chartype_vals, char32_vals, uint8_val) を返す。

    chartype_vals : list[int]   CharType の uint8 値列
    char32_vals   : list[int]   Unicode コードポイント列
    uint8_val     : int | None  run_len 系のときのみ
    """

    # --- ペイロードなし系 ---
    if key == "bias":
        return FT.BIAS, [], [], None
    if key == "kanji_pos_first":
        return FT.KANJI_POS_FIRST, [], [], None

    # --- run 系（uint8ペイロード）---
    m = re.fullmatch(r"kanji_run=(.+)", key)
    if m:
        return FT.KANJI_RUN_LEN, [], [], _RUN_LEN_MAP[m.group(1)]

    m = re.fullmatch(r"jnum_run=(.+)", key)
    if m:
        return FT.JAPANESE_NUMERIC_RUN_LEN, [], [], _RUN_LEN_MAP[m.group(1)]

    m = re.fullmatch(r"jnum_run_p1=(.+)", key)
    if m:
        return FT.PREV_JAPANESE_NUMERIC_RUN_LEN, [], [], _RUN_LEN_MAP[m.group(1)]

    # --- 人名フラグ系（uint8ペイロード: 1=B, 2=I）---
    m = re.fullmatch(r"name_s=([BI])", key)
    if m:
        return FT.NAME_FLAG_SELF, [], [], _NAME_FLAG_MAP[m.group(1)]

    m = re.fullmatch(r"name_p1=([BI])", key)
    if m:
        return FT.NAME_FLAG_PREV1, [], [], _NAME_FLAG_MAP[m.group(1)]

    m = re.fullmatch(r"name_n1=([BI])", key)
    if m:
        return FT.NAME_FLAG_NEXT1, [], [], _NAME_FLAG_MAP[m.group(1)]

    # --- type_tri （CharType×3）---
    m = re.fullmatch(r"type_tri_p2_p1_s=(.+)-(.+)-(.+)", key)
    if m:
        cts = [CHARTYPE_TO_INT[m.group(i)] for i in (1, 2, 3)]
        return FT.TYPE_TRI_PREV2_PREV1_SELF, cts, [], None

    m = re.fullmatch(r"type_tri_p1_s_n1=(.+)-(.+)-(.+)", key)
    if m:
        cts = [CHARTYPE_TO_INT[m.group(i)] for i in (1, 2, 3)]
        return FT.TYPE_TRI_PREV1_SELF_NEXT1, cts, [], None

    m = re.fullmatch(r"type_tri_s_n1_n2=(.+)-(.+)-(.+)", key)
    if m:
        cts = [CHARTYPE_TO_INT[m.group(i)] for i in (1, 2, 3)]
        return FT.TYPE_TRI_SELF_NEXT1_NEXT2, cts, [], None

    m = re.fullmatch(r"type_tri_p3_p2_p1=(.+)-(.+)-(.+)", key)
    if m:
        cts = [CHARTYPE_TO_INT[m.group(i)] for i in (1, 2, 3)]
        return FT.TYPE_TRI_PREV3_PREV2_PREV1, cts, [], None

    m = re.fullmatch(r"type_tri_n1_n2_n3=(.+)-(.+)-(.+)", key)
    if m:
        cts = [CHARTYPE_TO_INT[m.group(i)] for i in (1, 2, 3)]
        return FT.TYPE_TRI_NEXT1_NEXT2_NEXT3, cts, [], None

    # --- type_trans_p1_s （CharType×2）---
    m = re.fullmatch(r"type_trans_p1_s=(.+)->(.+)", key)
    if m:
        cts = [CHARTYPE_TO_INT[m.group(1)], CHARTYPE_TO_INT[m.group(2)]]
        return FT.TYPE_TRANSITION, cts, [], None

    # --- type 系（CharType×1）---
    m = re.fullmatch(r"type_s=(.+)", key)
    if m:
        return FT.TYPE_SELF, [CHARTYPE_TO_INT[m.group(1)]], [], None
    m = re.fullmatch(r"type_p1=(.+)", key)
    if m:
        return FT.TYPE_PREV1, [CHARTYPE_TO_INT[m.group(1)]], [], None
    m = re.fullmatch(r"type_p2=(.+)", key)
    if m:
        return FT.TYPE_PREV2, [CHARTYPE_TO_INT[m.group(1)]], [], None
    m = re.fullmatch(r"type_n1=(.+)", key)
    if m:
        return FT.TYPE_NEXT1, [CHARTYPE_TO_INT[m.group(1)]], [], None
    m = re.fullmatch(r"type_n2=(.+)", key)
    if m:
        return FT.TYPE_NEXT2, [CHARTYPE_TO_INT[m.group(1)]], [], None
    m = re.fullmatch(r"type_p3=(.+)", key)
    if m:
        return FT.TYPE_PREV3, [CHARTYPE_TO_INT[m.group(1)]], [], None
    m = re.fullmatch(r"type_n3=(.+)", key)
    if m:
        return FT.TYPE_NEXT3, [CHARTYPE_TO_INT[m.group(1)]], [], None

    # --- trigram（char32_t×3）---
    m = re.fullmatch(r"tri_p2_p1_s=(.)(.)(.)", key)
    if m:
        cps = [ord(m.group(i)) for i in (1, 2, 3)]
        return FT.TRIGRAM_PREV2_PREV1_SELF, [], cps, None

    m = re.fullmatch(r"tri_p1_s_n1=(.)(.)(.)", key)
    if m:
        cps = [ord(m.group(i)) for i in (1, 2, 3)]
        return FT.TRIGRAM_PREV1_SELF_NEXT1, [], cps, None

    m = re.fullmatch(r"tri_s_n1_n2=(.)(.)(.)", key)
    if m:
        cps = [ord(m.group(i)) for i in (1, 2, 3)]
        return FT.TRIGRAM_SELF_NEXT1_NEXT2, [], cps, None

    m = re.fullmatch(r"tri_p3_p2_p1=(.)(.)(.)", key)
    if m:
        cps = [ord(m.group(i)) for i in (1, 2, 3)]
        return FT.TRIGRAM_PREV3_PREV2_PREV1, [], cps, None

    m = re.fullmatch(r"tri_n1_n2_n3=(.)(.)(.)", key)
    if m:
        cps = [ord(m.group(i)) for i in (1, 2, 3)]
        return FT.TRIGRAM_NEXT1_NEXT2_NEXT3, [], cps, None

    # --- bigram（char32_t×2）---
    m = re.fullmatch(r"bi_p1_s=(.)(.)", key)
    if m:
        return FT.BIGRAM_PREV1_SELF, [], [ord(m.group(1)), ord(m.group(2))], None
    m = re.fullmatch(r"bi_p2_p1=(.)(.)", key)
    if m:
        return FT.BIGRAM_PREV2_PREV1, [], [ord(m.group(1)), ord(m.group(2))], None
    m = re.fullmatch(r"bi_s_n1=(.)(.)", key)
    if m:
        return FT.BIGRAM_SELF_NEXT1, [], [ord(m.group(1)), ord(m.group(2))], None
    m = re.fullmatch(r"bi_n1_n2=(.)(.)", key)
    if m:
        return FT.BIGRAM_NEXT1_NEXT2, [], [ord(m.group(1)), ord(m.group(2))], None
    m = re.fullmatch(r"bi_p3_p2=(.)(.)", key)
    if m:
        return FT.BIGRAM_PREV3_PREV2, [], [ord(m.group(1)), ord(m.group(2))], None
    m = re.fullmatch(r"bi_n2_n3=(.)(.)", key)
    if m:
        return FT.BIGRAM_NEXT2_NEXT3, [], [ord(m.group(1)), ord(m.group(2))], None

    # --- char_s（複合ユニット対応: 1〜3文字）---
    m = re.fullmatch(r"char_s=(.{3})", key)
    if m:
        return FT.CHAR_SELF_COMPOUND_3, [], [ord(c) for c in m.group(1)], None
    m = re.fullmatch(r"char_s=(.{2})", key)
    if m:
        return FT.CHAR_SELF_COMPOUND_2, [], [ord(c) for c in m.group(1)], None
    m = re.fullmatch(r"char_s=(.)", key)
    if m:
        return FT.CHAR_SELF, [], [ord(m.group(1))], None

    # --- char 系（char32_t×1）---
    m = re.fullmatch(r"char_p1=(.)", key)
    if m:
        return FT.CHAR_PREV1, [], [ord(m.group(1))], None
    m = re.fullmatch(r"char_p2=(.)", key)
    if m:
        return FT.CHAR_PREV2, [], [ord(m.group(1))], None
    m = re.fullmatch(r"char_n1=(.)", key)
    if m:
        return FT.CHAR_NEXT1, [], [ord(m.group(1))], None
    m = re.fullmatch(r"char_n2=(.)", key)
    if m:
        return FT.CHAR_NEXT2, [], [ord(m.group(1))], None
    m = re.fullmatch(r"char_p3=(.)", key)
    if m:
        return FT.CHAR_PREV3, [], [ord(m.group(1))], None
    m = re.fullmatch(r"char_n3=(.)", key)
    if m:
        return FT.CHAR_NEXT3, [], [ord(m.group(1))], None

    raise ValueError(f"未知の特徴量キー: {key!r}")


# =====================================================================
# int8 量子化
# =====================================================================


def quantize_to_int8(
    arr: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    float32 配列を int8 に量子化する。
    スケール係数は |最大値| / 127.0 。
    戻り値: (scale, int8_array)
    推論時のスコア: int8_val * scale ≈ 元の float32 値
    """
    max_abs = np.max(np.abs(arr))
    if max_abs == 0.0:
        return 1.0, np.zeros_like(arr, dtype=np.int8)
    scale = float(max_abs / 127.0)
    quantized = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return scale, quantized


def quantize_csr_per_row_to_int8(
    csr,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CSR 行列を行 (クラス) ごとに独立した scale で int8 量子化する。

    全体で 1 つの scale を使う `quantize_to_int8` と違い、行ごとに
    |最大値| / 127.0 を計算するため、重みの大きさがクラス間で偏っていても
    各クラスがほぼフルの int8 分解能を使える。

    戻り値: (scales[n_classes], int8_data[n_nonzero])
    推論時のスコア: data[j] * scales[row(j)] ≈ 元の float32 値
    """
    n_rows = csr.shape[0]
    data_f32 = csr.data.astype(np.float32)
    quantized = np.zeros_like(data_f32, dtype=np.int8)
    scales = np.zeros(n_rows, dtype=np.float32)
    for row in range(n_rows):
        start, end = csr.indptr[row], csr.indptr[row + 1]
        if start == end:
            scales[row] = 1.0
            continue
        scale, row_int8 = quantize_to_int8(data_f32[start:end])
        scales[row] = scale
        quantized[start:end] = row_int8
    return scales, quantized


# =====================================================================
# 差分 + varint 符号化（version 0x0A、flags bit1）
# =====================================================================


def _encode_varints(values) -> bytes:
    """非負整数列を LEB128 varint（7bit/byte、下位先行、MSB = 継続）で連結する。

    numpy でベクトル化してある。w7 の rowind は 400 万要素あり、Python のループでは
    秒単位かかるため。値ごとのバイト数を先に数え、出力の位置を前置和で決めてから、
    バイト位置 k ごとに一括で書く（最大 10 バイト = uint64）。
    """
    values = np.ascontiguousarray(values, dtype=np.uint64)
    if values.size == 0:
        return b""
    nbytes = np.ones(values.shape, dtype=np.int64)
    for k in range(1, 10):
        nbytes += values >= np.uint64(1) << np.uint64(7 * k)
    offsets = np.concatenate(([0], np.cumsum(nbytes)[:-1]))
    out = np.zeros(int(nbytes.sum()), dtype=np.uint8)
    for k in range(10):
        mask = nbytes > k
        if not mask.any():
            break
        chunk = (values[mask] >> np.uint64(7 * k)) & np.uint64(0x7F)
        cont = (nbytes[mask] > k + 1).astype(np.uint8) << np.uint8(7)
        out[offsets[mask] + k] = chunk.astype(np.uint8) | cont
    return out.tobytes()


def _encode_delta_varints(values, starts=None) -> bytes:
    """昇順の整数列を「先頭の値 + 隣との差分」の varint 列にする。

    `starts` を渡すと、その添字ごとに列を区切る（各区切りの先頭は差分でなく
    値そのものを書く）。CSC rowind を列ごとに、GBDT cats を split ごとに
    区切るときに使う。None なら全体を 1 本の列として扱う。
    """
    values = np.ascontiguousarray(values, dtype=np.uint64)
    if values.size == 0:
        return b""
    delta = values.copy()
    delta[1:] -= values[:-1]
    if starts is None:
        starts = np.zeros(1, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    starts = starts[starts < values.size]
    delta[starts] = values[starts]
    # 差分は uint64 で計算しているので、降順の箇所があれば巨大な値として現れる。
    if (delta > values.max()).any():
        raise ValueError("差分符号化する配列が区切り内で昇順になっていません")
    return _encode_varints(delta)


class VocabLayout:
    """統合語彙テーブル version 0x09 のレイアウト計画。

    v0x08 まではエントリごとに `feature_id` と `(cat_column, cat_code)` を書いていた
    （1 エントリ 13〜25 バイト）。0x09 では**採番し直して 3 つとも暗黙にする**ので、
    エントリは詰めた uint64 キー 8 バイトだけになる。

      feature_id = セクションの feature_id_base + セクション内の添字
                   （= 全体をキー順に並べたときの行位置）
      cat_code   = セクションの cat_code_base + セクション内の添字
      cat_column = セクションが 1 個だけ持つ

    そのぶん、書き出し側で 3 つの並べ替えが要る:

      1. CSC 重み行列の列を `old_fid_order` の順に並べ替える
      2. 線形境界の重みベクトルも同じ順に並べ替える
      3. GBDT の `cats` を `cat_remap` で新コードへ書き換える
    """

    __slots__ = ("sections", "old_fid_order", "cat_remap")

    def __init__(self, sections: list, old_fid_order: list, cat_remap: dict):
        self.sections = sections
        self.old_fid_order = old_fid_order
        self.cat_remap = cat_remap


def _build_csc_weight_bytes(
    csr, data, n_classes: int, n_features: int, col_order, delta: bool
) -> bytearray:
    """
    読みモデル重みを CSC 形式のバイト列にする（`.mbm` / `.mbmf` 共通）。

    `csr` の疎構造 (indptr/indices) と、それに整列した値配列 `data` を受け取り、
    列 (特徴量) 方向に転置して書き出す。`data` の dtype が そのまま書き出す値の型に
    なる（`.mbm` は int8、`.mbmf` は float32）。

    量子化 scale はクラス (行) ごとなので、量子化は行が連続する CSR で済ませて
    から、転置だけをここで行う。

    `col_order[new_fid] = old_fid` で列を並べ替える（version 0x09 で feature_id を
    キー順に採番し直したため）。

    レイアウト（version 0x09）:
      n_nonzero : uint32
      col_len   : uint16 × n_features   列ごとの非ゼロ数。読み手が前置和して colptr にする
      rowind    : uint16 × n_nonzero    （行インデックス = クラスID）
      data      : dtype(data) × n_nonzero

    0x08 までは colptr を uint32 × (n_features+1) の累積和で書いていた。1 列の非ゼロ数は
    高々 n_classes（<= 65536）なので uint16 の列長で足り、w4 で 640KB・w7 で 2.16MB 減る。

    `delta=True`（version 0x0A、flags bit1）のときは col_len を値ごとの varint、
    rowind を列ごとの差分 varint 列で書く。data は据え置き。
    """
    if n_classes > MAX_CLASSES:
        raise ValueError(
            f"n_classes={n_classes} が上限 {MAX_CLASSES} を超えています"
            f"（CSC の行インデックスを uint16 で書くため）"
        )

    csc = sparse.csr_matrix(
        (data, csr.indices, csr.indptr), shape=(n_classes, n_features)
    ).tocsc()
    csc = csc[:, np.asarray(col_order, dtype=np.int64)]
    csc.sort_indices()

    col_len = np.diff(csc.indptr)
    if col_len.max(initial=0) > 0xFFFF:
        raise ValueError(
            f"CSC の 1 列あたり非ゼロ数の最大 {int(col_len.max())} が uint16 を超えています"
        )

    out = bytearray()
    out += struct.pack("<I", csc.nnz)
    if delta:
        out += _encode_varints(col_len)
        out += _encode_delta_varints(csc.indices, starts=csc.indptr[:-1])
    else:
        out += col_len.astype("<u2").tobytes()
        out += csc.indices.astype("<u2").tobytes()  # rowind = クラスID
    out += csc.data.tobytes()
    return out


# =====================================================================
# .zip バンドルの読み込み（.mbm / .mbmf 共通）
# =====================================================================


def _load_bundle(zip_path: str) -> Tuple[Any, list, str]:
    """
    momo の .zip モデルから joblib バンドル・人名辞書・単一文字辞書を読み込む。

    戻り値: (bundle, name_entries, single_char_text)
      bundle            : joblib でロードした LRModelBundle
      name_entries      : [(表層形, ユニット別読み or None), ...]（辞書なしモデルは空）
      single_char_text  : 単一文字辞書 TSV の生テキスト
    """
    # 単一文字辞書のファイル名（モデルZIPへの同梱名・パッケージリソース名と共通）
    SINGLE_CHAR_DICT_FILENAME = "single_character_dic.tsv"
    tmp_dir = tempfile.mkdtemp()
    name_entries: list = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            import json

            version_info = json.loads(zf.read("version_info.json").decode("utf-8"))
            bundle_name = version_info["model_bundle"]
            zf.extract(bundle_name, tmp_dir)
            # 学習時に同梱された人名辞書（なければ空 = n_names 0 で書き出す）
            if NAME_DICT_FILENAME in zf.namelist():
                name_entries = parse_name_dict_text(
                    zf.read(NAME_DICT_FILENAME).decode("utf-8")
                )
            # 学習時に同梱された単一文字辞書（旧ZIPで無い場合はパッケージ内蔵で代替）
            if SINGLE_CHAR_DICT_FILENAME in zf.namelist():
                single_char_text = zf.read(SINGLE_CHAR_DICT_FILENAME).decode("utf-8")
            else:
                from importlib import resources

                single_char_text = (
                    resources.files("momo_py")
                    / f"resources/{SINGLE_CHAR_DICT_FILENAME}"
                ).read_text(encoding="utf-8")
                print(
                    "⚠️  ZIPに単一文字辞書が同梱されていないため、パッケージ内蔵の辞書を使用します。"
                )
        bundle = joblib.load(os.path.join(tmp_dir, bundle_name))
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
    return bundle, name_entries, single_char_text


def _build_tree_node_bytes(node: dict, cat_remap: dict, delta: bool) -> bytes:
    """LightGBMの木構造（dump_model()の1ノード）を再帰的にバイト列へ変換する。

    `cat_remap[(column, old_code)] = new_code` でカテゴリコードを version 0x09 の
    採番（キー順）へ書き換える。学習時に現れたが統合語彙に居場所がないコードは
    ありえない（`_build_cat_by_feature_id` が突合済み）ので、未知コードは失敗させる。

    `delta=True`（version 0x0A、flags bit1）のときは split_feature と n_cats を varint、
    cats を差分 varint 列で書く。node_tag / default_left / leaf_value は据え置き。
    """
    if "leaf_value" in node:
        return struct.pack("<Bf", 0, float(node["leaf_value"]))

    if node.get("decision_type") != "==":
        raise ValueError(
            "カテゴリカル分岐(decision_type='==')以外の分岐には対応していません: "
            f"decision_type={node.get('decision_type')!r}, split_feature={node.get('split_feature')}"
        )
    column = int(node["split_feature"])
    old_cats = [int(c) for c in str(node["threshold"]).split("||")]
    missing = [c for c in old_cats if (column, c) not in cat_remap]
    if missing:
        raise ValueError(
            f"境界モデルの木が持つカテゴリコード {missing} が列 {column} の"
            "統合語彙に存在しません（cat ⊆ 読み語彙 の前提が崩れています）"
        )
    cats = sorted(cat_remap[(column, c)] for c in old_cats)

    out = bytearray()
    out.append(1)  # node_tag: split
    if delta:
        out += _encode_varints([column])
        out.append(1 if node.get("default_left") else 0)
        out += _encode_varints([len(cats)])
        out += _encode_delta_varints(cats)
    else:
        out += struct.pack("<I", column)
        out.append(1 if node.get("default_left") else 0)
        out += struct.pack("<I", len(cats))
        for c in cats:
            out += struct.pack("<I", c)
    out += _build_tree_node_bytes(node["left_child"], cat_remap, delta)
    out += _build_tree_node_bytes(node["right_child"], cat_remap, delta)
    return bytes(out)


def _build_boundary_tree_bytes(booster, cat_remap: dict, delta: bool) -> bytes:
    """LightGBM Boosterの全木を再帰的にバイト列へ変換する。"""
    tree_info = booster.dump_model()["tree_info"]
    out = bytearray()
    out += struct.pack("<I", len(tree_info))
    for tree in tree_info:
        out += _build_tree_node_bytes(tree["tree_structure"], cat_remap, delta)
    return bytes(out)


def _build_boundary_bytes(
    bundle: Any, quantize: bool, layout: VocabLayout, delta: bool
) -> bytes:
    """境界モデルセクション（algo_tagプレフィックス付き）を構築する。

    quantize=True: .mbm 用（線形モデルはint8量子化）。
    quantize=False: .mbmf 用（線形モデルもfloat32のまま）。
    GBDT（木）は quantize に関わらず同一バイト列（量子化しない。leaf値・カテゴリ
    コードは元々小さい/離散的で量子化の対象にならないため）。
    """
    algo = getattr(bundle, "boundary_algo", "sgd")

    if algo == "sgd":
        model_b = bundle.model_boundary  # SGDClassifier
        # coef_ は (1, n_features) または (2, n_features)。modified_huber の2値分類
        # では coef_[0] がクラス1の重みベクトル。
        b_coef = model_b.coef_.astype(np.float32)
        if b_coef.ndim == 2:
            b_coef = b_coef[0]
        # 線形境界の重みは feature_id で引くので、CSC の列と同じ順に並べ替える。
        b_coef = b_coef[np.asarray(layout.old_fid_order, dtype=np.int64)]
        b_intercept = model_b.intercept_.astype(np.float32)  # shape: (2,) or (1,)
        if b_intercept.shape[0] == 1:
            # 2値分類で intercept が1要素のことがある
            b_intercept = np.array([0.0, float(b_intercept[0])], dtype=np.float32)

        out = bytearray()
        out.append(BOUNDARY_ALGO_LINEAR)
        if quantize:
            scale_b, b_int8 = quantize_to_int8(b_coef)
            out += struct.pack("<f", scale_b)
            out += bytes(b_int8.tobytes())
        else:
            out += bytes(b_coef.tobytes())
        out += struct.pack("<ff", b_intercept[0], b_intercept[1])
        return bytes(out)

    if algo == "gbdt":
        # version 0x07: cat_vocab は統合語彙テーブルに吸収したので、境界セクションは
        # n_columns と木だけ。カテゴリカル (column, code) は _build_vocab_bytes 側が書く。
        out = bytearray()
        out.append(BOUNDARY_ALGO_TREE)
        out += struct.pack("<I", len(bundle.boundary_cat_names))  # n_columns
        out += _build_boundary_tree_bytes(
            bundle.model_boundary.booster_, layout.cat_remap, delta
        )
        return bytes(out)

    raise ValueError(f"未知の boundary_algo です: {algo!r}")


# =====================================================================
# セクションのバイナリ構築（.mbm / .mbmf 共通）
# =====================================================================


def _canon_feature_key(key: str):
    """特徴量キー文字列を正準タプル (ft, ct_tuple, cp_tuple, u8) に落とす。

    読み語彙のキー文字列（例 "char_s=漢"）とカテゴリカル語彙のキー文字列
    （"{name}={value}"）は生の文字列が微妙に違っても、正準化すると同じキーは
    一致する（Rust 側の FeatureKey と同じ比較基準）。cat ⊆ 読み の突合に使う。
    """
    ft, ct_vals, cp_vals, u8_val = parse_feature_key(key)
    return (ft, tuple(ct_vals), tuple(cp_vals), u8_val)


def _build_cat_by_feature_id(vocab: dict, cat_names: list, cat_vocabs: dict) -> dict:
    """GBDT カテゴリカル `(column, code)` を feature_id ごとに引く辞書を作る。

    統合語彙テーブル（version 0x07）は**カテゴリカルキーが読み語彙の部分集合**である
    ことを前提とする。読み語彙に無いカテゴリカルキーがあれば、統合語彙に居場所が
    ないので `ValueError` で明示的に失敗させる（黙って欠損扱いにすると境界判定が
    静かに壊れるため）。
    """
    key_to_id = {}
    for key_str, fid in vocab.items():
        key_to_id[_canon_feature_key(key_str)] = fid

    cat_by_id: dict = {}
    for col_idx, name in enumerate(cat_names):
        for value_str, code in cat_vocabs[name].items():
            try:
                canon = _canon_feature_key(f"{name}={value_str}")
            except (ValueError, KeyError):
                canon = _canon_feature_key(name)
            fid = key_to_id.get(canon)
            if fid is None:
                raise ValueError(
                    f"境界モデルのカテゴリカルキー {name}={value_str!r} が読み語彙に"
                    "存在しません。統合語彙（version 0x07）はカテゴリカルキーが読み語彙の"
                    "部分集合であることを前提とします（読み・境界で特徴量抽出を"
                    "共有しているか確認してください）。"
                )
            cat_by_id[fid] = (col_idx, code)
    return cat_by_id


def _unified_cat(bundle: Any, vocab: dict, boundary_algo: str) -> Tuple[dict | None, int]:
    """統合語彙に埋め込むカテゴリカル写像とヘッダ flags を決める。

    GBDT 境界のときだけ `({feature_id: (column, code)}, FLAG_VOCAB_HAS_CAT)` を返す。
    線形境界のときは `(None, 0x00)`（統合語彙にカテゴリカルを書かない）。
    """
    if boundary_algo == "gbdt":
        cat_by_id = _build_cat_by_feature_id(
            vocab, bundle.boundary_cat_names, bundle.boundary_cat_vocabs
        )
        return cat_by_id, FLAG_VOCAB_HAS_CAT
    return None, 0x00


def _pack_vocab_key(ft: int, ct_vals: list, cp_vals: list, u8_val: int | None) -> int:
    """統合語彙のキーのペイロードを uint64 に詰める（version 0x09）。

    Rust 側 `vocab.rs::pack_key` と同じ規則。FeatureType のビットフィールド上、
    ペイロード種別（CharType×N / char32×M / uint8×1 / なし）は排他なので、
    どのタイプでも uint64 1 個で表せる。

      char32×M   : cp[0] を上位に CP_BITS ずつ（M <= 3 なので 63bit）
      chartype×N : ct[0] を上位に 8bit ずつ（N <= 3）
      uint8×1    : u8_val
      なし        : 0

    上位から詰めるので、同一タイプ内では詰めた値の昇順が
    `_vocab_row_sort_key` の順序と一致する。
    """
    m = char32_count(ft)
    if m:
        v = 0
        for i in range(m):
            cp = int(cp_vals[i])
            if not 0 <= cp < (1 << CP_BITS):
                raise ValueError(
                    f"コードポイント U+{cp:X} が {CP_BITS}bit に収まりません"
                    f"（特徴量タイプ 0x{ft:02X}）"
                )
            v = (v << CP_BITS) | cp
        return v

    n = chartype_count(ft)
    if n:
        v = 0
        for i in range(n):
            ct = int(ct_vals[i])
            if not 0 <= ct < 256:
                raise ValueError(f"CharType 値 {ct} が uint8 に収まりません")
            v = (v << 8) | ct
        return v

    if is_uint8_payload(ft):
        return int(u8_val or 0)

    return 0


def _vocab_row_sort_key(ft: int, u8_val: int | None, ct_vals: list, cp_vals: list) -> tuple:
    """統合語彙テーブルの行を並べる正準ソートキー。

    Rust `FeatureKey` の `derive(Ord)` と同じフィールド優先順位
    `(feature_type, u8val, ct[0..3], cp[0..3])` で比較できるよう、同じ順で
    タプルを組む（`_canon_feature_key()` はカテゴリカルキーの一致判定用の
    別の並びを返すため、ソート専用にこちらを使う）。

    `FeatureType`/`CharType` は Rust 側で宣言順と `#[repr(u8)]` 値の昇順が
    一致するよう維持されている（`char_type.rs` に明示コメントあり）ため、
    ここでの単純な数値タプル比較が Rust の `Ord` と同じ順序になる。
    """
    return (ft, u8_val if u8_val is not None else 0, tuple(ct_vals), tuple(cp_vals))


def _plan_vocab_layout(vocab: dict, cat_by_id: dict | None) -> VocabLayout:
    """`{key_str: feature_id}` から version 0x09 のレイアウトを組み立てる。

    `cat_by_id`（GBDT のとき `{feature_id: (column, code)}`）を渡すと、
    カテゴリカルコードもキー順に採番し直す。
    """
    rows = []
    for key_str, fid in vocab.items():
        try:
            ft, ct_vals, cp_vals, u8_val = parse_feature_key(key_str)
        except (ValueError, KeyError) as e:
            raise ValueError(
                f"特徴量キーの解析に失敗しました: {key_str!r} ({e!r})。"
                "学習TSVの列ズレや不正な文字種が混入していないか確認してください。"
            ) from None
        rows.append(
            (
                _vocab_row_sort_key(ft, u8_val, ct_vals, cp_vals),
                ft,
                _pack_vocab_key(ft, ct_vals, cp_vals, u8_val),
                int(fid),
            )
        )

    rows.sort(key=lambda r: r[0])

    # 列ごとのコードは「全体をキー順に走査した順」で採番する。1 つの列を複数の
    # タイプが共有する場合（char_s = CharSelf 0x90 と CharSelfCompound2 0xA6）でも、
    # 各タイプのぶんは連続した区間になるので `cat_code_base + 添字` で表せる。
    next_code: dict = {}
    cat_remap: dict = {}

    sections: list = []
    old_fid_order: list = []
    # 行ごとに割り当てた新コード（None = カテゴリカルなし）。つじつま確認に使う。
    new_code_of_row: list = []
    cur = None
    for _sort_key, ft, packed, old_fid in rows:
        old_fid_order.append(old_fid)

        column, old_code = (NO_CAT_COLUMN, 0)
        if cat_by_id is not None:
            column, old_code = cat_by_id.get(old_fid, (NO_CAT_COLUMN, 0))

        new_code = 0
        if column != NO_CAT_COLUMN:
            new_code = next_code.get(column, 0)
            next_code[column] = new_code + 1
            cat_remap[(column, old_code)] = new_code
            new_code_of_row.append(new_code)
        else:
            new_code_of_row.append(None)

        if cur is None or cur["feature_type"] != ft:
            cur = {
                "feature_type": ft,
                "cat_column": column,
                "cat_code_base": new_code,
                "feature_id_base": len(old_fid_order) - 1,
                "keys": [],
            }
            sections.append(cur)
        elif cur["cat_column"] != column:
            # 「列は feature_type ごとに 1 個」が崩れると、セクションが列を 1 個だけ
            # 持つ設計が成立しない。黙って一方を捨てず、明示的に失敗させる。
            raise ValueError(
                f"特徴量タイプ 0x{ft:02X} に複数のカテゴリカル列 "
                f"({cur['cat_column']} と {column}) が対応しています。"
                "momo_py.categorical は特徴量名ごとに 1 列を作る前提です。"
            )

        if cur["keys"] and packed <= cur["keys"][-1]:
            raise ValueError(
                f"特徴量タイプ 0x{ft:02X} のキーが昇順になっていないか重複しています "
                f"(0x{packed:016X})"
            )
        cur["keys"].append(packed)

    # 採番のつじつまを確認する。読み手（Rust の vocab.rs）は
    # `feature_id = feature_id_base + 添字`・`cat_code = cat_code_base + 添字` を
    # 前提に引くので、ここが崩れると推論結果が静かにずれる。
    for sec in sections:
        base_fid = sec["feature_id_base"]
        n = len(sec["keys"])
        if sec["cat_column"] == NO_CAT_COLUMN:
            continue
        actual = new_code_of_row[base_fid : base_fid + n]
        expected = list(range(sec["cat_code_base"], sec["cat_code_base"] + n))
        if actual != expected:
            raise ValueError(
                f"特徴量タイプ 0x{sec['feature_type']:02X} のカテゴリカルコードが"
                f"連続した区間になりませんでした（採番ロジックの不整合）: {actual[:8]}..."
            )

    if len(old_fid_order) != len(vocab):
        raise ValueError(
            f"統合語彙の行数 {len(old_fid_order)} が語彙サイズ {len(vocab)} と一致しません"
        )

    return VocabLayout(sections, old_fid_order, cat_remap)


def _build_vocab_bytes(layout: VocabLayout, delta: bool = False) -> bytes:
    """統合語彙テーブル（version 0x09）のバイト列を作る。

    レイアウト:
      n_sections : uint32
      以下 n_sections 個（feature_type 昇順）:
        feature_type  : uint8
        _pad          : uint8[3]
        count         : uint32
        cat_column    : uint32   0xFFFFFFFF = 列なし
        cat_code_base : uint32
      続いて、各セクションのキー配列（uint64 × count）をセクションの順に連結
      （`delta=True` のときはセクションごとの差分 varint 列）
    """
    out = bytearray()
    out += struct.pack("<I", len(layout.sections))
    for sec in layout.sections:
        out += struct.pack(
            "<BBBBIII",
            sec["feature_type"],
            0,
            0,
            0,
            len(sec["keys"]),
            sec["cat_column"],
            sec["cat_code_base"],
        )
    for sec in layout.sections:
        if delta:
            out += _encode_delta_varints(sec["keys"])
        else:
            out += np.asarray(sec["keys"], dtype="<u8").tobytes()
    return bytes(out)


def _build_label_bytes(read_classes) -> bytes:
    """読みラベルテーブルのバイト列を構築する。"""
    label_bytes = bytearray()
    for label in read_classes:
        encoded = label.encode("utf-8")
        assert len(encoded) <= 255, f"ラベルが長すぎます: {label!r}"
        label_bytes.append(len(encoded))
        label_bytes += encoded
    return bytes(label_bytes)


def _build_name_dict_bytes(name_entries) -> bytes:
    """人名辞書テーブルのバイト列を構築する。"""
    name_dict_bytes = bytearray()
    name_dict_bytes += struct.pack("<I", len(name_entries))
    for surface, readings in name_entries:
        encoded = surface.encode("utf-8")
        assert len(encoded) <= 255, f"人名が長すぎます: {surface!r}"
        name_dict_bytes.append(len(encoded))
        name_dict_bytes += encoded
        if readings is None:
            name_dict_bytes.append(0)
        else:
            assert len(readings) <= 255
            name_dict_bytes.append(len(readings))
            for reading in readings:
                r_enc = reading.encode("utf-8")
                assert len(r_enc) <= 255, f"読みが長すぎます: {reading!r}"
                name_dict_bytes.append(len(r_enc))
                name_dict_bytes += r_enc
    return bytes(name_dict_bytes)


def _build_single_char_dict_bytes(single_char_dict: dict) -> bytes:
    """単一文字辞書テーブルのバイト列を構築する。"""
    single_char_dict_bytes = bytearray()
    single_char_dict_bytes += struct.pack("<I", len(single_char_dict))
    for ch in sorted(single_char_dict):
        readings = single_char_dict[ch]
        encoded = ch.encode("utf-8")
        assert len(encoded) <= 255, f"文字キーが長すぎます: {ch!r}"
        single_char_dict_bytes.append(len(encoded))
        single_char_dict_bytes += encoded
        assert len(readings) <= 255, f"読みが多すぎます: {ch!r}"
        single_char_dict_bytes.append(len(readings))
        for reading in readings:
            r_enc = reading.encode("utf-8")
            assert len(r_enc) <= 255, f"読みが長すぎます: {reading!r}"
            single_char_dict_bytes.append(len(r_enc))
            single_char_dict_bytes += r_enc
    return bytes(single_char_dict_bytes)


def _write_sections(
    out_path: str,
    header: bytes,
    vocab_bytes: bytes,
    label_bytes: bytes,
    read_weight_bytes: bytes,
    intercept_r_bytes: bytes,
    boundary_bytes: bytes,
    name_dict_bytes: bytes,
    single_char_dict_bytes: bytes,
) -> None:
    print(f"💾 書き出し中: {out_path}")
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(vocab_bytes)
        f.write(label_bytes)
        f.write(read_weight_bytes)
        f.write(intercept_r_bytes)
        f.write(boundary_bytes)
        f.write(name_dict_bytes)
        f.write(single_char_dict_bytes)


# =====================================================================
# エクスポート本体
# =====================================================================


def export(zip_path: str, out_path: str, *, compact: bool = True) -> None:
    """
    momo の .zip モデルを C++/Rust 向け量子化バイナリ (.mbm) に変換して書き出す。

    `compact=True`（既定）はソート済み配列を差分 + varint で書く（flags bit1）。
    `compact=False` は固定幅レイアウト（mmap で借用ロードする道を残すため）。
    """
    print(f"📦 モデル読み込み中: {zip_path}")
    bundle, name_entries, single_char_text = _load_bundle(zip_path)

    vocab = bundle.vectorizer_read.vocabulary_  # {key_str: feature_id}
    coef_sparse = bundle.coef_read_sparse  # CSR (n_classes × n_features)
    intercept_r = bundle.intercept_read  # float32 (n_classes,)
    read_classes = bundle.read_classes  # str array (n_classes,)
    boundary_algo = getattr(bundle, "boundary_algo", "sgd")

    n_classes = len(read_classes)
    n_features = len(vocab)

    print(f"   クラス数    : {n_classes}")
    print(f"   特徴量次元数: {n_features}")

    # GBDT 境界のときだけ、カテゴリカル (column, code) を統合語彙に埋め込む。
    cat_by_id, flags = _unified_cat(bundle, vocab, boundary_algo)
    if compact:
        flags |= FLAG_DELTA_VARINT

    print(f"🔨 統合語彙テーブル変換中... (layout={'delta+varint' if compact else 'fixed'})")
    layout = _plan_vocab_layout(vocab, cat_by_id)
    vocab_bytes = _build_vocab_bytes(layout, delta=compact)

    print("🔨 読みラベルテーブル変換中...")
    label_bytes = _build_label_bytes(read_classes)

    # --- 読みモデル重み（CSR で行ごとに int8 量子化 → CSC に転置して書き出す）---
    print("🔨 読みモデル重み量子化中...")
    csr = coef_sparse.tocsr()
    scales_r, data_int8 = quantize_csr_per_row_to_int8(csr)

    read_weight_bytes = bytearray()
    read_weight_bytes += struct.pack(f"<{n_classes}f", *scales_r.tolist())
    read_weight_bytes += _build_csc_weight_bytes(
        csr, data_int8, n_classes, n_features, layout.old_fid_order, delta=compact
    )

    # --- 読みモデル intercept ---
    intercept_r_f32 = intercept_r.astype(np.float32)
    intercept_r_bytes = intercept_r_f32.tobytes()  # float32 × n_classes

    # --- 境界モデル（線形はint8量子化、GBDTは量子化なし）---
    print(f"🔨 境界モデル変換中... (algo={boundary_algo})")
    boundary_bytes = _build_boundary_bytes(
        bundle, quantize=True, layout=layout, delta=compact
    )

    print(f"🔨 人名辞書テーブル変換中... ({len(name_entries)} エントリ)")
    name_dict_bytes = _build_name_dict_bytes(name_entries)

    single_char_dict = parse_single_char_dict_tsv(single_char_text)
    print(f"🔨 単一文字辞書テーブル変換中... ({len(single_char_dict)} エントリ)")
    single_char_dict_bytes = _build_single_char_dict_bytes(single_char_dict)

    header = struct.pack(
        "<4sBBBBII",
        MAGIC_MBM,
        VERSION,
        flags,  # reserved[0] = flags
        0x00,
        0x00,  # reserved[1..2]
        n_classes,
        n_features,
    )

    _write_sections(
        out_path,
        header,
        vocab_bytes,
        label_bytes,
        read_weight_bytes,
        intercept_r_bytes,
        boundary_bytes,
        name_dict_bytes,
        single_char_dict_bytes,
    )

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"✅ 完了: {size_mb:.1f} MB")
    print(f"   語彙テーブル  : {len(vocab_bytes):>10,} bytes")
    print(f"   読みラベル    : {len(label_bytes):>10,} bytes")
    print(
        f"   読みモデル重み: {len(read_weight_bytes):>10,} bytes  (scale: min={scales_r.min():.6f} max={scales_r.max():.6f})"
    )
    print(f"   読み intercept: {len(intercept_r_bytes):>10,} bytes")
    print(f"   境界モデル    : {len(boundary_bytes):>10,} bytes  (algo={boundary_algo})")
    print(
        f"   人名辞書      : {len(name_dict_bytes):>10,} bytes  ({len(name_entries)} エントリ)"
    )
    print(
        f"   単一文字辞書  : {len(single_char_dict_bytes):>10,} bytes  ({len(single_char_dict)} エントリ)"
    )


def export_float(zip_path: str, out_path: str, *, compact: bool = True) -> None:
    """
    momo の .zip モデルを、量子化せず float32 のまま Rust 向けバイナリ (.mbmf) に
    変換して書き出す。`.mbm`（量子化後）と量子化前の状態を比較する用途のサイドカー。
    `compact` の意味は `export()` と同じ。

    セクション構成は `.mbm` と同一だが、読みモデル重み・境界モデル重みの2セクション
    だけ quant_scale を持たず、int8 の代わりに float32 でそのまま格納する。
    """
    print(f"📦 モデル読み込み中: {zip_path}")
    bundle, name_entries, single_char_text = _load_bundle(zip_path)

    vocab = bundle.vectorizer_read.vocabulary_
    coef_sparse = bundle.coef_read_sparse
    intercept_r = bundle.intercept_read
    read_classes = bundle.read_classes
    boundary_algo = getattr(bundle, "boundary_algo", "sgd")

    n_classes = len(read_classes)
    n_features = len(vocab)

    print(f"   クラス数    : {n_classes}")
    print(f"   特徴量次元数: {n_features}")

    # .mbm と同一の統合語彙（feature_id 暗黙 + 任意のカテゴリカル）。
    cat_by_id, flags = _unified_cat(bundle, vocab, boundary_algo)
    if compact:
        flags |= FLAG_DELTA_VARINT

    print(f"🔨 統合語彙テーブル変換中... (layout={'delta+varint' if compact else 'fixed'})")
    layout = _plan_vocab_layout(vocab, cat_by_id)
    vocab_bytes = _build_vocab_bytes(layout, delta=compact)

    print("🔨 読みラベルテーブル変換中...")
    label_bytes = _build_label_bytes(read_classes)

    # --- 読みモデル重み（CSC・float32・量子化なし）---
    print("🔨 読みモデル重み変換中 (float32、量子化なし)...")
    csr = coef_sparse.tocsr()
    data_f32 = csr.data.astype("<f4", copy=False)

    read_weight_bytes = _build_csc_weight_bytes(
        csr, data_f32, n_classes, n_features, layout.old_fid_order, delta=compact
    )

    # --- 読みモデル intercept ---
    intercept_r_f32 = intercept_r.astype(np.float32)
    intercept_r_bytes = intercept_r_f32.tobytes()

    # --- 境界モデル（線形はfloat32・量子化なし、GBDTは元々量子化なし）---
    print(f"🔨 境界モデル変換中... (algo={boundary_algo})")
    boundary_bytes = _build_boundary_bytes(
        bundle, quantize=False, layout=layout, delta=compact
    )

    print(f"🔨 人名辞書テーブル変換中... ({len(name_entries)} エントリ)")
    name_dict_bytes = _build_name_dict_bytes(name_entries)

    single_char_dict = parse_single_char_dict_tsv(single_char_text)
    print(f"🔨 単一文字辞書テーブル変換中... ({len(single_char_dict)} エントリ)")
    single_char_dict_bytes = _build_single_char_dict_bytes(single_char_dict)

    header = struct.pack(
        "<4sBBBBII",
        MAGIC_MBMF,
        VERSION,
        flags,  # reserved[0] = flags
        0x00,
        0x00,  # reserved[1..2]
        n_classes,
        n_features,
    )

    _write_sections(
        out_path,
        header,
        vocab_bytes,
        label_bytes,
        read_weight_bytes,
        intercept_r_bytes,
        boundary_bytes,
        name_dict_bytes,
        single_char_dict_bytes,
    )

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"✅ 完了: {size_mb:.1f} MB")
    print(f"   語彙テーブル  : {len(vocab_bytes):>10,} bytes")
    print(f"   読みラベル    : {len(label_bytes):>10,} bytes")
    print(
        f"   読みモデル重み: {len(read_weight_bytes):>10,} bytes  (float32、量子化なし)"
    )
    print(f"   読み intercept: {len(intercept_r_bytes):>10,} bytes")
    print(f"   境界モデル    : {len(boundary_bytes):>10,} bytes  (algo={boundary_algo}、量子化なし)")
    print(
        f"   人名辞書      : {len(name_dict_bytes):>10,} bytes  ({len(name_entries)} エントリ)"
    )
    print(
        f"   単一文字辞書  : {len(single_char_dict_bytes):>10,} bytes  ({len(single_char_dict)} エントリ)"
    )


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="momo モデル (.zip) を Rust/C++ 向けバイナリに変換する"
    )
    parser.add_argument("zip_path", help="入力: momo モデル ZIP ファイル")
    parser.add_argument("out_path", help="出力: バイナリファイル (.mbm または .mbmf)")
    parser.add_argument(
        "--float",
        action="store_true",
        help="量子化せず float32 のまま .mbmf として書き出す（.mbm との比較用サイドカー）",
    )
    parser.add_argument(
        "--fixed-width",
        action="store_true",
        help="ソート済み配列を差分 + varint で圧縮せず固定幅で書く（mmap 借用ロード向け）",
    )
    args = parser.parse_args()

    compact = not args.fixed_width
    if args.float:
        export_float(args.zip_path, args.out_path, compact=compact)
    else:
        export(args.zip_path, args.out_path, compact=compact)
