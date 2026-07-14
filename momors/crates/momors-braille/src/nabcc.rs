//! Unicode 点字 → NABCC ASCII の符号化。

/// NABCC の英字ケース。
///
/// Braille ASCII（NABCC の基になった 64 文字集合）が定義するのは ASCII 0x20..=0x5F、
/// すなわち大文字のみで、小文字は本来この集合の外にある。よって既定は [`Upper`]。
///
/// 一方、大文字 NABCC のバイト列を通常のテキストエディタで開いて点字ディスプレイで
/// 読むと、全セルに dot7 が付加されて極端に読みにくくなる。この用途では [`Lower`] が要る。
///
/// [`Upper`]: NabccCase::Upper
/// [`Lower`]: NabccCase::Lower
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NabccCase {
    /// 大文字。Braille ASCII の規定どおり。BASE / JBCC もこちら。
    Upper,
    /// 小文字。点字ディスプレイでの読みやすさを優先する場合に使う。
    Lower,
}

/// Unicode 点字（U+2800..=U+28FF）→ NABCC ASCII バイト変換テーブル（小文字）。
///
/// インデックスは `codepoint - 0x2800`（ドットパターンのビット値）。
/// C++ 実装の `utfbrl_to_nabcc_table` に対応する。
/// 未定義パターンはスペース（b' '）。
static BRAILLE_TO_NABCC: &[u8] =
    b" a1b'k2l`cif/msp\"e3h9o6r~djg>ntq,*5<-u8v.%{$+x!&;:4|0z7(_?w}#y)= A B K L@CIF MSP E H O R DJG NTQ     U V  [  X      \\ Z    W] Y";

/// Unicode 点字文字を NABCC ASCII バイトに変換する。
///
/// U+2800..=U+28FF の範囲外、または未定義のドットパターンはスペース（b' '）を返す。
pub(crate) fn braille_to_nabcc(c: char, case: NabccCase) -> u8 {
    let cp = c as u32;
    if cp < 0x2800 || cp > 0x28FF {
        return b' ';
    }
    let idx = (cp - 0x2800) as usize;
    let b = BRAILLE_TO_NABCC.get(idx).copied().unwrap_or(b' ');
    match case {
        NabccCase::Upper => b.to_ascii_uppercase(),
        NabccCase::Lower => b,
    }
}
