/// Unicode 点字（U+2800..=U+28FF）→ NABCC ASCII バイト変換テーブル。
///
/// インデックスは `codepoint - 0x2800`（ドットパターンのビット値）。
/// C++ 実装の `utfbrl_to_nabcc_table` に対応する。
/// 未定義パターンはスペース（b' '）。
static BRAILLE_TO_NABCC: &[u8] =
    b" a1b'k2l`cif/msp\"e3h9o6r~djg>ntq,*5<-u8v.%{$+x!&;:4|0z7(_?w}#y)= A B K L@CIF MSP E H O R DJG NTQ     U V  [  X      \\ Z    W] Y";

/// Unicode 点字文字を NABCC ASCII バイトに変換する（小文字版）。
///
/// U+2800..=U+28FF の範囲外、または未定義のドットパターンはスペース（b' '）を返す。
pub(crate) fn braille_to_nabcc(c: char) -> u8 {
    let cp = c as u32;
    if cp < 0x2800 || cp > 0x28FF {
        return b' ';
    }
    let idx = (cp - 0x2800) as usize;
    BRAILLE_TO_NABCC.get(idx).copied().unwrap_or(b' ')
}

/// Unicode 点字文字を Capital NABCC ASCII バイトに変換する。
///
/// 6 点のみを扱うプリンタ向け形式で広く使われる大文字版。
/// `braille_to_nabcc` の結果のうち `a`-`z` を `A`-`Z` に変換する。
pub(crate) fn braille_to_nabcc_capital(c: char) -> u8 {
    braille_to_nabcc(c).to_ascii_uppercase()
}
