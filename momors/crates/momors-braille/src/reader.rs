//! 点字ファイルの読み込み。
//!
//! 現在は BES バイナリ形式（[`writer::write_bes_file`](crate::writer) の逆変換）に対応する。
//!
//! BES は印刷イメージを保持する形式であり、ヘッダ行・強制/暗黙の改ページが
//! すべてファイル中に焼き込まれている。本リーダはバイト列をデコードし、
//! 編集可能な物理行リストへ復元する。方針:
//!
//! - 行末コード: `0x0D 0xFE` = 論理行終端、`0xFE` 単独 = 物理行のみ終端（論理行継続）、
//!   末尾の `0x0D` 単独 = ドキュメント終端（論理行終端扱い）。
//! - 改ページ `0xFD`: ページの生行数が `lines_per_page` に達していれば**暗黙改ページ**
//!   （リフォーマットで消えてよい）とみなして捨てる。`lines_per_page` 未満で終わって
//!   いれば**強制改ページ**として直前の物理行に印を付ける。
//! - ヘッダ行: 各ページ先頭行を位置で落とす。ただし数符 `⠼` を含む行に限定して誤爆を防ぐ。
//!   （タイトル・ページ番号の中身の復元は点字→数字の逆変換が入ってから。）

const HEADER_SIZE: usize = 512;
const EXT_HEADER_SIZE: usize = 512;
const PAGE_MARK: u8 = 0xFD;
const LINE_END: u8 = 0xFE;
const CR: u8 = 0x0D;
const EOF_MARK: u8 = 0xFF;
const CELL_BASE: u8 = 0xA0;
const CELL_MAX: u8 = 0xDF; // 0xA0 + 63
const BRAILLE_BASE: u32 = 0x2800;
const NUM_SIGN: char = '⠼'; // U+283C

/// BES から復元した1物理行。
#[derive(Debug, Clone, PartialEq)]
pub struct BesPhysicalLine {
    /// 点字セル文字列。
    pub content: String,
    /// この物理行で論理行（段落）が終わるか。
    pub logical_end: bool,
    /// この物理行の直後で強制改ページするか。
    pub page_break: bool,
}

/// BES から復元したドキュメント。
#[derive(Debug, Clone, PartialEq)]
pub struct BesDocument {
    /// 1行あたりのマス数（xl）。
    pub line_width: usize,
    /// 1ページあたりの行数（yl）。
    pub lines_per_page: usize,
    /// ヘッダ除去後、全ページ通しの物理行。
    pub lines: Vec<BesPhysicalLine>,
}

/// BES バイナリ列を読み込む。マジック不一致や明らかな破損では `None` を返す。
pub fn read_bes_file(bytes: &[u8]) -> Option<BesDocument> {
    if bytes.len() < HEADER_SIZE + EXT_HEADER_SIZE + 2 || &bytes[0..4] != b"%BET" {
        return None;
    }

    let line_width = parse_2digit(&bytes[0x16..0x18]).map(|v| v.saturating_sub(1))?;
    let lines_per_page = parse_2digit(&bytes[0x18..0x1A])?;
    if lines_per_page == 0 {
        return None;
    }

    // ヘッダ(512) + 拡張ヘッダ(512) の後にコントロールエリア。
    // writer は最小構成として 0xFF 1 バイトを書く。
    let mut pos = HEADER_SIZE + EXT_HEADER_SIZE;
    if pos < bytes.len() && bytes[pos] == EOF_MARK {
        pos += 1;
    }

    // サイズ前置きのページチャンクを順に読む。
    let mut pages: Vec<Vec<BesPhysicalLine>> = Vec::new();
    loop {
        if pos >= bytes.len() {
            break;
        }
        // 末尾の EOF マーカ（最終バイトの単独 0xFF）。
        if bytes[pos] == EOF_MARK && pos + 1 >= bytes.len() {
            break;
        }
        if pos + 2 > bytes.len() {
            break;
        }
        let size = bytes[pos] as usize | (bytes[pos + 1] as usize) << 8;
        if size < 3 {
            break;
        }
        let chunk_end = pos + size;
        if chunk_end > bytes.len() {
            break;
        }
        let chunk = &bytes[pos + 2..chunk_end];
        if chunk.first() != Some(&PAGE_MARK) {
            break;
        }
        pages.push(parse_page(&chunk[1..]));
        pos = chunk_end;
    }

    // 改ページの強制/暗黙判定 + ヘッダ行除去。
    let page_count = pages.len();
    let mut lines: Vec<BesPhysicalLine> = Vec::new();
    for (i, mut page) in pages.into_iter().enumerate() {
        let raw_count = page.len();

        // 各ページ先頭のヘッダ行を落とす（数符を含む行に限定）。
        if page
            .first()
            .map(|l| l.content.contains(NUM_SIGN))
            .unwrap_or(false)
        {
            page.remove(0);
        }

        // 生行数が lines_per_page 未満なら強制改ページ（最終ページは対象外）。
        let forced_break = i + 1 < page_count && raw_count < lines_per_page;
        if forced_break {
            if let Some(last) = page.last_mut() {
                last.page_break = true;
            }
        }

        lines.extend(page);
    }

    Some(BesDocument {
        line_width,
        lines_per_page,
        lines,
    })
}

/// ページチャンク（先頭 `0xFD` を除いた本体）を物理行に分解する。
fn parse_page(body: &[u8]) -> Vec<BesPhysicalLine> {
    let mut lines = Vec::new();
    let mut content = String::new();
    let mut saw_cr = false;

    for &b in body {
        match b {
            CELL_BASE..=CELL_MAX => {
                content.push(char::from_u32(BRAILLE_BASE + (b - CELL_BASE) as u32).unwrap());
            }
            CR => saw_cr = true,
            LINE_END => {
                lines.push(BesPhysicalLine {
                    content: std::mem::take(&mut content),
                    logical_end: saw_cr,
                    page_break: false,
                });
                saw_cr = false;
            }
            _ => {} // その他の制御バイトは無視
        }
    }

    // 末尾の 0x0D 単独（FE 省略）で終わる行を取りこぼさない。
    if !content.is_empty() || saw_cr {
        lines.push(BesPhysicalLine {
            content,
            logical_end: saw_cr,
            page_break: false,
        });
    }

    lines
}

/// 2 桁 ASCII 数字を usize に変換する。数字以外は `None`。
fn parse_2digit(b: &[u8]) -> Option<usize> {
    if b.len() != 2 || !b[0].is_ascii_digit() || !b[1].is_ascii_digit() {
        return None;
    }
    Some(((b[0] - b'0') * 10 + (b[1] - b'0')) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formatter::{BrailleFormatter, FormatterConfig};
    use crate::writer::OutputFormat;

    fn braille_str(n: usize) -> String {
        std::iter::repeat('⠁').take(n).collect()
    }

    /// テスト用 BES バイト列を組み立てる。
    /// `pages`: 各ページの (内容, logical_end) のリスト。
    fn make_bes(line_width: usize, lines_per_page: usize, pages: &[Vec<(&str, bool)>]) -> Vec<u8> {
        let mut out = vec![b' '; HEADER_SIZE];
        out[0..4].copy_from_slice(b"%BET");
        let xl_enc = line_width + 1;
        out[0x16] = b'0' + (xl_enc / 10) as u8;
        out[0x17] = b'0' + (xl_enc % 10) as u8;
        out[0x18] = b'0' + (lines_per_page / 10) as u8;
        out[0x19] = b'0' + (lines_per_page % 10) as u8;
        out.extend(std::iter::repeat(b' ').take(EXT_HEADER_SIZE));
        out.push(EOF_MARK); // コントロールエリア

        let last_page = pages.len().saturating_sub(1);
        for (pi, page) in pages.iter().enumerate() {
            let mut buf = vec![PAGE_MARK];
            let last_line = page.len().saturating_sub(1);
            for (li, (text, logical_end)) in page.iter().enumerate() {
                for ch in text.chars() {
                    let pattern = (ch as u32).saturating_sub(BRAILLE_BASE) as u8;
                    buf.push(CELL_BASE + pattern);
                }
                if pi == last_page && li == last_line {
                    buf.push(CR);
                } else if *logical_end {
                    buf.push(CR);
                    buf.push(LINE_END);
                } else {
                    buf.push(LINE_END);
                }
            }
            let size = 2 + buf.len();
            out.push((size & 0xFF) as u8);
            out.push(((size >> 8) & 0xFF) as u8);
            out.extend_from_slice(&buf);
        }
        out.push(EOF_MARK);
        out
    }

    #[test]
    fn rejects_non_bes() {
        assert!(read_bes_file(b"not a bes file at all............").is_none());
    }

    #[test]
    fn reads_xl_yl_from_header() {
        let bytes = make_bes(32, 22, &[vec![("⠁⠃⠉", true)]]);
        let doc = read_bes_file(&bytes).unwrap();
        assert_eq!(doc.line_width, 32);
        assert_eq!(doc.lines_per_page, 22);
    }

    #[test]
    fn maps_line_terminators() {
        // 物理行継続(FE) → logical_end=false、論理行終端(0D FE) → true
        let bytes = make_bes(32, 22, &[vec![("⠁⠃", false), ("⠉⠙", true), ("⠑⠋", true)]]);
        let doc = read_bes_file(&bytes).unwrap();
        assert_eq!(doc.lines.len(), 3);
        assert_eq!(doc.lines[0].content, "⠁⠃");
        assert!(!doc.lines[0].logical_end); // 物理行のみ終端 → 論理行は継続
        assert!(doc.lines[1].logical_end);
        assert!(doc.lines[2].logical_end); // 最終行（0D 単独）も論理行終端
    }

    #[test]
    fn drops_header_line_with_number_sign() {
        // 先頭行に数符を含むヘッダを置く → 落とされる
        let header = format!("⠞⠊{}⠼⠁", "⠀".repeat(3));
        let bytes = make_bes(32, 22, &[vec![(header.as_str(), true), ("⠁⠃⠉", true)]]);
        let doc = read_bes_file(&bytes).unwrap();
        assert_eq!(doc.lines.len(), 1);
        assert_eq!(doc.lines[0].content, "⠁⠃⠉");
    }

    #[test]
    fn full_page_break_is_implicit() {
        // ページ0 が lines_per_page=3 行ちょうど（満杯）→ 改ページは暗黙 → page_break なし
        let bytes = make_bes(
            32,
            3,
            &[
                vec![("⠁", false), ("⠃", false), ("⠉", true)],
                vec![("⠙", true)],
            ],
        );
        let doc = read_bes_file(&bytes).unwrap();
        assert_eq!(doc.lines.len(), 4);
        assert!(doc.lines.iter().all(|l| !l.page_break));
    }

    #[test]
    fn short_page_break_is_forced() {
        // ページ0 が lines_per_page=5 未満（2行）で終わる → 強制改ページ
        let bytes = make_bes(32, 5, &[vec![("⠁", false), ("⠃", true)], vec![("⠉", true)]]);
        let doc = read_bes_file(&bytes).unwrap();
        assert_eq!(doc.lines.len(), 3);
        assert!(doc.lines[1].page_break); // ページ0最終行に強制改ページ
        assert!(!doc.lines[0].page_break);
        assert!(!doc.lines[2].page_break); // 最終ページは対象外
    }

    #[test]
    fn roundtrip_from_writer_preserves_content() {
        // writer 出力（ヘッダ込み・自動改ページ）を読み戻し、本文と論理構造が一致する。
        let config = FormatterConfig {
            line_width: 32,
            lines_per_page: 25,
            page_header: false, // ヘッダ無しで本文のみを厳密比較
            title: None,
        };
        let paras = vec![braille_str(10), braille_str(40), braille_str(5)];
        let doc = BrailleFormatter::new(config).format(&paras);
        let bytes = OutputFormat::Bes.write(&doc);

        let read = read_bes_file(&bytes).unwrap();
        let original: Vec<String> = doc
            .pages()
            .iter()
            .flat_map(|p| p.iter())
            .map(|l| l.content.clone())
            .collect();
        let restored: Vec<String> = read.lines.iter().map(|l| l.content.clone()).collect();
        assert_eq!(restored, original);
    }
}
