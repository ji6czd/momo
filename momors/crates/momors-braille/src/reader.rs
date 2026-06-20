//! 点字ファイルの読み込み。バイト列を正本 [`BrailleDocument`] へ復元する。
//!
//! 現在は BES バイナリ形式（[`crate::writer`] の `Bes` 出力の逆変換）に対応する。
//!
//! BES は印刷イメージを保持する形式であり、ヘッダ行・強制/暗黙の改ページが
//! すべてファイル中に焼き込まれている。本リーダはバイト列をデコードし、
//! 段落（物理行の列）へ復元する。方針:
//!
//! - 行末コード: `0x0D 0xFE` = 論理行終端、`0xFE` 単独 = 物理行のみ終端（論理行継続）、
//!   末尾の `0x0D` 単独 = ドキュメント終端（論理行終端扱い）。
//! - 改ページ `0xFD`: ページの生行数が `lines_per_page` に達していれば**暗黙改ページ**
//!   （リフォーマットで消えてよい）とみなして捨てる。`lines_per_page` 未満で終わって
//!   いれば**強制改ページ**として直前の物理行に印を付ける。
//! - ヘッダ行: 各ページ先頭行を位置で落とす。ただし数符 `⠼` を含む行に限定して誤爆を防ぐ。
//!   （タイトル・ページ番号の中身の復元は点字→数字の逆変換が入ってから。）

use crate::document::{BrailleDocument, DocumentConfig, PageBreak, PhysicalLine};

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

/// 復元途中の1物理行（段落へグループ化する前）。
struct RawLine {
    content: String,
    logical_end: bool,
    page_break: bool,
}

/// BES バイナリ列を読み込んで正本ドキュメントへ復元する。
/// マジック不一致や明らかな破損では `None` を返す。
pub fn read_bes(bytes: &[u8]) -> Option<BrailleDocument> {
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
    let mut pages: Vec<Vec<RawLine>> = Vec::new();
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

    // 改ページの強制/暗黙判定 + ヘッダ行除去 → 通しの物理行へ。
    let page_count = pages.len();
    let mut raw: Vec<RawLine> = Vec::new();
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

        raw.extend(page);
    }

    // 物理行を logical_end で区切って段落へまとめる。
    let mut paragraphs: Vec<Vec<PhysicalLine>> = Vec::new();
    let mut current: Vec<PhysicalLine> = Vec::new();
    for line in raw {
        let logical_end = line.logical_end;
        current.push(PhysicalLine {
            content: line.content,
            logical_end,
            page_break: if line.page_break {
                Some(PageBreak::default())
            } else {
                None
            },
        });
        if logical_end {
            paragraphs.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        paragraphs.push(current);
    }

    let config = DocumentConfig {
        line_width,
        lines_per_page,
        page_header: true,
        // タイトルは読み込み時に落としている（点字→数字の逆変換が入るまで復元しない）。
        title: None,
        number_start: 1,
    };

    Some(BrailleDocument { paragraphs, config })
}

/// ページチャンク（先頭 `0xFD` を除いた本体）を物理行に分解する。
fn parse_page(body: &[u8]) -> Vec<RawLine> {
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
                lines.push(RawLine {
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
        lines.push(RawLine {
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
        assert!(read_bes(b"not a bes file at all............").is_none());
    }

    #[test]
    fn reads_xl_yl_from_header() {
        let bytes = make_bes(32, 22, &[vec![("⠁⠃⠉", true)]]);
        let doc = read_bes(&bytes).unwrap();
        assert_eq!(doc.config.line_width, 32);
        assert_eq!(doc.config.lines_per_page, 22);
    }

    #[test]
    fn groups_physical_lines_into_paragraphs() {
        // 物理行継続(FE) → 同一段落、論理行終端(0D FE) → 段落確定
        let bytes = make_bes(32, 22, &[vec![("⠁⠃", false), ("⠉⠙", true), ("⠑⠋", true)]]);
        let doc = read_bes(&bytes).unwrap();
        assert_eq!(doc.paragraphs.len(), 2);
        assert_eq!(doc.paragraphs[0].len(), 2); // 継続 + 終端
        assert_eq!(doc.paragraphs[0][0].content, "⠁⠃");
        assert!(!doc.paragraphs[0][0].logical_end);
        assert!(doc.paragraphs[0][1].logical_end);
        assert_eq!(doc.paragraphs[1].len(), 1);
    }

    #[test]
    fn drops_header_line_with_number_sign() {
        let header = format!("⠞⠊{}⠼⠁", "⠀".repeat(3));
        let bytes = make_bes(32, 22, &[vec![(header.as_str(), true), ("⠁⠃⠉", true)]]);
        let doc = read_bes(&bytes).unwrap();
        assert_eq!(doc.paragraphs.len(), 1);
        assert_eq!(doc.logical_text(0), "⠁⠃⠉");
    }

    #[test]
    fn full_page_break_is_implicit() {
        let bytes = make_bes(
            32,
            3,
            &[
                vec![("⠁", false), ("⠃", false), ("⠉", true)],
                vec![("⠙", true)],
            ],
        );
        let doc = read_bes(&bytes).unwrap();
        assert!(doc
            .paragraphs
            .iter()
            .flatten()
            .all(|l| l.page_break.is_none()));
    }

    #[test]
    fn short_page_break_is_forced() {
        let bytes = make_bes(32, 5, &[vec![("⠁", false), ("⠃", true)], vec![("⠉", true)]]);
        let doc = read_bes(&bytes).unwrap();
        // ページ0 最終行（段落0の末尾）に強制改ページ
        assert!(doc.paragraphs[0].last().unwrap().page_break.is_some());
        assert!(doc.paragraphs[1].last().unwrap().page_break.is_none());
    }

    #[test]
    fn roundtrip_from_writer_preserves_content() {
        // writer の Bes 出力（ヘッダ無し）を読み戻し、段落ごとの論理テキストが一致する。
        // doc 側は未折返しセグメント、read 側は BES の折返し行だが、連結すれば一致する。
        let config = DocumentConfig {
            line_width: 32,
            lines_per_page: 25,
            page_header: false,
            title: None,
            number_start: 1,
        };
        let paras = vec![braille_str(10), braille_str(40), braille_str(5)];
        let doc = BrailleDocument::from_paragraphs(&paras, config);
        let bytes = OutputFormat::Bes.write(&doc);

        let read = read_bes(&bytes).unwrap();
        assert_eq!(read.paragraphs.len(), doc.paragraphs.len());
        let restored: Vec<String> = (0..read.paragraphs.len())
            .map(|i| read.logical_text(i))
            .collect();
        let original: Vec<String> = (0..doc.paragraphs.len())
            .map(|i| doc.logical_text(i))
            .collect();
        assert_eq!(restored, original);
    }
}
