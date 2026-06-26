//! 正本 [`BrailleDocument`] をファイル形式のバイト列へ直列化する。
//!
//! - `Mbr`: 論理ドキュメントをそのまま MBR テキストへ（[`BrailleDocument::to_mbr`]）。
//! - `Bes` / `Base` / `BrailleText`: [`render`] で印刷イメージへ導出してから符号化する。

use crate::document::BrailleDocument;
use crate::formatter::{render, FormattedDocument};
use crate::nabcc::braille_to_nabcc_capital;

/// ドキュメントの出力形式。
pub enum OutputFormat {
    /// ページ間をフォームフィード (`\x0C`) で区切ったプレーンテキスト。拡張子: `.brf`
    ///
    /// 各行の末尾点字スペース (`⠀` U+2800) は落とす。これにより 1 行が必ず
    /// `line_width` マス以内になり、プリンタ印刷時に行幅超過で意図しない折返しが
    /// 入るのを防ぐ。代償として、この出力からは折返し位置（行末スペース）を
    /// 厳密には再現できない（印刷専用の非可逆形式であり読み戻しもしない）。
    BrailleText,
    /// PC-9801 用の点字ファイル形式。ヘッダとページデータを含む。拡張子: `.bse`
    Base,
    /// IBM 点字編集システム形式。独自バイナリ。拡張子: `.bes`
    Bes,
    /// Momo Braille Document。フロントマター + 段落本文の UTF-8 テキスト。拡張子: `.mbr`
    Mbr,
}

impl OutputFormat {
    /// 正本ドキュメントをこの形式のバイト列へ変換する。
    pub fn write(&self, doc: &BrailleDocument) -> Vec<u8> {
        match self {
            OutputFormat::Mbr => doc.to_mbr().into_bytes(),
            OutputFormat::BrailleText => write_braille_text(&render(doc)),
            OutputFormat::Base => write_base_file(&render(doc)),
            OutputFormat::Bes => write_bes_file(&render(doc)),
        }
    }
}

fn write_braille_text(doc: &FormattedDocument) -> Vec<u8> {
    let mut out = String::new();
    for (i, page) in doc.pages().iter().enumerate() {
        if i > 0 {
            out.push_str("\x0C\n");
        }
        // 行末の点字スペースを落とす（line_width 超過と印刷時の余分な折返しを防ぐ）。
        let lines: Vec<&str> = page
            .iter()
            .map(|l| l.content.as_str().trim_end_matches('⠀'))
            .collect();
        out.push_str(&lines.join("\n"));
        out.push('\n');
    }
    out.into_bytes()
}

/// BASE 形式。
/// ヘッダ: 512 バイト（末尾 8 バイトが "ppppccll"）+ 改行。
/// ページデータ: 各ページ必ず `lines_per_page` 行（不足は空行で埋める）。
fn write_base_file(doc: &FormattedDocument) -> Vec<u8> {
    let page_count = doc.page_count();
    let line_width = doc.line_width();
    let lines_per_page = doc.lines_per_page();

    let mut out = String::new();

    let format_info = format!("{:04}{:02}{:02}", page_count, line_width, lines_per_page);
    out.extend(std::iter::repeat(' ').take(504));
    out.push_str(&format_info);
    out.push('\n');

    for page in doc.pages() {
        for i in 0..lines_per_page {
            if let Some(line) = page.get(i) {
                for c in line.content.chars() {
                    out.push(braille_to_nabcc_capital(c) as char);
                }
            }
            out.push('\n');
        }
    }

    out.into_bytes()
}

/// BES 形式バイナリ書き出し。
fn write_bes_file(doc: &FormattedDocument) -> Vec<u8> {
    let xl = doc.line_width();
    let yl = doc.lines_per_page();
    let total_pages = doc.page_count();

    let mut out: Vec<u8> = Vec::new();

    // ── メインヘッダ (512 bytes) ──────────────────────────────
    let mut header = vec![b' '; 512];
    header[0..4].copy_from_slice(b"%BET");
    header[4] = b'4';
    header[5] = b'0';
    header[6] = b'0';
    header[7] = b'%';

    // 作成日付 YYYY/MM/DD（固定値; 実行時日付は依存なしで省略）
    header[8..18].copy_from_slice(b"2026/01/01");

    // xl エンコード: xl+1 を 2 桁 ASCII
    let xl_enc = xl + 1;
    header[0x16] = b'0' + (xl_enc / 10) as u8;
    header[0x17] = b'0' + (xl_enc % 10) as u8;

    // yl エンコード: そのまま 2 桁 ASCII
    header[0x18] = b'0' + (yl / 10) as u8;
    header[0x19] = b'0' + (yl % 10) as u8;

    // 総ページ数エンコード: (total_pages-1) を右詰め 3 桁 ASCII
    let page_enc = if total_pages > 0 { total_pages - 1 } else { 0 };
    let s = format!("{:3}", page_enc);
    header[0x1E..0x21].copy_from_slice(s.as_bytes());

    out.extend_from_slice(&header);

    // ── 拡張ヘッダ (512 bytes) ───────────────────────────────
    let mut ext = vec![b' '; 512];
    ext[0] = 0xFF;
    ext[1] = 0xFF;
    ext[2] = 0xFF;
    ext[3] = 0x54; // 'T'
    out.extend_from_slice(&ext);

    // ── コントロールエリア (最小構成: 0xFF 1 byte) ───────────
    out.push(0xFF);

    // ── ページデータ ──────────────────────────────────────────
    for (page_idx, page) in doc.pages().iter().enumerate() {
        let is_last_page = page_idx + 1 == total_pages;
        let last_line_idx = page.len().saturating_sub(1);

        let mut page_buf: Vec<u8> = Vec::new();
        page_buf.push(0xFD); // ページ開始マーカ

        for (line_idx, line) in page.iter().enumerate() {
            let is_last_line = line_idx == last_line_idx;

            // 点字セルを書き出す（各文字 U+2800+ → 0xA0+pattern）
            for ch in line.content.chars() {
                let pattern = (ch as u32).saturating_sub(0x2800) as u8;
                page_buf.push(0xA0 + pattern);
            }

            // 行末コード
            if is_last_page && is_last_line {
                // ページ末の最終行: 0x0D のみ (FE 省略可)
                page_buf.push(0x0D);
            } else if line.logical_end {
                // 論理行終わり: 0x0D + 0xFE
                page_buf.push(0x0D);
                page_buf.push(0xFE);
            } else {
                // 物理行のみ終わり (論理行は次行へ継続): 0xFE
                page_buf.push(0xFE);
            }
        }

        // サイズヘッダ (2 bytes little-endian、自身のサイズを含む)
        let chunk_total = 2 + page_buf.len();
        out.push((chunk_total & 0xFF) as u8);
        out.push(((chunk_total >> 8) & 0xFF) as u8);
        out.extend_from_slice(&page_buf);
    }

    // ── EOF マーカ ───────────────────────────────────────────
    out.push(0xFF);

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::DocumentConfig;

    fn config(page_header: bool) -> DocumentConfig {
        DocumentConfig {
            line_width: 32,
            lines_per_page: 22,
            page_header,
            title: None,
            number_start: 1,
        }
    }

    fn doc(paragraphs: &[&str], page_header: bool) -> BrailleDocument {
        let paras: Vec<String> = paragraphs.iter().map(|s| s.to_string()).collect();
        BrailleDocument::from_paragraphs(&paras, config(page_header))
    }

    #[test]
    fn mbr_uses_document_to_mbr() {
        let d = doc(&["⠁⠃⠉", "⠙⠑⠋"], true);
        let out = String::from_utf8(OutputFormat::Mbr.write(&d)).unwrap();
        assert_eq!(out, d.to_mbr());
        let body = out.split("---\n").nth(1).unwrap();
        assert_eq!(body, "⠁⠃⠉\n\n⠙⠑⠋\n");
    }

    #[test]
    fn bes_has_correct_magic() {
        let d = doc(&["⠁⠃"], false);
        let bytes = OutputFormat::Bes.write(&d);
        assert_eq!(&bytes[0..4], b"%BET");
        assert_eq!(bytes[4], b'4');
    }

    #[test]
    fn bes_xl_yl_encoding() {
        let d = doc(&["⠁"], false);
        let bytes = OutputFormat::Bes.write(&d);
        assert_eq!(bytes[0x16], b'3'); // xl=32 → 33
        assert_eq!(bytes[0x17], b'3');
        assert_eq!(bytes[0x18], b'2'); // yl=22
        assert_eq!(bytes[0x19], b'2');
    }

    #[test]
    fn bes_braille_cell_encoding() {
        let d = doc(&["⠁"], false);
        let bytes = OutputFormat::Bes.write(&d);
        let page_start = 0x401;
        assert_eq!(bytes[page_start + 2], 0xFD); // ページ開始マーカ
        assert_eq!(bytes[page_start + 3], 0xA1); // ⠁ = 0xA0 + 1
    }

    #[test]
    fn bes_logical_line_end_codes() {
        let d = doc(&["⠁", "⠃"], false);
        let bytes = OutputFormat::Bes.write(&d);
        let page_start = 0x401;
        // page_buf: FD A1 0D FE A3 0D
        assert_eq!(bytes[page_start + 2], 0xFD);
        assert_eq!(bytes[page_start + 3], 0xA1);
        assert_eq!(bytes[page_start + 4], 0x0D);
        assert_eq!(bytes[page_start + 5], 0xFE);
        assert_eq!(bytes[page_start + 6], 0xA3);
        assert_eq!(bytes[page_start + 7], 0x0D);
    }

    #[test]
    fn bes_eof_marker() {
        let d = doc(&["⠁"], false);
        let bytes = OutputFormat::Bes.write(&d);
        assert_eq!(*bytes.last().unwrap(), 0xFF);
    }

    #[test]
    fn braille_text_trims_trailing_space_within_line_width() {
        // line_width=5、ヘッダなし。"⠁⠁⠁⠁⠀⠃⠃" は 4 セル + スペース + 2 セル。
        // render は 1 行目を 5 セル埋め後の行末スペースを含めて折返す（"⠁⠁⠁⠁⠀"=5+1?）。
        // .brl 出力では行末スペースが落ちて各行が line_width 以内になる。
        let cfg = DocumentConfig {
            line_width: 5,
            lines_per_page: 25,
            page_header: false,
            title: None,
            number_start: 1,
        };
        let d = BrailleDocument::from_paragraphs(&["⠁⠁⠁⠁⠀⠃⠃".to_string()], cfg);
        let out = String::from_utf8(OutputFormat::BrailleText.write(&d)).unwrap();
        for line in out.split('\n') {
            assert!(!line.ends_with('⠀'), "行末スペースが残っている: {:?}", line);
            assert!(
                line.chars().count() <= 5,
                "line_width を超過: {:?} ({} セル)",
                line,
                line.chars().count()
            );
        }
    }

    #[test]
    fn braille_text_form_feed_between_pages() {
        // lines_per_page=1, ヘッダなし、2段落 → 2ページ → フォームフィードで区切る
        let cfg = DocumentConfig {
            lines_per_page: 1,
            page_header: false,
            ..config(false)
        };
        let d = BrailleDocument::from_paragraphs(&["⠁".to_string(), "⠃".to_string()], cfg);
        let out = String::from_utf8(OutputFormat::BrailleText.write(&d)).unwrap();
        assert!(out.contains('\x0C'));
    }
}
