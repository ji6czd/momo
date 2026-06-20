//! 正本 [`BrailleDocument`] から**印刷イメージ** [`FormattedDocument`] を導出する。
//!
//! - [`render`]: 各セグメントを行幅で**折り返し**た上で、ページ分割・ページヘッダ生成・
//!   ページ番号付与・改ページ上書きを行う。強制改行（セグメント境界）と改ページは尊重する。
//! - [`wrap_line`] / [`wrap_suffix`]: 1セグメントの点字文字列をワードラップして
//!   表示行に分割する低レベル関数。`render` 内部や FFI の逐次折返しに使う。

use crate::document::{BrailleDocument, PageBreak, PageNumberStyle, PhysicalLine};

const BRAILLE_SPACE: char = '⠀'; // U+2800
const BRAILLE_NUM_PREFIX: char = '⠼';

/// 標準スタイルの数字セル（0..=9）。
const DIGITS_STANDARD: [char; 10] = ['⠚', '⠁', '⠃', '⠉', '⠙', '⠑', '⠋', '⠛', '⠓', '⠊'];
/// 代替スタイルの数字セル（0..=9、下がり数字）。
const DIGITS_ALTERNATIVE: [char; 10] = ['⠴', '⠂', '⠆', '⠒', '⠲', '⠢', '⠖', '⠶', '⠦', '⠔'];

fn braille_page_num(n: u32, style: PageNumberStyle) -> String {
    let digits = match style {
        PageNumberStyle::Standard => &DIGITS_STANDARD,
        PageNumberStyle::Alternative => &DIGITS_ALTERNATIVE,
    };
    let mut s = String::new();
    s.push(BRAILLE_NUM_PREFIX);
    for ch in n.to_string().chars() {
        s.push(digits[(ch as u8 - b'0') as usize]);
    }
    s
}

// ============================================================
// FormattedDocument（印刷イメージ）
// ============================================================

/// 印刷イメージ上の1表示行（折返し後）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedLine {
    /// 点字セルの文字列（ヘッダ行ならヘッダ内容）。
    pub content: String,
    /// この行で論理行（段落）が終わるか。ヘッダ行は `true`。
    pub logical_end: bool,
    /// ページヘッダ行か。
    pub is_header: bool,
    /// この行の元になった**セグメント**の通し番号（全段落通し）。ヘッダ行は -1。
    /// エディタが表示行 ↔ 論理位置（セグメント+オフセット）を対応づけるのに使う。
    pub segment_index: i32,
}

/// 正本から導出された印刷イメージ。ページのリストとして保持する。
#[derive(Debug, Clone)]
pub struct FormattedDocument {
    pages: Vec<Vec<RenderedLine>>,
    line_width: usize,
    lines_per_page: usize,
}

impl FormattedDocument {
    /// ページのスライス。各要素が1ページ（物理行のリスト）。
    pub fn pages(&self) -> &[Vec<RenderedLine>] {
        &self.pages
    }

    /// 総ページ数。
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// 1行あたりのマス数。
    pub fn line_width(&self) -> usize {
        self.line_width
    }

    /// 1ページあたりの行数（ヘッダ行を含む）。
    pub fn lines_per_page(&self) -> usize {
        self.lines_per_page
    }
}

// ============================================================
// render
// ============================================================

/// 正本ドキュメントを印刷イメージへ導出する。
///
/// 各セグメントを行幅で**折り返し**た上で、ページ分割（暗黙/強制）とページヘッダ・
/// ページ番号を付与する。空ドキュメント（段落 0）は 0 ページ。
/// 折返しで分かれた行は同じ `segment_index` を持ち、最後の行だけが論理行終端/改ページを担う。
pub fn render(doc: &BrailleDocument) -> FormattedDocument {
    let cfg = &doc.config;
    let line_width = cfg.line_width;
    let lines_per_page = cfg.lines_per_page;

    // ヘッダが1行を占めるため、コンテンツ用の行数を1減らす。
    let content_per_page = if cfg.page_header {
        lines_per_page.saturating_sub(1).max(1)
    } else {
        lines_per_page.max(1)
    };

    // セグメントを折返して表示行へ平坦化する。
    // 各行: (content, logical_end, page_break, segment_index)
    let mut flat: Vec<(String, bool, Option<PageBreak>, i32)> = Vec::new();
    let mut seg_idx: i32 = 0;
    for para in &doc.paragraphs {
        if para.is_empty() {
            flat.push((String::new(), true, None, seg_idx));
            seg_idx += 1;
            continue;
        }
        for seg in para {
            // セグメントを折返す。空セグメントは空行1行。
            let mut rows = wrap_line(&seg.content, line_width);
            if rows.is_empty() {
                rows.push(PhysicalLine::new(String::new(), true));
            }
            let last = rows.len() - 1;
            for (ri, row) in rows.into_iter().enumerate() {
                let is_last = ri == last;
                flat.push((
                    row.content,
                    seg.logical_end && is_last,
                    if is_last {
                        seg.page_break.clone()
                    } else {
                        None
                    },
                    seg_idx,
                ));
            }
            seg_idx += 1;
        }
    }

    if flat.is_empty() {
        return FormattedDocument {
            pages: Vec::new(),
            line_width,
            lines_per_page,
        };
    }

    // パス1: ページへ分割する。各 raw ページに「そのページを開始させた改ページ上書き」を持たせる。
    let mut raw_pages: Vec<(Vec<(String, bool, i32)>, Option<PageBreak>)> = Vec::new();
    let mut cur: Vec<(String, bool, i32)> = Vec::new();
    let mut start_override: Option<PageBreak> = None;
    let mut count = 0usize;

    for (content, logical_end, page_break, sidx) in flat {
        cur.push((content, logical_end, sidx));
        count += 1;
        if let Some(pb) = page_break {
            // 強制改ページ: この行の後でページを送る。
            raw_pages.push((std::mem::take(&mut cur), start_override.take()));
            start_override = Some(pb);
            count = 0;
        } else if count >= content_per_page {
            // 暗黙改ページ。
            raw_pages.push((std::mem::take(&mut cur), start_override.take()));
            count = 0;
        }
    }
    if !cur.is_empty() {
        raw_pages.push((cur, start_override.take()));
    }

    // パス2: ページ番号・ヘッダを付与する。
    let mut pages: Vec<Vec<RenderedLine>> = Vec::with_capacity(raw_pages.len());
    let mut page_number = cfg.number_start;
    let mut style = PageNumberStyle::default();

    for (lines, ovr) in raw_pages {
        if let Some(o) = &ovr {
            if let Some(s) = o.number_style {
                style = s;
            }
            if let Some(n) = o.number_start {
                page_number = n;
            }
        }

        let mut page: Vec<RenderedLine> = Vec::with_capacity(lines.len() + 1);
        if cfg.page_header {
            let title: &str = match &ovr {
                Some(o) if o.header_override.is_some() => o.header_override.as_deref().unwrap(),
                _ => cfg.title.as_deref().unwrap_or(""),
            };
            page.push(RenderedLine {
                content: make_header(title, page_number, style, line_width),
                logical_end: true,
                is_header: true,
                segment_index: -1,
            });
        }
        for (content, logical_end, sidx) in lines {
            page.push(RenderedLine {
                content,
                logical_end,
                is_header: false,
                segment_index: sidx,
            });
        }
        pages.push(page);
        page_number += 1;
    }

    FormattedDocument {
        pages,
        line_width,
        lines_per_page,
    }
}

/// ページヘッダ行を生成する。
/// タイトル（左揃え） + 点字スペース埋め + ページ番号（`line_width - max(番号セル, 6)` から開始）。
fn make_header(title: &str, page_num: u32, style: PageNumberStyle, line_width: usize) -> String {
    let page_brl = braille_page_num(page_num, style);
    let page_cells = page_brl.chars().count();
    let page_start = line_width.saturating_sub(page_cells.max(6));

    let title_part: String = title.chars().take(page_start).collect();
    let title_cells = title_part.chars().count();

    let mut header = title_part;
    for _ in title_cells..page_start {
        header.push(BRAILLE_SPACE);
    }
    header.push_str(&page_brl);
    header
}

// ============================================================
// ワードラップ（低レベル）
// ============================================================

/// 1論理行（点字文字列）を `line_width` マスで折り返して物理行に分割する。
///
/// 空文字列は空リストを返す。最後の物理行は常に `logical_end = true`。
pub fn wrap_line(text: &str, line_width: usize) -> Vec<PhysicalLine> {
    wrap_suffix(text, line_width, line_width)
}

/// 論理行のサフィックスを折り返す。
///
/// `first_line_remaining`: 現在の物理行の残りマス数。
/// - `0` のとき: 先頭に空物理行を1行出力し、以降は通常幅で折り返す。
/// - `line_width` のとき: [`wrap_line`] と等価。
///
/// 空文字列は空リストを返す。最後の物理行は常に `logical_end = true`。
pub fn wrap_suffix(
    text: &str,
    line_width: usize,
    first_line_remaining: usize,
) -> Vec<PhysicalLine> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut lines = word_wrap_from(text, first_line_remaining, line_width);
    if let Some(last) = lines.last_mut() {
        last.logical_end = true;
    }
    lines
}

/// ワードラップの実装。`first_line_width` マスを最初の行に使い、以降は `full_width` マスを使う。
/// `first_line_width == 0` のとき先頭に空行を1行追加して以降は通常幅で処理する。
/// 返却する各行の `logical_end` はすべて `false`。
fn word_wrap_from(text: &str, first_line_width: usize, full_width: usize) -> Vec<PhysicalLine> {
    let chars: Vec<char> = text.chars().collect();
    let mut lines = Vec::new();
    let mut start = 0;
    let mut width = first_line_width;

    if width == 0 {
        lines.push(PhysicalLine::new(String::new(), false));
        width = full_width;
    }

    while start < chars.len() {
        if chars.len() - start <= width {
            lines.push(PhysicalLine::new(
                chars[start..].iter().collect::<String>(),
                false,
            ));
            break;
        }

        let end = start + width;
        if chars.get(end) == Some(&BRAILLE_SPACE) {
            // 内容がwidth埋まり、直後がスペース → trailing spaceとして含める
            lines.push(PhysicalLine::new(
                chars[start..=end].iter().collect::<String>(),
                false,
            ));
            start = end + 1;
        } else {
            match (start..end).rev().find(|&i| chars[i] == BRAILLE_SPACE) {
                Some(sp) => {
                    // 折り返し点のスペースは行末に残す
                    lines.push(PhysicalLine::new(
                        chars[start..=sp].iter().collect::<String>(),
                        false,
                    ));
                    start = sp + 1;
                }
                None => {
                    // スペースなし: 強制分割
                    lines.push(PhysicalLine::new(
                        chars[start..end].iter().collect::<String>(),
                        false,
                    ));
                    start = end;
                }
            }
        }
        // 次行の先頭スペースを削除
        while start < chars.len() && chars[start] == BRAILLE_SPACE {
            start += 1;
        }
        width = full_width;
    }

    lines
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::DocumentConfig;

    fn braille_str(n: usize) -> String {
        std::iter::repeat('⠁').take(n).collect()
    }

    fn braille_with_spaces(segments: &[usize]) -> String {
        segments
            .iter()
            .map(|&n| braille_str(n))
            .collect::<Vec<_>>()
            .join(&BRAILLE_SPACE.to_string())
    }

    fn config(line_width: usize, lines_per_page: usize, page_header: bool) -> DocumentConfig {
        DocumentConfig {
            line_width,
            lines_per_page,
            page_header,
            title: None,
            number_start: 1,
        }
    }

    fn doc_from(paragraphs: &[String], cfg: DocumentConfig) -> BrailleDocument {
        BrailleDocument::from_paragraphs(paragraphs, cfg)
    }

    // ---- wrap ----

    #[test]
    fn wrap_empty_returns_empty() {
        assert!(wrap_line("", 32).is_empty());
    }

    #[test]
    fn wrap_sets_logical_end_on_last() {
        let lines = wrap_line(&braille_with_spaces(&[10, 10]), 15);
        assert_eq!(lines.len(), 2);
        assert!(!lines[0].logical_end);
        assert!(lines[1].logical_end);
    }

    #[test]
    fn wrap_splits_at_space() {
        let lines = wrap_line(&braille_with_spaces(&[10, 10]), 15);
        assert_eq!(lines[0].content.chars().count(), 11); // 10 + trailing space
        assert_eq!(lines[1].content.chars().count(), 10);
    }

    #[test]
    fn wrap_no_leading_space_after_wrap() {
        let para = format!("{}⠀⠀{}", braille_str(4), braille_str(4));
        let lines = wrap_line(&para, 6);
        assert_eq!(lines.len(), 2);
        assert!(!lines[1].content.starts_with(BRAILLE_SPACE));
    }

    #[test]
    fn wrap_hard_break_when_no_space() {
        let lines = wrap_line(&braille_str(20), 10);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].content.chars().count(), 10);
        assert_eq!(lines[1].content.chars().count(), 10);
    }

    #[test]
    fn wrap_suffix_remaining_zero_produces_empty_first_line() {
        let lines = wrap_suffix(&braille_str(5), 10, 0);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].content, "");
        assert_eq!(lines[1].content.chars().count(), 5);
        assert!(lines[1].logical_end);
    }

    #[test]
    fn wrap_suffix_subsequent_lines_use_full_width() {
        let text = braille_with_spaces(&[2, 10, 2]);
        let lines = wrap_suffix(&text, 10, 3);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].content.chars().count(), 3);
        assert_eq!(lines[1].content.chars().count(), 11);
        assert_eq!(lines[2].content.chars().count(), 2);
        assert!(lines[2].logical_end);
    }

    // ---- render ----

    #[test]
    fn render_empty_document_has_no_pages() {
        let doc = doc_from(&[], config(32, 25, false));
        assert_eq!(render(&doc).page_count(), 0);
    }

    #[test]
    fn render_short_paragraph_one_line_no_header() {
        let doc = doc_from(&[braille_str(10)], config(32, 25, false));
        let f = render(&doc);
        assert_eq!(f.page_count(), 1);
        assert_eq!(f.pages()[0].len(), 1);
        assert!(!f.pages()[0][0].is_header);
        assert!(f.pages()[0][0].logical_end);
    }

    #[test]
    fn render_wraps_long_segment() {
        // 1段落＝1セグメント（未折返し）。render が行幅で折り返す。
        // 10 + space + 10 = 21 セル > width=15 → 2行、どちらも同じ segment_index=0。
        let doc = doc_from(&[braille_with_spaces(&[10, 10])], config(15, 25, false));
        let f = render(&doc);
        assert_eq!(f.page_count(), 1);
        assert_eq!(f.pages()[0].len(), 2);
        assert_eq!(f.pages()[0][0].segment_index, 0);
        assert_eq!(f.pages()[0][1].segment_index, 0);
        assert!(!f.pages()[0][0].logical_end); // 折返し継続
        assert!(f.pages()[0][1].logical_end); // 段落終端
    }

    #[test]
    fn render_segment_index_increments_per_segment() {
        // 段落0: 強制改行で2セグメント、段落1: 1セグメント → segment_index 0,1,2
        let mut doc = doc_from(&[braille_str(3), braille_str(3)], config(32, 25, false));
        doc.paragraphs[0] = vec![
            PhysicalLine::new(braille_str(3), false), // 強制改行（seg 0）
            PhysicalLine::new(braille_str(3), true),  // 段落終端（seg 1）
        ];
        // 段落1 は seg 2
        let f = render(&doc);
        let rows: Vec<i32> = f.pages()[0].iter().map(|l| l.segment_index).collect();
        assert_eq!(rows, vec![0, 1, 2]);
        assert!(!f.pages()[0][0].logical_end); // 強制改行
        assert!(f.pages()[0][1].logical_end); // 段落終端
        assert!(f.pages()[0][2].logical_end);
    }

    #[test]
    fn render_paginates_by_content_height() {
        // 10段落 × 1物理行、lines_per_page=3、ヘッダなし → 10/3 = 4ページ
        let paras: Vec<String> = std::iter::repeat(braille_str(5)).take(10).collect();
        let doc = doc_from(&paras, config(32, 3, false));
        assert_eq!(render(&doc).page_count(), 4);
    }

    #[test]
    fn render_header_uses_one_line() {
        let paras: Vec<String> = std::iter::repeat(braille_str(5)).take(4).collect();
        let doc = doc_from(&paras, config(32, 5, true));
        let f = render(&doc);
        // content_per_page = 4、4段落 → 1ページ、ヘッダ1 + 4
        assert_eq!(f.page_count(), 1);
        assert_eq!(f.pages()[0].len(), 5);
        assert!(f.pages()[0][0].is_header);
    }

    #[test]
    fn render_header_page_number_position() {
        let doc = doc_from(&[braille_str(1)], config(32, 25, true));
        let f = render(&doc);
        let header = &f.pages()[0][0].content;
        assert_eq!(header.chars().count(), 28); // 26 + 2 (⠼⠁)
        let tail: String = header
            .chars()
            .rev()
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        assert_eq!(tail, "⠼⠁");
    }

    #[test]
    fn render_forced_page_break_splits_page() {
        // 1段落2物理行、ヘッダなし、行/ページ=25。手動で1行目に強制改ページ。
        let mut doc = doc_from(&[braille_str(5)], config(32, 25, false));
        // 段落を2物理行にして1行目に page_break を付ける
        doc.paragraphs[0] = vec![
            {
                let mut l = PhysicalLine::new(braille_str(5), false);
                l.page_break = Some(PageBreak::default());
                l
            },
            PhysicalLine::new(braille_str(5), true),
        ];
        let f = render(&doc);
        assert_eq!(f.page_count(), 2);
    }

    #[test]
    fn render_page_break_number_start_override() {
        let mut doc = doc_from(&[braille_str(1)], config(32, 25, true));
        doc.paragraphs[0] = vec![
            {
                let mut l = PhysicalLine::new(braille_str(1), false);
                l.page_break = Some(PageBreak {
                    number_start: Some(10),
                    ..PageBreak::default()
                });
                l
            },
            PhysicalLine::new(braille_str(1), true),
        ];
        let f = render(&doc);
        assert_eq!(f.page_count(), 2);
        // 2ページ目のヘッダ末尾は ⠼⠁⠚ (=10)
        let h2 = &f.pages()[1][0].content;
        let tail: String = h2
            .chars()
            .rev()
            .take(3)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        assert_eq!(tail, "⠼⠁⠚");
    }

    #[test]
    fn render_number_start_offsets_first_page() {
        // 開始ページ番号を 5 にすると先頭ページの番号が 5 になる。
        let cfg = DocumentConfig {
            number_start: 5,
            ..config(32, 25, true)
        };
        let doc = doc_from(&[braille_str(1)], cfg);
        let f = render(&doc);
        let header = &f.pages()[0][0].content;
        // 標準スタイルの 5 = ⠑、prefix ⠼
        let tail: String = header
            .chars()
            .rev()
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        assert_eq!(tail, "⠼⠑");
    }

    #[test]
    fn render_alternative_page_number_style_override() {
        // ページ番号スタイルは改ページ単位の上書きで切り替える（文書全体の既定は標準）。
        let mut doc = doc_from(&[braille_str(1)], config(32, 25, true));
        doc.paragraphs[0] = vec![
            {
                let mut l = PhysicalLine::new(braille_str(1), false);
                l.page_break = Some(PageBreak {
                    number_start: Some(1),
                    number_style: Some(PageNumberStyle::Alternative),
                    ..PageBreak::default()
                });
                l
            },
            PhysicalLine::new(braille_str(1), true),
        ];
        let f = render(&doc);
        assert_eq!(f.page_count(), 2);
        // 2ページ目は代替スタイルで番号 1 = ⠼⠂
        let h2 = &f.pages()[1][0].content;
        let tail: String = h2
            .chars()
            .rev()
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        assert_eq!(tail, "⠼⠂");
    }

    #[test]
    fn render_page_break_header_override() {
        let mut doc = doc_from(&[braille_str(1)], config(32, 25, true));
        doc.config.title = Some(braille_str(3));
        doc.paragraphs[0] = vec![
            {
                let mut l = PhysicalLine::new(braille_str(1), false);
                l.page_break = Some(PageBreak {
                    header_override: Some("⠭⠭".to_string()),
                    ..PageBreak::default()
                });
                l
            },
            PhysicalLine::new(braille_str(1), true),
        ];
        let f = render(&doc);
        assert_eq!(f.page_count(), 2);
        assert!(f.pages()[1][0].content.starts_with("⠭⠭"));
        // 1ページ目はドキュメントタイトル
        assert!(f.pages()[0][0].content.starts_with(&braille_str(3)));
    }
}
