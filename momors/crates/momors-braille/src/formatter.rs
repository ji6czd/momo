//! 正本 [`BrailleDocument`] から**印刷イメージ** [`FormattedDocument`] を導出する。
//!
//! - [`render`]: 各セグメントを行幅で**折り返し**た上で、ページ分割・ページヘッダ生成・
//!   ページ番号付与・改ページ上書きを行う。強制改行（セグメント境界）と改ページは尊重する。
//!   ヘッダ表示有無・タイトル・ページ番号・番号スタイルは [`PageBreak`] による**継続的な状態**
//!   として扱われ、ヘッダ表示有無が変わるとページあたりのコンテンツ収容行数もそのページから
//!   動的に変わる。
//! - [`wrap_line`] / [`wrap_suffix`]: 1セグメントの点字文字列をワードラップして
//!   表示行に分割する低レベル関数。`render` 内部や FFI の逐次折返しに使う。

use crate::document::{BrailleDocument, PageBreak, PageNumberStyle, ParagraphEntry, PhysicalLine};

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

/// あるページで実効しているヘッダ関連の継続的状態。
///
/// エディタが「今表示されているページ以降に適用する設定」を編集する際、まずここから
/// 実効値を読み取ってダイアログの初期値にする用途を想定する（[`FormattedDocument::page_info`]）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageInfo {
    /// このページにヘッダ行が表示されているか。
    pub header_enabled: bool,
    /// このページのヘッダタイトル（点字文字列）。
    pub title: Option<String>,
    /// このページのページ番号。
    pub page_number: u32,
    /// このページのページ番号スタイル。
    pub style: PageNumberStyle,
}

/// 正本から導出された印刷イメージ。ページのリストとして保持する。
#[derive(Debug, Clone)]
pub struct FormattedDocument {
    pages: Vec<Vec<RenderedLine>>,
    page_info: Vec<PageInfo>,
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

    /// 指定ページで実効しているヘッダ関連の継続的状態。ページが存在しなければ `None`。
    pub fn page_info(&self, page: usize) -> Option<&PageInfo> {
        self.page_info.get(page)
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

/// ヘッダ表示有無から、1ページあたりのコンテンツ用行数を求める。
/// ヘッダ行が1行を占めるため、ヘッダ表示時は `lines_per_page - 1`（最低1）。
fn content_capacity(header_enabled: bool, lines_per_page: usize) -> usize {
    if header_enabled {
        lines_per_page.saturating_sub(1).max(1)
    } else {
        lines_per_page.max(1)
    }
}

/// パス1で解決された1ページ分。ヘッダ表示有無・タイトル・ページ番号・番号スタイルは、
/// このページの**先頭時点**で有効だった継続状態として確定済み。
struct ResolvedPage {
    lines: Vec<(String, bool, i32)>,
    header_enabled: bool,
    title: Option<String>,
    page_number: u32,
    style: PageNumberStyle,
}

/// パス0（平坦化）が出力する1単位。`Break` はコンテンツを一切持たない制御信号で、
/// どの `Row` にも属さない（[`ParagraphEntry::Break`] が独立したエントリであることに対応）。
enum FlatItem {
    Row {
        content: String,
        logical_end: bool,
        segment_index: i32,
    },
    Break(PageBreak),
}

/// 正本ドキュメントを印刷イメージへ導出する。
///
/// 各セグメントを行幅で**折り返し**た上で、ページ分割（暗黙/強制）とページヘッダ・
/// ページ番号を付与する。空ドキュメント（段落 0）は 0 ページ。
/// 折返しで分かれた行は同じ `segment_index` を持ち、最後の行だけが論理行終端/改ページを担う。
///
/// ヘッダ表示有無 (`header_enabled`) ・タイトル・ページ番号・番号スタイルは、強制改ページの
/// 上書きが無い限り前のページから引き継がれる（継続的な状態）。ヘッダ表示有無は
/// 1ページあたりのコンテンツ収容行数にも影響するため、各ページの収容行数は、その**ページの
/// 先頭で有効なヘッダ表示状態**に基づいて動的に決まる（改ページ上書きは、それを含む行が
/// 属する現在のページの収容行数には影響せず、次のページから適用される）。
pub fn render(doc: &BrailleDocument) -> FormattedDocument {
    let cfg = &doc.config;
    let line_width = cfg.line_width;
    let lines_per_page = cfg.lines_per_page;

    // パス0: エントリを折返して FlatItem へ平坦化する。Break エントリはコンテンツを
    // 持たない制御信号としてそのまま渡す（どの Row にも属さない）。
    let mut flat: Vec<FlatItem> = Vec::new();
    let mut seg_idx: i32 = 0;
    for entry in &doc.paragraphs {
        match entry {
            ParagraphEntry::Text(lines) => {
                if lines.is_empty() {
                    flat.push(FlatItem::Row {
                        content: String::new(),
                        logical_end: true,
                        segment_index: seg_idx,
                    });
                    seg_idx += 1;
                    continue;
                }
                for seg in lines {
                    // セグメントを折返す。空セグメントは空行1行。
                    let mut rows = wrap_line(&seg.content, line_width);
                    if rows.is_empty() {
                        rows.push(PhysicalLine::new(String::new(), true));
                    }
                    let last = rows.len() - 1;
                    for (ri, row) in rows.into_iter().enumerate() {
                        flat.push(FlatItem::Row {
                            content: row.content,
                            logical_end: seg.logical_end && ri == last,
                            segment_index: seg_idx,
                        });
                    }
                    seg_idx += 1;
                }
            }
            ParagraphEntry::Break(pb) => flat.push(FlatItem::Break(pb.clone())),
        }
    }

    if flat.is_empty() {
        return FormattedDocument {
            pages: Vec::new(),
            page_info: Vec::new(),
            line_width,
            lines_per_page,
        };
    }

    // パス1: ページへ分割しつつ、ヘッダ表示有無・タイトル・ページ番号・番号スタイルという
    // 継続的な状態を解決する。ヘッダ表示有無はページ収容行数にも影響するため、収容行数は
    // 「そのページの先頭で有効なヘッダ表示状態」から都度求め直す。
    let mut resolved_pages: Vec<ResolvedPage> = Vec::new();
    let mut header_enabled = cfg.page_header;
    let mut title: Option<String> = cfg.title.clone();
    let mut page_number = cfg.number_start;
    let mut style = cfg.number_style;
    let mut capacity = content_capacity(header_enabled, lines_per_page);

    let mut cur: Vec<(String, bool, i32)> = Vec::new();
    let mut count = 0usize;

    for item in flat {
        match item {
            FlatItem::Row {
                content,
                logical_end,
                segment_index,
            } => {
                cur.push((content, logical_end, segment_index));
                count += 1;
                if count >= capacity {
                    resolved_pages.push(ResolvedPage {
                        lines: std::mem::take(&mut cur),
                        header_enabled,
                        title: title.clone(),
                        page_number,
                        style,
                    });
                    page_number += 1;
                    count = 0;
                }
            }
            FlatItem::Break(pb) => {
                // Break はコンテンツを持たない制御信号。cur が空でも常にページを確定する
                // （改ページが連続する・文書先頭にあるなどの退化ケースも一律に扱う）。
                resolved_pages.push(ResolvedPage {
                    lines: std::mem::take(&mut cur),
                    header_enabled,
                    title: title.clone(),
                    page_number,
                    style,
                });
                page_number += 1;
                count = 0;

                // 改ページの上書きは、次のページから適用される。
                if let Some(v) = pb.header_enabled {
                    header_enabled = v;
                }
                if let Some(h) = pb.header_override {
                    title = Some(h);
                }
                if let Some(n) = pb.number_start {
                    page_number = n;
                }
                if let Some(s) = pb.number_style {
                    style = s;
                }
                capacity = content_capacity(header_enabled, lines_per_page);
            }
        }
    }
    if !cur.is_empty() {
        resolved_pages.push(ResolvedPage {
            lines: cur,
            header_enabled,
            title,
            page_number,
            style,
        });
    }

    // パス2: 解決済みの状態から RenderedLine を組み立てるだけの純粋な整形。
    let mut pages: Vec<Vec<RenderedLine>> = Vec::with_capacity(resolved_pages.len());
    let mut page_info: Vec<PageInfo> = Vec::with_capacity(resolved_pages.len());
    for rp in resolved_pages {
        page_info.push(PageInfo {
            header_enabled: rp.header_enabled,
            title: rp.title.clone(),
            page_number: rp.page_number,
            style: rp.style,
        });

        let mut page: Vec<RenderedLine> = Vec::with_capacity(rp.lines.len() + 1);
        if rp.header_enabled {
            let title_str = rp.title.as_deref().unwrap_or("");
            page.push(RenderedLine {
                content: make_header(title_str, rp.page_number, rp.style, line_width),
                logical_end: true,
                is_header: true,
                segment_index: -1,
            });
        }
        for (content, logical_end, sidx) in rp.lines {
            page.push(RenderedLine {
                content,
                logical_end,
                is_header: false,
                segment_index: sidx,
            });
        }
        pages.push(page);
    }

    FormattedDocument {
        pages,
        page_info,
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
            number_style: PageNumberStyle::Standard,
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
        doc.paragraphs[0] = ParagraphEntry::Text(vec![
            PhysicalLine::new(braille_str(3), false), // 強制改行（seg 0）
            PhysicalLine::new(braille_str(3), true),  // 段落終端（seg 1）
        ]);
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
        // Text/Break/Text の3エントリ。ヘッダなし、行/ページ=25。
        let mut doc = doc_from(&[braille_str(5)], config(32, 25, false));
        doc.paragraphs = vec![
            ParagraphEntry::Text(vec![PhysicalLine::new(braille_str(5), true)]),
            ParagraphEntry::Break(PageBreak::default()),
            ParagraphEntry::Text(vec![PhysicalLine::new(braille_str(5), true)]),
        ];
        let f = render(&doc);
        assert_eq!(f.page_count(), 2);
    }

    #[test]
    fn render_page_break_number_start_override() {
        let mut doc = doc_from(&[braille_str(1)], config(32, 25, true));
        doc.paragraphs = vec![
            ParagraphEntry::Text(vec![PhysicalLine::new(braille_str(1), true)]),
            ParagraphEntry::Break(PageBreak {
                number_start: Some(10),
                ..PageBreak::default()
            }),
            ParagraphEntry::Text(vec![PhysicalLine::new(braille_str(1), true)]),
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
    fn render_number_style_from_config_applies_to_first_page() {
        // DocumentConfig.number_style を代替スタイルにすると、改ページ上書き無しでも
        // 先頭ページから代替スタイルの番号が使われる。
        let cfg = DocumentConfig {
            number_style: PageNumberStyle::Alternative,
            ..config(32, 25, true)
        };
        let doc = doc_from(&[braille_str(1)], cfg);
        let f = render(&doc);
        let header = &f.pages()[0][0].content;
        // 代替スタイルの 1 = ⠂、prefix ⠼
        let tail: String = header
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
    fn render_alternative_page_number_style_override() {
        // ページ番号スタイルは改ページ単位の上書きで切り替える（文書全体の既定は標準）。
        let mut doc = doc_from(&[braille_str(1)], config(32, 25, true));
        doc.paragraphs = vec![
            ParagraphEntry::Text(vec![PhysicalLine::new(braille_str(1), true)]),
            ParagraphEntry::Break(PageBreak {
                number_start: Some(1),
                number_style: Some(PageNumberStyle::Alternative),
                ..PageBreak::default()
            }),
            ParagraphEntry::Text(vec![PhysicalLine::new(braille_str(1), true)]),
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
        doc.paragraphs = vec![
            ParagraphEntry::Text(vec![PhysicalLine::new(braille_str(1), true)]),
            ParagraphEntry::Break(PageBreak {
                header_override: Some("⠭⠭".to_string()),
                ..PageBreak::default()
            }),
            ParagraphEntry::Text(vec![PhysicalLine::new(braille_str(1), true)]),
        ];
        let f = render(&doc);
        assert_eq!(f.page_count(), 2);
        assert!(f.pages()[1][0].content.starts_with("⠭⠭"));
        // 1ページ目はドキュメントタイトル
        assert!(f.pages()[0][0].content.starts_with(&braille_str(3)));
    }

    #[test]
    fn render_title_override_persists_across_implicit_page_break() {
        // lines_per_page=3 → ヘッダあり時の収容行数は2。改ページ直後のページ(2枚目)だけでなく、
        // その後の暗黙のページ分割で生じるページ(3枚目)でもタイトル上書きが継続することを確認する。
        // 改ページ直前は1行だけにして、暗黙のページ確定（収容行数ちょうど）と重ならないようにする。
        let mut doc = doc_from(&[braille_str(1)], config(32, 3, true));
        let mut after: Vec<PhysicalLine> = (0..3)
            .map(|_| PhysicalLine::new(braille_str(1), false))
            .collect();
        after.push(PhysicalLine::new(braille_str(1), true));
        doc.paragraphs = vec![
            ParagraphEntry::Text(vec![PhysicalLine::new(braille_str(1), false)]),
            ParagraphEntry::Break(PageBreak {
                header_override: Some("⠭".to_string()),
                ..PageBreak::default()
            }),
            ParagraphEntry::Text(after),
        ];
        let f = render(&doc);
        assert_eq!(f.page_count(), 3);
        assert!(f.pages()[1][0].content.starts_with("⠭"));
        assert!(f.pages()[2][0].content.starts_with("⠭")); // 暗黙分割後も継続
    }

    #[test]
    fn render_header_enabled_toggle_changes_page_capacity() {
        // header_enabled=false のセクション（収容行数5）→ 強制改ページで header_enabled=Some(true)
        // に切り替え（収容行数4）。ヘッダ有無の切替が暗黙のページ分割をまたいで継続し、かつ
        // ページ収容行数に反映されることを確認する。改ページ前後の行数はどちらも収容行数の
        // 倍数からずらしてあり（7行/5、9行/4）、暗黙のページ確定と改ページ自体の確定が
        // 重ならないようにしている。
        let mut doc = doc_from(&[braille_str(1)], config(32, 5, false));
        let before: Vec<PhysicalLine> = (0..7)
            .map(|_| PhysicalLine::new(braille_str(1), false))
            .collect();
        let mut after: Vec<PhysicalLine> = (0..8)
            .map(|_| PhysicalLine::new(braille_str(1), false))
            .collect();
        after.push(PhysicalLine::new(braille_str(1), true));
        doc.paragraphs = vec![
            ParagraphEntry::Text(before),
            ParagraphEntry::Break(PageBreak {
                header_enabled: Some(true),
                ..PageBreak::default()
            }),
            ParagraphEntry::Text(after),
        ];

        let f = render(&doc);
        assert_eq!(f.page_count(), 5);
        // 前半: ヘッダ無し、収容行数5がそのまま反映される（7行 → 5行+2行）。
        assert_eq!(f.pages()[0].len(), 5);
        assert!(!f.pages()[0][0].is_header);
        assert_eq!(f.pages()[1].len(), 2);
        assert!(!f.pages()[1][0].is_header);
        // 後半: ヘッダ有り、収容行数は4に減る（9行 → 4行+4行+1行、それぞれ+ヘッダ1行）。
        assert_eq!(f.pages()[2].len(), 5);
        assert!(f.pages()[2][0].is_header);
        assert_eq!(f.pages()[3].len(), 5);
        assert!(f.pages()[3][0].is_header);
        assert_eq!(f.pages()[4].len(), 2);
        assert!(f.pages()[4][0].is_header);
    }

    #[test]
    fn render_show_header_marker_toggles_header_line() {
        // MBR の ==== show_header=false マーカー経由でも同じ状態遷移が機能することを確認する。
        let mut doc = doc_from(&[braille_str(1)], config(32, 25, true));
        doc.paragraphs = vec![
            ParagraphEntry::Text(vec![PhysicalLine::new(braille_str(1), true)]),
            ParagraphEntry::Break(PageBreak::from_marker("==== show_header=false")),
            ParagraphEntry::Text(vec![PhysicalLine::new(braille_str(1), true)]),
        ];
        let f = render(&doc);
        assert_eq!(f.page_count(), 2);
        assert!(f.pages()[0][0].is_header);
        assert!(!f.pages()[1][0].is_header);
    }

    #[test]
    fn page_info_reflects_config_defaults_on_first_page() {
        let mut cfg = config(32, 25, true);
        cfg.title = Some("⠞⠊".to_string());
        cfg.number_start = 3;
        cfg.number_style = PageNumberStyle::Alternative;
        let doc = doc_from(&[braille_str(1)], cfg);
        let f = render(&doc);
        let info = f.page_info(0).unwrap();
        assert!(info.header_enabled);
        assert_eq!(info.title.as_deref(), Some("⠞⠊"));
        assert_eq!(info.page_number, 3);
        assert_eq!(info.style, PageNumberStyle::Alternative);
        assert!(f.page_info(1).is_none());
    }

    #[test]
    fn page_info_tracks_continuous_state_across_pages() {
        // header_enabled/title/番号/スタイルの4項目すべてが、改ページ上書きの後、
        // 暗黙分割をまたいでも page_info に正しく反映され続けることを確認する。
        let mut doc = doc_from(&[braille_str(1)], config(32, 3, false));
        let mut after: Vec<PhysicalLine> = (0..3)
            .map(|_| PhysicalLine::new(braille_str(1), false))
            .collect();
        after.last_mut().unwrap().logical_end = true;
        doc.paragraphs = vec![
            ParagraphEntry::Text(vec![PhysicalLine::new(braille_str(1), false)]),
            ParagraphEntry::Break(PageBreak {
                header_enabled: Some(true),
                header_override: Some("⠭".to_string()),
                number_start: Some(9),
                number_style: Some(PageNumberStyle::Alternative),
            }),
            ParagraphEntry::Text(after),
        ];

        let f = render(&doc);
        assert_eq!(f.page_count(), 3);
        let p0 = f.page_info(0).unwrap();
        assert!(!p0.header_enabled);
        assert_eq!(p0.page_number, 1);

        let p1 = f.page_info(1).unwrap();
        assert!(p1.header_enabled);
        assert_eq!(p1.title.as_deref(), Some("⠭"));
        assert_eq!(p1.page_number, 9);
        assert_eq!(p1.style, PageNumberStyle::Alternative);

        // 3ページ目は暗黙分割で生じるが、状態は2ページ目から継続する。
        let p2 = f.page_info(2).unwrap();
        assert!(p2.header_enabled);
        assert_eq!(p2.title.as_deref(), Some("⠭"));
        assert_eq!(p2.page_number, 10);
        assert_eq!(p2.style, PageNumberStyle::Alternative);
    }

    // ---- Break の退化ケース（一律ルール、特別扱いしない） ----

    #[test]
    fn render_leading_page_break_produces_empty_first_page() {
        // 文書先頭が Break でも一律に扱い、空の最初のページができる。
        let mut doc = doc_from(&[braille_str(1)], config(32, 25, true));
        doc.paragraphs = vec![
            ParagraphEntry::Break(PageBreak::default()),
            ParagraphEntry::Text(vec![PhysicalLine::new(braille_str(1), true)]),
        ];
        let f = render(&doc);
        assert_eq!(f.page_count(), 2);
        assert_eq!(f.pages()[0].len(), 1); // ヘッダ行のみ
        assert!(f.pages()[0][0].is_header);
        assert_eq!(f.pages()[1].len(), 2); // ヘッダ+本文1行
    }

    #[test]
    fn render_adjacent_page_breaks_produce_empty_page_between() {
        // Break が連続しても一律に扱い、間に空ページができる。
        let mut doc = doc_from(&[braille_str(1)], config(32, 25, true));
        doc.paragraphs = vec![
            ParagraphEntry::Text(vec![PhysicalLine::new(braille_str(1), true)]),
            ParagraphEntry::Break(PageBreak::default()),
            ParagraphEntry::Break(PageBreak::default()),
            ParagraphEntry::Text(vec![PhysicalLine::new(braille_str(1), true)]),
        ];
        let f = render(&doc);
        assert_eq!(f.page_count(), 3);
        assert_eq!(f.pages()[1].len(), 1); // 中間ページはヘッダ行のみ（本文0行）
        assert!(f.pages()[1][0].is_header);
    }

    #[test]
    fn render_break_only_document_yields_one_empty_page() {
        // テキストを一切持たない文書でも、Break 1個からヘッダのみの1ページが導出される。
        let mut doc = doc_from(&[braille_str(1)], config(32, 25, true));
        doc.paragraphs = vec![ParagraphEntry::Break(PageBreak::default())];
        let f = render(&doc);
        assert_eq!(f.page_count(), 1);
        assert_eq!(f.pages()[0].len(), 1);
        assert!(f.pages()[0][0].is_header);
    }
}
