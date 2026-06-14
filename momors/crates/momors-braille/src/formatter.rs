const BRAILLE_SPACE: char = '⠀'; // U+2800
const BRAILLE_NUM_PREFIX: char = '⠼';

fn digit_char(d: u8) -> char {
    match d {
        0 => '⠚',
        1 => '⠁',
        2 => '⠃',
        3 => '⠉',
        4 => '⠙',
        5 => '⠑',
        6 => '⠋',
        7 => '⠛',
        8 => '⠓',
        9 => '⠊',
        _ => BRAILLE_SPACE,
    }
}

fn braille_page_num(n: u32) -> String {
    let mut s = String::new();
    s.push(BRAILLE_NUM_PREFIX);
    for ch in n.to_string().chars() {
        s.push(digit_char(ch as u8 - b'0'));
    }
    s
}

// ============================================================
// FormatterConfig
// ============================================================

/// フォーマッタの設定。
pub struct FormatterConfig {
    /// 1行あたりのマス数（点字セル数）。
    pub line_width: usize,
    /// 1ページあたりの行数（ヘッダ行を含む）。
    pub lines_per_page: usize,
    /// ページヘッダ行を生成するか。`true` の場合、タイトルの有無にかかわらずヘッダ行を出力する。
    pub page_header: bool,
    /// ページヘッダのタイトル（点字文字列）。`None` の場合はページ番号のみ。
    pub title: Option<String>,
}

impl Default for FormatterConfig {
    fn default() -> Self {
        Self {
            line_width: 32,
            lines_per_page: 22,
            page_header: true,
            title: None,
        }
    }
}

// ============================================================
// PhysicalLine / FormattedDocument
// ============================================================

/// 物理行1行のデータ。
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalLine {
    /// 点字セルの文字列。
    pub content: String,
    /// この物理行で論理行（段落）が終わるか。
    pub logical_end: bool,
}

impl PhysicalLine {
    fn new(content: String, logical_end: bool) -> Self {
        Self {
            content,
            logical_end,
        }
    }
}

/// フォーマット済み点字ドキュメント。
///
/// ページのリストとして保持する。各ページは物理行の `Vec<PhysicalLine>`。
#[derive(Debug, Clone)]
pub struct FormattedDocument {
    pages: Vec<Vec<PhysicalLine>>,
    line_width: usize,
    lines_per_page: usize,
}

impl FormattedDocument {
    /// ページのスライスを返す。各要素が1ページ（物理行のリスト）。
    pub fn pages(&self) -> &[Vec<PhysicalLine>] {
        &self.pages
    }

    /// 総ページ数。
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// 1行あたりのマス数（点字セル数）。
    pub fn line_width(&self) -> usize {
        self.line_width
    }

    /// 1ページあたりの行数（ヘッダ行を含む）。
    pub fn lines_per_page(&self) -> usize {
        self.lines_per_page
    }
}

// ============================================================
// BrailleFormatter
// ============================================================

/// 点字段落のリストを受け取り、折り返し・ページ分割・ページヘッダ生成を行う。
pub struct BrailleFormatter {
    config: FormatterConfig,
}

impl BrailleFormatter {
    pub fn new(config: FormatterConfig) -> Self {
        Self { config }
    }

    /// 段落リスト（論理行）をフォーマットして [`FormattedDocument`] を返す。
    ///
    /// - `paragraphs`: 各要素が1論理行（段落）の点字文字列。
    /// - 空文字列の要素は空行として扱う。
    pub fn format(&self, paragraphs: &[String]) -> FormattedDocument {
        // 各段落を物理行に展開。最後の物理行に logical_end=true を付与。
        let mut physical_lines: Vec<PhysicalLine> = Vec::new();
        for para in paragraphs {
            if para.is_empty() {
                physical_lines.push(PhysicalLine::new(String::new(), true));
            } else {
                let mut wrapped = self.word_wrap(para);
                if let Some(last) = wrapped.last_mut() {
                    last.logical_end = true;
                }
                physical_lines.extend(wrapped);
            }
        }

        if physical_lines.is_empty() {
            return FormattedDocument {
                pages: vec![],
                line_width: self.config.line_width,
                lines_per_page: self.config.lines_per_page,
            };
        }

        // ヘッダ行がある場合はコンテンツ用の行数を1減らす
        let content_height = if self.config.page_header {
            self.config.lines_per_page.saturating_sub(1)
        } else {
            self.config.lines_per_page
        };
        let content_height = content_height.max(1);

        let pages = physical_lines
            .chunks(content_height)
            .enumerate()
            .map(|(i, chunk)| {
                let page_num = (i + 1) as u32;
                let mut page = Vec::new();
                if self.config.page_header {
                    let title = self.config.title.as_deref().unwrap_or("");
                    page.push(PhysicalLine::new(self.make_header(title, page_num), true));
                }
                page.extend_from_slice(chunk);
                page
            })
            .collect();

        FormattedDocument {
            pages,
            line_width: self.config.line_width,
            lines_per_page: self.config.lines_per_page,
        }
    }

    /// 1段落をワードラップして物理行のリストに変換する。
    /// 返却する各行の `logical_end` はすべて `false`。呼び出し側が最後の行を `true` に設定する。
    fn word_wrap(&self, text: &str) -> Vec<PhysicalLine> {
        let width = self.config.line_width;
        let chars: Vec<char> = text.chars().collect();
        let mut lines = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            if chars.len() - start <= width {
                lines.push(PhysicalLine::new(chars[start..].iter().collect(), false));
                break;
            }

            // start+width より手前の最後のスペースで折る
            let end = start + width;
            match (start..end).rev().find(|&i| chars[i] == BRAILLE_SPACE) {
                Some(sp) => {
                    // 折り返し点のスペースは行末に残す
                    lines.push(PhysicalLine::new(chars[start..=sp].iter().collect(), false));
                    start = sp + 1;
                }
                None => {
                    // スペースなし: 強制分割
                    lines.push(PhysicalLine::new(chars[start..end].iter().collect(), false));
                    start = end;
                }
            }
            // 次行の先頭スペースを削除
            while start < chars.len() && chars[start] == BRAILLE_SPACE {
                start += 1;
            }
        }

        lines
    }

    /// ページヘッダ行を生成する。
    /// タイトル（左揃え） + 点字スペース埋め + ページ番号（`line_width - 6` の位置から開始）。
    fn make_header(&self, title: &str, page_num: u32) -> String {
        let page_brl = braille_page_num(page_num);
        let page_cells = page_brl.chars().count();
        let page_start = self.config.line_width.saturating_sub(page_cells.max(6));

        let title_part: String = title.chars().take(page_start).collect();
        let title_cells = title_part.chars().count();

        let mut header = title_part;
        for _ in title_cells..page_start {
            header.push(BRAILLE_SPACE);
        }
        header.push_str(&page_brl);
        header
    }
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(line_width: usize, lines_per_page: usize) -> BrailleFormatter {
        BrailleFormatter::new(FormatterConfig {
            line_width,
            lines_per_page,
            page_header: false,
            title: None,
        })
    }

    fn braille_str(n: usize) -> String {
        // n個の点字セル（⠁ で埋めた文字列）
        std::iter::repeat('⠁').take(n).collect()
    }

    fn braille_with_spaces(segments: &[usize]) -> String {
        // セグメントをスペースで繋いだ点字文字列
        segments
            .iter()
            .map(|&n| braille_str(n))
            .collect::<Vec<_>>()
            .join(&BRAILLE_SPACE.to_string())
    }

    #[test]
    fn empty_input_returns_empty_document() {
        let doc = fmt(32, 25).format(&[]);
        assert_eq!(doc.page_count(), 0);
    }

    #[test]
    fn short_paragraph_stays_one_line() {
        let para = braille_str(10);
        let doc = fmt(32, 25).format(&[para.clone()]);
        assert_eq!(doc.page_count(), 1);
        assert_eq!(doc.pages()[0], vec![PhysicalLine::new(para, true)]);
    }

    #[test]
    fn word_wrap_splits_at_space() {
        // 10+space+10 = 21 chars > width=15 → 2行になる
        // 折り返し点のスペースは行末に残るので1行目は11文字
        let para = braille_with_spaces(&[10, 10]);
        let doc = fmt(15, 25).format(&[para]);
        let page = &doc.pages()[0];
        assert_eq!(page.len(), 2);
        assert_eq!(page[0].content.chars().count(), 11); // 10 + trailing space
        assert_eq!(page[1].content.chars().count(), 10);
        assert!(!page[0].logical_end);
        assert!(page[1].logical_end);
    }

    #[test]
    fn word_wrap_no_leading_space_after_wrap() {
        // 連続スペース後のセグメントが次行の先頭にスペースを持たないことを確認
        // "AAAA⠀⠀BBBB" (4+space+space+4=10), width=6
        // → wrap at last space in 0..6 (pos 5): line1="AAAA⠀⠀" (6chars), start=6
        // → chars[6..]="BBBB" → line2="BBBB" (no leading space)
        let para = format!("{}⠀⠀{}", braille_str(4), braille_str(4));
        let doc = fmt(6, 25).format(&[para]);
        let page = &doc.pages()[0];
        assert_eq!(page.len(), 2);
        assert!(
            !page[1].content.starts_with(BRAILLE_SPACE),
            "2行目の先頭がスペースになっている"
        );
        assert_eq!(page[1].content.chars().count(), 4);
    }

    #[test]
    fn word_wrap_hard_break_no_leading_space() {
        // 強制分割後、次の先頭がスペースの場合も削除される
        // "AAAAAAAAAA⠀⠀BBBB" (10+space+space+4=16), width=10
        // → no space in 0..10 → hard break: line1="AAAAAAAAAA", start=10
        //   chars[10]='⠀' → skip → chars[11]='⠀' → skip → start=12, "BBBB"
        // → line2="BBBB"
        let para = format!("{}⠀⠀{}", braille_str(10), braille_str(4));
        let doc = fmt(10, 25).format(&[para]);
        let page = &doc.pages()[0];
        assert_eq!(page.len(), 2);
        assert!(
            !page[1].content.starts_with(BRAILLE_SPACE),
            "強制分割後の2行目の先頭がスペースになっている"
        );
        assert_eq!(page[1].content.chars().count(), 4);
    }

    #[test]
    fn word_wrap_hard_break_when_no_space() {
        // スペースなし20文字 > width=10 → 強制分割
        let para = braille_str(20);
        let doc = fmt(10, 25).format(&[para]);
        let page = &doc.pages()[0];
        assert_eq!(page.len(), 2);
        assert_eq!(page[0].content.chars().count(), 10);
        assert_eq!(page[1].content.chars().count(), 10);
    }

    #[test]
    fn logical_end_flags_are_correct() {
        // 1段落が2物理行に分割 → 1行目 logical_end=false、2行目 logical_end=true
        let para = braille_with_spaces(&[10, 10]); // width=15 で2行
        let doc = fmt(15, 25).format(&[para]);
        let page = &doc.pages()[0];
        assert!(!page[0].logical_end);
        assert!(page[1].logical_end);
    }

    #[test]
    fn empty_paragraph_has_logical_end() {
        let doc = fmt(32, 25).format(&[String::new()]);
        assert!(doc.pages()[0][0].logical_end);
    }

    #[test]
    fn multiple_paragraphs_logical_ends() {
        // 2段落それぞれ1行 → 両方 logical_end=true
        let doc = fmt(32, 25).format(&[braille_str(5), braille_str(5)]);
        let page = &doc.pages()[0];
        assert_eq!(page.len(), 2);
        assert!(page[0].logical_end);
        assert!(page[1].logical_end);
    }

    #[test]
    fn pagination_splits_into_pages() {
        // 10段落 × 3行/段落 = 30物理行、lines_per_page=10 → 3ページ
        let para = braille_with_spaces(&[5, 5, 5]); // width=32 内に収まる 17chars → 1物理行
        // 各paragraphを折り返し後1行にするため、幅を十分大きく
        let paragraphs: Vec<String> = std::iter::repeat(para).take(10).collect();
        let doc = fmt(32, 3).format(&paragraphs);
        // 10行 / 3 = 4ページ (3+3+3+1)
        assert_eq!(doc.page_count(), 4);
    }

    #[test]
    fn page_header_uses_one_line() {
        let f = BrailleFormatter::new(FormatterConfig {
            line_width: 32,
            lines_per_page: 5,
            page_header: true,
            title: Some(braille_str(5)),
        });
        let paragraphs: Vec<String> = std::iter::repeat(braille_str(5)).take(4).collect();
        let doc = f.format(&paragraphs);
        // content_height = 5-1 = 4、4段落 → 1ページ
        assert_eq!(doc.page_count(), 1);
        assert_eq!(doc.pages()[0].len(), 5); // 1(header) + 4(content)
    }

    #[test]
    fn header_page_num_starts_at_line_width_minus_6() {
        // line_width=32 → page_start=26、ページ番号 ⠼⠁ (2cells) → 合計28
        let f = BrailleFormatter::new(FormatterConfig {
            line_width: 32,
            lines_per_page: 25,
            page_header: true,
            title: Some(braille_str(5)),
        });
        let doc = f.format(&[braille_str(1)]);
        let header = &doc.pages()[0][0].content;
        assert_eq!(header.chars().count(), 28); // 26 + 2
        // 末尾2文字がページ番号 ⠼⠁
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
    fn braille_page_num_format() {
        assert_eq!(braille_page_num(1), "⠼⠁");
        assert_eq!(braille_page_num(10), "⠼⠁⠚");
        assert_eq!(braille_page_num(100), "⠼⠁⠚⠚");
    }
}
