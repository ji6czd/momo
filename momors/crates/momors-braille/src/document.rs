//! 点字ドキュメントの**正本（論理モデル）**。
//!
//! すべての入出力（MBR / BES / BASE / BrailleText、CLI・FFI・wasm・エディタ）は
//! この [`BrailleDocument`] を経由する。reader はバイト列をこの型へ復元し、
//! writer はこの型からバイト列を生成する。印刷イメージ
//! （[`crate::formatter::FormattedDocument`]）は、この正本から
//! [`crate::formatter::render`] で一方向に導出される派生物。
//!
//! ## レベルの区別
//!
//! - **論理ドキュメント**（本モジュール）: 段落 = 物理行の列。手動改行・強制改ページ・
//!   ページ番号上書きなど、編集で保持すべき情報をすべて持つ。可逆。
//! - **印刷イメージ**（[`crate::formatter`]）: ページ×折返し済み行×ヘッダ。非可逆。
//!
//! 段落を「物理行の列」として保持するのは、ユーザーが置いた強制物理改行を
//! 再ワードラップで失わないため。新規にテキストから組む場合は
//! [`BrailleDocument::from_reflowed_paragraphs`] が折返しを行う。

use crate::formatter::wrap_line;

const SEPARATOR: &str = "---";
const PAGE_BREAK_MARKER: &str = "====";

// ============================================================
// 基本型
// ============================================================

/// ページ番号のスタイル。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageNumberStyle {
    /// 標準（下がり数字なし）: ⠼ + ⠚⠁⠃…
    Standard,
    /// 代替スタイル: ⠼ + ⠴⠂⠆…
    Alternative,
}

impl Default for PageNumberStyle {
    fn default() -> Self {
        PageNumberStyle::Standard
    }
}

/// 強制改ページに付随する上書き情報。
///
/// 直前の物理行の**後ろ**でページを送る。各フィールドは、その改ページで
/// 始まる新しいページに対する上書きを表す（`None` は上書きなし）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PageBreak {
    /// このページのヘッダタイトルを上書きする（点字文字列）。
    pub header_override: Option<String>,
    /// このページのページ番号をこの値から再開する。
    pub number_start: Option<u32>,
    /// このページ以降のページ番号スタイルを切り替える。
    pub number_style: Option<PageNumberStyle>,
}

impl PageBreak {
    /// 上書き情報を一切持たない（既定の）強制改ページか。
    pub fn is_plain(&self) -> bool {
        self.header_override.is_none() && self.number_start.is_none() && self.number_style.is_none()
    }

    /// MBR の `====` マーカー行（プレフィックス込み）へ直列化する。
    pub fn to_marker(&self) -> String {
        format_page_break(self)
    }

    /// `====` マーカー行（プレフィックス有無どちらでも可）を解析する。
    pub fn from_marker(line: &str) -> Self {
        let rest = line.strip_prefix(PAGE_BREAK_MARKER).unwrap_or(line);
        parse_page_break(rest)
    }
}

/// 論理ドキュメント上の1物理行。手動改行を保持できる単位。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalLine {
    /// 点字セルの文字列。
    pub content: String,
    /// この物理行で論理行（段落）が終わるか。
    pub logical_end: bool,
    /// この物理行の直後で強制改ページするか（上書き情報込み）。
    pub page_break: Option<PageBreak>,
}

impl PhysicalLine {
    /// 改ページ無し・指定 `logical_end` の物理行を作る。
    pub fn new(content: impl Into<String>, logical_end: bool) -> Self {
        Self {
            content: content.into(),
            logical_end,
            page_break: None,
        }
    }
}

/// ドキュメント設定（フロントマター相当）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentConfig {
    /// 1行あたりのマス数（点字セル数）。
    pub line_width: usize,
    /// 1ページあたりの行数（ヘッダ行を含む）。
    pub lines_per_page: usize,
    /// ページヘッダ行を生成するか。
    pub page_header: bool,
    /// ページヘッダのタイトル（点字文字列）。`None` でページ番号のみ。
    pub title: Option<String>,
    /// 既定のページ番号スタイル（[`PageBreak::number_style`] で上書き可能）。
    pub number_style: PageNumberStyle,
}

impl Default for DocumentConfig {
    fn default() -> Self {
        Self {
            line_width: 32,
            lines_per_page: 22,
            page_header: true,
            title: None,
            number_style: PageNumberStyle::Standard,
        }
    }
}

// ============================================================
// BrailleDocument
// ============================================================

/// 点字ドキュメントの正本。段落 = 物理行の列。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrailleDocument {
    /// 段落のリスト。各要素が1論理行（段落）の物理行リスト。
    pub paragraphs: Vec<Vec<PhysicalLine>>,
    /// ドキュメント設定。
    pub config: DocumentConfig,
}

impl Default for BrailleDocument {
    fn default() -> Self {
        Self {
            paragraphs: Vec::new(),
            config: DocumentConfig::default(),
        }
    }
}

impl BrailleDocument {
    /// 空のドキュメント（段落なし）。
    pub fn empty() -> Self {
        Self::default()
    }

    /// 段落 `index` の論理テキスト（物理行を連結したもの）を返す。
    pub fn logical_text(&self, index: usize) -> String {
        self.paragraphs[index]
            .iter()
            .map(|l| l.content.as_str())
            .collect()
    }

    /// 段落ごとの点字文字列を、設定の `line_width` でワードラップして組む。
    ///
    /// プレディクタ出力など「まだ折り返されていない」点字段落から
    /// ドキュメントを作るときに使う。各段落は折返し後の物理行リストになる。
    /// 空段落（空文字列）は空の物理行1行になる。
    pub fn from_reflowed_paragraphs(paragraphs: &[String], config: DocumentConfig) -> Self {
        let width = config.line_width;
        let paragraphs = paragraphs
            .iter()
            .map(|para| {
                let mut lines = wrap_line(para, width);
                if lines.is_empty() {
                    lines.push(PhysicalLine::new(String::new(), true));
                }
                lines
            })
            .collect();
        Self { paragraphs, config }
    }

    // -------------------------------------------------------
    // MBR（Momo Braille Document）テキスト形式
    // -------------------------------------------------------

    /// MBR テキストを解析して正本ドキュメントを返す。
    ///
    /// フォーマット:
    /// ```text
    /// line_width = 32
    /// lines_per_page = 22
    /// page_header = true
    /// title = ⠞⠊⠞⠇⠑          # 省略可
    /// ---
    /// ⠁⠃⠉                     # 論理行1・物理行1
    /// ==== start=5 style=alt   # 直前の物理行の後で強制改ページ（上書き可）
    /// ⠁⠃⠉                     # 論理行1・物理行2（改ページ後）
    ///                          # 空行 = 論理行の区切り
    /// ⠑⠋⠛                     # 論理行2
    /// ```
    pub fn parse_mbr(text: &str) -> Self {
        let lines: Vec<&str> = text.split('\n').map(|l| l.trim_end_matches('\r')).collect();
        let sep_idx = lines.iter().position(|&l| l == SEPARATOR);

        let (header_lines, body_lines): (&[&str], &[&str]) = match sep_idx {
            Some(i) => (&lines[..i], &lines[i + 1..]),
            None => (&[], &lines[..]),
        };

        let config = parse_config(header_lines);

        // 空行で区切りながら論理行グループを構築。
        let mut groups: Vec<Vec<&str>> = Vec::new();
        let mut current: Vec<&str> = Vec::new();
        for &line in body_lines {
            if line.is_empty() {
                groups.push(std::mem::take(&mut current));
            } else {
                current.push(line);
            }
        }
        if !current.is_empty() {
            groups.push(current);
        }
        // 末尾の空グループを除去。
        while groups.last().map(|g| g.is_empty()).unwrap_or(false) {
            groups.pop();
        }

        let mut paragraphs: Vec<Vec<PhysicalLine>> = Vec::new();
        for group in groups {
            if group.is_empty() {
                paragraphs.push(vec![PhysicalLine::new(String::new(), true)]);
                continue;
            }

            let mut phys: Vec<PhysicalLine> = Vec::new();
            for line in group {
                if let Some(rest) = line.strip_prefix(PAGE_BREAK_MARKER) {
                    // ==== マーカー: 直前の物理行に改ページを付ける。
                    // グループ先頭（直前の物理行がない）なら無視。
                    if let Some(last) = phys.last_mut() {
                        last.page_break = Some(parse_page_break(rest));
                    }
                } else {
                    phys.push(PhysicalLine::new(line, false));
                }
            }

            if phys.is_empty() {
                phys.push(PhysicalLine::new(String::new(), true));
            } else {
                phys.last_mut().unwrap().logical_end = true;
            }
            paragraphs.push(phys);
        }

        Self { paragraphs, config }
    }

    /// 正本ドキュメントを MBR テキストへ直列化する。
    pub fn to_mbr(&self) -> String {
        let c = &self.config;
        let mut out = String::new();
        out.push_str(&format!("line_width = {}\n", c.line_width));
        out.push_str(&format!("lines_per_page = {}\n", c.lines_per_page));
        out.push_str(&format!(
            "page_header = {}\n",
            if c.page_header { "true" } else { "false" }
        ));
        if let Some(title) = &c.title {
            if !title.is_empty() {
                out.push_str(&format!("title = {}\n", title));
            }
        }
        out.push_str(SEPARATOR);
        out.push('\n');

        for (i, para) in self.paragraphs.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            let is_empty = para.is_empty() || (para.len() == 1 && para[0].content.is_empty());
            if !is_empty {
                for line in para {
                    out.push_str(&line.content);
                    out.push('\n');
                    if let Some(pb) = &line.page_break {
                        out.push_str(&format_page_break(pb));
                        out.push('\n');
                    }
                }
            }
        }
        out
    }
}

// ============================================================
// MBR ヘルパ
// ============================================================

fn parse_config(lines: &[&str]) -> DocumentConfig {
    let mut config = DocumentConfig::default();
    for raw in lines {
        // 行内コメント（#）を落とす。
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let Some(eq) = line.find('=') else { continue };
        let key = line[..eq].trim();
        let value = line[eq + 1..].trim();
        match key {
            "line_width" => {
                if let Ok(v) = value.parse() {
                    config.line_width = v;
                }
            }
            "lines_per_page" => {
                if let Ok(v) = value.parse() {
                    config.lines_per_page = v;
                }
            }
            "page_header" => config.page_header = value == "true",
            "title" => {
                config.title = if value.is_empty() {
                    None
                } else {
                    Some(value.to_string())
                }
            }
            _ => {}
        }
    }
    config
}

/// `====` の後ろ（マーカーを除いた残り）を解析する。
fn parse_page_break(rest: &str) -> PageBreak {
    let mut pb = PageBreak::default();
    for token in rest.trim().split(' ').filter(|t| !t.is_empty()) {
        let Some(eq) = token.find('=') else { continue };
        let key = &token[..eq];
        let val = &token[eq + 1..];
        match key {
            "start" => {
                if let Ok(n) = val.parse() {
                    pb.number_start = Some(n);
                }
            }
            "style" => {
                pb.number_style = Some(if val == "alt" {
                    PageNumberStyle::Alternative
                } else {
                    PageNumberStyle::Standard
                });
            }
            "header" => pb.header_override = Some(val.to_string()),
            _ => {}
        }
    }
    pb
}

fn format_page_break(pb: &PageBreak) -> String {
    let mut s = String::from(PAGE_BREAK_MARKER);
    if let Some(n) = pb.number_start {
        s.push_str(&format!(" start={}", n));
    }
    if let Some(style) = pb.number_style {
        s.push_str(match style {
            PageNumberStyle::Alternative => " style=alt",
            PageNumberStyle::Standard => " style=standard",
        });
    }
    if let Some(h) = &pb.header_override {
        s.push_str(&format!(" header={}", h));
    }
    s
}

// ============================================================
// テスト
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mbr_roundtrip_two_paragraphs() {
        let mbr = "line_width = 32\nlines_per_page = 22\npage_header = true\n---\n⠁⠃⠉\n\n⠙⠑⠋\n";
        let doc = BrailleDocument::parse_mbr(mbr);
        assert_eq!(doc.paragraphs.len(), 2);
        assert_eq!(doc.logical_text(0), "⠁⠃⠉");
        assert_eq!(doc.logical_text(1), "⠙⠑⠋");
        assert_eq!(doc.to_mbr(), mbr);
    }

    #[test]
    fn mbr_parses_front_matter() {
        let mbr = "line_width = 40\nlines_per_page = 18\npage_header = false\ntitle = ⠞⠊\n---\n⠁\n";
        let doc = BrailleDocument::parse_mbr(mbr);
        assert_eq!(doc.config.line_width, 40);
        assert_eq!(doc.config.lines_per_page, 18);
        assert!(!doc.config.page_header);
        assert_eq!(doc.config.title.as_deref(), Some("⠞⠊"));
    }

    #[test]
    fn mbr_comments_ignored_in_front_matter() {
        let mbr = "line_width = 32 # マス数\n# まるごとコメント\nlines_per_page = 22\n---\n⠁\n";
        let doc = BrailleDocument::parse_mbr(mbr);
        assert_eq!(doc.config.line_width, 32);
        assert_eq!(doc.config.lines_per_page, 22);
    }

    #[test]
    fn mbr_multi_physical_line_paragraph_preserved() {
        let mbr = "line_width = 32\nlines_per_page = 22\npage_header = true\n---\n⠁⠃\n⠉⠙\n\n⠑\n";
        let doc = BrailleDocument::parse_mbr(mbr);
        assert_eq!(doc.paragraphs.len(), 2);
        assert_eq!(doc.paragraphs[0].len(), 2);
        assert!(!doc.paragraphs[0][0].logical_end);
        assert!(doc.paragraphs[0][1].logical_end);
        assert_eq!(doc.to_mbr(), mbr);
    }

    #[test]
    fn mbr_page_break_with_overrides_roundtrip() {
        let mbr = "line_width = 32\nlines_per_page = 22\npage_header = true\n---\n⠁⠃\n==== start=5 style=alt header=⠭\n⠉⠙\n";
        let doc = BrailleDocument::parse_mbr(mbr);
        assert_eq!(doc.paragraphs.len(), 1);
        let pb = doc.paragraphs[0][0].page_break.as_ref().unwrap();
        assert_eq!(pb.number_start, Some(5));
        assert_eq!(pb.number_style, Some(PageNumberStyle::Alternative));
        assert_eq!(pb.header_override.as_deref(), Some("⠭"));
        assert!(doc.paragraphs[0][1].logical_end);
        assert_eq!(doc.to_mbr(), mbr);
    }

    #[test]
    fn mbr_page_break_at_group_start_ignored() {
        // グループ先頭の ==== は無視される（直前の物理行がない）。
        let mbr =
            "line_width = 32\nlines_per_page = 22\npage_header = true\n---\n==== start=2\n⠁\n";
        let doc = BrailleDocument::parse_mbr(mbr);
        assert_eq!(doc.paragraphs.len(), 1);
        assert_eq!(doc.paragraphs[0].len(), 1);
        assert!(doc.paragraphs[0][0].page_break.is_none());
    }

    #[test]
    fn mbr_empty_paragraph_roundtrip() {
        // 空段落 = 空行。2段落のうち中間が空。
        let mbr = "line_width = 32\nlines_per_page = 22\npage_header = true\n---\n⠁\n\n\n⠃\n";
        let doc = BrailleDocument::parse_mbr(mbr);
        assert_eq!(doc.paragraphs.len(), 3);
        assert_eq!(doc.logical_text(0), "⠁");
        assert_eq!(doc.logical_text(1), "");
        assert_eq!(doc.logical_text(2), "⠃");
        assert_eq!(doc.to_mbr(), mbr);
    }

    #[test]
    fn mbr_no_separator_treats_all_as_body() {
        let doc = BrailleDocument::parse_mbr("⠁\n⠃\n");
        assert_eq!(doc.paragraphs.len(), 1);
        assert_eq!(doc.paragraphs[0].len(), 2);
    }

    #[test]
    fn from_reflowed_paragraphs_wraps() {
        let config = DocumentConfig {
            line_width: 5,
            ..DocumentConfig::default()
        };
        // 4 + space + 4 = 9 セル > 5 → 2物理行
        let para = "⠁⠁⠁⠁⠀⠃⠃⠃⠃".to_string();
        let doc = BrailleDocument::from_reflowed_paragraphs(&[para], config);
        assert_eq!(doc.paragraphs.len(), 1);
        assert_eq!(doc.paragraphs[0].len(), 2);
        assert!(doc.paragraphs[0][1].logical_end);
    }

    #[test]
    fn from_reflowed_paragraphs_empty_becomes_empty_line() {
        let doc =
            BrailleDocument::from_reflowed_paragraphs(&[String::new()], DocumentConfig::default());
        assert_eq!(doc.paragraphs.len(), 1);
        assert_eq!(doc.paragraphs[0].len(), 1);
        assert_eq!(doc.paragraphs[0][0].content, "");
        assert!(doc.paragraphs[0][0].logical_end);
    }
}
