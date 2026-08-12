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
//! [`BrailleDocument::from_paragraphs`] で組み、折返しは [`crate::formatter::render`] が行う。

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
/// 始まるページ以降に適用される**継続的な状態変更**を表す（`None` は「現在の
/// 状態を維持（上書きしない）」の意味）。一度上書きされた値は、以降のページに
/// そのまま引き継がれ、次の [`PageBreak`] で再び上書きされるまで有効であり続ける
/// （暗黙のページ分割をまたいでも保持される）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PageBreak {
    /// このページ以降のヘッダタイトルを上書きする（点字文字列）。次に上書きされるまで継続。
    pub header_override: Option<String>,
    /// このページ以降、ページヘッダ行の表示有無を切り替える。`None` は現在の状態を維持。
    pub header_enabled: Option<bool>,
    /// このページのページ番号をこの値から再開する（以降は自動的に1ずつ増加する）。
    pub number_start: Option<u32>,
    /// このページ以降のページ番号スタイルを切り替える。次に上書きされるまで継続。
    pub number_style: Option<PageNumberStyle>,
}

impl PageBreak {
    /// 上書き情報を一切持たない（既定の）強制改ページか。
    pub fn is_plain(&self) -> bool {
        self.header_override.is_none()
            && self.header_enabled.is_none()
            && self.number_start.is_none()
            && self.number_style.is_none()
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

/// 論理ドキュメント上の1**セグメント**（強制改行・段落で区切られる単位）。
///
/// `content` は**折返し前**のテキスト。ソフト折返しは保持せず、表示・印刷時に
/// [`crate::formatter::render`] が行幅で折り返す。段落内に複数セグメントがある場合、
/// セグメント境界は**強制改行**（[`logical_end`](Self::logical_end) = false）を表す。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalLine {
    /// 点字セルの文字列（折返し前）。
    pub content: String,
    /// このセグメントで論理行（段落）が終わるか。`false` は次セグメントへの強制改行。
    pub logical_end: bool,
    /// このセグメントの直後で強制改ページするか（上書き情報込み）。
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
///
/// `page_header` / `title` / `number_start` は先頭ページの初期状態であり、
/// [`PageBreak`] の対応フィールド（[`PageBreak::header_enabled`] /
/// [`PageBreak::header_override`] / [`PageBreak::number_start`] /
/// [`PageBreak::number_style`]）によって、以降のページで継続的に上書きできる
/// （複数セクションに分かれた文書の、セクションごとのヘッダ・ページ番号再開に使う）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentConfig {
    /// 1行あたりのマス数（点字セル数）。
    pub line_width: usize,
    /// 1ページあたりの行数（ヘッダ行を含む）。
    pub lines_per_page: usize,
    /// ページヘッダ行を生成するか（先頭ページの初期状態。[`PageBreak::header_enabled`] で以降を上書き可能）。
    pub page_header: bool,
    /// ページヘッダのタイトル（点字文字列。先頭ページの初期状態。`None` でページ番号のみ。
    /// [`PageBreak::header_override`] で以降を上書き可能）。
    pub title: Option<String>,
    /// 開始ページ番号（先頭ページに付く番号。[`PageBreak::number_start`] で以降を上書き可能）。
    pub number_start: u32,
}

impl Default for DocumentConfig {
    fn default() -> Self {
        Self {
            line_width: 32,
            lines_per_page: 22,
            page_header: true,
            title: None,
            number_start: 1,
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

    /// 段落 `index` の論理テキスト（セグメントを連結したもの）を返す。
    pub fn logical_text(&self, index: usize) -> String {
        self.paragraphs[index]
            .iter()
            .map(|l| l.content.as_str())
            .collect()
    }

    /// 1段落＝1論理テキストの単純なドキュメントを組む（強制改行・改ページなし）。
    ///
    /// 各段落は単一セグメント（折返し前のテキスト）になる。折返しは保存・表示時に
    /// [`crate::formatter::render`] が行う。プレディクタ出力や CLI の取り込みに使う。
    pub fn from_paragraphs(paragraphs: &[String], config: DocumentConfig) -> Self {
        let paragraphs = paragraphs
            .iter()
            .map(|para| vec![PhysicalLine::new(para.clone(), true)])
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
    /// ==== start=5 style=alt show_header=false   # 直前の物理行の後で強制改ページ（上書き可。以降のページへ継続）
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
        if c.number_start != 1 {
            out.push_str(&format!("number_start = {}\n", c.number_start));
        }
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
            "number_start" => {
                if let Ok(v) = value.parse() {
                    config.number_start = v;
                }
            }
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
            "show_header" => pb.header_enabled = Some(val == "true"),
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
    if let Some(show) = pb.header_enabled {
        s.push_str(if show {
            " show_header=true"
        } else {
            " show_header=false"
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
    fn mbr_number_start_roundtrip() {
        // number_start は既定(1)以外のときだけ前付けに出力され、往復で保たれる。
        let mbr =
            "line_width = 32\nlines_per_page = 22\npage_header = true\nnumber_start = 5\n---\n⠁\n";
        let doc = BrailleDocument::parse_mbr(mbr);
        assert_eq!(doc.config.number_start, 5);
        assert_eq!(doc.to_mbr(), mbr);
    }

    #[test]
    fn mbr_default_number_start_omitted() {
        // 既定(1)のときは前付けに number_start を出さない。
        let doc = BrailleDocument::parse_mbr(
            "line_width = 32\nlines_per_page = 22\npage_header = true\n---\n⠁\n",
        );
        assert_eq!(doc.config.number_start, 1);
        assert!(!doc.to_mbr().contains("number_start"));
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
    fn mbr_page_break_show_header_roundtrip() {
        let mbr_off = "line_width = 32\nlines_per_page = 22\npage_header = true\n---\n⠁⠃\n==== show_header=false\n⠉⠙\n";
        let doc = BrailleDocument::parse_mbr(mbr_off);
        let pb = doc.paragraphs[0][0].page_break.as_ref().unwrap();
        assert_eq!(pb.header_enabled, Some(false));
        assert_eq!(doc.to_mbr(), mbr_off);

        let mbr_on = "line_width = 32\nlines_per_page = 22\npage_header = false\n---\n⠁⠃\n==== show_header=true\n⠉⠙\n";
        let doc = BrailleDocument::parse_mbr(mbr_on);
        let pb = doc.paragraphs[0][0].page_break.as_ref().unwrap();
        assert_eq!(pb.header_enabled, Some(true));
        assert_eq!(doc.to_mbr(), mbr_on);
    }

    #[test]
    fn mbr_page_break_all_overrides_combined_roundtrip() {
        // 表紙(ヘッダ無し)からセクション開始(ヘッダ有り・番号再開・タイトル変更)への遷移を模す。
        let mbr = "line_width = 32\nlines_per_page = 22\npage_header = false\n---\n⠁⠃\n==== start=5 style=alt show_header=true header=⠭\n⠉⠙\n";
        let doc = BrailleDocument::parse_mbr(mbr);
        let pb = doc.paragraphs[0][0].page_break.as_ref().unwrap();
        assert_eq!(pb.number_start, Some(5));
        assert_eq!(pb.number_style, Some(PageNumberStyle::Alternative));
        assert_eq!(pb.header_enabled, Some(true));
        assert_eq!(pb.header_override.as_deref(), Some("⠭"));
        assert_eq!(doc.to_mbr(), mbr);
    }

    #[test]
    fn page_break_is_plain_accounts_for_header_enabled() {
        assert!(PageBreak::default().is_plain());
        assert!(
            !PageBreak {
                header_enabled: Some(true),
                ..PageBreak::default()
            }
            .is_plain()
        );
        assert!(
            !PageBreak {
                header_enabled: Some(false),
                ..PageBreak::default()
            }
            .is_plain()
        );
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
    fn from_paragraphs_one_segment_each_no_wrap() {
        let config = DocumentConfig {
            line_width: 5,
            ..DocumentConfig::default()
        };
        // 折返しは render の責務。ここでは1段落＝1セグメント（未折返し）。
        let para = "⠁⠁⠁⠁⠀⠃⠃⠃⠃".to_string();
        let doc = BrailleDocument::from_paragraphs(&[para.clone()], config);
        assert_eq!(doc.paragraphs.len(), 1);
        assert_eq!(doc.paragraphs[0].len(), 1);
        assert_eq!(doc.paragraphs[0][0].content, para);
        assert!(doc.paragraphs[0][0].logical_end);
    }

    #[test]
    fn from_paragraphs_empty_becomes_empty_segment() {
        let doc = BrailleDocument::from_paragraphs(&[String::new()], DocumentConfig::default());
        assert_eq!(doc.paragraphs.len(), 1);
        assert_eq!(doc.paragraphs[0].len(), 1);
        assert_eq!(doc.paragraphs[0][0].content, "");
        assert!(doc.paragraphs[0][0].logical_end);
    }

    #[test]
    fn mbr_hard_break_within_paragraph_roundtrip() {
        // 段落内に強制改行（隣接行・空行なし）。2セグメント、最初は logical_end=false。
        let mbr = "line_width = 32\nlines_per_page = 22\npage_header = true\n---\n⠁⠃\n⠉⠙\n";
        let doc = BrailleDocument::parse_mbr(mbr);
        assert_eq!(doc.paragraphs.len(), 1);
        assert_eq!(doc.paragraphs[0].len(), 2);
        assert!(!doc.paragraphs[0][0].logical_end); // 強制改行
        assert!(doc.paragraphs[0][1].logical_end); // 段落終端
        assert_eq!(doc.to_mbr(), mbr);
    }
}
