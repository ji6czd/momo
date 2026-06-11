use crate::formatter::FormattedDocument;

/// フォーマット済み点字ドキュメントの出力形式。
pub enum OutputFormat {
    /// ページ間をフォームフィード (`\x0C`) で区切ったプレーンテキスト。ファイル拡張子: `.brl`
    BrailleText,
    /// 点字プリンタ向け BASE 形式。512 バイトヘッダ＋ページデータ。ファイル拡張子: `.bse`
    Base,
}

impl OutputFormat {
    /// ドキュメントをこの形式のテキストに変換して返す。
    pub fn write(&self, doc: &FormattedDocument) -> String {
        match self {
            OutputFormat::BrailleText => write_braille_text(doc),
            OutputFormat::Base => write_base_file(doc),
        }
    }
}

fn write_braille_text(doc: &FormattedDocument) -> String {
    let mut out = String::new();
    for (i, page) in doc.pages().iter().enumerate() {
        if i > 0 {
            out.push_str("\x0C\n");
        }
        out.push_str(&page.join("\n"));
        out.push('\n');
    }
    out
}

/// BASE 形式。
/// ヘッダ: 512 バイト（末尾 8 バイトが "ppppccll"）+ 改行。
/// ページデータ: 各ページ必ず `lines_per_page` 行（不足は空行で埋める）。
fn write_base_file(doc: &FormattedDocument) -> String {
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
                out.push_str(line);
            }
            out.push('\n');
        }
    }

    out
}
