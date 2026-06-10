use crate::formatter::FormattedDocument;

/// フォーマット済み点字ドキュメントをファイル出力用テキストに変換する。
pub struct BrailleWriter;

impl BrailleWriter {
    /// ページ間をフォームフィード (`\x0C`) で区切ったプレーンテキストを生成する。
    pub fn braille_text(doc: &FormattedDocument) -> String {
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
}
