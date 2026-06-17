using System.Text;

namespace MomoEditor;

/// <summary>
/// Momo Braille Document (.mbr) 形式の読み書き。
///
/// フォーマット:
///   line_width = 32
///   lines_per_page = 22
///   page_header = true
///   title = ⠞⠊⠞⠇⠑          # 省略可
///   ---
///   ⠁⠃⠉⠙⠑⠋⠛⠓⠊⠚             # 論理行1・物理行1
///   ====                     # 強制改ページ（直前の物理行の後でページが変わる）
///   ⠁⠃⠉⠙                    # 論理行1・物理行2（改ページ後）
///                            # 空行 = 論理行の区切り
///   ⠑⠋⠛⠓⠊⠚                  # 論理行2（物理行1のみ）
///
/// ==== マーカーはオプションのパラメータを持てる:
///   ==== start=5             # ページ番号を5から再開
///   ==== style=alt           # 代替ページ番号スタイルに切り替え
///   ==== header=⠁⠃⠉⠙        # ヘッダ文字列をオーバーライド
///   （複数組み合わせ可）
///
/// 空行が論理行の区切りを兼ねる。
/// 2連続空行は「論理行の区切り＋空論理行の区切り」を意味する。
/// </summary>
static class MbrFormat
{
    private const string Separator = "---";
    private const string PageBreakMarker = "====";

    // ---- 読み込み ----

    public static BrailleDocument Read(string text)
    {
        var lines = text.Split('\n').Select(l => l.TrimEnd('\r')).ToArray();
        var sepIdx = Array.FindIndex(lines, l => l == Separator);

        IEnumerable<string> headerLines = sepIdx >= 0 ? lines[..sepIdx] : [];
        IEnumerable<string> bodyLines   = sepIdx >= 0 ? lines[(sepIdx + 1)..] : lines;

        var doc = new BrailleDocument
        {
            Config = ParseConfig(headerLines),
        };

        // 空行で区切りながら論理行グループを構築
        var groups = new List<List<string>>();
        var currentGroup = new List<string>();
        foreach (var line in bodyLines)
        {
            if (line.Length == 0)
            {
                groups.Add([.. currentGroup]);
                currentGroup.Clear();
            }
            else
            {
                currentGroup.Add(line);
            }
        }
        if (currentGroup.Count > 0)
            groups.Add([.. currentGroup]);

        // 末尾の空論理行を除去
        while (groups.Count > 0 && groups[^1].Count == 0)
            groups.RemoveAt(groups.Count - 1);

        foreach (var group in groups)
        {
            if (group.Count == 0)
            {
                doc.Paragraphs.Add([new PhysicalLine("", true)]);
                continue;
            }

            var physLines = new List<PhysicalLine>();
            foreach (var line in group)
            {
                if (line.StartsWith(PageBreakMarker))
                {
                    // ==== マーカー: 直前の物理行に PageBreak を付ける
                    // グループ先頭にある場合（前の物理行がない）は無視する
                    if (physLines.Count > 0)
                        physLines[^1] = physLines[^1] with { PageBreak = ParsePageBreak(line) };
                }
                else
                {
                    physLines.Add(new PhysicalLine(line, false));
                }
            }

            if (physLines.Count == 0)
                physLines.Add(new PhysicalLine("", true));
            else
                physLines[^1] = physLines[^1] with { LogicalEnd = true };

            doc.Paragraphs.Add(physLines);
        }

        return doc;
    }

    private static PageBreakInfo ParsePageBreak(string line)
    {
        string? headerOverride = null;
        int? numberStart = null;
        PageNumberStyle? numberStyle = null;

        var tokens = line[PageBreakMarker.Length..].Trim()
            .Split(' ', StringSplitOptions.RemoveEmptyEntries);

        foreach (var token in tokens)
        {
            var eq = token.IndexOf('=');
            if (eq < 0) continue;
            var key = token[..eq];
            var val = token[(eq + 1)..];
            switch (key)
            {
                case "start":
                    if (int.TryParse(val, out int n)) numberStart = n;
                    break;
                case "style":
                    numberStyle = val == "alt" ? PageNumberStyle.Alternative : PageNumberStyle.Standard;
                    break;
                case "header":
                    headerOverride = val;
                    break;
            }
        }

        return new PageBreakInfo(headerOverride, numberStart, numberStyle);
    }

    private static FormatterConfig ParseConfig(IEnumerable<string> lines)
    {
        var config = new FormatterConfig();
        foreach (var raw in lines)
        {
            var line = raw.Split('#')[0].Trim();
            if (line.Length == 0) continue;

            var eq = line.IndexOf('=');
            if (eq < 0) continue;

            var key   = line[..eq].Trim();
            var value = line[(eq + 1)..].Trim();

            config = key switch
            {
                "line_width"    => config with { LineWidth    = int.TryParse(value, out var v) ? v : config.LineWidth },
                "lines_per_page"=> config with { LinesPerPage = int.TryParse(value, out var v) ? v : config.LinesPerPage },
                "page_header"   => config with { PageHeader   = value == "true" },
                "title"         => config with { Title        = value.Length > 0 ? value : null },
                _               => config,
            };
        }
        return config;
    }

    // ---- 書き出し ----

    public static string Write(BrailleDocument doc)
    {
        var sb = new StringBuilder();
        var c = doc.Config;

        sb.AppendLine($"line_width = {c.LineWidth}");
        sb.AppendLine($"lines_per_page = {c.LinesPerPage}");
        sb.AppendLine($"page_header = {(c.PageHeader ? "true" : "false")}");
        if (c.Title is { Length: > 0 } title)
            sb.AppendLine($"title = {title}");
        sb.AppendLine(Separator);

        bool first = true;
        foreach (var paragraph in doc.Paragraphs)
        {
            if (!first) sb.AppendLine();
            first = false;

            bool isEmpty = paragraph.Count == 0
                || (paragraph.Count == 1 && paragraph[0].Content.Length == 0);
            if (!isEmpty)
            {
                foreach (var line in paragraph)
                {
                    sb.AppendLine(line.Content);
                    if (line.PageBreak != null)
                        sb.AppendLine(FormatPageBreak(line.PageBreak));
                }
            }
        }

        return sb.ToString();
    }

    private static string FormatPageBreak(PageBreakInfo pb)
    {
        var sb = new StringBuilder(PageBreakMarker);
        if (pb.NumberStart.HasValue)
            sb.Append($" start={pb.NumberStart.Value}");
        if (pb.NumberStyle.HasValue)
            sb.Append(pb.NumberStyle.Value == PageNumberStyle.Alternative ? " style=alt" : " style=standard");
        if (pb.HeaderOverride != null)
            sb.Append($" header={pb.HeaderOverride}");
        return sb.ToString();
    }
}
