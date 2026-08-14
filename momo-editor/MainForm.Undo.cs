using System.Windows.Forms.Automation;
using Momo;

namespace MomoEditor;

// 元に戻す/やり直し（スナップショットベース）。_document 全体のディープクローンを
// スタックに積む方式。個々の編集操作の逆操作を書かない代わりに、常に
// 「編集前の状態そのもの」を保持するため、正しさの検証が容易。
public partial class MainForm
{
    private const int MaxUndoSteps = 64; // 将来の設定ダイアログで可変化予定（今は固定）

    private sealed record EditSnapshot(BrailleDocument Document, int CursorSegment, int CursorOffset);

    // 直近の ReformatAndRender 完了直後の状態（次の編集が来たとき Undo スタックへ積む対象）。
    // _view == null（フォールバック時）は常に null。
    private EditSnapshot? _lastSnapshot;

    private readonly List<EditSnapshot> _undoStack = [];
    private readonly List<EditSnapshot> _redoStack = [];

    private static void PushCapped(List<EditSnapshot> stack, EditSnapshot snapshot)
    {
        stack.Add(snapshot);
        if (stack.Count > MaxUndoSteps) stack.RemoveAt(0);
    }

    private static BrailleDocument CloneDocument(BrailleDocument doc)
    {
        var clone = new BrailleDocument { Config = doc.Config }; // FormatterConfig は不変 record なので共有可
        foreach (var e in doc.Entries) clone.Entries.Add(CloneEntry(e));
        return clone;
    }

    private static DocEntry CloneEntry(DocEntry entry) => entry switch
    {
        TextSegment t => new TextSegment(t.Text, t.ParagraphEnd),
        PageBreakEntry p => new PageBreakEntry(p.Marker),
        _ => throw new NotSupportedException($"Unknown DocEntry type: {entry.GetType()}"),
    };

    private void SmartUndo()
    {
        if (_view == null)
        {
            if (!richTextBox.CanUndo)
            {
                Announce("これ以上元に戻せません", AutomationNotificationProcessing.ImportantMostRecent);
                return;
            }
            richTextBox.Undo();
            Announce("元に戻しました", AutomationNotificationProcessing.ImportantMostRecent);
            return;
        }
        if (_undoStack.Count == 0 || _lastSnapshot == null)
        {
            Announce("これ以上元に戻せません", AutomationNotificationProcessing.ImportantMostRecent);
            return;
        }
        var target = _undoStack[^1];
        _undoStack.RemoveAt(_undoStack.Count - 1);
        PushCapped(_redoStack, _lastSnapshot);
        RestoreSnapshot(target);
        Announce("元に戻しました", AutomationNotificationProcessing.ImportantMostRecent);
    }

    private void SmartRedo()
    {
        if (_view == null)
        {
            if (!richTextBox.CanRedo)
            {
                Announce("これ以上やり直せません", AutomationNotificationProcessing.ImportantMostRecent);
                return;
            }
            richTextBox.Redo();
            Announce("やり直しました", AutomationNotificationProcessing.ImportantMostRecent);
            return;
        }
        if (_redoStack.Count == 0 || _lastSnapshot == null)
        {
            Announce("これ以上やり直せません", AutomationNotificationProcessing.ImportantMostRecent);
            return;
        }
        var target = _redoStack[^1];
        _redoStack.RemoveAt(_redoStack.Count - 1);
        PushCapped(_undoStack, _lastSnapshot);
        RestoreSnapshot(target);
        Announce("やり直しました", AutomationNotificationProcessing.ImportantMostRecent);
    }

    private void RestoreSnapshot(EditSnapshot snapshot)
    {
        // snapshot.Document は積んだ時点で独立クローン済みなので、_lastSnapshot として
        // そのまま使い回してよい（_document は別途フレッシュにクローンし直すため、
        // 以後 _document を書き換えても snapshot 側は影響を受けない）。
        _document = CloneDocument(snapshot.Document);
        RenderDocumentToEditor(snapshot.CursorSegment, snapshot.CursorOffset);
        _lastSnapshot = snapshot;
    }
}
