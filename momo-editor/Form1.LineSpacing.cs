using System.Runtime.InteropServices;

namespace MomoEditor;

// RichTextBox の行間制御。RichTextBox には行間プロパティが無いため、
// EM_SETPARAFORMAT (PARAFORMAT2) を直接送って段落の行間を設定する。
public partial class Form1
{
    // 行間の倍率（1.0 = 標準の1行）。後でユーザー設定できるようにする予定。
    private double _lineSpacing = 1.3;

    private const int EM_SETPARAFORMAT = 0x0447;
    private const uint PFM_LINESPACING = 0x00000100;
    // bLineSpacingRule = 5: dyLineSpacing / 20 を「行数」として解釈する（比例行間）。
    private const byte LINE_SPACING_MULTIPLE = 5;

    [StructLayout(LayoutKind.Sequential)]
    private struct PARAFORMAT2
    {
        public int cbSize;
        public uint dwMask;
        public short wNumbering;
        public short wReserved;
        public int dxStartIndent;
        public int dxRightIndent;
        public int dxOffset;
        public short wAlignment;
        public short cTabCount;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 32)]
        public int[] rgxTabs;
        public int dySpaceBefore;
        public int dySpaceAfter;
        public int dyLineSpacing;
        public short sStyle;
        public byte bLineSpacingRule;
        public byte bOutlineLevel;
        public short wShadingWeight;
        public short wShadingStyle;
        public short wNumberingStart;
        public short wNumberingStyle;
        public short wNumberingTab;
        public short wBorderSpace;
        public short wBorderWidth;
        public short wBorders;
    }

    [DllImport("user32.dll", CharSet = CharSet.Auto)]
    private static extern IntPtr SendMessage(IntPtr hWnd, int msg, IntPtr wParam, ref PARAFORMAT2 lp);

    /// <summary>RichTextBox 全体に行間（_lineSpacing 倍）を適用する。</summary>
    private void ApplyLineSpacing()
    {
        int savedStart = richTextBox.SelectionStart;
        int savedLength = richTextBox.SelectionLength;

        richTextBox.SelectAll();
        var fmt = new PARAFORMAT2
        {
            cbSize = Marshal.SizeOf<PARAFORMAT2>(),
            dwMask = PFM_LINESPACING,
            bLineSpacingRule = LINE_SPACING_MULTIPLE,
            dyLineSpacing = (int)Math.Round(20 * _lineSpacing),
        };
        SendMessage(richTextBox.Handle, EM_SETPARAFORMAT, IntPtr.Zero, ref fmt);

        richTextBox.Select(savedStart, savedLength);
    }
}
