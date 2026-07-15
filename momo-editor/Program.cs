namespace MomoEditor;

static class Program
{
    /// <summary>
    ///  The main entry point for the application.
    /// </summary>
    [STAThread]
    static void Main(string[] args)
    {
        // To customize application configuration such as set high DPI settings or default font,
        // see https://aka.ms/applicationconfiguration.
        ApplicationConfiguration.Initialize();
        // 先頭の引数を開くファイルパスとして扱う（Explorer の「プログラムから開く」や
        // 関連付けから渡される）。引数が無ければ null を渡して新規ドキュメントで起動する。
        string? initialPath = args.Length > 0 ? args[0] : null;
        Application.Run(new MainForm(initialPath));
    }
}