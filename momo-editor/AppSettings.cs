using System.Text.Json;

namespace MomoEditor;

class AppSettings
{
    public List<string> RecentFiles { get; set; } = [];
    public int UndoStackSize { get; set; } = 64;

    private const int MaxRecentFiles = 10;

    private static string SettingsPath => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "MomoEditor", "settings.json");

    /// <summary>設定ファイルを読み込む。存在しない・壊れている・読み込めない場合は既定値を返す
    /// （設定は無くても動く付随機能のため、読み込み失敗で起動を妨げない）。</summary>
    public static AppSettings Load()
    {
        try
        {
            if (File.Exists(SettingsPath))
            {
                var json = File.ReadAllText(SettingsPath);
                var settings = JsonSerializer.Deserialize<AppSettings>(json);
                if (settings != null) return settings;
            }
        }
        catch (Exception)
        {
            // 破損・アクセス不可時は既定値にフォールバック
        }
        return new AppSettings();
    }

    public void Save()
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(SettingsPath)!);
            var json = JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(SettingsPath, json);
        }
        catch (Exception)
        {
            // 保存失敗は致命的ではないため無視する
        }
    }

    /// <summary>指定パスを最近使ったファイルの先頭に追加する（既にあれば移動）。
    /// 上限を超えたら古いものを捨てる。呼び出しのたびに保存する。</summary>
    public void AddRecentFile(string path)
    {
        RecentFiles.RemoveAll(p => string.Equals(p, path, StringComparison.OrdinalIgnoreCase));
        RecentFiles.Insert(0, path);
        if (RecentFiles.Count > MaxRecentFiles)
            RecentFiles.RemoveRange(MaxRecentFiles, RecentFiles.Count - MaxRecentFiles);
        Save();
    }
}
