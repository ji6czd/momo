using System.Media;
using System.Text;

namespace MomoEditor;

/// <summary>
/// 操作に対する短いサウンドフィードバック（イヤコン）を鳴らす。
/// サイン波を PCM WAV としてメモリ上に生成し、System.Media.SoundPlayer で非同期再生する。
/// 各イヤコンは起動時に一度だけ生成してキャッシュするため、再生時の遅延と割り当てを抑えられる。
/// </summary>
internal static class Earcon
{
    private const int SampleRate = 44100;

    // 点字入力が成立したときのクリック音（2000Hz / 0.01秒のサイン波）。
    private static readonly SoundPlayer ClickPlayer = CreateSinePlayer(2000.0, 0.01, 0.4);

    /// <summary>点字セルが1つ入力されたときのクリック音を鳴らす。</summary>
    public static void PlayClick() => ClickPlayer.Play();

    /// <summary>
    /// 指定した周波数・長さ・音量のサイン波を再生する <see cref="SoundPlayer"/> を生成する。
    /// </summary>
    /// <param name="frequencyHz">周波数（Hz）。</param>
    /// <param name="durationSec">長さ（秒）。</param>
    /// <param name="volume">振幅（0.0～1.0）。</param>
    private static SoundPlayer CreateSinePlayer(double frequencyHz, double durationSec, double volume)
    {
        var wav = BuildSineWav(frequencyHz, durationSec, volume);
        // SoundPlayer.Play は非同期再生のため、ストリームは再生中も生かしておく必要がある。
        // SoundPlayer ごと static で保持するので問題ない。
        var player = new SoundPlayer(new MemoryStream(wav));
        player.Load();
        return player;
    }

    /// <summary>サイン波の 16bit モノラル PCM WAV をバイト列として生成する。</summary>
    private static byte[] BuildSineWav(double frequencyHz, double durationSec, double volume)
    {
        const short channels = 1;
        const short bitsPerSample = 16;
        const short blockAlign = channels * bitsPerSample / 8;
        const int byteRate = SampleRate * blockAlign;

        int sampleCount = (int)(SampleRate * durationSec);
        int dataSize = sampleCount * blockAlign;

        using var ms = new MemoryStream(44 + dataSize);
        using var bw = new BinaryWriter(ms);

        // RIFF ヘッダー
        bw.Write(Encoding.ASCII.GetBytes("RIFF"));
        bw.Write(36 + dataSize);
        bw.Write(Encoding.ASCII.GetBytes("WAVE"));
        // fmt チャンク
        bw.Write(Encoding.ASCII.GetBytes("fmt "));
        bw.Write(16);             // チャンクサイズ
        bw.Write((short)1);       // PCM
        bw.Write(channels);
        bw.Write(SampleRate);
        bw.Write(byteRate);
        bw.Write(blockAlign);
        bw.Write(bitsPerSample);
        // data チャンク
        bw.Write(Encoding.ASCII.GetBytes("data"));
        bw.Write(dataSize);

        // 始端・終端のプチノイズ（クリックノイズ）を抑えるための短いフェード。
        int fade = Math.Min(sampleCount / 8, SampleRate / 200);
        for (int i = 0; i < sampleCount; i++)
        {
            double envelope = 1.0;
            if (i < fade) envelope = (double)i / fade;
            else if (i >= sampleCount - fade) envelope = (double)(sampleCount - i) / fade;

            double sample = Math.Sin(2.0 * Math.PI * frequencyHz * i / SampleRate) * volume * envelope;
            bw.Write((short)(sample * short.MaxValue));
        }

        bw.Flush();
        return ms.ToArray();
    }
}
