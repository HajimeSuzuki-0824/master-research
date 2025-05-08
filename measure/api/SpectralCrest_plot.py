import librosa
import numpy as np
import matplotlib.pyplot as plt

def analyze_SpectralCrest_plot(filename, outputImage_path):
    """
    Spectral Crestの計算（MedianおよびIQR）を行い、プロット画像を保存する関数。

    Args:
        filename (str): 入力音声ファイルのパス
        outputImage_path (str): 出力プロット画像の保存パス

    Returns:
        tuple: (Spectral Crest Median, Spectral Crest IQR)
    """
    # 音声読み込み
    y, sr = librosa.load(filename, sr=None)

    # Timbre Toolboxと同様のパラメータ
    hop_size_sec = 0.0058
    win_size_sec = 0.0232
    hop_length = int(sr * hop_size_sec)
    n_fft = int(sr * win_size_sec)

    # 振幅スペクトルを取得
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # Spectral Crest の計算（最大値 / 総和）
    crest = np.max(S, axis=0) / (np.sum(S, axis=0) + 1e-10)

    # 無音区間（総和=0）のNaNを0で置換
    crest[np.isnan(crest)] = 0

    # 時間軸
    times = np.arange(crest.shape[0]) * hop_size_sec

    # プロット
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(np.arange(len(y)) / sr, y, color='gray')
    axs[0].set_title("Waveform")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    axs[1].plot(times, crest, color='red', label="Spectral Crest")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Crest")
    axs[1].legend()
    axs[1].set_title("Spectral Crest Over Time")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(outputImage_path, format='png')

    # 統計量の計算
    median_val = np.median(crest)
    iqr_val = np.percentile(crest, 75) - np.percentile(crest, 25)

    return median_val, iqr_val
