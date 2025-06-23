import librosa
import numpy as np
import matplotlib.pyplot as plt

filename = 'sounds/スケール1.wav'
outputImage_path = 'png/test.png'

def analyze_SpectralCentroid_plot(filename, outputImage_path):
    """
    Spectral Centroidの計算を行い、プロットする

    Args:
        filename (str): 読み込む音声ファイルのパス
        outputImage_path (str): 作成したプロット画像を保存するパス

    Returns:
        numpy.ndarray: 計算されたSpectral Centroid（Hz単位）
    """
    # 音声ファイルの読み込み
    y, sr = librosa.load(filename, sr=None)

    # Timbre Toolboxと同じウィンドウ/ホップサイズに設定
    hop_size_sec = 0.0058
    win_size_sec = 0.0232
    hop_length = int(sr * hop_size_sec)
    n_fft = int(sr * win_size_sec)

    # スペクトル重心の計算（librosaの公式関数を使用）
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]

    # 波形プロット用時間軸
    time_wave = np.arange(len(y)) / sr
    # スペクトル重心プロット用時間軸
    time_cent = np.arange(spec_cent.shape[-1]) * hop_size_sec

    # プロット
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(time_wave, y, color='gray')
    axs[0].set_title("Waveform")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    axs[1].plot(time_cent, spec_cent, color='b', label="Spectral Centroid")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Frequency (Hz)")
    axs[1].set_title("Spectral Centroid")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(outputImage_path, format='png')
    return float(np.median(spec_cent))


# a = analyze_SpectralCent_plot(filename, outputImage_path)
# print(f"Spectral Centroid Median: {np.median(a)}")
