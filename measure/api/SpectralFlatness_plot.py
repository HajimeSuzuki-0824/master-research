import librosa
import numpy as np
import matplotlib.pyplot as plt

def analyze_SpectralFlatness_plot(filename, outputImage_path):
    y, sr = librosa.load(filename, sr=None)

    # Timbre Toolboxに準拠したパラメータ
    hop_size_sec = 0.0058
    win_size_sec = 0.0232
    hop_length = int(sr * hop_size_sec)
    n_fft = int(sr * win_size_sec)

    # 振幅スペクトルの取得
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # Spectral Flatness の計算（Matlabと同じ定義）
    eps = 1e-10
    flatness = np.exp(np.mean(np.log(S + eps), axis=0)) / (np.mean(S, axis=0) + eps)

    # 無音フレームを0に
    flatness[np.sum(S, axis=0) == 0] = 0

    # 時間軸の設定
    time_wave = np.arange(len(y)) / sr
    time_spec = np.arange(flatness.shape[0]) * hop_size_sec

    # プロット
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(time_wave, y, color='gray')
    axs[0].set_title("Waveform")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    axs[1].plot(time_spec, flatness, color='purple', label="Spectral Flatness")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Flatness")
    axs[1].set_title("Spectral Flatness")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(outputImage_path, format='png')
    return float(np.median(flatness))
