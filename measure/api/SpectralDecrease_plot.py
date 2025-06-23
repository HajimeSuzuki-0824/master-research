import librosa
import numpy as np
import matplotlib.pyplot as plt

def analyze_SpectralDecrease_plot(filename, outputImage_path):
    y, sr = librosa.load(filename, sr=None)

    # Timbre Toolboxと同様のウィンドウ/ホップサイズ
    hop_size_sec = 0.0058
    win_size_sec = 0.0232
    hop_length = int(sr * hop_size_sec)
    n_fft = int(sr * win_size_sec)

    # 振幅スペクトル
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # Numerator: 各バンドの差分（1番目のバンドとの差）
    numerator = S[1:, :] - S[0:1, :]

    # Denominator: 1/(k - 1) の係数（Peetersの定義）
    k = np.arange(1, S.shape[0])  # index: 1 to N-1
    denominator = 1.0 / k[:, np.newaxis]

    # Spectral Decrease 計算
    spectral_decrease = np.sum(denominator * numerator, axis=0) / (np.sum(S[1:, :], axis=0) + 1e-10)

    # 無音フレーム（0割り）を0にする
    silent = np.sum(S[1:, :], axis=0) == 0
    spectral_decrease[silent] = 0

    # 時間軸
    time_wave = np.arange(len(y)) / sr
    time_spec = np.arange(S.shape[1]) * hop_size_sec

    # プロット
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(time_wave, y, color='gray')
    axs[0].set_title("Waveform")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    axs[1].plot(time_spec, spectral_decrease, color='r', label="Spectral Decrease")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Decrease")
    axs[1].set_title("Spectral Decrease")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(outputImage_path, format='png')
    return float(np.median(spectral_decrease))
