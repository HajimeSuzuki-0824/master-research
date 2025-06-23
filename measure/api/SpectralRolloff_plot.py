import librosa
import numpy as np
import matplotlib.pyplot as plt

def analyze_SpectralRolloff_plot(filename, outputImage_path, roll_percent=0.95):
    # 音声読み込み
    y, sr = librosa.load(filename, sr=None)

    # Timbre Toolboxに準じたパラメータ
    hop_size_sec = 0.0058
    win_size_sec = 0.0232
    hop_length = int(sr * hop_size_sec)
    n_fft = int(sr * win_size_sec)

    # Rolloff計算（Hz単位）
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft,
                                               hop_length=hop_length,
                                               roll_percent=roll_percent)[0]

    # 時間軸
    time_wave = np.arange(len(y)) / sr
    time_spec = np.arange(len(rolloff)) * hop_size_sec

    # プロット
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(time_wave, y, color='gray')
    axs[0].set_title("Waveform")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    axs[1].plot(time_spec, rolloff, color='orange', label=f"Spectral Rolloff ({int(roll_percent*100)}%)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Rolloff (Hz)")
    axs[1].set_title("Spectral Rolloff")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(outputImage_path, format='png')

    # IQR計算
    q75, q25 = np.percentile(rolloff, [75, 25])
    iqr = q75 - q25

    return float(iqr)
