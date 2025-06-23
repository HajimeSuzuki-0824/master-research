import librosa
import numpy as np
import matplotlib.pyplot as plt

def analyze_SpectralVariation_plot(filename, outputImage_path):
    y, sr = librosa.load(filename, sr=None)

    # Timbre Toolbox互換のSTFTパラメータ
    hop_size_sec = 0.0058
    win_size_sec = 0.0232
    hop_length = int(sr * hop_size_sec)
    n_fft = int(sr * win_size_sec)

    # STFTの振幅スペクトル
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # 遅延スペクトル（1フレーム前）を作成（ゼロパディング）
    delayed = np.pad(S[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=0)

    # 分子：フレーム同士の内積
    num = np.sum(S * delayed, axis=0)

    # 分母：各ベクトルのL2ノルムの積
    denom = np.sum(S**2, axis=0) * np.sum(delayed**2, axis=0)

    # Spectral Variation の定義（1 - コサイン類似度）
    spec_var = 1 - (num / (np.sqrt(denom) + 1e-10))
    spec_var[denom == 0] = 0  # 無音フレーム処理

    # 時間軸の作成
    time_wave = np.arange(len(y)) / sr
    time_spec = np.arange(len(spec_var)) * hop_size_sec

    # プロット
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(time_wave, y, color='gray')
    axs[0].set_title("Waveform")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    axs[1].plot(time_spec, spec_var, color='teal', label="Spectral Variation")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Variation")
    axs[1].set_title("Spectral Variation")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(outputImage_path, format='png')

    # 中央値とIQRを計算
    median = float(np.median(spec_var))
    q75, q25 = np.percentile(spec_var, [75, 25])
    iqr = float(q75 - q25)

    return median, iqr