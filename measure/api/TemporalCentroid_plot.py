import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def analyze_TemporalCentroid_plot(filename, outputImage_path, noise_thd_db=-60):
    # 音声ファイルの読み込み
    y, sr = librosa.load(filename, sr=None)

    # ヒルベルト変換でパワーエンベロープを取得
    envelope = np.abs(scipy.signal.hilbert(y))
    power_envelope = envelope ** 2
    power_thd = 10 ** (noise_thd_db / 10)

    # 有効な範囲の検出（PowerEnvelope > THD）
    stc = np.argmax(power_envelope > power_thd)
    etc = len(power_envelope) - np.argmax(power_envelope[::-1] > power_thd) - 1

    if etc <= stc:
        return None  # 有効区間がない場合

    # Temporal Centroidの計算（重心位置）
    envtc = power_envelope[stc:etc + 1]
    nsample = np.arange(1, len(envtc) + 1)
    tc = np.sum(envtc * nsample) / (np.sum(envtc) + 1e-10)
    tc_sec = tc / sr

    # プロット生成
    time_axis = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_axis, y, color='gray', label='Waveform')
    ax.plot(time_axis, 10 * np.log10(power_envelope + 1e-10), color='blue', alpha=0.5, label='Power Envelope (dB)')
    ax.axvline(stc / sr, color='green', linestyle='--', label='Start')
    ax.axvline(etc / sr, color='red', linestyle='--', label='End')
    ax.axvline(tc_sec, color='orange', linestyle='-', label='Temporal Centroid')
    ax.set_title("Temporal Centroid")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude / Power (dB)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(outputImage_path)

    return float(tc_sec)
