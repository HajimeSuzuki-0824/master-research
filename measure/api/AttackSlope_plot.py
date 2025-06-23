import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def analyze_AttackSlope_plot(filename, outputImage_path, noise_thd_db=-48, attack_thd_db=-30, incr_step_db=6):
    """
    Attack Slope（局所傾きの平均）を計算し、プロット画像を保存する。

    Args:
        filename (str): 音声ファイルのパス
        outputImage_path (str): プロットの保存パス
        noise_thd_db (float): Attack 開始閾値（dBFS）
        attack_thd_db (float): Attack 終了閾値（dBFS）
        incr_step_db (float): 傾き計算時のdBステップ

    Returns:
        float or None: Attack Slope（dB/s）。該当区間がない場合は None。
    """
    # 音声読み込み
    y, sr = librosa.load(filename, sr=None)

    # Rectified envelope（ヒルベルト変換ベース）
    envelope = np.abs(scipy.signal.hilbert(y))
    envelope_db = 20 * np.log10(envelope + 1e-10)  # dBスケール

    # Attack 領域の検出
    sat_indices = np.where(envelope_db > noise_thd_db)[0]
    eat_indices = np.where(envelope_db >= attack_thd_db)[0]
    if len(sat_indices) == 0 or len(eat_indices) == 0:
        return None
    sat = sat_indices[0]
    eat = eat_indices[0]
    if eat <= sat:
        return None

    # Attack区間でのしきい値列（例：-48→-42→...→-20）
    satv = envelope_db[sat]
    eatv = envelope_db[eat]
    thds = np.arange(max(satv, noise_thd_db), eatv + 1e-6, incr_step_db)
    thd_times = []

    for thd in thds:
        idx = np.argmax(envelope_db[sat:eat] >= thd)
        thd_times.append((sat + idx) / sr)

    # 重複を排除し、傾きを計算
    thd_times, idx_unique = np.unique(thd_times, return_index=True)
    thds = thds[idx_unique]
    if len(thds) < 2:
        return None

    slopes = np.diff(thds) / np.diff(thd_times)
    attack_slope = float(np.mean(slopes))

    # プロット
    time_axis = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_axis, envelope_db, label="Envelope (dB)", color='orange')
    ax.set_title("Attack Slope Estimation")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (dB)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(outputImage_path)

    return attack_slope
