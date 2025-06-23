import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

SR = 44100
HOP_LENGTH = 512
FRAME_LENGTH = 2048

# =====================================
# パスの指定とファイル名生成
# =====================================
sound_num = 1  # 0:スケール, 1:Classic, 2:Jazz
valence_num = 2  # 0:スケール, 1:smile, 2:sad
valence_list = ["スケール", "smile", "sad"]
sound_name = ['Ssax', 'Tp', 'Hr', 'Tb', 'Euph']
if sound_num == 0:
    valence_num = 0
num_val = f"_{sound_num}_{valence_list[valence_num]}"
for i in range(len(sound_name)):
    sound_name[i] = sound_name[i] + num_val
if sound_num == 0:
    onsets_count = 17
elif sound_num == 1:
    onsets_count = 12
else:
    onsets_count = 15

sound_name = ['Tp_1_daw', 'Ssax_1_daw', 'Tb_1_daw']

def load_audio(filepath):
    y, _ = librosa.load(filepath, sr=SR)
    return y


def compute_features(y):
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    S = np.abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
    flux = np.pad(flux, (1, 0), mode='constant')  # 時系列を揃える
    return rms, flux


def compute_score(rms, flux):
    drms = np.abs(np.diff(rms, prepend=rms[0]))
    dflux = np.abs(np.diff(flux, prepend=flux[0]))

    norm_drms = librosa.util.normalize(drms)
    norm_dflux = librosa.util.normalize(dflux)

    # 単純な加重スコア
    score = 0.6 * norm_dflux + 0.4 * norm_drms
    return score


def detect_onsets(score, y, count):
    peaks, _ = find_peaks(score, distance=int(0.15 * SR / HOP_LENGTH))  # 約150ms以上離す
    peak_scores = score[peaks]
    top_indices = peaks[np.argsort(peak_scores)[-count:]]  # 上位N個を選択
    onset_times = librosa.frames_to_time(top_indices, sr=SR, hop_length=HOP_LENGTH)
    onset_times.sort()
    return onset_times


def save_plot(score, onsets, out_path, name):
    times = librosa.frames_to_time(np.arange(len(score)), sr=SR, hop_length=HOP_LENGTH)
    plt.figure(figsize=(12, 5))
    plt.plot(times, score, label='Combined Score')
    plt.vlines(onsets, 0, 1, color='r', linestyle='--', label='Detected Onsets')
    plt.title(f"Simple Onset Detection: {name}")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    for name in sound_name:
        filename = f"sounds/{name}.wav"
        out_csv = f"outputs/csv/{name}.csv"
        out_plot = f"outputs/png/{name}.png"

        y = load_audio(filename)
        rms, flux = compute_features(y)
        score = compute_score(rms, flux)
        onsets = detect_onsets(score, y, onsets_count)

        pd.DataFrame({'onset_times': onsets}).to_csv(out_csv, index=False)
        print(f"[INFO] Onset times saved to: {out_csv}")

        save_plot(score, onsets, out_plot, name)
        print(f"[INFO] Plot saved to: {out_plot}")


if __name__ == "__main__":
    main()
