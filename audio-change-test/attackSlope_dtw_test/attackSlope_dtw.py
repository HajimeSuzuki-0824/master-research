import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import savgol_filter
from dtw import dtw
import os

HOP_LENGTH = 1024
FRAME_LENGTH = HOP_LENGTH * 4

def load_audio(filepath):
    y, sr = sf.read(filepath)
    return y, sr

def compute_attack_slope(y, sr):
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    return rms

def extract_profiles(feature_seq, sr, onsets, offsets):
    frame_rate = sr / HOP_LENGTH
    profiles = {}
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        start = int(onset * frame_rate)
        end = int(offset * frame_rate)
        if end > len(feature_seq):
            continue
        profiles[i] = feature_seq[start:end]
    return profiles

def stretch_profile(profile, target_len):
    x_original = np.linspace(0, 1, len(profile))
    x_target = np.linspace(0, 1, target_len)
    return np.interp(x_target, x_original, profile)

def apply_profiles(y, sr, onsets, offsets, mapped_profiles):
    frame_rate = sr / HOP_LENGTH
    y_mod = y.copy()
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        if i not in mapped_profiles:
            continue
        start_frame = int(onset * frame_rate)
        end_frame = int(offset * frame_rate)
        profile = mapped_profiles[i]
        n_frames = end_frame - start_frame
        if n_frames <= 0:
            continue
        stretched = stretch_profile(profile, n_frames)
        if len(stretched) >= 9:
            stretched = savgol_filter(stretched, window_length=9, polyorder=2)
        max_val = np.max(stretched) + 1e-6
        for j in range(n_frames):
            gain = stretched[j] / max_val
            gain = max(gain, 0.1)
            frame_start = (start_frame + j) * HOP_LENGTH
            frame_end = frame_start + HOP_LENGTH
            if frame_end >= len(y_mod):
                break
            gain = stretched[j] / (np.max(stretched) + 1e-6)
            y_mod[frame_start:frame_end] *= gain
    return y_mod

def main(A_name, B_name, out_path):
    onset_offset_df_a = pd.read_csv(f"csv/{A_name}.csv")
    onset_offset_df_b = pd.read_csv(f"csv/{B_name}.csv")

    onsets_a = onset_offset_df_a["onset_times"].astype(float).to_numpy()
    offsets_a = onset_offset_df_a["offset_times"].astype(float).to_numpy()

    onsets_b = onset_offset_df_b["onset_times"].astype(float).to_numpy()
    offsets_b = onset_offset_df_b["offset_times"].astype(float).to_numpy()

    y_a, sr_a = load_audio(f"sounds/inputs/{A_name}.wav")
    y_b, sr_b = load_audio(f"sounds/inputs/{B_name}.wav")

    attack_a = compute_attack_slope(y_a, sr_a)
    profiles_a = extract_profiles(attack_a, sr_a, onsets_a, offsets_a)

    onsets_a_ = onsets_a.reshape(-1, 1)
    onsets_b_ = onsets_b.reshape(-1, 1)
    dist_fn = lambda x, y: np.abs(x - y)
    alignment = dtw(onsets_a_, onsets_b_, dist_method=dist_fn, step_pattern="symmetric1", keep_internals=True)

    path_a, path_b = alignment.index1, alignment.index2
    mapped_profiles = {}
    for ai, bi in zip(path_a, path_b):
        if ai not in profiles_a:
            continue
        mapped_profiles[bi] = profiles_a[ai]

    y_b_mod = apply_profiles(y_b, sr_b, onsets_b, offsets_b, mapped_profiles)
    sf.write(out_path, y_b_mod, sr_b)
    print(f"[INFO] Saved to {out_path}")

if __name__ == "__main__":
    A_name = "Ssax_1_smile"
    B_name = "Ssax_1_daw"
    out_path = f"sounds/outputs/{B_name}_attackTrans_smile.wav"
    main(A_name, B_name)
