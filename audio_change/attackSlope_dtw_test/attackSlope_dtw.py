import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from dtw import dtw
import os

HOP_LENGTH = 512
FRAME_LENGTH = 2048


def load_audio(filepath):
    y, sr = sf.read(filepath)
    return y, sr


def compute_db_envelope(y, sr):
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH, frame_length=FRAME_LENGTH)[0]
    env_db = 20 * np.log10(rms + 1e-6)
    return env_db


def extract_profiles(y, sr, onsets, window_sec=0.1):
    env_db = compute_db_envelope(y, sr)
    frame_rate = sr / HOP_LENGTH
    window_len = int(window_sec * frame_rate)
    profiles = {}
    for i, onset in enumerate(onsets):
        start = int(onset * frame_rate)
        end = start + window_len
        if end >= len(env_db):
            continue
        profiles[i] = env_db[start:end]
    return profiles


def stretch_profile(profile, target_len):
    x_original = np.linspace(0, 1, len(profile))
    x_target = np.linspace(0, 1, target_len)
    return np.interp(x_target, x_original, profile)


def apply_profiles(y, sr, onsets, mapped_profiles):
    frame_rate = sr / HOP_LENGTH
    y_mod = y.copy()
    for i, onset in enumerate(onsets):
        if i not in mapped_profiles:
            continue
        start_frame = int(onset * frame_rate)
        profile_db = mapped_profiles[i]
        n_frames = len(profile_db)
        start_sample = start_frame * HOP_LENGTH
        for j in range(n_frames):
            frame_start = start_sample + j * HOP_LENGTH
            frame_end = frame_start + HOP_LENGTH
            if frame_end >= len(y_mod):
                break
            seg = y_mod[frame_start:frame_end]
            rms = np.sqrt(np.mean(seg ** 2)) + 1e-6
            target_linear = 10 ** (profile_db[j] / 20)
            gain = target_linear / rms
            y_mod[frame_start:frame_end] *= gain
    return y_mod


def main():
    A_name = "Euph_0_スケール"
    B_name = "Tp_0_スケール"
    
    onsets_a = pd.read_csv(f"csv/{A_name}.csv")['onset_times'].to_numpy()
    onsets_b = pd.read_csv(f"csv/{B_name}.csv")['onset_times'].to_numpy()

    y_a, sr_a = load_audio(f"sounds/{A_name}.wav")
    y_b, sr_b = load_audio(f"sounds/{B_name}.wav")
    assert sr_a == sr_b

    profiles_a = extract_profiles(y_a, sr_a, onsets_a)

    onsets_a_ = onsets_a.reshape(-1, 1)
    onsets_b_ = onsets_b.reshape(-1, 1)
    dist_fn = lambda x, y: np.abs(x - y)
    alignment = dtw(onsets_a_, onsets_b_, dist_method=dist_fn, step_pattern="symmetric1", keep_internals=True)

    path_a, path_b = alignment.index1, alignment.index2
    mapped_profiles = {}
    for ai, bi in zip(path_a, path_b):
        if ai not in profiles_a:
            continue
        target_len = int(0.1 * sr_a / HOP_LENGTH)
        stretched = stretch_profile(profiles_a[ai], target_len)
        mapped_profiles[bi] = stretched

    y_b_mod = apply_profiles(y_b, sr_b, onsets_b, mapped_profiles)
    out_path = f"sounds/{B_name}_profile_transferred.wav"
    sf.write(out_path, y_b_mod, sr_b)
    print(f"[INFO] Saved to {out_path}")


if __name__ == "__main__":
    main()