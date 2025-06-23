import numpy as np
import pandas as pd
import soundfile as sf
import scipy.optimize
import librosa
from dtw import dtw
import os
import gc

HOP_LENGTH = 512
FRAME_LENGTH = HOP_LENGTH * 4

def load_audio(filepath):
    y, sr = sf.read(filepath)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y, sr

def compute_spectral_centroid(y, sr, hop_length=512, n_fft=2048):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).reshape(-1, 1)
    numerator = np.sum(freqs * S, axis=0)
    denominator = np.sum(S, axis=0) + 1e-6
    centroid = numerator / denominator
    return centroid

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

def apply_profiles(y, sr, onsets, offsets, mapped_profiles, n_bands=8):
    y_mod = np.zeros_like(y)

    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        if i not in mapped_profiles:
            continue

        start_sample = int(onset * sr)
        end_sample = int(offset * sr)
        if end_sample > len(y):
            continue

        y_seg = y[start_sample:end_sample]
        if y_seg.ndim > 1:
            y_seg = np.mean(y_seg, axis=1)
        if len(y_seg) < 2048:
            continue

        profile = mapped_profiles[i]
        S = librosa.stft(y_seg, n_fft=2048, hop_length=HOP_LENGTH, dtype=np.complex64)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        mag_avg = np.mean(np.abs(S), axis=1)
        mag_cumsum = np.cumsum(np.sort(mag_avg)[::-1])
        mag_total = mag_cumsum[-1]
        cutoff = 0.90 * mag_total
        sorted_mag = np.sort(mag_avg)[::-1]
        threshold = sorted_mag[np.searchsorted(mag_cumsum, cutoff)]
        valid_bins = np.where(mag_avg >= threshold)[0]
        if len(valid_bins) == 0:
            continue
        f_min = max(freqs[valid_bins[0]], 1)
        f_max = freqs[valid_bins[-1]]

        log_min = np.log10(f_min)
        log_max = np.log10(f_max)
        band_edges = np.logspace(log_min, log_max, n_bands + 1)
        band_edges[-1] = np.inf

        stretched = stretch_profile(profile, S.shape[1])

        for t in range(S.shape[1]):
            S_t = np.abs(S[:, t])
            phase = np.angle(S[:, t])
            SC_target = stretched[t]

            def loss(w):
                weights = np.zeros_like(freqs)
                for b_idx in range(len(band_edges)-1):
                    f_min = band_edges[b_idx]
                    f_max = band_edges[b_idx+1]
                    weights += w[b_idx] * ((freqs >= f_min) & (freqs < f_max))

                S_mod = S_t * weights
                SC = np.sum(freqs * S_mod) / (np.sum(S_mod) + 1e-6)
                return (SC - SC_target) ** 2 + 0.001 * np.sum((w - 1) ** 2)

            result = scipy.optimize.minimize(
                loss, x0=np.ones(n_bands),
                method='L-BFGS-B',
                bounds=[(0.5, 2)] * n_bands,
                options={'maxiter': 20}
            )

            if result.success:
                w_opt = result.x
                weights = np.zeros_like(freqs)
                for b_idx in range(len(band_edges)-1):
                    f_min = band_edges[b_idx]
                    f_max = band_edges[b_idx+1]
                    weights += w_opt[b_idx] * ((freqs >= f_min) & (freqs < f_max))

                S[:, t] = (S_t * weights) * np.exp(1j * phase)
                del result
                gc.collect()

        y_seg_mod = librosa.istft(S, hop_length=HOP_LENGTH, length=len(y_seg))
        y_mod[start_sample:end_sample] += y_seg_mod[:end_sample - start_sample]

    return y_mod


def main():
    A_name = "Ssax_1_smile"
    B_name = "Ssax_1_daw"

    df_a = pd.read_csv(f"csv/{A_name}.csv")
    df_b = pd.read_csv(f"csv/{B_name}.csv")

    onsets_a = df_a["onset_times"].astype(float).to_numpy()
    offsets_a = df_a["offset_times"].astype(float).to_numpy()

    onsets_b = df_b["onset_times"].astype(float).to_numpy()
    offsets_b = df_b["offset_times"].astype(float).to_numpy()

    y_a, sr_a = load_audio(f"sounds/inputs/{A_name}.wav")
    y_b, sr_b = load_audio(f"sounds/inputs/{B_name}.wav")

    centroid_a = compute_spectral_centroid(y_a, sr_a)
    profiles_a = extract_profiles(centroid_a, sr_a, onsets_a, offsets_a)

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
    out_path = f"sounds/outputs/{B_name}_centTrans_smile.wav"
    sf.write(out_path, y_b_mod, sr_b)
    print(f"[INFO] Saved to {out_path}")

if __name__ == "__main__":
    main()
