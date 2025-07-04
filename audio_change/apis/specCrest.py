# apis/specCrest.py
import numpy as np
import librosa
import scipy.optimize
import gc
from typing import Dict, Tuple

from apis import load

N_FFT = load.HOP_LENGTH * 4

def compute_spectral_crest(y: np.ndarray, sr: int) -> np.ndarray:
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=load.HOP_LENGTH))
    crest = np.max(S, axis=0) / (np.sum(S, axis=0) + 1e-10)
    crest[np.isnan(crest)] = 0
    return crest


def softmax_crest(S_mod: np.ndarray, beta: float = 10) -> float:
    weights = np.exp(beta * S_mod)
    weights /= np.sum(weights)
    return np.sum(weights * S_mod) / (np.sum(S_mod) + 1e-6)


def apply_spectral_crest_profiles(y: np.ndarray, sr: int, onsets: np.ndarray, offsets: np.ndarray, mapped_profiles: Dict[int, np.ndarray], n_bands: int = 8) -> np.ndarray:
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
            continue  # STFT不可

        profile = mapped_profiles[i]
        S = librosa.stft(y_seg, n_fft=2048, hop_length=load.HOP_LENGTH, dtype=np.complex64)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # 自動帯域推定（平均スペクトルから上位90%）
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

        stretched = load.stretch_profile(profile, S.shape[1])

        for t in range(S.shape[1]):
            S_t = np.abs(S[:, t])
            phase = np.angle(S[:, t])
            SC_target = stretched[t]

            def loss(w):
                weights = np.zeros_like(freqs)
                for b_idx in range(len(band_edges) - 1):
                    f_low = band_edges[b_idx]
                    f_high = band_edges[b_idx + 1]
                    weights += w[b_idx] * ((freqs >= f_low) & (freqs < f_high))
                S_mod = S_t * weights
                crest = softmax_crest(S_mod)
                return (crest - SC_target) ** 2 + 0.001 * np.sum((w - 1) ** 2)

            result = scipy.optimize.minimize(
                loss, x0=np.ones(n_bands),
                method='L-BFGS-B',
                bounds=[(0.5, 2)] * n_bands,
                options={'maxiter': 15}
            )

            if result.success:
                w_opt = result.x
                weights = np.zeros_like(freqs)
                for b_idx in range(len(band_edges) - 1):
                    f_low = band_edges[b_idx]
                    f_high = band_edges[b_idx + 1]
                    weights += w_opt[b_idx] * ((freqs >= f_low) & (freqs < f_high))
                S[:, t] = (S_t * weights) * np.exp(1j * phase)
                del result
                gc.collect()

        y_seg_mod = librosa.istft(S, hop_length=load.HOP_LENGTH, length=len(y_seg))
        y_mod[start_sample:end_sample] += y_seg_mod[:end_sample - start_sample]

    return y_mod


def transform_spectral_crest(source_name: str, target_name: str, output_path: str, base_dir: str = ".") -> None:
    # データ読み込み
    source_data = load.load_data(source_name, base_dir)
    target_data = load.load_data(target_name, base_dir)
    
    # スペクトルクレストを計算
    crest_source = compute_spectral_crest(source_data["y"], source_data["sr"])
    
    # プロファイル抽出
    profiles_source = load.extract_profiles(
        crest_source, source_data["sr"], 
        source_data["onsets"], source_data["offsets"]
    )
    
    # DTWアライメント
    path_a, path_b = load.perform_dtw_alignment(source_data["onsets"], target_data["onsets"])
    
    # プロファイルマッピング
    mapped_profiles = load.create_profile_mapping(profiles_source, path_a, path_b)
    
    # 変換適用
    y_modified = apply_spectral_crest_profiles(
        target_data["y"], target_data["sr"],
        target_data["onsets"], target_data["offsets"],
        mapped_profiles
    )
    
    # 保存
    load.save_audio(y_modified, target_data["sr"], output_path)


def api_transform_spectral_crest(source_name: str, target_name: str, output_name: str = None, base_dir: str = ".") -> str:
    if output_name is None:
        output_name = f"{target_name}_crestTrans_{source_name.split('_')[-1]}.wav"
    
    output_path = f"{base_dir}/outputs/{output_name}"
    transform_spectral_crest(source_name, target_name, output_path, base_dir)
    return output_path