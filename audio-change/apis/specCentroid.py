# apis/specCentroid.py
import numpy as np
import librosa
import scipy.optimize
import gc
from typing import Dict, Tuple

from apis import load

def compute_spectral_centroid(y: np.ndarray, sr: int, hop_length: int = load.HOP_LENGTH, n_fft: int = 2048) -> np.ndarray:
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).reshape(-1, 1)
    numerator = np.sum(freqs * S, axis=0)
    denominator = np.sum(S, axis=0) + 1e-6
    centroid = numerator / denominator
    return centroid


def apply_spectral_centroid_profiles(y: np.ndarray, sr: int, onsets: np.ndarray, offsets: np.ndarray, mapped_profiles: Dict[int, np.ndarray], n_bands: int = 8) -> np.ndarray:
    y_mod = np.zeros_like(y)

    # ターゲットのSpectral Centroidを事前計算
    target_centroid = compute_spectral_centroid(y, sr)
    target_profiles = load.extract_profiles(target_centroid, sr, onsets, offsets)

    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        if i not in mapped_profiles or i not in target_profiles:
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

        source_profile = mapped_profiles[i]
        target_profile = target_profiles[i]
        S = librosa.stft(y_seg, n_fft=2048, hop_length=load.HOP_LENGTH, dtype=np.complex64)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # 帯域設定
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

        # 相対変化パターンを計算
        source_stretched = load.stretch_profile(source_profile, S.shape[1])
        target_stretched = load.stretch_profile(target_profile, S.shape[1])
        
        source_mean = np.mean(source_stretched) + 1e-6
        target_mean = np.mean(target_stretched) + 1e-6

        for t in range(S.shape[1]):
            S_t = np.abs(S[:, t])
            phase = np.angle(S[:, t])
            
            # 相対的な変化パターンから目標値を計算
            source_ratio = source_stretched[t] / source_mean
            SC_target = target_stretched[t] * source_ratio

            def loss(w):
                weights = np.zeros_like(freqs)
                for b_idx in range(len(band_edges)-1):
                    f_min_band = band_edges[b_idx]
                    f_max_band = band_edges[b_idx+1]
                    weights += w[b_idx] * ((freqs >= f_min_band) & (freqs < f_max_band))

                S_mod = S_t * weights
                SC = np.sum(freqs * S_mod) / (np.sum(S_mod) + 1e-6)
                return (SC - SC_target) ** 2 + 0.001 * np.sum((w - 1) ** 2)

            result = scipy.optimize.minimize(
                loss, x0=np.ones(n_bands),
                method='L-BFGS-B',
                bounds=[(0.5, 1.0)] * n_bands,
                options={'maxiter': 20}
            )

            if result.success:
                w_opt = result.x
                
                # 重みが1.0を超えないように正規化
                max_weight = np.max(w_opt)
                if max_weight > 1.0:
                    w_opt = w_opt / max_weight
                
                weights = np.zeros_like(freqs)
                for b_idx in range(len(band_edges)-1):
                    f_min_band = band_edges[b_idx]
                    f_max_band = band_edges[b_idx+1]
                    weights += w_opt[b_idx] * ((freqs >= f_min_band) & (freqs < f_max_band))

                S[:, t] = (S_t * weights) * np.exp(1j * phase)
                del result
                gc.collect()

        y_seg_mod = librosa.istft(S, hop_length=load.HOP_LENGTH, length=len(y_seg))
        y_mod[start_sample:end_sample] += y_seg_mod[:end_sample - start_sample]

    return y_mod


def transform_spectral_centroid(source_name: str, target_name: str, output_path: str, base_dir: str = ".") -> None:    
    # データ読み込み
    source_data = load.load_data(source_name, base_dir)
    target_data = load.load_data(target_name, base_dir)
    
    # スペクトル重心を計算
    centroid_source = compute_spectral_centroid(source_data["y"], source_data["sr"])
    
    # プロファイル抽出
    profiles_source = load.extract_profiles(
        centroid_source, source_data["sr"], 
        source_data["onsets"], source_data["offsets"]
    )
    
    # DTWアライメント
    path_a, path_b = load.perform_dtw_alignment(source_data["onsets"], target_data["onsets"])
    
    # プロファイルマッピング
    mapped_profiles = load.create_profile_mapping(profiles_source, path_a, path_b)
    
    # 変換適用
    y_modified = apply_spectral_centroid_profiles(
        target_data["y"], target_data["sr"],
        target_data["onsets"], target_data["offsets"],
        mapped_profiles
    )
    
    # 保存
    load.save_audio(y_modified, target_data["sr"], output_path)


def api_transform_spectral_centroid(source_name: str, target_name: str, output_name: str = None, base_dir: str = ".") -> str:
    if output_name is None:
        output_name = f"{target_name}_centTrans_{source_name.split('_')[-1]}.wav"
    
    output_path = f"{base_dir}/outputs/{output_name}"
    transform_spectral_centroid(source_name, target_name, output_path, base_dir)
    return output_path