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

def normalize_note_regions(feature_seq: np.ndarray, sr: int, onsets: np.ndarray, offsets: np.ndarray, hop_length: int = load.HOP_LENGTH) -> np.ndarray:
    """音符部分のみをMin-Max正規化"""
    frame_rate = sr / hop_length
    note_values = []
    
    # 音符部分の値を収集
    for onset, offset in zip(onsets, offsets):
        start = int(onset * frame_rate)
        end = int(offset * frame_rate)
        if end <= len(feature_seq):
            note_values.extend(feature_seq[start:end])
    
    if len(note_values) == 0:
        return feature_seq
    
    # Min-Max正規化のパラメータを計算
    min_val = np.min(note_values)
    max_val = np.max(note_values)
    
    if max_val - min_val == 0:
        return feature_seq
    
    # 正規化された特徴量配列を作成
    normalized_seq = feature_seq.copy()
    
    # 音符部分のみを正規化
    for onset, offset in zip(onsets, offsets):
        start = int(onset * frame_rate)
        end = int(offset * frame_rate)
        if end <= len(feature_seq):
            normalized_seq[start:end] = (feature_seq[start:end] - min_val) / (max_val - min_val)
    
    return normalized_seq

def apply_spectral_centroid_transformation_note_wise(source_audio: np.ndarray, target_profiles: Dict[int, np.ndarray],
                                                   note_index: int, sr: int, n_bands: int = 8) -> np.ndarray:
    """音符単位でのSpectral Centroid変換（正規化済みプロファイル対応）"""
    if note_index not in target_profiles:
        return source_audio
    
    y_seg = source_audio
    if y_seg.ndim > 1:
        y_seg = np.mean(y_seg, axis=1)
    if len(y_seg) < 2048:
        # パディングして処理
        y_seg_padded = np.pad(y_seg, (0, 2048 - len(y_seg)), 'constant')
        S = librosa.stft(y_seg_padded, n_fft=2048, hop_length=load.HOP_LENGTH, dtype=np.complex64)
    else:
        S = librosa.stft(y_seg, n_fft=2048, hop_length=load.HOP_LENGTH, dtype=np.complex64)
    
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    # 帯域設定（旧システムと同じ）
    mag_avg = np.mean(np.abs(S), axis=1)
    mag_cumsum = np.cumsum(np.sort(mag_avg)[::-1])
    mag_total = mag_cumsum[-1]
    cutoff = 0.90 * mag_total
    sorted_mag = np.sort(mag_avg)[::-1]
    threshold = sorted_mag[np.searchsorted(mag_cumsum, cutoff)]
    valid_bins = np.where(mag_avg >= threshold)[0]
    
    if len(valid_bins) == 0:
        return source_audio
        
    f_min = max(freqs[valid_bins[0]], 1)
    f_max = freqs[valid_bins[-1]]
    log_min = np.log10(f_min)
    log_max = np.log10(f_max)
    band_edges = np.logspace(log_min, log_max, n_bands + 1)
    band_edges[-1] = np.inf

    # ターゲットプロファイルを取得し、現在の音声の長さに合わせて調整
    target_profile = target_profiles[note_index]
    target_stretched = load.stretch_profile(target_profile, S.shape[1])
    
    # 正規化済みのプロファイルを使用
    # ターゲットプロファイルは既に0-1に正規化済みなので、適切なスケールに変換
    source_centroid = compute_spectral_centroid(y_seg, sr)
    source_mean = np.mean(source_centroid) + 1e-6
    
    # 正規化済みプロファイルを実際の周波数範囲にスケール
    centroid_range = np.max(source_centroid) - np.min(source_centroid)
    centroid_min = np.min(source_centroid)
    scaled_target = target_stretched * centroid_range + centroid_min

    for t in range(S.shape[1]):
        S_t = np.abs(S[:, t])
        phase = np.angle(S[:, t])
        
        # 目標のスペクトル重心（スケール済みターゲットプロファイルから）
        SC_target = scaled_target[t]

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
    
    # 元の長さに合わせて調整
    if len(y_seg_mod) > len(source_audio):
        y_seg_mod = y_seg_mod[:len(source_audio)]
    elif len(y_seg_mod) < len(source_audio):
        y_seg_mod = np.pad(y_seg_mod, (0, len(source_audio) - len(y_seg_mod)), 'constant')
    
    return y_seg_mod

def get_target_centroid_profiles(target_name: str, base_dir: str = ".") -> Dict[int, np.ndarray]:
    """ターゲット音声からSpectral Centroidプロファイルを抽出（正規化対応）"""
    target_data = load.load_data_with_suffix(target_name, "spectral", base_dir)
    target_centroid = compute_spectral_centroid(target_data["y"], target_data["sr"])
    
    # 音符部分のみを正規化
    normalized_centroid = normalize_note_regions(
        target_centroid, target_data["sr"], 
        target_data["onsets"], target_data["offsets"]
    )
    
    # 正規化された特徴量から音符単位でプロファイルを抽出
    profiles = load.extract_profiles(
        normalized_centroid, target_data["sr"], 
        target_data["onsets"], target_data["offsets"]
    )
    
    return profiles