import numpy as np
import librosa
from scipy.signal import savgol_filter
from typing import Dict, Tuple
from apis import load


def compute_attack_slope(y: np.ndarray, sr: int) -> np.ndarray:
    rms = librosa.feature.rms(y=y, frame_length=load.FRAME_LENGTH, hop_length=load.HOP_LENGTH)[0]
    return rms

def apply_attack_slope_profiles(y: np.ndarray, sr: int, onsets: np.ndarray, offsets: np.ndarray, mapped_profiles: Dict[int, np.ndarray]) -> np.ndarray:
    frame_rate = sr / load.HOP_LENGTH
    y_mod = y.copy()
    
    # ターゲットのAttack Slopeを事前計算
    target_rms = librosa.feature.rms(y=y, frame_length=load.FRAME_LENGTH, hop_length=load.HOP_LENGTH)[0]
    target_profiles = load.extract_profiles(target_rms, sr, onsets, offsets)
    
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        if i not in mapped_profiles or i not in target_profiles:
            continue
            
        start_frame = int(onset * frame_rate)
        end_frame = int(offset * frame_rate)
        source_profile = mapped_profiles[i]
        target_profile = target_profiles[i]
        n_frames = end_frame - start_frame
        
        if n_frames <= 0 or len(target_profile) == 0:
            continue
            
        # ソースの相対変化率を計算
        source_stretched = load.stretch_profile(source_profile, n_frames)
        target_stretched = load.stretch_profile(target_profile, n_frames)
        
        source_mean = np.mean(source_stretched) + 1e-6
        target_mean = np.mean(target_stretched) + 1e-6
        
        if len(source_stretched) >= 9:
            source_stretched = savgol_filter(source_stretched, window_length=9, polyorder=2)
            
        # 全体の最大ゲインを事前計算して正規化
        max_gain = 0.0
        normalized_gains = []
        for j in range(n_frames):
            source_ratio = source_stretched[j] / source_mean
            target_value = target_stretched[j] * source_ratio
            raw_gain = target_value / (target_stretched[j] + 1e-6)
            normalized_gains.append(raw_gain)
            max_gain = max(max_gain, raw_gain)

        # 最大ゲインが1.0を超える場合は全体を正規化
        if max_gain > 1.0:
            normalized_gains = [g / max_gain for g in normalized_gains]

        for j in range(n_frames):
            gain = max(normalized_gains[j], 0.1)  # 最小値制限のみ
            
            frame_start = (start_frame + j) * load.HOP_LENGTH
            frame_end = frame_start + load.HOP_LENGTH
            
            if frame_end >= len(y_mod):
                break
                
            y_mod[frame_start:frame_end] *= gain
            
    return y_mod

def transform_attack_slope(source_name: str, target_name: str, output_path: str, base_dir: str = ".") -> None:
    # データ読み込み
    source_data = load.load_data(source_name, base_dir)
    target_data = load.load_data(target_name, base_dir)
    
    # アタック傾斜を計算
    attack_source = compute_attack_slope(source_data["y"], source_data["sr"])
    
    # プロファイル抽出
    profiles_source = load.extract_profiles(
        attack_source, source_data["sr"], 
        source_data["onsets"], source_data["offsets"]
    )
    
    # DTWアライメント
    path_a, path_b = load.perform_dtw_alignment(source_data["onsets"], target_data["onsets"])
    
    # プロファイルマッピング
    mapped_profiles = load.create_profile_mapping(profiles_source, path_a, path_b)
    
    # 変換適用
    y_modified = apply_attack_slope_profiles(
        target_data["y"], target_data["sr"],
        target_data["onsets"], target_data["offsets"],
        mapped_profiles
    )
    
    # 保存
    load.save_audio(y_modified, target_data["sr"], output_path)

def api_transform_attack_slope(source_name: str, target_name: str, output_name: str = None, base_dir: str = ".") -> str:
    if output_name is None:
        output_name = f"{target_name}_attackTrans_{source_name.split('_')[-1]}.wav"
    
    output_path = f"{base_dir}/outputs/{output_name}"
    transform_attack_slope(source_name, target_name, output_path, base_dir)
    return output_path