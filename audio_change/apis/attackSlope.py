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
    
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        if i not in mapped_profiles:
            continue
            
        start_frame = int(onset * frame_rate)
        end_frame = int(offset * frame_rate)
        profile = mapped_profiles[i]
        n_frames = end_frame - start_frame
        
        if n_frames <= 0:
            continue
            
        stretched = load.stretch_profile(profile, n_frames)
        
        # Savitzky-Golayフィルタで平滑化
        if len(stretched) >= 9:
            stretched = savgol_filter(stretched, window_length=9, polyorder=2)
            
        max_val = np.max(stretched) + 1e-6
        
        for j in range(n_frames):
            gain = stretched[j] / max_val
            gain = max(gain, 0.1)  # 最小ゲインを保証
            
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