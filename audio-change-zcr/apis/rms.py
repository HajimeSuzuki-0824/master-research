import numpy as np
import librosa
from scipy.signal import savgol_filter
from typing import Dict, Tuple
from apis import load

def compute_rms(y: np.ndarray, sr: int) -> np.ndarray:
    rms = librosa.feature.rms(y=y, frame_length=load.FRAME_LENGTH, hop_length=load.HOP_LENGTH)[0]
    return rms

def apply_rms_transformation_note_wise(source_audio: np.ndarray, target_profiles: Dict[int, np.ndarray], 
                                     note_index: int, sr: int) -> np.ndarray:
    """音符単位でのRMS変換（旧システムベース）"""
    if note_index not in target_profiles:
        return source_audio
    
    frame_rate = sr / load.HOP_LENGTH
    
    # ソースのRMS計算
    source_rms = librosa.feature.rms(y=source_audio, frame_length=load.FRAME_LENGTH, hop_length=load.HOP_LENGTH)[0]
    
    # ターゲットプロファイルを取得
    target_profile = target_profiles[note_index]
    
    # ソース音声の長さに合わせてターゲットプロファイルを調整
    n_frames = len(source_rms)
    if n_frames <= 0 or len(target_profile) == 0:
        return source_audio
    
    # ターゲットプロファイルを線形補間で調整
    target_stretched = load.stretch_profile(target_profile, n_frames)
    
    # 相対変化率を計算
    source_mean = np.mean(source_rms) + 1e-6
    target_mean = np.mean(target_stretched) + 1e-6
    
    # Savitzky-Golay フィルタによる平滑化（旧システムと同じ）
    if len(target_stretched) >= 9:
        target_stretched = savgol_filter(target_stretched, window_length=9, polyorder=2)
    
    y_mod = source_audio.copy()
    
    # 全体の最大ゲインを事前計算して正規化
    max_gain = 0.0
    normalized_gains = []
    for j in range(n_frames):
        source_ratio = source_rms[j] / source_mean
        target_value = target_stretched[j] * source_ratio
        raw_gain = target_value / (source_rms[j] + 1e-6)
        normalized_gains.append(raw_gain)
        max_gain = max(max_gain, raw_gain)

    # 最大ゲインが1.0を超える場合は全体を正規化
    if max_gain > 1.0:
        normalized_gains = [g / max_gain for g in normalized_gains]

    # フレーム単位でゲインを適用
    for j in range(n_frames):
        gain = max(normalized_gains[j], 0.1)  # 最小値制限
        
        frame_start = j * load.HOP_LENGTH
        frame_end = frame_start + load.HOP_LENGTH
        
        if frame_end >= len(y_mod):
            break
            
        y_mod[frame_start:frame_end] *= gain
    
    return y_mod

def get_target_rms_profiles(target_name: str, base_dir: str = ".") -> Dict[int, np.ndarray]:
    """ターゲット音声からRMSプロファイルを抽出"""
    target_data = load.load_data_with_suffix(target_name, "rms", base_dir)
    target_rms = compute_rms(target_data["y"], target_data["sr"])
    
    # 音符単位でプロファイルを抽出
    profiles = load.extract_profiles(
        target_rms, target_data["sr"], 
        target_data["onsets"], target_data["offsets"]
    )
    
    return profiles