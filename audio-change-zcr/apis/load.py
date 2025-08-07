import numpy as np
import pandas as pd
import soundfile as sf
import os
from typing import Tuple, Dict, Any

HOP_LENGTH = 512
FRAME_LENGTH = HOP_LENGTH * 4

def load_audio(filepath) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(filepath)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y, sr

def load_csv_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    onsets = df["onset_times"].astype(float).to_numpy()
    offsets = df["offset_times"].astype(float).to_numpy()
    return onsets, offsets

def load_data(name: str, base_dir: str = ".") -> Dict[str, Any]:
    audio_path = os.path.join(base_dir, "inputs", "sounds", f"{name}.wav")
    csv_path = os.path.join(base_dir, "inputs", "csv", f"{name}.csv")
    
    y, sr = load_audio(audio_path)
    onsets, offsets = load_csv_data(csv_path)
    
    return {
        "y": y,
        "sr": sr,
        "onsets": onsets,
        "offsets": offsets
    }

def load_data_with_suffix(name: str, suffix: str, base_dir: str = ".") -> Dict[str, Any]:
    audio_path = os.path.join(base_dir, "inputs", "sounds", f"{name}.wav")
    csv_path = os.path.join(base_dir, "inputs", "csv", f"{name}_{suffix}.csv")
    
    y, sr = load_audio(audio_path)
    onsets, offsets = load_csv_data(csv_path)
    
    return {
        "y": y,
        "sr": sr,
        "onsets": onsets,
        "offsets": offsets
    }

def extract_profiles(feature_seq: np.ndarray, sr: int, onsets: np.ndarray, offsets: np.ndarray, hop_length: int = HOP_LENGTH) -> Dict[int, np.ndarray]:
    """音符単位で特徴量プロファイルを抽出"""
    frame_rate = sr / hop_length
    profiles = {}
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        start = int(onset * frame_rate)
        end = int(offset * frame_rate)
        if end > len(feature_seq):
            continue
        profiles[i] = feature_seq[start:end]
    return profiles

def stretch_profile(profile: np.ndarray, target_len: int) -> np.ndarray:
    """線形補間による特徴量プロファイルの長さ調整"""
    if len(profile) == 0:
        return np.zeros(target_len)
    x_original = np.linspace(0, 1, len(profile))
    x_target = np.linspace(0, 1, target_len)
    return np.interp(x_target, x_original, profile)

def save_audio(y: np.ndarray, sr: int, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, y, sr)
    print(f"[INFO] Saved to {output_path}")

def save_temp_audio(y: np.ndarray, sr: int, temp_path: str) -> None:
    """一時ファイル保存用（ログ出力なし）"""
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    sf.write(temp_path, y, sr)