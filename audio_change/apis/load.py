# apis/load.py
import numpy as np
import pandas as pd
import soundfile as sf
from dtw import dtw
import os
from typing import Tuple, Dict, Any

HOP_LENGTH = 512
FRAME_LENGTH = HOP_LENGTH * 4

def load_audio(filepath) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(filepath)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y, sr


def load_csv_data(csv_path: str)-> Tuple[np.ndarray, np.ndarray]:
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


def extract_profiles(feature_seq: np.ndarray, sr: int, onsets: np.ndarray, offsets: np.ndarray, hop_length: int = HOP_LENGTH) -> Dict[int, np.ndarray]:
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
    if len(profile) == 0:
        return np.zeros(target_len)
    x_original = np.linspace(0, 1, len(profile))
    x_target = np.linspace(0, 1, target_len)
    return np.interp(x_target, x_original, profile)

def perform_dtw_alignment(onsets_a: np.ndarray, onsets_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    onsets_a_ = onsets_a.reshape(-1, 1)
    onsets_b_ = onsets_b.reshape(-1, 1)
    dist_fn = lambda x, y: np.abs(x - y)
    alignment = dtw(onsets_a_, onsets_b_, dist_method=dist_fn, step_pattern="symmetric1", keep_internals=True)
    return alignment.index1, alignment.index2


def create_profile_mapping(profiles_a: Dict[int, np.ndarray], path_a: np.ndarray, path_b: np.ndarray) -> Dict[int, np.ndarray]:
    mapped_profiles = {}
    for ai, bi in zip(path_a, path_b):
        if ai in profiles_a:
            mapped_profiles[bi] = profiles_a[ai]
    return mapped_profiles


def save_audio(y: np.ndarray, sr: int, output_path: str) -> None:
    """
    音声ファイルを保存する
    
    Args:
        y: 音声データ
        sr: サンプリングレート
        output_path: 出力パス
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, y, sr)
    print(f"[INFO] Saved to {output_path}")