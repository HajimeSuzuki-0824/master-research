import librosa
import numpy as np
import soundfile as sf
import os

# -------------------------------
# 設定
# -------------------------------
input_name = "input1"
input_path = f"inputs/{input_name}.wav"  # 元音源ファイルを指定（例: 単音や短いメロディ）
sr = 44100                # サンプリングレート
n_fft = 2048
hop_length = 512
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# 音声読み込み＆STFT
# -------------------------------
y, sr = librosa.load(input_path, sr=sr)
S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
mag, phase = np.abs(S), np.angle(S)

# -------------------------------
# Spectral Centroid 操作
# -------------------------------

def modify_centroid(mag, mode='high'):
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).reshape(-1, 1)
    if mode == 'high':
        weights = np.linspace(1.0, 3.0, mag.shape[0]).reshape(-1, 1)
    elif mode == 'low':
        weights = np.linspace(3.0, 1.0, mag.shape[0]).reshape(-1, 1)
    return mag * weights

# -------------------------------
# Spectral Crest 操作
# -------------------------------

def modify_crest(mag, mode='high'):
    if mode == 'high':
        mag = mag ** 1.2  # ピーク強調
    elif mode == 'low':
        mag = mag ** 0.8  # フラット化
    return mag

# -------------------------------
# 変換＆保存関数
# -------------------------------

def process_and_save(mag, phase, name):
    S_modified = mag * np.exp(1j * phase)
    y_modified = librosa.istft(S_modified, hop_length=hop_length)
    sf.write(os.path.join(output_dir, f"{name}.wav"), y_modified, sr)

# -------------------------------
# 各パターン出力
# -------------------------------
mag_c_high = modify_centroid(mag, mode='high')
mag_c_low = modify_centroid(mag, mode='low')
mag_cr_high = modify_crest(mag, mode='high')
mag_cr_low = modify_crest(mag, mode='low')

# 保存
process_and_save(mag_c_high, phase, f"{input_name}_centroid_high")
process_and_save(mag_c_low, phase, f"{input_name}_centroid_low")
process_and_save(mag_cr_high, phase, f"{input_name}_crest_high")
process_and_save(mag_cr_low, phase, f"{input_name}_crest_low")
