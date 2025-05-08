'''
A音源（A_audio）とB音源（B_audio）をDTW(Dynamic Time Warping)で同期させる
'''

# ライブラリのインポート
import librosa
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import os

# ファイルパス設定
A_audio = 'sounds/Ssax_1_smile.wav'
B_audio = 'sounds/Euph_1_smile.wav'


class AudioAligner:
    # パラメータ初期化
    def __init__(self, frame_length=2048, hop_length=512, prominence=0.001, output_dir='output_images'):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.prominence = prominence
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    # 音声ファイル読み込み
    def load_audio(self, filepath):
        y, sr = librosa.load(filepath, sr=None)
        return y, sr
    
    # エネルギー推移計算
    def compute_energy_envelope(self, y):
        return np.array([
            np.sum(np.abs(y[i:i+self.frame_length])**2)
            for i in range(0, len(y) - self.frame_length, self.hop_length)
        ])
    
    # アタックタイミング検出
    def detect_attack_times(self, energy, sr):
        peaks, _ = scipy.signal.find_peaks(energy, prominence=self.prominence)
        attack_times = []
        for peak_idx in peaks:
            threshold = 0.10 * energy[peak_idx]
            pre_peak = energy[:peak_idx]
            attack_idx_candidates = np.where(pre_peak <= threshold)[0]
            if len(attack_idx_candidates) > 0:
                attack_idx = attack_idx_candidates[-1]
                attack_time = attack_idx * (self.hop_length / sr)
                attack_times.append(attack_time)
        return np.array(attack_times)
    
    # 時系列を同一長に補間
    def resample_attack_times(self, attack_times, length):
        if len(attack_times) < 2:
            raise ValueError(f"アタックタイミング数が不十分 です({len(attack_times)}点)。prominenceを下げるか、長い音源を使用して下さい。")
        return np.interp(
            np.linspace(0, 1, length),
            np.linspace(0, 1, len(attack_times)),
            attack_times
        )
    
    # DTWで整列
    def align_sequences(self, seq1, seq2):
        D, wp = librosa.sequence.dtw(X=seq1[:, np.newaxis], Y=seq2[:, np.newaxis], metric='euclidean')
        return D, wp
    
    # 整列前アタック列をプロット
    def plot_attack_sequences(self, seq1, seq2):
        plt.figure(figsize=(8, 4))
        plt.scatter(range(len(seq1)), seq1, label='A Attack Times', s=30)
        plt.scatter(range(len(seq2)), seq2, label='B Attack Times', s=30)
        plt.xlabel("Attack Index (Note Number)")
        plt.ylabel("Time (seconds)")
        plt.title("Attack Time Sequences Before Alignment")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'attack_sequences.png'))
        plt.close()
    
    # 整列パスをプロット
    def plot_warping_path(self, wp):
        plt.figure(figsize=(8, 4))
        plt.plot([p[0] for p in wp], [p[1] for p in wp], marker='o')
        plt.title('DTW Warping Path')
        plt.xlabel('A Frame index')
        plt.ylabel('B Frame Index')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'warping_path.png'))


if __name__ == "__main__":
    # Alignerのインスタンスの作成
    aligner = AudioAligner()

    # 音源を読み込み
    y_a, sr_a = aligner.load_audio(A_audio)
    y_b, sr_b = aligner.load_audio(B_audio)

    # エネルギー包絡計算
    energy_a = aligner.compute_energy_envelope(y_a)
    energy_b = aligner.compute_energy_envelope(y_b)

    # アタックタイミング検出
    attack_times_a = aligner.detect_attack_times(energy_a, sr_a)
    attack_times_b = aligner.detect_attack_times(energy_b, sr_b)

    # アタックタイムリサンプリング
    length = max(len(attack_times_a), len(attack_times_b))
    attack_seq_a = aligner.resample_attack_times(attack_times_a, length)
    attack_seq_b = aligner.resample_attack_times(attack_times_b, length)

    # 整列
    D, wp = aligner.align_sequences(attack_seq_a, attack_seq_b)

    # プロット、ファイル保存
    aligner.plot_attack_sequences(attack_seq_a, attack_seq_b)
    aligner.plot_warping_path(wp)

    # 整列結果の一部を表示
    print(f"対応するフレーム数: {len(wp)}")
    step = max(1, len(wp)//10)
    for (i, j) in wp[::step]:
        print(f"A音源 frame {i} ↔ B音源 frame {j}")