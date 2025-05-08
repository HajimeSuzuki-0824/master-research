'''
音源の動的重み付けを用いた複合オンセット検出
'''

# ライブラリのインポート
import librosa
import numpy as np
import scipy.signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pandas as pd
import os

# パスの指定
num_val = "_0_スケール"
sound_name = ['Ssax','Tp','Hr','Tb','Euph']
for i in range(len(sound_name)):
    sound_name[i] = sound_name[i] + num_val


base_config = {
    'hop_length': 512,
    'df0_threshold': 0.02, # df0が判定する閾値
    'df0_weight_high': 0.15, # df0変化が大きいときの寄与重み
    'df0_weight_low': 0.1, # df0が小さい時の最低寄与
    'onset_min_time': 0.2, # 検出開始時間
    'onset_max_time_margin': 1.0, # 検出終了時間（後ろから何秒）
    'distance_seconds': 0.1,
    # グリッドサーチ用 sigma 範囲
    'sigma_min': 0.01,
    'sigma_max': 1.2,
    'sigma_step': 0.05,
    # グリッドサーチ用 prominence 範囲
    'prominence_min': 0.01,
    'prominence_max': 0.2,
    'prominence_step': 0.01,
    # 目標オンセット数
    'target_onsets': 17,
}

# グローバル config を初期化
config = base_config.copy()

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-6)

def dynamic_weighted_onset_detection(audio_path, output_plot_path=None, output_csv_path=None):
    y, sr = librosa.load(audio_path, sr=None)
    hop_length = config['hop_length']

    # 特徴量計算
    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    flux = librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(S, ref=np.max), sr=sr, hop_length=hop_length
    )
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)

    # F0 推定と変化量
    f0 = np.array([
        pitch[np.argmax(mag)]
        for mag, pitch in zip(magnitudes.T, pitches.T)
    ])
    d_f0 = np.abs(np.diff(f0, prepend=f0[0]))

    # 正規化
    norm_energy = normalize(o_env) + 1e-3
    norm_energy = np.clip(norm_energy, 0, 1)
    norm_flux = normalize(flux)
    norm_df0 = normalize(d_f0)

    # 重み付きスコア合成
    weights_flux = 1.0 - norm_energy
    weights_energy = norm_energy
    df0_weight = np.where(
        norm_df0 > config['df0_threshold'],
        config['df0_weight_high'],
        config['df0_weight_low']
    )
    combined_score = (
        weights_flux * norm_flux +
        weights_energy * norm_energy +
        df0_weight * norm_df0
    )

    # 平滑化
    combined_score = gaussian_filter1d(
        combined_score,
        sigma=config['sigma']
    )

    # 固定 absolute prominence
    dynamic_prominence = config['prominence']

    # ピーク検出
    distance = sr * config['distance_seconds'] / hop_length
    peaks, _ = scipy.signal.find_peaks(
        combined_score,
        prominence=dynamic_prominence,
        distance=distance
    )
    onset_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    # 有効時間範囲フィルタ
    duration = librosa.get_duration(y=y, sr=sr)
    valid_mask = (
        (onset_times > config['onset_min_time']) &
        (onset_times < (duration - config['onset_max_time_margin']))
    )
    onset_times = onset_times[valid_mask]
    peaks = peaks[valid_mask]

    # プロット出力
    if output_plot_path:
        times = librosa.frames_to_time(
            np.arange(len(combined_score)), sr=sr, hop_length=hop_length
        )
        plt.figure(figsize=(10, 4))
        plt.plot(times, combined_score, label='Combined Onset Score')
        plt.plot(onset_times, combined_score[peaks], 'rx', label='Detected Onsets')
        plt.xlabel('Time (s)')
        plt.ylabel('Onset Score')
        plt.title('Dynamic Weighted Onset Detection')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_plot_path)
        plt.close()

    # CSV 出力
    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        pd.DataFrame({'onset_times': onset_times}).to_csv(
            output_csv_path, index=False
        )

    return onset_times


def grid_search_for(audio_path):
    """
    指定音源に対して sigma と prominence の最適組み合わせを探索
    """
    global config
    # sigma と prominence のリストを範囲から生成
    sigma_list = np.arange(
        base_config['sigma_min'],
        base_config['sigma_max'] + base_config['sigma_step'],
        base_config['sigma_step']
    )
    prominence_list = np.arange(
        base_config['prominence_min'],
        base_config['prominence_max'] + base_config['prominence_step'],
        base_config['prominence_step']
    )

    # target_onsets と一致するパラメータ集合
    zero_params = []

    for sigma in sigma_list:
        for prom in prominence_list:
            # グローバル config をリセット
            config.clear()
            config.update(base_config)
            config['sigma'] = float(sigma)
            config['prominence'] = float(prom)

            onsets = dynamic_weighted_onset_detection(audio_path)
            if len(onsets) == base_config['target_onsets']:
                zero_params.append((float(sigma), float(prom)))

    # 一致組み合わせがあれば最適選択
    if zero_params:
        # sigma→prominence の昇順でソートし先頭を選択
        best_sigma, best_prom = sorted(zero_params, key=lambda x: (x[0], x[1]))[0]
        return {'sigma': best_sigma, 'prominence': best_prom}

    # 見つからない場合は従来の最小誤差探索にフォールバック
    best_params = {}
    best_error = float('inf')
    for sigma in sigma_list:
        for prom in prominence_list:
            config.clear()
            config.update(base_config)
            config['sigma'] = float(sigma)
            config['prominence'] = float(prom)
            onsets = dynamic_weighted_onset_detection(audio_path)
            error = abs(len(onsets) - base_config['target_onsets'])
            if error < best_error:
                best_error = error
                best_params = {'sigma': float(sigma), 'prominence': float(prom)}
    return best_params


if __name__ == "__main__":
    # 音源ごとに最適パラメータ探索＆実行
    for instr in sound_name:
        audio_path = f"sounds/{instr}.wav"
        # グリッドサーチで最適化
        optimal = grid_search_for(audio_path)
        # config を最適パラメータで更新
        config = base_config.copy()
        config.update(optimal)

        print(f"=== {instr} Best Parameters ===")
        print(optimal)

        # 検出結果出力
        plot_path = f"outputs/png/{instr}.png"
        csv_path = f"outputs/csv/{instr}.csv"
        onsets = dynamic_weighted_onset_detection(audio_path, plot_path, csv_path)
        print(f"--- {instr} onset_times ({len(onsets)}) ---")
        print(onsets)