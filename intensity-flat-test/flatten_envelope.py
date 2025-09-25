import argparse
import numpy as np
import soundfile as sf
from scipy.signal import hilbert

def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    # 端はパディングして歪みを抑える
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(xpad, kernel, mode="same")[pad:-pad]

def build_gain_curve(env_smooth: np.ndarray, sr: int, start_s: float, end_s: float,
                     target_dbfs: float | None, fade_ms: float, max_gain: float) -> np.ndarray:
    n = env_smooth.shape[-1]
    start_i = max(0, int(round(start_s * sr)))
    end_i   = min(n, int(round(end_s * sr)))
    if end_i <= start_i:
        raise ValueError("end は start より後にしてください。")
    eps = 1e-8

    if target_dbfs is None:
        target = float(np.mean(env_smooth[start_i:end_i]) + eps)
    else:
        # dBFS → 線形振幅。ここでは 0dBFS ≒ 振幅1.0 を想定
        target = 10 ** (target_dbfs / 20.0)

    gain = np.ones(n, dtype=float)
    raw = target / (env_smooth + eps)
    raw = np.clip(raw, 1.0 / max_gain, max_gain)  # 極端な増幅/減衰を制限

    # 区間に適用
    gain[start_i:end_i] = raw[start_i:end_i]

    # 境界フェード
    fade_len = max(1, int(sr * fade_ms / 1000.0))
    # 立ち上がり
    if start_i > 0 and fade_len > 1:
        ramp = np.linspace(1.0, gain[start_i], fade_len)
        s0 = max(0, start_i - fade_len)
        gain[s0:start_i] = np.linspace(1.0, ramp[0], start_i - s0)
        gain[start_i:start_i + min(fade_len, n - start_i)] = ramp[:max(0, min(fade_len, n - start_i))]
    # 立ち下がり
    if end_i < n and fade_len > 1:
        ramp = np.linspace(gain[end_i - 1], 1.0, fade_len)
        e1 = min(n, end_i + fade_len)
        gain[end_i:e1] = ramp[:e1 - end_i]
    return gain

def process_file(inp: str, outp: str, start: float, end: float,
                 smooth_ms: float, fade_ms: float, target_dbfs: float | None,
                 max_gain: float):
    # mono or stereo でもOK
    y, sr = sf.read(inp, always_2d=True)  # shape: (n_samples, n_channels)
    y = y.T  # (C, N)
    C, N = y.shape

    # Hilbert でエンベロープ（チャンネルごとに）
    env = np.abs(hilbert(y, axis=1))
    # 平滑化（移動平均）
    win = max(1, int(round(sr * smooth_ms / 1000.0)))
    env_s = np.vstack([moving_average(env[c], win) for c in range(C)])

    # モノラル相当のゲイン曲線（全ch同じゲインにする方が自然なことが多い）
    env_mono = np.mean(env_s, axis=0)

    gain = build_gain_curve(env_mono, sr, start, end, target_dbfs, fade_ms, max_gain)

    # 適用（各chに同じゲイン）
    y_out = (y * gain[None, :])

    # クリップ回避（安全に±1.0へ）
    peak = np.max(np.abs(y_out))
    if peak > 0.999:
        y_out = y_out / peak * 0.999

    # 書き出し
    sf.write(outp, y_out.T, sr, subtype="PCM_16")

def main():
    import os
    import csv

    # ===== パス設定=====
    inputs_dir = "inputs"
    outputs_dir = "outputs"
    csv_path = os.path.join(inputs_dir, "onsets.csv")

    # ===== 共通パラメータ =====
    smooth_ms = 50.0      # エンベロープ平滑化窓 [ms]
    fade_ms = 10.0        # 区間境界のフェード [ms]
    target_dbfs = None    # None → 区間の平均エンベロープに合わせる
    max_gain = 6.0        # 増幅/減衰の上限倍率（安全のため）

    # ===== onsets.csv（1行）から start/end を取得 =====
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"onsets.csv が見つかりません: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        try:
            row = next(reader)
        except StopIteration:
            raise ValueError("onsets.csv にデータ行がありません（1行必要です）")

        # ヘッダは基本 'start','end' を想定。
        # もし別名なら以下の候補を使用（例: 'onset','offset' 等）
        start_key_candidates = ["onset", "start_sec", "onset"]
        end_key_candidates   = ["offset", "end_sec", "offset"]

        def pick(d, keys):
            for k in keys:
                if k in d and d[k] != "":
                    return float(d[k])
            raise KeyError(f"CSVに必要な列が見つかりません。候補: {keys}")

        start = pick(row, start_key_candidates)
        end   = pick(row, end_key_candidates)

    if end <= start:
        raise ValueError(f"onsets.csv の秒数が不正です（end <= start）。start={start}, end={end}")

    # ===== 出力ディレクトリ作成 =====
    os.makedirs(outputs_dir, exist_ok=True)

    # ===== inputs 内の全 .wav を処理 =====
    wavs = sorted(fn for fn in os.listdir(inputs_dir) if fn.lower().endswith(".wav"))
    if not wavs:
        print(f"⚠️ 入力 .wav が見つかりません: {inputs_dir}")

    for fname in wavs:
        inp = os.path.join(inputs_dir, fname)
        outp = os.path.join(outputs_dir, fname)
        print(f"▶ {fname}: {start:.3f}–{end:.3f} 秒を平坦化 → {outp}")

        process_file(
            inp=inp,
            outp=outp,
            start=start,
            end=end,
            smooth_ms=smooth_ms,
            fade_ms=fade_ms,
            target_dbfs=target_dbfs,
            max_gain=max_gain,
        )

    print("すべて完了しました")

if __name__ == "__main__":
    main()