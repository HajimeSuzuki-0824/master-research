# -*- coding: utf-8 -*-

import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import soundfile as sf

from main import AudioProcessor as BaseAudioProcessor

class DAWAudioProcessor(BaseAudioProcessor):
    """特徴量変換を行わず DAW 素材だけで合成する派生プロセッサ"""

    def _check_required_files(self):
        """派生版では、ターゲット参照音源や特徴量 CSV は不要。最低限のファイルのみ検査する。"""
        required_files = [
            self.sheet_dir / f"{self.song_number}.csv",
            self.sheet_dir / "music_info.csv",
            self.daw_sounds_dir / "onsets.csv",
        ]
        missing = [p for p in required_files if not p.exists()]
        if missing:
            print("ERROR: 以下の必須ファイルが見つかりません:")
            for p in missing:
                print(f"  - {p}")
            sys.exit(1)

        # 少なくとも 1 個は wav があるか軽く確認（名称は find_daw_audio_file が柔軟探索する）
        if not any(self.daw_sounds_dir.glob("*.wav")):
            print(f"ERROR: DAW 素材が見つかりません: {self.daw_sounds_dir}")
            sys.exit(1)

    def check_note_count_consistency(self, sheet_data: pd.DataFrame) -> None:
        """main.py では参照音源の音符数と照合していたが、本派生では照合を行わない。"""
        return  # no-op

    def apply_fade_in_out(self, y: np.ndarray, sr: int, fade_duration: float = 0.02) -> np.ndarray:
        """多チャネル対応のフェードイン/アウト。極端に短い音でも破綻しないようにガード。"""
        y = np.asarray(y, dtype=np.float32)
        if y.size == 0:
            return y

        n = y.shape[0]
        fade_samples = int(max(1, sr * fade_duration))

        # 極端に短いノートではフェード長を短縮
        if 2 * fade_samples > n:
            fade_samples = max(1, n // 4)

        if y.ndim == 1:
            # mono
            fade_in  = np.linspace(0.0, 1.0, fade_samples, endpoint=False, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_samples, endpoint=False, dtype=np.float32)
            y[:fade_samples] *= fade_in
            y[-fade_samples:] *= fade_out
            return y

        # stereo/multichannel: (N, C)
        c = y.shape[1]
        fade_in  = np.linspace(0.0, 1.0, fade_samples, endpoint=False, dtype=np.float32)[:, None]  # (F,1)
        fade_out = np.linspace(1.0, 0.0, fade_samples, endpoint=False, dtype=np.float32)[:, None]
        y[:fade_samples, :]  *= fade_in   # (F,C)
        y[-fade_samples:, :] *= fade_out  # (F,C)
        return y

    def process_audio(self) -> str:
        """特徴量変換なしで、楽譜に従って DAW 素材を切り出し→結合する。"""
        print(f"処理開始(DAW 直結版): target={self.target_file}  instrument={self.daw_instrument}")

        # データ読み込み（テンポ・切り出し窓・譜面）
        sheet_data = pd.read_csv(self.sheet_dir / f"{self.song_number}.csv")
        music_info = pd.read_csv(self.sheet_dir / "music_info.csv")
        daw_onsets = pd.read_csv(self.daw_sounds_dir / "onsets.csv")

        # 整合性チェック（no-op だが、API互換のため呼び出し）
        self.check_note_count_consistency(sheet_data)

        tempo = float(music_info.iloc[0]["tempo"])
        qnote = 60.0 / tempo  # 四分音符秒
        daw_onset_start = float(daw_onsets.iloc[0]["onset"])
        daw_onset_end = float(daw_onsets.iloc[0]["offset"])

        total_len = float(sheet_data["length"].sum())
        expected_dur = total_len * qnote
        print(f"テンポ: {tempo:.1f} BPM  / 四分音符: {qnote:.3f}s")
        print(f"譜面総長: {total_len}  → 予想曲長 ≈ {expected_dur:.2f}s")

        # 準備
        self.cleanup_temp_files()
        temp_files: List[Path] = []
        current_sr = None

        start_ts = time.time()
        total_notes = len(sheet_data)
        processed = 0

        try:
            for idx, row in sheet_data.iterrows():
                # 音名
                raw_pn = row.get("PitchName", None)
                pitch_name = "rest" if pd.isna(raw_pn) else str(raw_pn).strip()

                # 長さ→秒
                length = float(row["length"])
                duration = length * qnote

                # Hr/Tb/Euph は 1 オクターブ下げ（main.py と同じ仕様）
                if self.daw_instrument in {"Hr", "Tb", "Euph"} and isinstance(pitch_name, str):
                    pn = pitch_name.replace("♭", "b").replace("♯", "#")
                    if pn.lower() != "rest":
                        import re as _re

                        m = _re.search(r"(.*?)(-?\d+)$", pn)
                        if m:
                            note_part, octv = m.group(1), m.group(2)
                            try:
                                pitch_name = f"{note_part}{int(octv) - 1}"
                            except Exception:
                                pass  # 失敗時は元のまま

                # 進捗表示
                processed += 1
                print(f"\n[{processed}/{total_notes}] Pitch={pitch_name} length={length} → {duration:.3f}s")

                # 一時保存先
                tmp = self.temp_dir / f"note_{idx:03d}.wav"

                # 休符（無音生成）
                if isinstance(pitch_name, str) and pitch_name.lower() == "rest":
                    if current_sr is None:
                        current_sr = 44100
                    silence = self.create_silence(duration, current_sr)
                    sf.write(tmp, silence, current_sr)
                    temp_files.append(tmp)
                    print("  -> 休符: 無音を書き出し")
                    continue

                # 素材 wav を探索
                wav_path = self.find_daw_audio_file(pitch_name)
                if wav_path is None or (not wav_path.exists()):
                    print(f"ERROR: 音名 '{pitch_name}' に対応する DAW 素材が見つかりません")
                    sys.exit(1)

                # 読み込み
                audio, sr = sf.read(str(wav_path), always_2d=False)
                current_sr = sr if current_sr is None else current_sr

                # 切り出し（SAT〜EAT）+ 所望の長さへ
                # - main.py の cut_audio_at_duration()（ゼロクロス考慮）をそのまま再利用
                note_audio = self.cut_audio_at_duration(
                    audio=np.asarray(audio, dtype=np.float32),
                    sr=sr,
                    duration=duration,
                    onset_start=daw_onset_start,
                    onset_end=daw_onset_end,
                )

                # 安定化処理とフェード
                note_audio = self.validate_audio(note_audio)
                note_audio = self.apply_fade_in_out(note_audio, sr, fade_duration=0.02)

                # 書き出し
                sf.write(tmp, note_audio, sr)
                temp_files.append(tmp)
                print(f"  -> {wav_path.name} を {duration:.3f}s に切り出し保存")

            # 連結（ゼロクロス補正 + コサインカーブのクロスフェード。main.pyの実装を再利用）
            print("\n結合処理中...")
            merged = self.smooth_concatenate_audio(temp_files, current_sr or 44100)

            # 最終フェード
            merged = self.apply_fade_in_out(merged, current_sr or 44100, fade_duration=0.02)

            # 出力名: 「_daw」を付与して区別
            out_path = self.outputs_dir / f"{self.target_file}_daw.wav"
            sf.write(str(out_path), merged, current_sr or 44100)
            print(f"\n✅ 完了: {out_path}  (総時間 {len(merged)/(current_sr or 44100):.2f}s)")

            return str(out_path)

        finally:
            # 中間ファイルを掃除
            self.cleanup_temp_files()


def main():
    # main.py と同じ環境変数を利用
    daw_instrument = os.getenv("DAW_INSTRUMENT", "Sax")
    target_file = os.getenv("TARGET_FILE", "Ssax_1_sad")
    song_number = os.getenv("SONG_NUMBER", "1")

    print("=== DAW 直結合成スクリプト (main_daw.py) ===")
    print(f"  DAW楽器 : {daw_instrument}")
    print(f"  ターゲット: {target_file}")
    print(f"  楽曲番号 : {song_number}")
    print("=" * 48)

    try:
        proc = DAWAudioProcessor(daw_instrument, target_file, song_number)
        out = proc.process_audio()
        print(f"\n出力ファイル: {out}")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()