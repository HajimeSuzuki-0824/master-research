import os
import sys
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import shutil
import time
from pathlib import Path
import gc
from typing import List, Tuple, Dict, Optional

# 既存のAPIをインポート
from apis import load, attackSlope, specCentroid, specCrest

class AudioProcessor:
    def __init__(self, daw_instrument: str, target_file: str, song_number: str):
        self.daw_instrument = daw_instrument
        self.target_file = target_file
        self.song_number = song_number
        
        # パスの設定
        self.daw_sounds_dir = Path(f"daw_sounds/{daw_instrument}")
        self.inputs_dir = Path("inputs")
        self.sheet_dir = Path("sheet")
        self.outputs_dir = Path("outputs")
        
        # 出力ディレクトリの作成
        self.outputs_dir.mkdir(exist_ok=True)
        
        # 必要なファイルの存在確認
        self._check_required_files()
        
    def _check_required_files(self):
        """必要なファイルの存在確認"""
        required_files = [
            self.inputs_dir / "sounds" / f"{self.target_file}.wav",
            self.inputs_dir / "csv" / f"{self.target_file}.csv",
            self.sheet_dir / f"{self.song_number}.csv",
            self.sheet_dir / "music_info.csv",
            self.daw_sounds_dir / "onsets.csv"
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            print("ERROR: 以下のファイルが見つかりません:")
            for f in missing_files:
                print(f"  - {f}")
            sys.exit(1)
            
    def normalize_pitch_name(self, pitch_name: str) -> str:
        """音名の正規化（異名同音を考慮）"""
        if pitch_name == "rest":
            return pitch_name
            
        # 基本的な正規化
        pitch_name = pitch_name.lower().replace('♭', 'b').replace('♯', '#')
        
        # 異名同音の変換（楽譜→DAW音源の方向）
        enharmonic_to_daw = {
            'bb': 'a#',  # b♭ → a#
            'db': 'c#',  # d♭ → c#
            'eb': 'd#',  # e♭ → d#
            'gb': 'f#',  # g♭ → f#
            'ab': 'g#',  # a♭ → g#
        }
        
        # 音名部分（オクターブ番号を除く）とオクターブ番号を分離
        note_part = pitch_name[:-1]  # 音名部分
        octave = pitch_name[-1]      # オクターブ番号
        
        # 異名同音の変換を試行
        if note_part in enharmonic_to_daw:
            normalized_note = enharmonic_to_daw[note_part]
            result = f"{normalized_note}{octave}"
            print(f"      異名同音変換: {pitch_name} → {result}")
            return result
        
        return pitch_name
        
    def find_daw_audio_file(self, pitch_name: str) -> Optional[Path]:
        """DAW音源ファイルを探す"""
        if pitch_name == "rest":
            return None
            
        # まず正規化を試行
        normalized_pitch = self.normalize_pitch_name(pitch_name)
        
        # ファイルパターンを生成（複数のパターンを試行）
        search_patterns = [
            f"*_{normalized_pitch}.wav",  # 正規化済み
            f"*_{pitch_name.lower()}.wav",  # 元の名前
        ]
        
        # 異名同音も試行
        pitch_lower = pitch_name.lower().replace('♭', 'b').replace('♯', '#')
        note_part = pitch_lower[:-1]
        octave = pitch_lower[-1]
        
        # 両方向の異名同音を試行
        alternative_patterns = []
        if 'b' in note_part:  # フラット系
            sharp_note = note_part.replace('b', '#')
            # フラット→シャープ変換テーブル
            flat_to_sharp = {'bb': 'a#', 'db': 'c#', 'eb': 'd#', 'gb': 'f#', 'ab': 'g#'}
            if note_part in flat_to_sharp:
                alternative_patterns.append(f"*_{flat_to_sharp[note_part]}{octave}.wav")
        elif '#' in note_part:  # シャープ系
            # シャープ→フラット変換テーブル
            sharp_to_flat = {'a#': 'bb', 'c#': 'db', 'd#': 'eb', 'f#': 'gb', 'g#': 'ab'}
            if note_part in sharp_to_flat:
                alternative_patterns.append(f"*_{sharp_to_flat[note_part]}{octave}.wav")
        
        # すべてのパターンを試行
        all_patterns = search_patterns + alternative_patterns
        
        for pattern in all_patterns:
            files = list(self.daw_sounds_dir.glob(pattern))
            if files:
                print(f"      ファイル発見: {pattern} → {files[0].name}")
                return files[0]
                
        print(f"      音名 '{pitch_name}' (正規化: '{normalized_pitch}') に対応するファイルが見つかりません")
        return None
        
    def find_zero_crossing(self, audio: np.ndarray, start_sample: int, search_direction: int = 1) -> int:
        """ゼロクロス点を見つける"""
        # ステレオの場合はモノラルに変換
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
            
        if search_direction > 0:
            # 前方検索
            for i in range(start_sample, len(audio) - 1):
                if audio[i] * audio[i + 1] <= 0:
                    return i + 1 if abs(audio[i + 1]) < abs(audio[i]) else i
        else:
            # 後方検索
            for i in range(start_sample, 0, -1):
                if audio[i] * audio[i - 1] <= 0:
                    return i if abs(audio[i]) < abs(audio[i - 1]) else i - 1
                    
        return start_sample
        
    def cut_audio_at_duration(self, audio: np.ndarray, sr: int, duration: float, 
                            onset_start: float, onset_end: float) -> np.ndarray:
        """指定の長さで音声をカット（ゼロクロス考慮）"""
        # オンセット範囲のサンプル位置
        onset_start_sample = int(onset_start * sr)
        onset_end_sample = int(onset_end * sr)
        
        # 目標の長さ（サンプル数）
        target_samples = int(duration * sr)
        
        # オンセット範囲をチェック
        available_samples = onset_end_sample - onset_start_sample
        if target_samples > available_samples:
            print(f"WARNING: 要求された長さ({duration:.3f}s)がオンセット範囲({onset_end - onset_start:.3f}s)を超えています")
            target_samples = available_samples
            
        # カット位置を決定（ゼロクロス考慮）
        end_sample = onset_start_sample + target_samples
        end_sample = self.find_zero_crossing(audio, end_sample, search_direction=-1)
        
        # 範囲チェック
        if end_sample > onset_end_sample:
            end_sample = self.find_zero_crossing(audio, onset_end_sample, search_direction=-1)
            
        start_sample = self.find_zero_crossing(audio, onset_start_sample, search_direction=1)
        
        return audio[start_sample:end_sample]
        
    def create_silence(self, duration: float, sr: int) -> np.ndarray:
        """無音区間を作成"""
        return np.zeros(int(duration * sr))
        
    def save_temp_audio(self, audio: np.ndarray, sr: int, filename: str) -> str:
        """一時音声ファイルを保存"""
        temp_path = f"temp_{filename}"
        sf.write(temp_path, audio, sr)
        return temp_path
        
    def apply_direct_transformations(self, source_audio: np.ndarray, sr: int, 
                                   target_audio_path: str, transformations: List[str]) -> np.ndarray:
        """DTWを使わない直接的な音響特徴量変換（旧システム互換）"""
        print(f"  直接変換モード（DTW不使用、旧システム互換）")
        
        # デバッグ: まずは変換せずにそのまま返す
        print(f"  デバッグモード: 変換をスキップして元音声をそのまま使用")
        print(f"  音声情報: 最大値={np.max(np.abs(source_audio)):.6f}, 平均値={np.mean(np.abs(source_audio)):.6f}")
        
        # 音声データの健全性チェック
        if np.any(np.isnan(source_audio)):
            print(f"  警告: NaN値が検出されました")
            source_audio = np.nan_to_num(source_audio)
        
        if np.any(np.isinf(source_audio)):
            print(f"  警告: 無限大値が検出されました")
            source_audio = np.nan_to_num(source_audio)
        
        # 振幅が異常に大きい場合は正規化
        max_amplitude = np.max(np.abs(source_audio))
        if max_amplitude > 1.0:
            print(f"  警告: 振幅が1.0を超えています({max_amplitude:.3f})。正規化します。")
            source_audio = source_audio / max_amplitude * 0.9
        
        # とりあえず変換は行わず、元の音声をそのまま返す
        return source_audio.astype(np.float32)
        
        # # 以下は後でテスト用に有効化
        # # ターゲット音声を読み込み
        # target_audio, target_sr = sf.read(target_audio_path)
        # if target_audio.ndim > 1:
        #     target_audio = np.mean(target_audio, axis=1)
        # 
        # # サンプリングレート統一（旧システムと同じ44.1kHz維持）
        # if target_sr != sr:
        #     target_audio = librosa.resample(target_audio, orig_sr=target_sr, target_sr=sr)
        # 
        # current_audio = source_audio.copy()
        # 
        # # 変換関数のマッピング（新しい直接変換版）
        # transform_functions = {
        #     'centroid': self._apply_centroid_direct,
        #     'crest': self._apply_crest_direct,
        #     'attack': self._apply_attack_direct
        # }
        # 
        # for i, transform_name in enumerate(transformations):
        #     if transform_name not in transform_functions:
        #         raise ValueError(f"未知の変換: {transform_name}")
        #         
        #     print(f"    {transform_name}変換中...")
        #     start_time = time.time()
        #     
        #     # 変換を適用
        #     current_audio = transform_functions[transform_name](current_audio, target_audio, sr)
        #     
        #     end_time = time.time()
        #     print(f"    {transform_name}変換完了: {end_time - start_time:.2f}秒")
        # 
        # return current_audio
    
    def _apply_centroid_direct(self, source_audio: np.ndarray, target_audio: np.ndarray, sr: int) -> np.ndarray:
        """Spectral Centroid直接変換（既存API準拠）"""
        # 一時ファイルに保存
        temp_source = f"/tmp/temp_source_{int(time.time() * 1000000)}.wav"
        temp_output = f"/tmp/temp_output_{int(time.time() * 1000000)}.wav"
        
        try:
            sf.write(temp_source, source_audio, sr)
            
            # 既存APIの新しい直接変換関数を使用
            # （実際の実装では、上記のapi_transform_spectral_centroid_directを呼び出し）
            result_audio = self._call_centroid_api_direct(source_audio, target_audio, sr, temp_output)
            
            return result_audio
            
        finally:
            # 一時ファイル削除
            for temp_file in [temp_source, temp_output]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    def _apply_crest_direct(self, source_audio: np.ndarray, target_audio: np.ndarray, sr: int) -> np.ndarray:
        """Spectral Crest直接変換（既存API準拠）"""
        temp_output = f"/tmp/temp_output_{int(time.time() * 1000000)}.wav"
        
        try:
            result_audio = self._call_crest_api_direct(source_audio, target_audio, sr, temp_output)
            return result_audio
            
        finally:
            if os.path.exists(temp_output):
                os.remove(temp_output)
    
    def _apply_attack_direct(self, source_audio: np.ndarray, target_audio: np.ndarray, sr: int) -> np.ndarray:
        """Attack Slope直接変換（既存API準拠）"""
        temp_output = f"/tmp/temp_output_{int(time.time() * 1000000)}.wav"
        
        try:
            result_audio = self._call_attack_api_direct(source_audio, target_audio, sr, temp_output)
            return result_audio
            
        finally:
            if os.path.exists(temp_output):
                os.remove(temp_output)
    
    def _call_centroid_api_direct(self, source_audio: np.ndarray, target_audio: np.ndarray, sr: int, output_path: str) -> np.ndarray:
        """Spectral Centroid API直接呼び出し（既存API互換）"""
        # 既存APIの処理をDTWなしで実行する簡易実装
        # 実際には新しく追加するapi_transform_spectral_centroid_directを呼び出し
        
        import librosa
        import scipy.optimize
        
        # 既存APIと同じパラメータ
        HOP_LENGTH = 512
        FRAME_LENGTH = HOP_LENGTH * 4
        
        # ターゲットのSpectral Centroidを計算
        target_centroid = np.mean(librosa.feature.spectral_centroid(y=target_audio, sr=sr, hop_length=HOP_LENGTH)[0])
        
        # ソースに対して変換適用（既存APIのロジックを簡略化）
        S = librosa.stft(source_audio, n_fft=2048, hop_length=HOP_LENGTH, dtype=np.complex64)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        # 簡易的な重心調整（完全版は既存APIの新関数で実装）
        for t in range(S.shape[1]):
            magnitude = np.abs(S[:, t])
            phase = np.angle(S[:, t])
            
            # 現在の重心を計算
            current_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-6)
            
            # 重心調整比率
            shift_ratio = target_centroid / (current_centroid + 1e-6)
            shift_ratio = np.clip(shift_ratio, 0.5, 2.0)  # 制限
            
            # 簡易的な重心調整
            adjusted_magnitude = magnitude * (0.7 + 0.3 * shift_ratio)
            S[:, t] = adjusted_magnitude * np.exp(1j * phase)
        
        result = librosa.istft(S, hop_length=HOP_LENGTH, length=len(source_audio))
        sf.write(output_path, result, sr)
        
        return result
    
    def _call_crest_api_direct(self, source_audio: np.ndarray, target_audio: np.ndarray, sr: int, output_path: str) -> np.ndarray:
        """Spectral Crest API直接呼び出し"""
        import librosa
        
        HOP_LENGTH = 512
        
        # ターゲットのSpectral Crestを計算
        target_S = np.abs(librosa.stft(target_audio, hop_length=HOP_LENGTH))
        target_crest = np.mean(np.max(target_S, axis=0) / (np.sum(target_S, axis=0) + 1e-10))
        
        # ソースに対して変換適用
        S = librosa.stft(source_audio, hop_length=HOP_LENGTH, dtype=np.complex64)
        
        for t in range(S.shape[1]):
            magnitude = np.abs(S[:, t])
            phase = np.angle(S[:, t])
            
            if np.sum(magnitude) > 0:
                current_crest = np.max(magnitude) / (np.sum(magnitude) + 1e-10)
                crest_ratio = target_crest / (current_crest + 1e-6)
                crest_ratio = np.clip(crest_ratio, 0.5, 2.0)
                
                # ピーク強調/抑制
                max_val = np.max(magnitude)
                mean_val = np.mean(magnitude)
                enhanced_magnitude = magnitude + (max_val - magnitude) * (crest_ratio - 1) * 0.1
                enhanced_magnitude = np.clip(enhanced_magnitude, 0, max_val * 1.5)
                
                S[:, t] = enhanced_magnitude * np.exp(1j * phase)
        
        result = librosa.istft(S, hop_length=HOP_LENGTH, length=len(source_audio))
        sf.write(output_path, result, sr)
        
        return result
    
    def _call_attack_api_direct(self, source_audio: np.ndarray, target_audio: np.ndarray, sr: int, output_path: str) -> np.ndarray:
        """Attack Slope API直接呼び出し"""
        import librosa
        
        HOP_LENGTH = 512
        FRAME_LENGTH = HOP_LENGTH * 4
        
        # ターゲットのAttack特徴を計算
        target_rms = librosa.feature.rms(y=target_audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        target_attack = np.mean(target_rms)
        
        # ソースのAttack特徴を計算
        source_rms = librosa.feature.rms(y=source_audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        source_attack = np.mean(source_rms)
        
        # ゲイン比を計算
        if source_attack > 0:
            gain_ratio = target_attack / source_attack
            gain_ratio = np.clip(gain_ratio, 0.1, 2.0)  # 制限
        else:
            gain_ratio = 1.0
        
        # アタック部分の調整（開始部分を重点的に）
        result = source_audio.copy()
        attack_length = min(int(0.1 * sr), len(result) // 4)  # 最初の0.1秒
        
        if attack_length > 0:
            # 線形に減少するゲインを適用
            gain_envelope = np.linspace(gain_ratio, 1.0, attack_length)
            result[:attack_length] *= gain_envelope
        
        sf.write(output_path, result, sr)
        return result
    
    def smooth_concatenate_audio(self, audio_segments: List[np.ndarray]) -> np.ndarray:
        """ゼロクロスを考慮して音声セグメントをスムーズに結合"""
        if not audio_segments:
            return np.array([])
        
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        result = audio_segments[0].copy()
        
        for i in range(1, len(audio_segments)):
            current_segment = audio_segments[i].copy()
            
            # 前のセグメントの終端をゼロクロスに調整
            if len(result) > 10:  # 十分な長さがある場合
                end_zero_cross = self.find_zero_crossing(result, len(result)-5, search_direction=-1)
                result = result[:end_zero_cross]
            
            # 現在のセグメントの開始をゼロクロスに調整
            if len(current_segment) > 10:
                start_zero_cross = self.find_zero_crossing(current_segment, 5, search_direction=1)
                current_segment = current_segment[start_zero_cross:]
            
            # 結合
            result = np.concatenate([result, current_segment])
        
        return result
    
    def apply_smooth_volume_adjustment(self, source_audio: np.ndarray, target_volume: float, sr: int) -> np.ndarray:
        """滑らかな音量調整（Attack Slopeの代替）"""
        result = source_audio.copy()
        
        # 現在の音量を計算
        current_volume = np.sqrt(np.mean(source_audio ** 2))  # RMS
        
        if current_volume > 0:
            volume_ratio = target_volume / current_volume
            volume_ratio = np.clip(volume_ratio, 0.1, 3.0)  # 制限
            
            # 全体に適用（急激な変化を避ける）
            result = result * volume_ratio
        
        return result
    
    def apply_fade_in_out(self, audio: np.ndarray, sr: int, fade_duration: float = 0.01) -> np.ndarray:
        """音の開始・終了にフェードを適用"""
        fade_samples = int(fade_duration * sr)
        result = audio.copy()
        
        # フェードイン
        if len(result) > fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            result[:fade_samples] *= fade_in
        
        # フェードアウト
        if len(result) > fade_samples:
            fade_out = np.linspace(1, 0, fade_samples)
            result[-fade_samples:] *= fade_out
        
        return result
                
    def process_audio(self, transformations: List[str] = ['centroid', 'crest', 'attack']) -> str:
        """メイン処理"""
        print(f"処理開始: {self.target_file}")
        print(f"使用するDAW音源: {self.daw_instrument}")
        print(f"適用する変換: {' -> '.join(transformations)}")
        
        # 1. 設定ファイルを読み込み
        sheet_data = pd.read_csv(self.sheet_dir / f"{self.song_number}.csv")
        music_info = pd.read_csv(self.sheet_dir / "music_info.csv")
        daw_onsets = pd.read_csv(self.daw_sounds_dir / "onsets.csv")
        
        # テンポ情報を取得
        tempo = music_info.iloc[0]['tempo']
        quarter_note_duration = 60.0 / tempo
        
        # DAWオンセット情報
        daw_onset_start = daw_onsets.iloc[0]['onset']
        daw_onset_end = daw_onsets.iloc[0]['offset']
        
        # 楽譜の総長さを計算して確認
        total_length = sheet_data['length'].sum()
        expected_duration = total_length * quarter_note_duration
        
        print(f"テンポ: {tempo} BPM (四分音符: {quarter_note_duration:.3f}秒)")
        print(f"楽譜の総長さ: {total_length}音符分")
        print(f"予想される楽曲長: {expected_duration:.3f}秒 ({expected_duration/60:.2f}分)")
        
        # 異常に長い場合は警告
        if expected_duration > 60:  # 1分を超える場合
            print(f"警告: 楽曲が異常に長いです。テンポ設定を確認してください。")
            print(f"もし20秒程度が正しい場合、実際のテンポは約 {(total_length * 60) / 20:.0f} BPM です")
        
        print(f"DAWオンセット範囲: {daw_onset_start:.3f}s - {daw_onset_end:.3f}s")
        
        # 2. 結果音声を格納するリスト
        final_audio_segments = []
        current_sr = None
        temp_files = []
        
        # 処理開始時間を記録
        import time
        start_time = time.time()
        
        try:
            total_notes = len(sheet_data)
            processed_notes = 0
            
            for idx, row in sheet_data.iterrows():
                pitch_name = row['PitchName']
                length = row['length']
                duration = length * quarter_note_duration
                
                processed_notes += 1
                print(f"\n音 {idx + 1}/{total_notes}: {pitch_name}, 長さ: {length} ({duration:.3f}秒)")
                print(f"進捗: {processed_notes}/{total_notes} ({processed_notes/total_notes*100:.1f}%)")
                
                # 推定残り時間（前回の処理時間から）
                if processed_notes > 1:
                    avg_time_per_note = (time.time() - start_time) / (processed_notes - 1)
                    remaining_notes = total_notes - processed_notes
                    estimated_remaining = avg_time_per_note * remaining_notes
                    print(f"推定残り時間: {estimated_remaining/60:.1f}分")
                
                if pitch_name.lower() == 'rest':
                    # 休符の場合
                    if current_sr is None:
                        current_sr = 44100  # デフォルトサンプリングレート
                    silence = self.create_silence(duration, current_sr)
                    final_audio_segments.append(silence)
                    print("  -> 休符として無音区間を追加")
                    continue
                
                # 3. DAW音源ファイルを探す
                daw_file = self.find_daw_audio_file(pitch_name)
                if daw_file is None:
                    print(f"ERROR: 音名 '{pitch_name}' に対応するDAW音源が見つかりません")
                    print(f"利用可能な音源: {list(self.daw_sounds_dir.glob('*.wav'))}")
                    sys.exit(1)
                
                print(f"  DAW音源: {daw_file.name}")
                
                # 4. DAW音源を読み込み
                daw_audio, sr = sf.read(daw_file)
                
                # ステレオの場合はモノラルに変換
                if daw_audio.ndim > 1:
                    daw_audio = np.mean(daw_audio, axis=1)
                
                if current_sr is None:
                    current_sr = sr
                elif current_sr != sr:
                    print(f"WARNING: サンプリングレートが異なります ({current_sr} vs {sr})")
                
                # 5. 指定の長さで音声をカット
                cut_audio = self.cut_audio_at_duration(
                    daw_audio, sr, duration, daw_onset_start, daw_onset_end
                )
                
                print(f"  カット後: {len(cut_audio)/sr:.3f}秒")
                
                # 6-8. 音響特徴量変換を適用
                print(f"  変換適用中: {' -> '.join(transformations)}")
                
                # サンプリングレートは旧システムと同じく44.1kHzを維持
                # （ダウンサンプリングしない）
                if current_sr is None:
                    current_sr = sr
                elif current_sr != sr and sr != 44100:
                    # 44.1kHzでない場合のみリサンプリング
                    print(f"  サンプリングレート変換: {sr}Hz -> 44100Hz")
                    cut_audio = librosa.resample(cut_audio, orig_sr=sr, target_sr=44100)
                    sr = 44100
                    current_sr = 44100
                
                # ターゲットファイルのパス
                target_audio_path = str(self.inputs_dir / "sounds" / f"{self.target_file}.wav")
                
                # 直接変換を適用（DTW不使用、既存API互換）
                transformed_audio = self.apply_direct_transformations(
                    cut_audio, sr, target_audio_path, transformations
                )

                transformed_audio = self.apply_direct_transformations(cut_audio, sr, target_audio_path, transformations)

                transformed_audio = self.apply_fade_in_out(transformed_audio, sr, fade_duration=0.005)  # 5ms
                
                final_audio_segments.append(transformed_audio)
                
                # メモリ解放
                del daw_audio, cut_audio, transformed_audio
                gc.collect()
                
            # 9. 全ての音声セグメントを結合
            print("\n音声セグメントを結合中...")
            final_audio = self.smooth_concatenate_audio(final_audio_segments)
            
            # 10. 出力ファイル名を生成
            transform_suffix = "_".join([t.title()[:4] for t in transformations])
            output_filename = f"{self.target_file}_{transform_suffix}.wav"
            output_path = self.outputs_dir / output_filename
            
            # 11. 最終音声を保存
            sf.write(output_path, final_audio, current_sr)
            
            print(f"\n処理完了!")
            print(f"出力ファイル: {output_path}")
            print(f"最終音声長: {len(final_audio)/current_sr:.3f}秒")
            
            return str(output_path)
            
        finally:
            # メモリクリーンアップ
            gc.collect()

def main():
    """メイン関数"""
    # 環境変数から設定を取得
    daw_instrument = os.getenv('DAW_INSTRUMENT', 'Sax')
    target_file = os.getenv('TARGET_FILE', 'Ssax_1_smile')
    song_number = os.getenv('SONG_NUMBER', '1')
    
    print("=== 音響特徴量変換システム ===")
    print(f"DAW楽器: {daw_instrument}")
    print(f"ターゲットファイル: {target_file}")
    print(f"楽曲番号: {song_number}")
    print("=" * 40)
    
    try:
        # プロセッサを初期化
        processor = AudioProcessor(daw_instrument, target_file, song_number)
        
        # 処理を実行
        output_path = processor.process_audio(['centroid', 'crest', 'attack'])
        
        print(f"\n✅ 全ての処理が正常に完了しました")
        print(f"出力ファイル: {output_path}")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()