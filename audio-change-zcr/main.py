import os
import sys
import pandas as pd
import numpy as np
import soundfile as sf
import time
from pathlib import Path
import gc
import tempfile
import shutil
from typing import List, Optional, Dict
import re

from apis import load, rms, specCentroid, specCrest

class AudioProcessor:
    def __init__(self, daw_instrument: str, target_file: str, song_number: str):
        self.daw_instrument = daw_instrument
        self.target_file = target_file
        self.song_number = song_number
        
        instrument_alias_for_daw = {
            "Euph" : "Tb",
        }
        self.daw_sounds_instrument = instrument_alias_for_daw.get(daw_instrument, daw_instrument)
        self.daw_sounds_dir = Path(f"daw_sounds/{self.daw_sounds_instrument}")
        self.inputs_dir = Path("inputs")
        self.sheet_dir = Path("sheet")
        self.outputs_dir = Path("outputs")
        self.temp_dir = Path("temp_notes")
        
        self.outputs_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        self._check_required_files()
        
    def _check_required_files(self):
        required_files = [
            self.inputs_dir / "sounds" / f"{self.target_file}.wav",
            self.inputs_dir / "csv" / f"{self.target_file}_rms.csv",
            self.inputs_dir / "csv" / f"{self.target_file}_spectral.csv",
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
            
    def cleanup_temp_files(self):
        """一時ファイルのクリーンアップ"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
            
    def normalize_pitch_name(self, pitch_name: str) -> str:
        if pitch_name == "rest":
            return pitch_name
            
        pitch_name = pitch_name.lower().replace('♭', 'b').replace('♯', '#')
        
        enharmonic_to_daw = {
            'bb': 'a#',
            'db': 'c#',
            'eb': 'd#',
            'gb': 'f#',
            'ab': 'g#',
        }
        
        note_part = pitch_name[:-1]
        octave = pitch_name[-1]
        
        if note_part in enharmonic_to_daw:
            normalized_note = enharmonic_to_daw[note_part]
            result = f"{normalized_note}{octave}"
            print(f"      異名同音変換: {pitch_name} → {result}")
            return result
        
        return pitch_name
        
    def find_daw_audio_file(self, pitch_name: str) -> Optional[Path]:
        if pitch_name == "rest":
            return None
            
        normalized_pitch = self.normalize_pitch_name(pitch_name)
        
        search_patterns = [
            f"*_{normalized_pitch}.wav",
            f"*_{pitch_name.lower()}.wav",
        ]
        
        pitch_lower = pitch_name.lower().replace('♭', 'b').replace('♯', '#')
        note_part = pitch_lower[:-1]
        octave = pitch_lower[-1]
        
        alternative_patterns = []
        if 'b' in note_part:
            flat_to_sharp = {'bb': 'a#', 'db': 'c#', 'eb': 'd#', 'gb': 'f#', 'ab': 'g#'}
            if note_part in flat_to_sharp:
                alternative_patterns.append(f"*_{flat_to_sharp[note_part]}{octave}.wav")
        elif '#' in note_part:
            sharp_to_flat = {'a#': 'bb', 'c#': 'db', 'd#': 'eb', 'f#': 'gb', 'g#': 'ab'}
            if note_part in sharp_to_flat:
                alternative_patterns.append(f"*_{sharp_to_flat[note_part]}{octave}.wav")
        
        all_patterns = search_patterns + alternative_patterns
        
        for pattern in all_patterns:
            files = list(self.daw_sounds_dir.glob(pattern))
            if files:
                print(f"      ファイル発見: {pattern} → {files[0].name}")
                return files[0]
                
        print(f"      音名 '{pitch_name}' (正規化: '{normalized_pitch}') に対応するファイルが見つかりません")
        return None
        
    def find_zero_crossing(self, audio: np.ndarray, start_sample: int, search_direction: int = 1, max_search_range: int = 1000) -> int:
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # より広い範囲でゼロクロスを検索
        if search_direction > 0:
            end_range = min(start_sample + max_search_range, len(audio) - 1)
            for i in range(start_sample, end_range):
                if i < len(audio) - 1 and audio[i] * audio[i + 1] <= 0:
                    # より正確なゼロクロス点を選択
                    return i + 1 if abs(audio[i + 1]) < abs(audio[i]) else i
        else:
            start_range = max(start_sample - max_search_range, 1)
            for i in range(start_sample, start_range, -1):
                if i > 0 and audio[i] * audio[i - 1] <= 0:
                    return i if abs(audio[i]) < abs(audio[i - 1]) else i - 1
        
        # ゼロクロスが見つからない場合、最も振幅の小さい点を探す
        if search_direction > 0:
            search_range = range(start_sample, min(start_sample + max_search_range, len(audio)))
        else:
            search_range = range(max(start_sample - max_search_range, 0), start_sample)
        
        if len(list(search_range)) > 0:
            min_idx = min(search_range, key=lambda x: abs(audio[x]) if x < len(audio) else float('inf'))
            return min_idx
        
        return start_sample
        
    def cut_audio_at_duration(self, audio: np.ndarray, sr: int, duration: float, 
                            onset_start: float, onset_end: float) -> np.ndarray:
        onset_start_sample = int(onset_start * sr)
        onset_end_sample = int(onset_end * sr)
        
        target_samples = int(duration * sr)
        
        available_samples = onset_end_sample - onset_start_sample
        if target_samples > available_samples:
            print(f"WARNING: 要求された長さ({duration:.3f}s)がオンセット範囲({onset_end - onset_start:.3f}s)を超えています")
            target_samples = available_samples
            
        end_sample = onset_start_sample + target_samples
        end_sample = self.find_zero_crossing(audio, end_sample, search_direction=-1)
        
        if end_sample > onset_end_sample:
            end_sample = self.find_zero_crossing(audio, onset_end_sample, search_direction=-1)
            
        start_sample = self.find_zero_crossing(audio, onset_start_sample, search_direction=1)
        
        return audio[start_sample:end_sample]
        
    def create_silence(self, duration: float, sr: int) -> np.ndarray:
        return np.zeros(int(duration * sr))
        
    def validate_audio(self, audio: np.ndarray) -> np.ndarray:
        if np.any(np.isnan(audio)):
            print(f"  警告: NaN値が検出されました")
            audio = np.nan_to_num(audio)
        
        if np.any(np.isinf(audio)):
            print(f"  警告: 無限大値が検出されました")
            audio = np.nan_to_num(audio)
        
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 1.0:
            print(f"  警告: 振幅が1.0を超えています({max_amplitude:.3f})。正規化します。")
            audio = audio / max_amplitude * 0.9
        
        return audio.astype(np.float32)

    def apply_transformations_note_wise(self, source_audio: np.ndarray, note_index: int, sr: int, 
                                      rms_profiles: Dict[int, np.ndarray],
                                      centroid_profiles: Dict[int, np.ndarray], 
                                      crest_profiles: Dict[int, np.ndarray],
                                      transformations: List[str]) -> np.ndarray:
        """音符単位での変換適用（A方式: DAW音源切り出し → 各変換適用 → 次の音符）"""
        current_audio = source_audio.copy()
        
        for transform_name in transformations:
            if transform_name == 'attack':
                print(f"      RMS変換適用中...")
                current_audio = rms.apply_rms_transformation_note_wise(
                    current_audio, rms_profiles, note_index, sr
                )
            elif transform_name == 'centroid':
                print(f"      Centroid変換適用中...")
                current_audio = specCentroid.apply_spectral_centroid_transformation_note_wise(
                    current_audio, centroid_profiles, note_index, sr
                )
            elif transform_name == 'crest':
                print(f"      Crest変換適用中...")
                current_audio = specCrest.apply_spectral_crest_transformation_note_wise(
                    current_audio, crest_profiles, note_index, sr
                )
            else:
                print(f"  警告: 未知の変換 '{transform_name}' をスキップします")
                continue
        
        return current_audio
    
    def smooth_concatenate_audio(self, temp_files: List[Path], sr: int) -> np.ndarray:
        """一時ファイルから音声を読み込んで滑らかに結合"""
        if not temp_files:
            return np.array([])
        
        audio_segments = []
        for temp_file in temp_files:
            if temp_file.exists():
                audio, _ = sf.read(temp_file)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                audio_segments.append(audio)
        
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        result = audio_segments[0].copy()
        
        for i in range(1, len(audio_segments)):
            current_segment = audio_segments[i].copy()
            
            # クロスフェード用のオーバーラップ区間を設定
            crossfade_samples = min(int(0.01 * sr), len(result) // 4, len(current_segment) // 4)
            crossfade_samples = max(crossfade_samples, 10)  # 最小10サンプル
            
            if len(result) > crossfade_samples and len(current_segment) > crossfade_samples:
                # ゼロクロス点での調整
                end_zero_cross = self.find_zero_crossing(result, len(result) - crossfade_samples, search_direction=-1, max_search_range=crossfade_samples)
                start_zero_cross = self.find_zero_crossing(current_segment, crossfade_samples, search_direction=1, max_search_range=crossfade_samples)
                
                # クロスフェード処理
                overlap_end = result[end_zero_cross-crossfade_samples:end_zero_cross]
                overlap_start = current_segment[start_zero_cross:start_zero_cross+crossfade_samples]
                
                if len(overlap_end) == len(overlap_start) and len(overlap_end) > 0:
                    # クロスフェード係数（コサインカーブ）
                    fade_out = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, len(overlap_end))))
                    fade_in = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, len(overlap_start))))
                    
                    # オーバーラップ区間をミックス
                    mixed_overlap = overlap_end * fade_out + overlap_start * fade_in
                    
                    # 結合
                    result = np.concatenate([
                        result[:end_zero_cross-crossfade_samples],
                        mixed_overlap,
                        current_segment[start_zero_cross+crossfade_samples:]
                    ])
                else:
                    # クロスフェードできない場合は従来通り
                    result = result[:end_zero_cross]
                    result = np.concatenate([result, current_segment[start_zero_cross:]])
            else:
                # セグメントが短すぎる場合は従来通り
                if len(result) > 10:
                    end_zero_cross = self.find_zero_crossing(result, len(result)-5, search_direction=-1)
                    result = result[:end_zero_cross]
                
                if len(current_segment) > 10:
                    start_zero_cross = self.find_zero_crossing(current_segment, 5, search_direction=1)
                    current_segment = current_segment[start_zero_cross:]
                
                result = np.concatenate([result, current_segment])
        
        return result
    
    def apply_fade_in_out(self, audio: np.ndarray, sr: int, fade_duration: float = 0.02) -> np.ndarray:
        """より滑らかなフェード処理"""
        fade_samples = int(fade_duration * sr)
        result = audio.copy()
        
        if len(result) <= fade_samples * 2:
            # 音声が短すぎる場合は全体にフェードを適用
            fade_samples = len(result) // 4
        
        if fade_samples > 0 and len(result) > fade_samples:
            # コサインカーブを使用したより滑らかなフェード
            fade_in = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, fade_samples)))
            fade_out = 0.5 * (1 - np.cos(np.pi * np.linspace(1, 0, fade_samples)))
            
            result[:fade_samples] *= fade_in
            result[-fade_samples:] *= fade_out
        
        return result
    
    def check_note_count_consistency(self, sheet_data: pd.DataFrame) -> None:
        """楽譜音符数と参考音源音符数の整合性チェック（休符を除く）"""
        # RMSプロファイルから音符数を取得
        rms_data = load.load_data_with_suffix(self.target_file, "rms", ".")
        rms_note_count = len(rms_data["onsets"])
        
        # Spectralプロファイルから音符数を取得
        spectral_data = load.load_data_with_suffix(self.target_file, "spectral", ".")
        spectral_note_count = len(spectral_data["onsets"])
        
        # 楽譜の休符以外の音符数を計算
        non_rest_notes = sheet_data[sheet_data['PitchName'].str.lower() != 'rest']
        sheet_non_rest_count = len(non_rest_notes)
        sheet_total_count = len(sheet_data)
        rest_count = sheet_total_count - sheet_non_rest_count
        
        print(f"音符数チェック:")
        print(f"  楽譜総数: {sheet_total_count}音符")
        print(f"  楽譜音符（休符除く）: {sheet_non_rest_count}音符")
        print(f"  楽譜休符: {rest_count}音符")
        print(f"  RMS参考音源: {rms_note_count}音符")
        print(f"  Spectral参考音源: {spectral_note_count}音符")
        
        if sheet_non_rest_count != rms_note_count:
            print(f"ERROR: 楽譜音符数（休符除く）({sheet_non_rest_count})とRMS参考音源音符数({rms_note_count})が一致しません")
            sys.exit(1)
            
        if sheet_non_rest_count != spectral_note_count:
            print(f"ERROR: 楽譜音符数（休符除く）({sheet_non_rest_count})とSpectral参考音源音符数({spectral_note_count})が一致しません")
            sys.exit(1)
        
        if rms_note_count != spectral_note_count:
            print(f"ERROR: RMS参考音源音符数({rms_note_count})とSpectral参考音源音符数({spectral_note_count})が一致しません")
            sys.exit(1)
        
        print(f"✓ 音符数の整合性確認完了")
                
    def process_audio(self, transformations: List[str] = ['centroid', 'crest', 'attack']) -> str:
        print(f"処理開始: {self.target_file}")
        print(f"使用するDAW音源: {self.daw_instrument}")
        print(f"適用する変換: {' -> '.join(transformations)}")
        
        # データ読み込み
        sheet_data = pd.read_csv(self.sheet_dir / f"{self.song_number}.csv")
        music_info = pd.read_csv(self.sheet_dir / "music_info.csv")
        daw_onsets = pd.read_csv(self.daw_sounds_dir / "onsets.csv")
        
        # 音符数の整合性チェック
        self.check_note_count_consistency(sheet_data)
        
        # 特徴量プロファイルを事前に取得
        print("特徴量プロファイル取得中...")
        rms_profiles = rms.get_target_rms_profiles(self.target_file, ".")
        centroid_profiles = specCentroid.get_target_centroid_profiles(self.target_file, ".")
        crest_profiles = specCrest.get_target_crest_profiles(self.target_file, ".")
        
        tempo = music_info.iloc[0]['tempo']
        quarter_note_duration = 60.0 / tempo
        
        daw_onset_start = daw_onsets.iloc[0]['onset']
        daw_onset_end = daw_onsets.iloc[0]['offset']
        
        total_length = sheet_data['length'].sum()
        expected_duration = total_length * quarter_note_duration
        
        print(f"テンポ: {tempo} BPM (四分音符: {quarter_note_duration:.3f}秒)")
        print(f"楽譜の総長さ: {total_length}音符分")
        print(f"予想される楽曲長: {expected_duration:.3f}秒 ({expected_duration/60:.2f}分)")
        print(f"DAWオンセット範囲: {daw_onset_start:.3f}s - {daw_onset_end:.3f}s")
        
        if expected_duration > 60:
            print(f"警告: 楽曲が異常に長いです。テンポ設定を確認してください。")
            print(f"もし20秒程度が正しい場合、実際のテンポは約 {(total_length * 60) / 20:.0f} BPM です")
        
        # 一時ファイルリスト
        temp_files = []
        current_sr = None
        
        start_time = time.time()
        
        try:
            total_notes = len(sheet_data)
            processed_notes = 0
            reference_note_index = 0  # 参考音源の音符インデックス（休符をスキップ）
            
            for idx, row in sheet_data.iterrows():
                # まず pitch_name を必ず決める（欠損は rest 扱い）
                raw_pn = row['PitchName'] if 'PitchName' in row.index else None
                if pd.isna(raw_pn):
                    pitch_name = 'rest'
                else:
                    pitch_name = str(raw_pn).strip()

                # 音価なども先に取得
                length = row['length']
                duration = length * quarter_note_duration

                # --- 楽器によるオクターブ補正（Hr / Tb / Euph は 1 オクターブ下げる）---
                if self.daw_instrument in {"Hr", "Tb", "Euph"} and isinstance(pitch_name, str):
                    pn = pitch_name
                    if pn.lower() != 'rest':
                        # 末尾の数字（例: C#4 の "4"）を正規表現で取得（全角♭/♯も事前に半角へ）
                        norm = pn.replace('♭', 'b').replace('♯', '#')
                        m = re.search(r"(.*?)(-?\d+)$", norm)
                        if m:
                            note_part, octv = m.group(1), m.group(2)
                            try:
                                new_oct = int(octv) - 1
                                pitch_name = f"{note_part}{new_oct}"
                                print(f"  楽器補正: {self.daw_instrument} のため {pn} -> {pitch_name} (1オクターブ下げ)")
                            except ValueError:
                                # 数字に変換できない場合は変更しない
                                pass

                processed_notes += 1
                print(f"\n音 {idx + 1}/{total_notes}: {pitch_name}, 長さ: {length} ({duration:.3f}秒)")
                print(f"進捗: {processed_notes}/{total_notes} ({processed_notes/total_notes*100:.1f}%)")

                if processed_notes > 1:
                    avg_time_per_note = (time.time() - start_time) / (processed_notes - 1)
                    remaining_notes = total_notes - processed_notes
                    estimated_remaining = avg_time_per_note * remaining_notes
                    print(f"推定残り時間: {estimated_remaining/60:.1f}分")

                # 一時ファイルパス
                temp_file = self.temp_dir / f"note_{idx:03d}.wav"

                # 休符処理（ここで必ず文字列として比較できる）
                if pitch_name.lower() == 'rest':
                    if current_sr is None:
                        current_sr = 44100
                    silence = self.create_silence(duration, current_sr)
                    sf.write(temp_file, silence, current_sr)
                    temp_files.append(temp_file)
                    print("  -> 休符として無音区間を保存")
                    # 休符の場合は reference_note_index をインクリメントしない
                    continue
                
                daw_file = self.find_daw_audio_file(pitch_name)
                if daw_file is None:
                    print(f"ERROR: 音名 '{pitch_name}' に対応するDAW音源が見つかりません")
                    print(f"利用可能な音源: {list(self.daw_sounds_dir.glob('*.wav'))}")
                    sys.exit(1)
                
                print(f"  DAW音源: {daw_file.name}")
                print(f"  参考音源音符インデックス: {reference_note_index}")
                
                daw_audio, sr = sf.read(daw_file)
                
                if daw_audio.ndim > 1:
                    daw_audio = np.mean(daw_audio, axis=1)
                
                if current_sr is None:
                    current_sr = sr
                elif current_sr != sr:
                    print(f"WARNING: サンプリングレートが異なります ({current_sr} vs {sr})")
                
                # DAW音源を指定長さにカット
                cut_audio = self.cut_audio_at_duration(
                    daw_audio, sr, duration, daw_onset_start, daw_onset_end
                )
                
                print(f"  カット後: {len(cut_audio)/sr:.3f}秒")
                
                # 音符単位での変換適用（A方式）- 参考音源のインデックスを使用
                print(f"  変換適用中: {' -> '.join(transformations)}")
                transformed_audio = self.apply_transformations_note_wise(
                    cut_audio, reference_note_index, sr, rms_profiles, centroid_profiles, crest_profiles, transformations
                )
                
                # 音声の検証とフェード処理
                transformed_audio = self.validate_audio(transformed_audio)
                transformed_audio = self.apply_fade_in_out(transformed_audio, sr, fade_duration=0.02)
                
                # 一時ファイルに保存
                sf.write(temp_file, transformed_audio, sr)
                temp_files.append(temp_file)
                
                # 音符（休符以外）の場合のみ参考音源インデックスをインクリメント
                reference_note_index += 1
                
                # メモリクリーンアップ
                del daw_audio, cut_audio, transformed_audio
                gc.collect()
                
            print("\n音声セグメントを結合中...")
            final_audio = self.smooth_concatenate_audio(temp_files, current_sr)
            
            transform_suffix = "_".join([t.title()[:4] for t in transformations])
            output_filename = f"{self.target_file}_{transform_suffix}.wav"
            output_path = self.outputs_dir / output_filename
            
            sf.write(output_path, final_audio, current_sr)
            
            print(f"\n処理完了!")
            print(f"出力ファイル: {output_path}")
            print(f"最終音声長: {len(final_audio)/current_sr:.3f}秒")
            
            return str(output_path)
            
        finally:
            # 一時ファイルのクリーンアップ
            print("\n一時ファイルクリーンアップ中...")
            self.cleanup_temp_files()
            gc.collect()

def main():
    daw_instrument = os.getenv('DAW_INSTRUMENT', 'Sax')
    target_file = os.getenv('TARGET_FILE', 'Ssax_1_sad')
    song_number = os.getenv('SONG_NUMBER', '1')
    
    print("=== 音響特徴量変換システム（音符単位変換版） ===")
    print(f"DAW楽器: {daw_instrument}")
    print(f"ターゲットファイル: {target_file}")
    print(f"楽曲番号: {song_number}")
    print("=" * 50)
    
    try:
        processor = AudioProcessor(daw_instrument, target_file, song_number)
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