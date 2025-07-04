# main.py
import os
import shutil
from datetime import datetime, timedelta
from itertools import permutations
from apis import attackSlope, specCentroid, specCrest

DIFF_JST_FROM_UTC = 9


instrument = 'Hr'
song_num = 2
valence_list = ['smile', 'sad']

song_num = str(song_num)
TARGET_NAME = instrument + "_" + song_num + "_" + "daw"


transform_functions = {
    'attack': attackSlope.api_transform_attack_slope,
    'centroid': specCentroid.api_transform_spectral_centroid,
    'crest': specCrest.api_transform_spectral_crest
}


def apply_single_transform(source_name, target_name, transform_type):
    """単一の変換を適用"""
    print(f"  [{transform_type}変換] {source_name} -> {target_name}")
    
    func = transform_functions[transform_type]
    output_name = f"{target_name}_{transform_type}Trans_{source_name.split('_')[-1]}.wav"
    output_path = func(source_name, target_name, output_name)
    
    return output_path

def copy_temp_file_to_inputs(temp_file_path, temp_name, target_name):
    """一時ファイルとCSVファイルをinputsにコピー"""
    input_sounds_dir = "./inputs/sounds"
    input_csv_dir = "./inputs/csv"
    
    # 音声ファイルのコピー
    dest_audio_path = os.path.join(input_sounds_dir, f"{temp_name}.wav")
    if os.path.exists(temp_file_path):
        shutil.copy2(temp_file_path, dest_audio_path)
        print(f"  [COPY] {temp_file_path} -> {dest_audio_path}")
    else:
        raise FileNotFoundError(f"一時ファイルが見つかりません: {temp_file_path}")
    
    # CSVファイルのコピー（ターゲット音声のCSVを使用）
    target_csv_path = os.path.join(input_csv_dir, f"{target_name}.csv")
    dest_csv_path = os.path.join(input_csv_dir, f"{temp_name}.csv")
    
    if os.path.exists(target_csv_path):
        shutil.copy2(target_csv_path, dest_csv_path)
        print(f"  [COPY] {target_csv_path} -> {dest_csv_path}")
    else:
        raise FileNotFoundError(f"ターゲットCSVファイルが見つかりません: {target_csv_path}")
    
    return dest_audio_path

def apply_double_transform(source_name, target_name, transform1, transform2):
    """2つの変換を順次適用"""
    print(f"  [{transform1}->{transform2}変換] {source_name} -> {target_name}")
    
    # 1回目の変換
    func1 = transform_functions[transform1]
    temp_name = f"{target_name}_temp_{transform1}"
    temp_output_name = f"{temp_name}.wav"
    temp_path = func1(source_name, target_name, temp_output_name)
    
    # 一時ファイルをinputs/soundsにコピー（ターゲットのCSVを使用）
    try:
        copy_temp_file_to_inputs(temp_path, temp_name, target_name)
        
        # 2回目の変換
        func2 = transform_functions[transform2]
        final_name = f"{target_name}_{transform1}_{transform2}Trans_{source_name.split('_')[-1]}.wav"
        final_path = func2(source_name, temp_name, final_name)
        
        return final_path
        
    except Exception as e:
        raise Exception(f"2重変換エラー: {e}")

def apply_triple_transform(source_name, target_name, transform1, transform2, transform3):
    """3つの変換を順次適用"""
    print(f"  [{transform1}->{transform2}->{transform3}変換] {source_name} -> {target_name}")
    
    # 1回目の変換
    func1 = transform_functions[transform1]
    temp1_name = f"{target_name}_temp_{transform1}"
    temp1_output_name = f"{temp1_name}.wav"
    temp1_path = func1(source_name, target_name, temp1_output_name)
    
    try:
        # 1回目の一時ファイルをinputs/soundsにコピー（ターゲットのCSVを使用）
        copy_temp_file_to_inputs(temp1_path, temp1_name, target_name)
        
        # 2回目の変換
        func2 = transform_functions[transform2]
        temp2_name = f"{target_name}_temp_{transform1}_{transform2}"
        temp2_output_name = f"{temp2_name}.wav"
        temp2_path = func2(source_name, temp1_name, temp2_output_name)
        
        # 2回目の一時ファイルをinputs/soundsにコピー（ターゲットのCSVを使用）
        copy_temp_file_to_inputs(temp2_path, temp2_name, target_name)
        
        # 3回目の変換
        func3 = transform_functions[transform3]
        final_name = f"{target_name}_{transform1}_{transform2}_{transform3}Trans_{source_name.split('_')[-1]}.wav"
        final_path = func3(source_name, temp2_name, final_name)
        
        return final_path
        
    except Exception as e:
        raise Exception(f"3重変換エラー: {e}")

def cleanup_temp_files():
    """一時ファイルをクリーンアップ"""
    input_sounds_dir = "./inputs/sounds"
    input_csv_dir = "./inputs/csv"
    temp_files = []
    
    # 音声ファイルの一時ファイル
    if os.path.exists(input_sounds_dir):
        for file in os.listdir(input_sounds_dir):
            if "temp_" in file:
                temp_files.append(os.path.join(input_sounds_dir, file))
    
    # CSVファイルの一時ファイル
    if os.path.exists(input_csv_dir):
        for file in os.listdir(input_csv_dir):
            if "temp_" in file:
                temp_files.append(os.path.join(input_csv_dir, file))
    
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            print(f"  [CLEANUP] {temp_file}")
        except Exception as e:
            print(f"  [CLEANUP ERROR] {temp_file}: {e}")

def process_valence(valence):
    """感情ごとの処理"""
    source_name = f"{instrument}_{song_num}_{valence}"
    
    print(f"\n=== {instrument} - {valence} ===")
    results = []
    
    try:
        # 1. 単一変換（3種類）
        print("1. 単一変換:")
        for transform_type in ['attack', 'centroid', 'crest']:
            try:
                output_path = apply_single_transform(source_name, TARGET_NAME, transform_type)
                results.append(f"✓ {transform_type}変換完了")
            except Exception as e:
                results.append(f"✗ {transform_type}変換失敗: {e}")
        
        # 2. 2つの変換組み合わせ（6種類）
        print("2. 2つの変換組み合わせ:")
        transform_types = ['attack', 'centroid', 'crest']
        for combo in permutations(transform_types, 2):
            try:
                output_path = apply_double_transform(source_name, TARGET_NAME, combo[0], combo[1])
                results.append(f"✓ {combo[0]}→{combo[1]}変換完了")
            except Exception as e:
                results.append(f"✗ {combo[0]}→{combo[1]}変換失敗: {e}")
        
        # 3. 3つの変換組み合わせ（6種類）
        print("3. 3つの変換組み合わせ:")
        for combo in permutations(transform_types, 3):
            try:
                output_path = apply_triple_transform(source_name, TARGET_NAME, combo[0], combo[1], combo[2])
                results.append(f"✓ {combo[0]}→{combo[1]}→{combo[2]}変換完了")
            except Exception as e:
                results.append(f"✗ {combo[0]}→{combo[1]}→{combo[2]}変換失敗: {e}")
                
    except Exception as e:
        results.append(f"✗ 処理中にエラー: {e}")
    
    # 各感情処理後に一時ファイルをクリーンアップ
    print("一時ファイルクリーンアップ:")
    cleanup_temp_files()
    
    return results

def main():
    try:
        print("音響特徴量変換処理を開始...")
        print(f"楽器: {instrument}")
        print(f"楽曲番号: {song_num}")
        print(f"ターゲット: {TARGET_NAME}")
        print(f"感情リスト: {valence_list}")
        
        # 出力ディレクトリの確認・作成
        os.makedirs("outputs", exist_ok=True)
        
        all_results = []
        
        # 各感情に対して処理実行
        for valence in valence_list:
            results = process_valence(valence)
            all_results.extend(results)
        
        # 最終クリーンアップ
        print("\n最終クリーンアップ:")
        cleanup_temp_files()
        
        # 結果サマリー
        print("\n" + "="*60)
        print("処理結果サマリー")
        print("="*60)
        
        success_count = sum(1 for r in all_results if r.startswith("✓"))
        error_count = sum(1 for r in all_results if r.startswith("✗"))
        
        print(f"成功: {success_count}件")
        print(f"失敗: {error_count}件")
        print(f"合計処理数: {len(all_results)}件")
        
        if error_count > 0:
            print("\n失敗した処理:")
            for result in all_results:
                if result.startswith("✗"):
                    print(f"  {result}")
        
        print(f"\n生成予定ファイル数: {len(valence_list)} × 15種類 = {len(valence_list) * 15}ファイル")
        print("処理完了!")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    start = datetime.utcnow() + timedelta(hours=DIFF_JST_FROM_UTC)
    print("実行開始：", start.strftime("%Y-%m-%d %H:%M:%S"))

    main()

    end = datetime.utcnow() + timedelta(hours=DIFF_JST_FROM_UTC)
    print("実行終了：", end.strftime("%Y-%m-%d %H:%M:%S"))
    execution_time = end - start
    print(f"実行時間： {execution_time}")