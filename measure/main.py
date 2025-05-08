from api.SpectralCentM_plot import analyze_SpectralCent_plot
from api.SpectralCrest_plot import analyze_SpectralCrest_plot

def main():
    # ファイル名の指定
    filename = 'スケール1'

    # 音源ファイルのパスを指定
    wav_name = f"sounds/{filename}.wav"

    # 出力画像のパス
    output_name = [
        f'png/{filename}_SpecCent.png',
        f'png/{filename}_SPecCrest.png'
    ]

    # 音響特徴量の計算
    SpectralCentroidMedian = analyze_SpectralCent_plot(wav_name, output_name[0])
    SpectralCrestMedian, SpectralCrestIQR = analyze_SpectralCrest_plot(wav_name, output_name[1])

    # 結果の表示
    print("=== 音響特徴量 ===")
    print(f"Spectral Centroid Median: {SpectralCentroidMedian:.2f} Hz")
    print(f"Spectral Crest Median: {SpectralCrestMedian:.4f}")
    print(f"Spectral Crest IQR: {SpectralCrestIQR:.4f}")

if __name__ == '__main__':
    main()