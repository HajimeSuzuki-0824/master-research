from api.SpectralCentroid_plot import analyze_SpectralCentroid_plot
from api.SpectralCrest_plot import analyze_SpectralCrest_plot
from api.SpectralDecrease_plot import analyze_SpectralDecrease_plot
from api.SpectralFlatness_plot import analyze_SpectralFlatness_plot
from api.SpectralRolloff_plot import analyze_SpectralRolloff_plot
from api.SpectralSpread_plot import analyze_SpectralSpread_plot
from api.SpectralVariation_plot import analyze_SpectralVariation_plot
from api.AttackSlope_plot import analyze_AttackSlope_plot
from api.TemporalCentroid_plot import analyze_TemporalCentroid_plot

def main():
    # ファイル名の指定
    filename = 'SSsax_1_smile'
    wav_name = f"sounds/{filename}.wav"

    # 出力画像パスの定義
    output_paths = {
        "SpecCent": f"png/{filename}_SpecCent.png",
        "SpecCrest": f"png/{filename}_SpecCrest.png",
        "SpecDecr": f"png/{filename}_SpecDecr.png",
        "SpecFlat": f"png/{filename}_SpecFlat.png",
        "SpecRoll": f"png/{filename}_SpecRoll.png",
        "SpecSpread": f"png/{filename}_SpecSpread.png",
        "SpecVar": f"png/{filename}_SpecVar.png",
        "Attack": f"png/{filename}_Attack.png",
        "TempCent": f"png/{filename}_TempCent.png"
    }

    # 音響特徴量の計算
    spec_cent_median = analyze_SpectralCentroid_plot(wav_name, output_paths["SpecCent"])
    spec_crest_median, spec_crest_iqr = analyze_SpectralCrest_plot(wav_name, output_paths["SpecCrest"])
    spec_decr_median = analyze_SpectralDecrease_plot(wav_name, output_paths["SpecDecr"])
    spec_flat_median = analyze_SpectralFlatness_plot(wav_name, output_paths["SpecFlat"])
    spec_roll_iqr = analyze_SpectralRolloff_plot(wav_name, output_paths["SpecRoll"])
    spec_spread_iqr = analyze_SpectralSpread_plot(wav_name, output_paths["SpecSpread"])
    spec_var_median, spec_var_iqr = analyze_SpectralVariation_plot(wav_name, output_paths["SpecVar"])
    attack_slope = analyze_AttackSlope_plot(wav_name, output_paths["Attack"])
    temporal_centroid = analyze_TemporalCentroid_plot(wav_name, output_paths["TempCent"])

    # 結果の表示
    print("=== 音響特徴量の統計量 ===")
    print(f"Spectral Centroid Median: {spec_cent_median:.2f} Hz")
    print(f"Spectral Crest Median: {spec_crest_median:.4f}, IQR: {spec_crest_iqr:.4f}")
    print(f"Spectral Decrease Median: {spec_decr_median:.4f}")
    print(f"Spectral Flatness Median: {spec_flat_median:.4f}")
    print(f"Spectral Rolloff IQR: {spec_roll_iqr:.2f} Hz")
    print(f"Spectral Spread IQR: {spec_spread_iqr:.2f} Hz")
    print(f"Spectral Variation Median: {spec_var_median:.4f}, IQR: {spec_var_iqr:.4f}")
    if attack_slope is not None:
        print(f"Attack Slope (dB/s): {attack_slope:.2f}")
    else:
        print("Attack Slope: 該当区間なし")
    if temporal_centroid is not None:
        print(f"Temporal Centroid (s): {temporal_centroid:.3f}")
    else:
        print("Temporal Centroid: 該当区間なし")

if __name__ == '__main__':
    main()
