from normalization import normalize_video_landmarks, NormalizationConfig

normalized = normalize_video_landmarks(
    video_path="data/with_audio.mp4",
    detector=detector,
    output_csv="output/normalization/landmarks.csv",
    config=NormalizationConfig(one_euro_beta=0.5),
)