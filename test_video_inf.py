from src.utils import get_model_and_config

from src.inference import VideoPredictor

model_name = "heatmap"
model, cfg = get_model_and_config(model_name)
video_path = "examples/tennis_match_shortened.mp4"
load_model = "sample_images/trial1/heatmap_pretrained_resnet18_672x448.pth"
detector = VideoPredictor(cfg, model, load_model)


# detector.batch_video_prediction(
#     detector=detector,
#     video_path=video_path,
#     output_path="output_video.mp4",
#     batch_size=16
# )

detector.predict_video(
    detector=detector,
    video_path=video_path,
    output_path="output_video.mp4",
)