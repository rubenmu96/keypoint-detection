from src.utils import load_model_inference
from src.inference import VideoPredictor
from config import dict_to_config
import json

def find_format(media):
    pass

def inference(media, folder):
    # check if .jpg, .png, .mp4 etc. then perform appropriate function aka auto detect file format
    # find .json and .pth from folder
    pass

model_name = "heatmap"
with open(f'{model_name}_config.json', 'r') as f:
    loaded_dict = json.load(f)

cfg = dict_to_config(loaded_dict)

model = load_model_inference("heatmap", cfg)

video_path = "examples/tennis_match.mp4"
load_model = "heatmap_pretrained_resnet34_672x448.pth"
detector = VideoPredictor(cfg, model, load_model)

detector.predict_video(
    detector=detector,
    video_path=video_path,
    output_path="output_video.mp4",
)