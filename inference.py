from src.utils import load_model_inference
from src.inference import VideoPredictor, ImagePredictor
from config import dict_to_config
import torch
import json
import glob
import os
import argparse

def get_file_type(filename):
    """Determine file type based on extension"""
    filename = filename.lower()
    
    image_extensions = {
        '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'
    }
    video_extensions = {
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'
    }
    
    ext = os.path.splitext(filename)[1]
    
    if ext in image_extensions:
        return 'image'
    elif ext in video_extensions:
        return 'video'
    else:
        return 'unknown'

def predictor(name, media, folder, use_amp=True):
    format = get_file_type(filename=media)

    if use_amp:
        load_model = glob.glob(f"{folder}*_fp16.pth")[0]
    else:
        load_model = glob.glob(f"{folder}*_fp32.pth")[0]
    cfg_path = glob.glob(f"{folder}*.json")[0]

    with open(cfg_path, 'r') as f:
        loaded_dict = json.load(f)

    cfg = dict_to_config(loaded_dict)
    model = load_model_inference(name, cfg)

    print(f"Using {'FP16' if use_amp else 'FP32'} precision for {cfg.model_name}")

    if format == "video":
        detector = VideoPredictor(cfg, model, load_model, use_fp16=use_amp)
        detector.predict_video(
            detector=detector,
            video_path=media,
            output_path="output_video.mp4",
        )
    elif format == "image":
        image_predictor = ImagePredictor(cfg, model, load_model, use_fp16=use_amp)
        keypoints = image_predictor.predict(media)
        image_predictor.draw_keypoints(media, keypoints)
    else:
        print("Please provide an image or video file.")


if __name__ == "__main__":
    """
    How to run:
    python inference.py --media PATH_TO_IMAGE/VIDEO --folder PATH_TO_MODEL [--fp32]

    Examples:
    python inference.py --media "examples/tennis_match_shortened.mp4" --folder "models/fp_16model/"
    python inference.py --media "examples/tennis_match_shortened.mp4" --folder "models/heatmap_model/" --fp32
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="heatmap", help="resnet, heatmap or rcnn")
    parser.add_argument('--media', type=str, help="Path to media (image or video)")
    parser.add_argument('--folder', type=str, help="Path to folder")
    parser.add_argument('--fp32', action='store_true', help="Use FP32 instead of FP16")
    args = parser.parse_args()

    use_amp = not args.fp32

    if not torch.cuda.is_available():
        use_amp = False
    
    predictor(args.name, args.media, args.folder, use_amp=use_amp)