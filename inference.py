import os
import torch
import json
import glob
import argparse

from config import dict_to_config
from src.inference import KeypointPredictor
from src.utils import (
    load_model_inference,
    get_file_type,
    predict_folder
)

def predictor(args, name, media, folder, use_amp=True):
    format = get_file_type(filename=media)

    cfg_path = glob.glob(f"{folder}*.json")[0]

    with open(cfg_path, 'r') as f:
        loaded_dict = json.load(f)

    cfg = dict_to_config(loaded_dict)

    onnx_model = None
    if use_amp:
        load_model = glob.glob(f"{folder}*_fp16.pth")[0]
        if args.use_onnx:
            print("Using FP16 onnx model")
            onnx_model = os.path.join(cfg.folder, f"{cfg.onnx_save_path}_fp16.onnx")
    else:
        load_model = glob.glob(f"{folder}*_fp32.pth")[0]
        if args.use_onnx:
            print("Using FP32 onnx model")
            onnx_model = os.path.join(cfg.folder, f"{cfg.onnx_save_path}_fp32.onnx")

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        cfg.device = "cpu"

    model = load_model_inference(name, cfg)

    print(f"Using {'FP16' if use_amp else 'FP32'} precision for {cfg.model_name}")
    
    predictor = KeypointPredictor(
        cfg=cfg,
        model=model,
        load_model=load_model,
        use_fp16=use_amp,
        onnx_path=onnx_model
    )

    if os.path.isdir(media):
        predict_folder(predictor, media, args.batch_size, output_dir=args.output_dir)
    else:
        format = get_file_type(filename=media)
        if format == "video":
            video_path = os.path.join(args.output_dir, "predictions.mp4")
            predictor.predict_video(media, video_path, limit_fps=True)
        elif format == "image":
            # TODO: save image in predictions folder
            keypoints = predictor.predict(media)
            predictor.draw_keypoints(media, keypoints, show=True)
        else:
            print("Please provide an image, video file, or folder.")



if __name__ == "__main__":
    """
    How to run:
    python inference.py --media PATH_TO_FOLDER/IMAGE/VIDEO --model_folder PATH_TO_MODEL [--fp32] [--use_onnx]

    python inference.py --media "examples/tennis_match_shortened.mp4" --model_folder "models/resnet18-hm/" --use_onnx
    python inference.py --media "examples/tennis_match_shortened.mp4" --model_folder "models/resnet18-hm/" --use_onnx --fp32
    python inference.py --media "dataset/sample_images/clay.jpg" --model_folder "models/resnet18-hm/" --use_onnx
    python inference.py --media "dataset/sample_images/" --model_folder "models/resnet18-hm/" --use_onnx
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="heatmap", help="resnet, heatmap or rcnn")
    parser.add_argument('--media', type=str, help="Path to media (image or video)")
    parser.add_argument('--model_folder', type=str, help="Path to model folder")
    parser.add_argument('--use_onnx', action='store_true', help="Use onnx") # maybe just --onnx?
    parser.add_argument('--fp32', action='store_true', help="Use FP32 instead of FP16")
    parser.add_argument('--output_dir', type=str, default='predictions/', help="Output direction of image, video, folder")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for folder prediction")
    args = parser.parse_args()

    use_amp = not args.fp32

    if not torch.cuda.is_available():
        use_amp = False
    
    predictor(args, args.name, args.media, args.model_folder, use_amp=use_amp)