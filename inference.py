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

def predictor(args, media, folder, use_amp=True):
    folder = os.path.join(folder, "") # will put "/" at the end if missing
    # Match only *_config.json to avoid picking up other JSON files (e.g. results).
    # Model type is determined from cfg.task saved in the config, no --name needed.
    cfg_path = glob.glob(f"{folder}*_config.json")[0]

    with open(cfg_path, 'r') as f:
        loaded_dict = json.load(f)

    cfg = dict_to_config(loaded_dict)

    # Loading model, onnx if --use_onnx, FP16 if use_amp.
    onnx_model = None
    if use_amp:
        load_model = glob.glob(f"{folder}*_fp16.pth")[0]
        if args.use_onnx:
            print("Using FP16 onnx model")
            onnx_model = os.path.join(
                cfg.folder, f"{cfg.onnx_save_path}_fp16.onnx"
            )
    else:
        load_model = glob.glob(f"{folder}*_fp32.pth")[0]
        if args.use_onnx:
            print("Using FP32 onnx model")
            onnx_model = os.path.join(
                cfg.folder, f"{cfg.onnx_save_path}_fp32.onnx"
            )

    # Swaps to CPU if CUDA does not exist.
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        cfg.device = "cpu"

    # Get model and inference class
    model = load_model_inference(cfg.task, cfg)
    predictor = KeypointPredictor(
        cfg=cfg,
        model=model,
        load_model=load_model,
        use_fp16=use_amp,
        onnx_path=onnx_model
    )

    # Run detection - supporting images, folders, and video
    if os.path.isdir(media):
        predict_folder(predictor, media, args.batch_size, output_dir=args.output_dir)
    else:
        format = get_file_type(filename=media)
        if format == "video":
            video_path = os.path.join(args.output_dir, "predictions.mp4")
            predictor.predict_video(media, video_path, limit_fps=True)
        elif format == "image":
            os.makedirs(args.output_dir, exist_ok=True)
            filename = os.path.splitext(os.path.basename(media))[0]
            output_path = os.path.join(args.output_dir, f"{filename}.png")
            keypoints = predictor.predict(media)
            predictor.draw_keypoints(media, keypoints, save_path=output_path, show=True)
        else:
            print("Please provide an image, video file, or folder.")


if __name__ == "__main__":
    """
    How to run:
    python inference.py --media PATH_TO_FOLDER/IMAGE/VIDEO --model_folder PATH_TO_MODEL [--fp32] [--use_onnx]

    python inference.py --media "examples/tennis_match1.mp4" --model_folder "models/resnet18-hm/" --use_onnx
    python inference.py --media "examples/tennis_match1.mp4" --model_folder "models/resnet18-hm/" --use_onnx --fp32
    python inference.py --media "examples/test-images/clay.jpg" --model_folder "models/resnet18-hm/" --use_onnx
    python inference.py --media "examples/test-images/" --model_folder "models/resnet18-hm/" --use_onnx

    python inference.py --media "examples/tennis_match1.mp4" --model_folder "models/keypoint-rcnn/" --fp32
    python inference.py --media "examples/test-images" --model_folder "models/keypoint-rcnn" --fp32
    """
    parser = argparse.ArgumentParser()
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
    
    predictor(args, args.media, args.model_folder, use_amp=use_amp)