"""Keypoint detection inference - image, video or folder."""
import os
import json
import glob
import argparse

import torch

from config import dict_to_config
from src.inference import KeypointPredictor
from src.utils import (
    load_model_inference,
    get_file_type,
    predict_folder
)

USAGE_EXAMPLE = """\
example:
  python inference.py --media "examples/tennis_match1.mp4" \\
      --model_folder "models/resnet18-hm/" --use_onnx
"""


def _find_one(pattern, description):
    """Return the first file matching pattern, or fail with a readable error."""
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No {description} found matching '{pattern}'")
    return matches[0]


def load_predictor(model_folder, *, use_onnx=False, use_amp=True) -> KeypointPredictor:
    """Build a KeypointPredictor from a trained model folder.

    The folder must contain the *_config.json written by main.py and a *_fp16.pth
    or *_fp32.pth checkpoint, plus the matching .onnx file when use_onnx is set.
    """
    folder = os.path.join(model_folder, "")
    with open(_find_one(f"{folder}*_config.json", "config file"), encoding="utf-8") as f:
        cfg = dict_to_config(json.load(f))

    # FP16 needs CUDA, so fall back to the FP32 weights on CPU.
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU with FP32.")
        cfg.device = "cpu"
        use_amp = False

    precision = "fp16" if use_amp else "fp32"
    load_model = _find_one(f"{folder}*_{precision}.pth", f"{precision.upper()} checkpoint")

    onnx_path = None
    if use_onnx:
        print(f"Using {precision.upper()} onnx model")
        # Look next to the checkpoints rather than in cfg.folder, which is the
        # training-time path and goes stale if the model folder is moved.
        onnx_path = os.path.join(folder, f"{cfg.onnx_save_path}_{precision}.onnx")

    model = load_model_inference(cfg.task, cfg)
    return KeypointPredictor(
        cfg=cfg,
        model=model,
        load_model=load_model,
        use_fp16=use_amp,
        onnx_path=onnx_path
    )


def predictor(
        media, model_folder, *, output_dir="predictions/", use_onnx=False,
        use_amp=True, batch_size=4, limit_fps=True, show=True
    ) -> None:
    """Run keypoint detection on an image, a video or a folder of images.

    Results are saved to output_dir. limit_fps plays a video back at its native
    frame rate instead of as fast as inference allows. show opens a window with
    the result; folders are always saved without being displayed.
    """
    kps_predictor = load_predictor(model_folder, use_onnx=use_onnx, use_amp=use_amp)

    if os.path.isdir(media):
        predict_folder(kps_predictor, media, batch_size, output_dir=output_dir)
        return

    media_format = get_file_type(filename=media)
    if media_format == "video":
        # cv2.VideoWriter writes nothing, without raising, if the directory is missing.
        os.makedirs(output_dir, exist_ok=True)
        video_path = os.path.join(output_dir, "predictions.mp4")
        kps_predictor.predict_video(media, video_path, limit_fps=limit_fps, show=show)
    elif media_format == "image":
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(media))[0]
        output_path = os.path.join(output_dir, f"{filename}.png")
        keypoints = kps_predictor.predict(media)
        kps_predictor.draw_keypoints(media, keypoints, save_path=output_path, show=show)
    else:
        print("Please provide an image, video file, or folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=USAGE_EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--media', type=str, required=True,
        help="Path to an image, a video or a folder of images"
    )
    parser.add_argument(
        '--model_folder', type=str, required=True,
        help="Path to the trained model folder"
    )
    parser.add_argument('--use_onnx', action='store_true', help="Use onnx")
    parser.add_argument('--fp32', action='store_true', help="Use FP32 instead of FP16")
    parser.add_argument(
        '--output_dir', type=str, default='predictions/',
        help="Output directory for images, video and folders"
    )
    parser.add_argument(
        '--batch_size', type=int, default=4, help="Batch size for folder prediction"
    )
    parser.add_argument(
        '--no_limit_fps', action='store_true',
        help="Process video as fast as possible instead of at its native frame rate"
    )
    parser.add_argument(
        '--no_show', action='store_true',
        help="Save results without opening a display window"
    )
    arguments = parser.parse_args()

    predictor(
        arguments.media,
        arguments.model_folder,
        output_dir=arguments.output_dir,
        use_onnx=arguments.use_onnx,
        use_amp=not arguments.fp32,
        batch_size=arguments.batch_size,
        limit_fps=not arguments.no_limit_fps,
        show=not arguments.no_show,
    )
