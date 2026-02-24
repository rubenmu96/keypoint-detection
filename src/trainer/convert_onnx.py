import os
import glob
import json

import torch

from config import dict_to_config
from src.utils import load_model_inference
from src.inference import load_fp16_model

def convert_onnx_fp32(folder):
    """Convert FP32 model to onnx"""
    folder = os.path.join(folder, "")

    cfg_path = glob.glob(f"{folder}*.json")[0]
    with open(cfg_path, 'r') as f:
        loaded_dict = json.load(f)
    cfg = dict_to_config(loaded_dict)

    load_model_path = glob.glob(f"{folder}*_fp32.pth")[0]
    model = load_model_inference(cfg.task, cfg)
    model = load_fp16_model(model, load_model_path, cfg.device)
    model = model.to("cpu")
    model.eval()

    dummy_input = torch.randn(1, 3, cfg.height, cfg.width)
    
    onnx_save_path = os.path.join(cfg.folder, f"{cfg.onnx_save_path}_fp32.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_save_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=[cfg.task],
        dynamic_axes={
            "input": {0: "batch_size"},
            cfg.task: {0: "batch_size"}
        }, dynamo=False
    )
    
    return True


def convert_onnx_fp16(folder):
    """Convert FP16 model to onnx"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping FP16 export")
        return
    
    cfg_path = glob.glob(f"{folder}*.json")[0]
    with open(cfg_path, 'r') as f:
        loaded_dict = json.load(f)
    cfg = dict_to_config(loaded_dict)
    
    # Try loading FP16 checkpoint first, fall back to FP32
    fp16_paths = glob.glob(f"{folder}*_fp16.pth")
    fp32_paths = glob.glob(f"{folder}*_fp32.pth")
    
    if fp16_paths:
        load_model_path = fp16_paths[0]
    else:
        load_model_path = fp32_paths[0]
    
    model = load_model_inference(cfg.task, cfg)
    model = load_fp16_model(model, load_model_path, cfg.device)
    model = model.half().cuda()
    model.eval()
    
    dummy_input = torch.randn(1, 3, cfg.height, cfg.width).half().cuda()
    
    onnx_save_path = os.path.join(cfg.folder, f"{cfg.onnx_save_path}_fp16.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_save_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=[cfg.task],
        dynamic_axes={
            "input": {0: "batch_size"},
            cfg.task: {0: "batch_size"}
        }, dynamo=False
    )