import torch
import time

@torch.no_grad()
def test_fp16_performance():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"Testing on: {props.name}")
    
    # Create a reasonably sized model for testing
    model_fp32 = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(128, 256, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(256, 17*2)  # 17 keypoints * 2 coordinates
    ).cuda()
    
    model_fp16 = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(128, 256, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(256, 17*2)
    ).cuda().half()  # Convert to FP16
    
    # Copy weights
    model_fp16.load_state_dict(model_fp32.state_dict())
    
    # Test input
    input_fp32 = torch.randn(1, 3, 256, 256).cuda()
    input_fp16 = input_fp32.half()
    
    # Warmup
    for _ in range(10):
        _ = model_fp32(input_fp32)
        _ = model_fp16(input_fp16)
    torch.cuda.synchronize()
    
    # Benchmark FP32
    num_runs = 500
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        _ = model_fp32(input_fp32)
    torch.cuda.synchronize()
    fp32_time = (time.time() - start) / num_runs * 1000
    
    # Benchmark FP16
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        _ = model_fp16(input_fp16)
    torch.cuda.synchronize()
    fp16_time = (time.time() - start) / num_runs * 1000
    
    # Benchmark with autocast
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            _ = model_fp32(input_fp32)
    torch.cuda.synchronize()
    autocast_time = (time.time() - start) / num_runs * 1000
    
    print(f"\nResults (average over {num_runs} runs):")
    print(f"FP32:           {fp32_time:.2f}ms ({1000/fp32_time:.1f} FPS)")
    print(f"FP16:           {fp16_time:.2f}ms ({1000/fp16_time:.1f} FPS)")
    print(f"FP32+Autocast:  {autocast_time:.2f}ms ({1000/autocast_time:.1f} FPS)")
    print(f"\nSpeedup:")
    print(f"FP16 vs FP32:        {fp32_time/fp16_time:.2f}x")
    print(f"Autocast vs FP32:    {fp32_time/autocast_time:.2f}x")
    
    if fp32_time/fp16_time < 1.2:
        print("\n⚠️  Low speedup detected. This might indicate:")
        print("- Model is too small to benefit from FP16")
        print("- Memory bandwidth bound workload")
        print("- Driver/CUDA issues")
    else:
        print(f"\n✅ Good speedup! Your {props.name} is working well with FP16")

if __name__ == "__main__":
    test_fp16_performance()