import requests
import time
import statistics
import psutil

# ============= FUNCTION 1: TIME BENCHMARK =============

def benchmark_time(num_images=10, width=512, height=512, steps=20):
    """Measure image generation time"""
    
    URL = "http://127.0.0.1:8188"
    times = []
    prompts = ["a landscape", "a cat", "a city", "abstract art", "a forest"]
    
    for i in range(num_images):
        workflow = {
            "3": {"inputs": {"seed": int(time.time()*1000)%(2**31), "steps": steps, "cfg": 8.0,
                  "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                  "model": ["4",0], "positive": ["6",0], "negative": ["7",0], "latent_image": ["5",0]},
                  "class_type": "KSampler"},
            "4": {"inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}, "class_type": "CheckpointLoaderSimple"},
            "5": {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"},
            "6": {"inputs": {"text": prompts[i%len(prompts)], "clip": ["4",1]}, "class_type": "CLIPTextEncode"},
            "7": {"inputs": {"text": "bad", "clip": ["4",1]}, "class_type": "CLIPTextEncode"},
            "8": {"inputs": {"samples": ["3",0], "vae": ["4",2]}, "class_type": "VAEDecode"},
            "9": {"inputs": {"filename_prefix": "bench", "images": ["8",0]}, "class_type": "SaveImage"}
        }
        
        start = time.time()
        r = requests.post(f"{URL}/prompt", json={"prompt": workflow})
        prompt_id = r.json()["prompt_id"]
        
        while prompt_id not in requests.get(f"{URL}/history/{prompt_id}").json():
            time.sleep(0.2)
        
        times.append(time.time() - start)
    
    return {
        "avg_time": statistics.mean(times),
        "min_time": min(times),
        "max_time": max(times),
        "total_time": sum(times)
    }


# ============= FUNCTION 2: RESOURCE MONITORING =============

def monitor_resources(duration=10):
    """Monitor CPU, RAM, and GPU usage for specified duration (seconds)"""
    
    samples = {"cpu": [], "ram": [], "gpu": []}
    gpu_available = False
    
    # Check for GPU (updated for nvidia-ml-py)
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName
        from pynvml import nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
        
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        gpu_name = nvmlDeviceGetName(handle)
        gpu_available = True
    except:
        gpu_name = None
    
    start = time.time()
    while time.time() - start < duration:
        # CPU and RAM
        samples["cpu"].append(psutil.cpu_percent(interval=0.1))
        samples["ram"].append(psutil.virtual_memory().percent)
        
        # GPU if available
        if gpu_available:
            try:
                from pynvml import nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
                util = nvmlDeviceGetUtilizationRates(handle).gpu
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                mem_percent = (mem_info.used / mem_info.total) * 100
                samples["gpu"].append({"usage": util, "memory": mem_percent})
            except:
                pass
        
        time.sleep(0.5)
    
    results = {
        "cpu": {"avg": statistics.mean(samples["cpu"]), "max": max(samples["cpu"])},
        "ram": {"avg": statistics.mean(samples["ram"]), "max": max(samples["ram"])}
    }
    
    if gpu_available and samples["gpu"]:
        gpu_usage = [s["usage"] for s in samples["gpu"]]
        gpu_memory = [s["memory"] for s in samples["gpu"]]
        results["gpu"] = {
            "name": gpu_name,
            "avg_usage": statistics.mean(gpu_usage),
            "max_usage": max(gpu_usage),
            "avg_memory": statistics.mean(gpu_memory),
            "max_memory": max(gpu_memory)
        }
    else:
        results["gpu"] = None
    
    return results


# ============= USAGE EXAMPLE =============

if __name__ == "__main__":
    # Test 1: Measure generation time
    print("1. Timing benchmark...")
    time_results = benchmark_time(10, 512, 512, 20)
    print(f"   Avg: {time_results['avg_time']:.2f}s")
    
    # Test 2: Monitor resources
    print("\n2. Resource monitoring (10 seconds)...")
    resource_results = monitor_resources(10)
    print(f"   CPU avg: {resource_results['cpu']['avg']:.1f}%")
    print(f"   RAM avg: {resource_results['ram']['avg']:.1f}%")
    if resource_results['gpu']:
        print(f"   GPU avg: {resource_results['gpu']['avg_usage']:.1f}%")
