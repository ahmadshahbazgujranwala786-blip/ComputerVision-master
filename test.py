# ============================================================
# T4 TEXT-TO-VIDEO BENCHMARK
# FastWan2.1-T2V-1.3B / FastVideo
# ============================================================

import os
import sys
import time
import json
import math
import gc
import platform
import subprocess
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_ID = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"

OUTPUT_DIR = Path("t4_video_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target workload
TARGET_DURATION = 20

# Start with a short clip.
# We benchmark this first, then extrapolate to 20 seconds.
FPS = 16
NUM_FRAMES = 81

# Model-friendly 480P configuration
WIDTH = 832
HEIGHT = 480

# FastWan distilled configuration
NUM_INFERENCE_STEPS = 3

SEED = 1234

PROMPT = """
A futuristic AI assistant helping a software engineer
in a modern office, cinematic lighting, realistic motion,
subtle camera movement, high quality
"""


# ============================================================
# 2. SYSTEM INFORMATION
# ============================================================

print("=" * 70)
print("T4 TEXT-TO-VIDEO BENCHMARK")
print("=" * 70)

print("\nSystem")
print("-" * 70)
print("Python:", sys.version)
print("Platform:", platform.platform())


# ============================================================
# 3. PYTORCH / CUDA CHECK
# ============================================================

try:
    import torch
except ImportError:
    print("\nPyTorch is not installed.")
    print("Install it first:")
    print("pip install torch")
    raise


print("\nCUDA")
print("-" * 70)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU was not detected. "
        "Make sure the notebook/runtime has a T4 GPU enabled."
    )


GPU_NAME = torch.cuda.get_device_name(0)

GPU_PROPERTIES = torch.cuda.get_device_properties(0)

VRAM_GB = GPU_PROPERTIES.total_memory / (1024 ** 3)

print("GPU:", GPU_NAME)
print(f"VRAM: {VRAM_GB:.2f} GB")
print(
    "CUDA capability:",
    f"{GPU_PROPERTIES.major}.{GPU_PROPERTIES.minor}"
)


# ============================================================
# 4. REQUIREMENT CHECK
# ============================================================

if "T4" not in GPU_NAME.upper():
    print("\nWARNING:")
    print("The detected GPU is not a Tesla T4.")
    print("This benchmark is intended for a T4.")


# ============================================================
# 5. GPU MEMORY HELPERS
# ============================================================

def gpu_memory():
    """
    Returns current GPU memory statistics.

    The function reports allocated and reserved CUDA memory
    so that VRAM consumption during video generation can be
    inspected and compared against the T4's available memory.
    """

    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)

    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
    }


def print_gpu_memory(label):
    """
    Prints current GPU memory usage.

    This helper is used before and after model loading and
    video generation to identify the VRAM footprint of the
    complete inference pipeline.
    """

    memory = gpu_memory()

    print(
        f"{label}: "
        f"allocated={memory['allocated_gb']:.2f} GB | "
        f"reserved={memory['reserved_gb']:.2f} GB"
    )


# ============================================================
# 6. INSTALL FASTVIDEO
# ============================================================

print("\nInstalling / updating FastVideo...")
print("-" * 70)

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-U",
        "fastvideo",
    ]
)

print("FastVideo installation completed.")


# ============================================================
# 7. IMPORT FASTVIDEO
# ============================================================

print("\nLoading FastVideo...")
print("-" * 70)

from fastvideo import VideoGenerator


# ============================================================
# 8. MODEL LOADING BENCHMARK
# ============================================================

print("\nLoading model...")
print("-" * 70)

torch.cuda.empty_cache()
gc.collect()

print_gpu_memory("Before model")

model_load_start = time.perf_counter()

generator = VideoGenerator.from_pretrained(
    MODEL_ID,
    num_gpus=1,
)

torch.cuda.synchronize()

model_load_time = time.perf_counter() - model_load_start

print_gpu_memory("After model")

print(
    f"\nModel loading time: "
    f"{model_load_time:.2f} seconds"
)


# ============================================================
# 9. GENERATION REQUEST
# ============================================================

request = {
    "prompt": PROMPT,

    "sampling": {
        "seed": SEED,
        "num_inference_steps": NUM_INFERENCE_STEPS,
    },

    "output": {
        "save_video": True,
        "fps": FPS,
        "width": WIDTH,
        "height": HEIGHT,
        "num_frames": NUM_FRAMES,
    },
}


# ============================================================
# 10. WARM-UP GENERATION
# ============================================================

print("\n")
print("=" * 70)
print("WARM-UP GENERATION")
print("=" * 70)

print(
    f"Resolution: {WIDTH}x{HEIGHT}"
)

print(
    f"Frames: {NUM_FRAMES}"
)

print(
    f"Duration: {NUM_FRAMES / FPS:.2f} seconds"
)

print(
    f"Inference steps: {NUM_INFERENCE_STEPS}"
)

print_gpu_memory("Before warm-up")


warmup_start = time.perf_counter()

warmup_result = generator.generate(request)

torch.cuda.synchronize()

warmup_time = time.perf_counter() - warmup_start

print_gpu_memory("After warm-up")

print(
    f"\nWarm-up time: "
    f"{warmup_time:.2f} seconds"
)

print(
    "\nWarm-up result:"
)

print(warmup_result)


# ============================================================
# 11. CLEAR CACHE BEFORE REAL BENCHMARK
# ============================================================

gc.collect()
torch.cuda.empty_cache()

print("\n")
print("=" * 70)
print("REAL BENCHMARK")
print("=" * 70)

print_gpu_memory("Before benchmark")


# ============================================================
# 12. ACTUAL GENERATION
# ============================================================

benchmark_start = time.perf_counter()

result = generator.generate(request)

torch.cuda.synchronize()

generation_time = time.perf_counter() - benchmark_start


# ============================================================
# 13. RESULTS
# ============================================================

print_gpu_memory("After benchmark")

print("\nGeneration completed.")

print(
    f"Generation time: "
    f"{generation_time:.2f} seconds"
)

print(
    f"Generated duration: "
    f"{NUM_FRAMES / FPS:.2f} seconds"
)


# ============================================================
# 14. LOCATE OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("OUTPUT")
print("=" * 70)

print(result)


# ============================================================
# 15. 20-SECOND ESTIMATION
# ============================================================

generated_duration = NUM_FRAMES / FPS

chunks_needed = math.ceil(
    TARGET_DURATION / generated_duration
)

estimated_generation_time = (
    chunks_needed * generation_time
)

print("\n")
print("=" * 70)
print("20-SECOND ESTIMATE")
print("=" * 70)

print(
    f"Benchmark clip duration: "
    f"{generated_duration:.2f} sec"
)

print(
    f"Target duration: "
    f"{TARGET_DURATION} sec"
)

print(
    f"Required chunks: "
    f"{chunks_needed}"
)

print(
    f"Measured generation time/chunk: "
    f"{generation_time:.2f} sec"
)

print(
    f"Estimated generation time for 20 sec: "
    f"{estimated_generation_time:.2f} sec"
)

print(
    f"Estimated generation time: "
    f"{estimated_generation_time / 60:.2f} minutes"
)


# ============================================================
# 16. 2-MINUTE TARGET CHECK
# ============================================================

TARGET_TIME = 120

print("\n")
print("=" * 70)
print("2-MINUTE TARGET")
print("=" * 70)

if estimated_generation_time <= TARGET_TIME:

    print(
        "PASS: Estimated 20-second generation "
        "is within the 2-minute target."
    )

    print(
        f"Time remaining: "
        f"{TARGET_TIME - estimated_generation_time:.2f} sec"
    )

else:

    print(
        "FAIL: Estimated 20-second generation "
        "exceeds the 2-minute target."
    )

    print(
        f"Over target by: "
        f"{estimated_generation_time - TARGET_TIME:.2f} sec"
    )


# ============================================================
# 17. THROUGHPUT
# ============================================================

video_seconds_per_second = (
    generated_duration / generation_time
)

print("\n")
print("=" * 70)
print("THROUGHPUT")
print("=" * 70)

print(
    f"Video generated per real second: "
    f"{video_seconds_per_second:.4f} sec/sec"
)

print(
    f"Video generation speed: "
    f"{video_seconds_per_second * 100:.2f}% realtime"
)


# ============================================================
# 18. BENCHMARK REPORT
# ============================================================

report = {
    "model": MODEL_ID,

    "gpu": GPU_NAME,

    "vram_gb": round(VRAM_GB, 2),

    "resolution": {
        "width": WIDTH,
        "height": HEIGHT,
    },

    "fps": FPS,

    "frames": NUM_FRAMES,

    "generated_duration_seconds": generated_duration,

    "inference_steps": NUM_INFERENCE_STEPS,

    "model_load_seconds": round(
        model_load_time,
        3
    ),

    "warmup_seconds": round(
        warmup_time,
        3
    ),

    "generation_seconds": round(
        generation_time,
        3
    ),

    "estimated_20_second_generation_seconds": round(
        estimated_generation_time,
        3
    ),

    "estimated_20_second_generation_minutes": round(
        estimated_generation_time / 60,
        3
    ),

    "two_minute_target": TARGET_TIME,

    "two_minute_target_pass": (
        estimated_generation_time <= TARGET_TIME
    ),

    "throughput_video_seconds_per_real_second": round(
        video_seconds_per_second,
        5
    ),
}


report_path = OUTPUT_DIR / "benchmark_report.json"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )


# ============================================================
# 19. FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL BENCHMARK REPORT")
print("=" * 70)

print(
    json.dumps(
        report,
        indent=4
    )
)

print("\nReport saved to:")
print(report_path)

print("\nBenchmark finished.")