"""Auto-detect GPU capacity and configure parallel workers.

Measures actual GPU memory per inference process, then calculates
optimal parallelism with a safety margin.
"""
import logging
import subprocess
import time

logger = logging.getLogger(__name__)

# Safety margin: reserve this fraction of total VRAM for OS/driver/overhead
VRAM_SAFETY_MARGIN = 0.15
# Minimum free VRAM (MB) to keep after allocating workers
VRAM_MIN_FREE_MB = 2048
# GPU compute contention factor: each additional process gets this fraction
# of single-process throughput (empirical: 2 processes ≈ 0.65x each)
COMPUTE_EFFICIENCY = [1.0, 0.65, 0.50, 0.42]


def get_gpu_info(gpu_id: int = 0) -> dict:
    """Query GPU memory and utilization via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             f"--id={gpu_id}",
             "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None

        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "name": parts[0],
            "total_mb": int(parts[1]),
            "used_mb": int(parts[2]),
            "free_mb": int(parts[3]),
            "utilization_pct": int(parts[4]),
        }
    except Exception as e:
        logger.warning(f"Failed to query GPU info: {e}")
        return None


def get_process_gpu_memory(gpu_id: int = 0) -> int:
    """Get current GPU memory usage by inference processes (MB)."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             f"--id={gpu_id}",
             "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return 0

        total = 0
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split(",")
                if len(parts) >= 2:
                    total += int(parts[1].strip())
        return total
    except Exception:
        return 0


def calculate_optimal_workers(
    gpu_id: int = 0,
    measured_per_process_mb: int = 0,
    max_workers: int = 4,
) -> dict:
    """
    Calculate optimal number of parallel workers for GPU inference.

    Args:
        gpu_id: GPU device index
        measured_per_process_mb: If known, the measured memory per process.
                                 If 0, uses current process memory or default estimate.
        max_workers: Hard cap on parallel workers

    Returns:
        dict with: workers, per_process_mb, total_mb, reasoning
    """
    info = get_gpu_info(gpu_id)
    if not info:
        return {
            "workers": 1,
            "per_process_mb": 0,
            "total_mb": 0,
            "reasoning": "Cannot query GPU, defaulting to 1 worker",
        }

    total_mb = info["total_mb"]

    # Determine per-process memory
    if measured_per_process_mb > 0:
        per_proc = measured_per_process_mb
    else:
        # Check if there's a running process to measure
        current_usage = get_process_gpu_memory(gpu_id)
        if current_usage > 0:
            per_proc = current_usage
        else:
            # Default estimate for MimicMotion SVD inference
            per_proc = 7000  # ~7GB based on observations

    # Calculate available memory for workers
    reserved = int(total_mb * VRAM_SAFETY_MARGIN)
    available = total_mb - reserved - VRAM_MIN_FREE_MB
    available = max(available, per_proc)  # at least 1 worker

    # How many workers fit in memory?
    mem_workers = min(available // per_proc, max_workers)
    mem_workers = max(mem_workers, 1)

    # Calculate effective throughput for each parallelism level
    best_workers = 1
    best_throughput = 1.0

    for n in range(1, mem_workers + 1):
        # Per-worker efficiency decreases with more workers
        if n - 1 < len(COMPUTE_EFFICIENCY):
            efficiency = COMPUTE_EFFICIENCY[n - 1]
        else:
            efficiency = COMPUTE_EFFICIENCY[-1] * 0.9  # diminishing returns

        throughput = n * efficiency
        if throughput > best_throughput:
            best_throughput = throughput
            best_workers = n

    reasoning = (
        f"GPU: {info['name']}, VRAM: {total_mb}MB total, "
        f"{per_proc}MB/process, {available}MB available. "
        f"Memory allows {mem_workers} workers, "
        f"optimal {best_workers} workers (throughput {best_throughput:.1f}x). "
        f"Efficiency per worker: {COMPUTE_EFFICIENCY[best_workers-1] if best_workers-1 < len(COMPUTE_EFFICIENCY) else '~0.4'}x"
    )

    return {
        "workers": best_workers,
        "per_process_mb": per_proc,
        "total_mb": total_mb,
        "available_mb": available,
        "mem_max_workers": mem_workers,
        "expected_throughput": round(best_throughput, 2),
        "reasoning": reasoning,
    }


def log_gpu_recommendation(gpu_id: int = 0, measured_per_process_mb: int = 0):
    """Log GPU auto-parallel recommendation."""
    rec = calculate_optimal_workers(gpu_id, measured_per_process_mb)
    logger.info(f"GPU auto-parallel: {rec['reasoning']}")
    return rec
